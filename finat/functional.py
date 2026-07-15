r"""Symbolic representation of degrees of freedom.

A :class:`PhysicallyMappedFunctional` represents a degree of freedom in the form

.. math:: \\ell(f) = \\sum_q w_q \\langle D, \\nabla^m f(x_q) \\rangle,

where the points :math:`x_q` and quadrature/moment weights :math:`w_q`
are numeric, and the direction tensor :math:`D` (of rank equal to the
derivative order :math:`m`) may be numeric, for functionals defined on
the reference cell, or a GEM expression carrying physical geometry.

This representation is the foundation for automating the transformation
theory of Kirby (2017): degrees of freedom of any FIAT element are
converted to this common form directly from their point and derivative
dictionaries, with the derivative direction recovered numerically, so
that no dispatch over FIAT functional types is required.

:meth:`PhysicallyMappedFunctional.evaluate` computes one row of
:math:`B_{ij} = \\ell_i(\\hat\\psi_j)`, the generalized Vandermonde
matrix of a physical node :math:`\\ell_i` against the reference nodal
basis :math:`\\hat\\psi_j`, transplanted to physical space by the
affine cell map :math:`x = J\\hat{x} + v_0`.  Physical derivatives of
the transplanted basis are the tabulated reference ones, recombined by
the chain rule, :math:`\\nabla_x^m(\\hat\\psi_j\\circ F) =
(J^{-1})^{\\otimes m} : \\hat\\nabla^m\\hat\\psi_j`.  Since :math:`B`
relates physical nodes to reference basis functions, the transformation
matrix :math:`V = B^{-1}` still needs assembling from these rows; that
inversion, exploiting the block structure of :math:`B` by topological
entity, is done by the caller (:mod:`finat.zany`), not by this class.

The physical counterpart of a reference functional is assumed to share
its points and weights: integral moments must be measure-intrinsic
(e.g. integral averages), following the reference node convention of
Brubeck & Kirby (2025).
"""

from math import factorial, prod

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.polynomial_set import mis
from FIAT.functional import Functional as FIATFunctional
from gem import Node, Zero
from finat.physically_mapped import adjugate, determinant


def multiindices(sd: int, order: int) -> list:
    """Multi-indices of a given order, with axis ordering for order 1."""
    return sorted(mis(sd, order), reverse=True)


class PhysicallyMappedFunctional:
    """Symbolic degree of freedom with a single derivative direction.

    Parameters
    ----------
    points :
        Tuple of reference-cell points.
    weights :
        Numeric weight for each point.
    order :
        The derivative order :math:`m`.
    direction :
        For ``order > 0``, the direction tensor of rank ``order``,
        either numeric (a reference-cell functional) or a GEM
        expression (a physical functional); ``None`` for ``order == 0``.
    rank :
        Value rank, for vector- or tensor-valued point evaluations.
    J :
        GEM expression for the cell Jacobian, set once ``direction``
        carries physical geometry; used by :meth:`evaluate` to convert
        physical derivatives to reference ones by the chain rule.

    """

    def __init__(self, points: tuple, weights: numpy.ndarray,
                 order: int = 0, direction=None, rank: int = 0, J: Node = None):
        self.points = points
        self.weights = weights
        self.order = order
        self.direction = direction
        self.rank = rank
        self.J = J

    @classmethod
    def from_fiat(cls, node: FIATFunctional, tol: float = 1e-12) -> "PhysicallyMappedFunctional":
        """Construct a symbolic PhysicallyMappedFunctional from a FIAT functional.

        The construction only inspects the point and derivative
        dictionaries: the derivative order and the (common) direction of
        differentiation are recovered numerically by factorizing the
        matrix of derivative weights.

        Parameters
        ----------
        node :
            The FIAT functional.
        tol :
            Relative tolerance for the rank-one factorization of the
            derivative weights.

        Returns
        -------
        PhysicallyMappedFunctional
            The symbolic representation of the FIAT functional.

        """
        if node.pt_dict and node.deriv_dict:
            raise NotImplementedError(
                f"{type(node).__name__} mixes value and derivative weights.")

        if not node.deriv_dict:
            points = tuple(node.pt_dict)
            comps = {comp for pt in points for w, comp in node.pt_dict[pt]}
            rank = len(max(comps))
            if rank == 0:
                weights = numpy.asarray([w for pt in points
                                         for w, comp in node.pt_dict[pt]])
                return cls(points, weights)
            # value weight profile: one row of component weights per point
            sd = node.ref_el.get_spatial_dimension()
            weights = numpy.zeros((len(points), sd**rank))
            shape = (sd,) * rank
            for q, pt in enumerate(points):
                for w, comp in node.pt_dict[pt]:
                    weights[q, numpy.ravel_multi_index(comp, shape)] += w
            return cls(points, weights, rank=rank)

        sd = node.ref_el.get_spatial_dimension()
        order = node.max_deriv_order
        alphas = multiindices(sd, order)
        lookup = {alpha: k for k, alpha in enumerate(alphas)}

        points = tuple(node.deriv_dict)
        W = numpy.zeros((len(points), len(alphas)))
        for q, pt in enumerate(points):
            for w, alpha, comp in node.deriv_dict[pt]:
                if comp != tuple():
                    raise NotImplementedError(
                        f"{type(node).__name__} has vector components.")
                W[q, lookup[tuple(alpha)]] += w

        # Factor the weights as a common direction times scalar weights
        u, s, vt = numpy.linalg.svd(W)
        if any(s[1:] > tol * s[0]):
            raise NotImplementedError(
                f"{type(node).__name__} has no common derivative direction.")
        direction = vt[0]
        weights = u[:, 0] * s[0]
        return cls(points, weights, order=order, direction=direction)

    def with_direction(self, direction, J: Node = None) -> "PhysicallyMappedFunctional":
        """Return the same functional with another direction tensor.

        Parameters
        ----------
        direction :
            The new direction tensor.
        J :
            GEM expression for the cell Jacobian, stored for use by
            :meth:`evaluate` when ``direction`` carries physical
            geometry.

        """
        return type(self)(self.points, self.weights,
                          order=self.order, direction=direction,
                          rank=self.rank, J=J)

    def evaluate(self, fiat_element: FiniteElement) -> numpy.ndarray:
        r"""Apply this functional to the nodal basis of a FIAT element.

        This is the generalized Vandermonde computation: the restriction
        of a functional :math:`\\ell` to the polynomial space satisfies
        :math:`\\pi \\ell = \\sum_j \\ell(\\psi_j)\\, \\pi n_j`, i.e. a row
        of :math:`B_{ij} = \\ell_i(\\hat\\psi_j)`.  When :attr:`direction`
        carries physical geometry (:attr:`J` is set), the reference
        nodal basis :math:`\\hat\\psi_j` is understood as transplanted to
        physical space by the affine cell map, so its physical
        derivative tensor is the tabulated reference one, recombined by
        the chain rule, :math:`\\nabla_x^m(\\hat\\psi_j\\circ F) =
        (J^{-1})^{\\otimes m} : \\hat\\nabla^m\\hat\\psi_j`.  Rather than
        transforming :attr:`direction`, this recombination is applied to
        the tabulation itself, and contracted against the weighted
        direction tensor :math:`W_q = w_q D` at every point, unmodified.
        Only the basis functions with a nonzero tabulated derivative at
        these points are carried through the contraction: for a nodal
        basis this is typically a small fraction of the full row.

        Parameters
        ----------
        fiat_element :
            The FIAT element providing the nodal basis.

        Returns
        -------
        numpy.ndarray
            The vector of values of this functional on the nodal basis.

        """
        tol = 1E-12
        sd = fiat_element.get_reference_element().get_spatial_dimension()
        tab = fiat_element.tabulate(self.order, self.points)
        if self.rank > 0:
            if self.order > 0:
                raise NotImplementedError(
                    "A derivative of a Piola-mapped value is not yet supported.")
            # The reference nodal basis is transplanted to physical
            # space by the element's own pullback (contracting each
            # value axis with J^{-T} or J/det J, per fiat_element.
            # mapping()); :attr:`weights` carries the *test* function's
            # own, separately built, physical geometry (see
            # :meth:`finat.zany.ZanyPhysicallyMappedElement._physical_weights`).
            raw = tab[(0,) * sd]  # shape (ndof,) + value_shape + (npoints,)
            raw[abs(raw) < tol] = 0
            support = numpy.flatnonzero(numpy.any(raw, axis=tuple(range(1, raw.ndim))))

            result = numpy.full(raw.shape[0], Zero(), dtype=object)
            if len(support) == 0:
                return result
            T = raw[support]
            if self.J is not None:
                T = _pullback_values(T, fiat_element.mapping()[0], self.J, sd)
            T = T.reshape(T.shape[0], -1, len(self.points))
            result[support] = numpy.einsum("jcq,qc->j", T, self.weights)
            return result

        if self.order == 0:
            raw = tab[(0,) * sd]  # shape (ndof,) + value_shape + (npoints,)
            raw[abs(raw) < tol] = 0
            support = numpy.flatnonzero(numpy.any(raw, axis=tuple(range(1, raw.ndim))))
            T = raw[support]
            Tw = T @ self.weights

            result = numpy.full(raw.shape[0], Zero(), dtype=object)
            result[support] = Tw
            return result

        order = self.order
        shape = (sd,) * order
        alphas = multiindices(sd, order)

        # A nodal basis is mostly zero at any given handful of points:
        # only contract the dofs with some nonzero tabulated derivative
        # here, following the sparsity pattern of Walkington's V.
        raw = numpy.array([tab[alpha] for alpha in alphas])  # (nalpha, ndof, npoints)
        raw[abs(raw) < tol] = 0
        support = numpy.flatnonzero(numpy.any(raw, axis=(0, 2)))
        result = numpy.full(raw.shape[1], Zero(), dtype=object)
        if len(support) == 0:
            return result

        # Full (uncompressed) reference derivative tensor, restricted to
        # the support: tab[alpha] repeated at every index ordering of its
        # multi-index, one per surviving basis function and point.
        Tab = numpy.full((len(support), len(self.points)) + shape, Zero())
        for index in numpy.ndindex(shape):
            alpha = _index_alpha(index, sd)
            Tab[(..., *index)] = tab[alpha][support]

        if self.J is not None:
            Jnp = numpy.array([[self.J[i, k] for k in range(sd)] for i in range(sd)],
                              dtype=object)
            Jinv = adjugate(Jnp) / determinant(Jnp)
            Tab = Tab.astype(object)
            # Contract each tensor slot with J^{-1} in turn (chain rule).
            # Contracting the axis right after the (support, npoints) prefix
            # cycles a not-yet-contracted slot into that position each
            # time, mirroring how the untransformed slots collapse in an
            # unbatched symmetric-tensor contraction.
            batch = Tab.ndim - order
            for _ in range(order):
                Tab = numpy.tensordot(Tab, Jinv, axes=(batch, 0))

        # Full direction tensor (including the multiplicity of each
        # multi-index), weighted at every point: W_q = w_q D.
        lookup = {alpha: k for k, alpha in enumerate(alphas)}
        D = numpy.empty(shape, dtype=object)
        for index in numpy.ndindex(shape):
            alpha = _index_alpha(index, sd)
            scale = prod(map(factorial, alpha)) / factorial(order)
            D[index] = self.direction[lookup[alpha]] * scale
        W = numpy.tensordot(self.weights, D, axes=0)

        result[support] = numpy.tensordot(Tab, W, axes=(tuple(range(1, order + 2)),
                                                        tuple(range(order + 1))))

        return result


def _index_alpha(index: tuple, sd: int) -> tuple:
    """Convert a tensor index into a derivative multi-index."""
    alpha = [0] * sd
    for k in index:
        alpha[k] += 1
    return tuple(alpha)


#: Value-axis pullback of each FIAT mapping, as the sequence of per-axis
#: pullback codes used by :func:`FIAT.macro.pullback`: 1 for a covariant
#: axis (contracted with :math:`J^{-T}`), 2 for a contravariant one
#: (contracted with :math:`J/\det J`).
_PULLBACK_FORMDEGREE = {
    "affine": (),
    "covariant piola": (1,),
    "contravariant piola": (2,),
    "double covariant piola": (1, 1),
    "double contravariant piola": (2, 2),
    "covariant contravariant piola": (1, 2),
    "contravariant covariant piola": (2, 1),
}


def _pullback_values(T: numpy.ndarray, mapping: str, J: Node, sd: int) -> numpy.ndarray:
    r"""Map the value axes of a reference tabulation to physical space.

    Generalizes :func:`FIAT.macro.pullback` to a symbolic Jacobian: each
    value axis of ``T`` is contracted in turn with :math:`J^{-T}`
    (covariant) or :math:`J/\det J` (contravariant), following
    ``mapping``, by temporarily swapping that axis with the last one.

    :arg T: Tabulation, shape ``(ndof,) + value_shape + (npoints,)``.
    :arg mapping: A FIAT mapping string, e.g. ``"contravariant piola"``.
    :arg J: GEM expression for the cell Jacobian.
    :arg sd: Spatial dimension of the cell.
    :returns: The physical-space tabulation, the same shape as ``T``.
    """
    try:
        formdegree = _PULLBACK_FORMDEGREE[mapping]
    except KeyError:
        raise NotImplementedError(f"Unrecognized mapping {mapping!r}.")
    if not formdegree:
        return T

    Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)], dtype=object)
    Jinv = adjugate(Jnp) / determinant(Jnp)
    K = Jnp / determinant(Jnp)
    pullback_matrix = {1: Jinv.T, 2: K}

    T = T.astype(object)
    ndim = T.ndim
    for i, k in enumerate(formdegree):
        # The batch axis (dofs) is axis 0; the value axes immediately
        # follow it, before the trailing points axis.
        axis = i + 1
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        T = T.transpose(perm)
        T = numpy.tensordot(T, pullback_matrix[k], axes=(-1, 1))
        T = T.transpose(perm)
    return T
