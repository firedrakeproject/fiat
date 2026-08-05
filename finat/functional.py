r"""Symbolic representation of degrees of freedom.

A :class:`PhysicallyMappedFunctional` represents a degree of freedom in the form

.. math:: \\ell(f) = \\sum_q w_q \\langle D, \\nabla^m f(x_q) \\rangle,

where the points :math:`x_q` and quadrature/moment weights :math:`w_q`
are numeric, and the direction tensor :math:`D` (of rank equal to the
derivative order :math:`m`) may be numeric, for functionals defined on
the reference cell, or a GEM expression, for functionals carrying
physical geometry.

This representation is the foundation for automating the transformation
theory of Kirby (2017): degrees of freedom of any FIAT element are
converted to this common form directly from their point and derivative
dictionaries, with the derivative direction recovered numerically, so
that no dispatch over FIAT functional types is required.

The physical counterpart of a reference functional is assumed to share
its points and weights: integral moments must be measure-intrinsic
(e.g. integral averages), following the reference node convention of
Brubeck & Kirby (2025).
"""

from functools import reduce
from math import factorial, prod
from operator import mul

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.polynomial_set import mis
from FIAT.functional import Functional as FIATFunctional
from gem import Node, Zero


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
        either numeric or a GEM expression; ``None`` for ``order == 0``.
    divergence :
        Whether this is a divergence functional: the trace, at each
        point, of the tensor pairing an order-1 derivative multi-index
        with a rank-1 value component.  Unlike an ordinary derivative,
        this does not have a single ``direction``, since the trace
        pattern spans the full derivative and value index ranges; it is
        recovered and handled as its own case because it commutes with
        the (contravariant) Piola pullback up to the Jacobian
        determinant, independently of the entity it sits on.
    mapping :
        The FIAT mapping string of the basis functions this functional
        is dual to (``"affine"``, ``"contravariant piola"``, ...): the
        type tag of the value slots, deciding which matrix each value
        slot of the weight tensor is contracted with under push-forward,
        exactly as the derivative slots of ``direction`` are contracted
        with the Jacobian.

    """

    def __init__(self, points: tuple, weights: numpy.ndarray,
                 order: int = 0, direction=None, rank: int = 0,
                 divergence: bool = False, mapping: str = "affine"):
        self.points = points
        self.weights = weights
        self.order = order
        self.direction = direction
        self.rank = rank
        self.divergence = divergence
        self.mapping = mapping

    @classmethod
    def from_fiat(cls, node: FIATFunctional, tol: float = 1e-12,
                  mapping: str = "affine") -> "PhysicallyMappedFunctional":
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
        mapping :
            The FIAT mapping string of the basis functions this
            functional is dual to.

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
                return cls(points, weights, mapping=mapping)
            # value weight profile: one row of component weights per point
            sd = node.ref_el.get_spatial_dimension()
            weights = numpy.zeros((len(points), sd**rank))
            shape = (sd,) * rank
            for q, pt in enumerate(points):
                for w, comp in node.pt_dict[pt]:
                    weights[q, numpy.ravel_multi_index(comp, shape)] += w
            return cls(points, weights, rank=rank, mapping=mapping)

        sd = node.ref_el.get_spatial_dimension()
        order = node.max_deriv_order
        alphas = multiindices(sd, order)
        lookup = {alpha: k for k, alpha in enumerate(alphas)}

        points = tuple(node.deriv_dict)
        has_comp = any(comp != tuple() for pt in points
                       for w, alpha, comp in node.deriv_dict[pt])
        if has_comp:
            # A divergence: at each point, the weights pairing an
            # order-1 derivative multi-index with a rank-1 value
            # component must form a scalar multiple of the trace.
            if order != 1:
                raise NotImplementedError(
                    f"{type(node).__name__} has vector components.")
            weights = numpy.zeros(len(points))
            for q, pt in enumerate(points):
                Wq = numpy.zeros((sd, sd))
                for w, alpha, comp in node.deriv_dict[pt]:
                    if len(comp) != 1:
                        raise NotImplementedError(
                            f"{type(node).__name__} has vector components.")
                    Wq[lookup[tuple(alpha)], comp[0]] += w
                if not numpy.allclose(Wq, Wq[0, 0] * numpy.eye(sd), atol=tol):
                    raise NotImplementedError(
                        f"{type(node).__name__} is not a divergence functional.")
                weights[q] = Wq[0, 0]
            return cls(points, weights, order=order, divergence=True,
                       mapping=mapping)

        W = numpy.zeros((len(points), len(alphas)))
        for q, pt in enumerate(points):
            for w, alpha, comp in node.deriv_dict[pt]:
                W[q, lookup[tuple(alpha)]] += w

        # Factor the weights as a common direction times scalar weights
        u, s, vt = numpy.linalg.svd(W)
        if any(s[1:] > tol * s[0]):
            raise NotImplementedError(
                f"{type(node).__name__} has no common derivative direction.")
        direction = vt[0]
        weights = u[:, 0] * s[0]
        return cls(points, weights, order=order, direction=direction,
                   mapping=mapping)

    def with_direction(self, direction) -> "PhysicallyMappedFunctional":
        """Return the same functional with another direction tensor."""
        return type(self)(self.points, self.weights,
                          order=self.order, direction=direction,
                          mapping=self.mapping)

    def pullback(self, A) -> "PhysicallyMappedFunctional":
        r"""Contract each derivative slot of the direction with a matrix.

        By the chain rule, reference derivatives of a pullback are
        physical derivatives contracted with the Jacobian, so viewing
        this reference functional as acting on physical functions
        amounts to contracting each slot of the direction tensor with
        the cell Jacobian; more general per-slot matrices arise when
        the direction is instead mapped through a physical frame.

        Parameters
        ----------
        A :
            The per-slot matrix: a GEM expression of shape ``(sd, sd)``,
            or an ``(sd, sd)`` array with numeric or GEM scalar entries.

        Returns
        -------
        PhysicallyMappedFunctional
            The functional with direction :math:`A \\otimes \\dots
            \\otimes A : D`, collapsed back onto multi-index
            coefficients.

        """
        if self.order == 0:
            return self
        if isinstance(A, Node):
            sd = A.shape[0]
            A = numpy.array([[A[i, k] for k in range(sd)] for i in range(sd)],
                            dtype=object)
        A = numpy.asarray(A)
        sd = A.shape[0]
        symbolic = A.dtype == object
        alphas = multiindices(sd, self.order)
        lookup = {alpha: k for k, alpha in enumerate(alphas)}

        direction = numpy.full(len(alphas), Zero() if symbolic else 0.0,
                               dtype=object if symbolic else float)
        shape = (sd,) * self.order
        for index in numpy.ndindex(shape):
            # Distribute the multi-index coefficients over a symmetric tensor
            alpha = _index_alpha(index, sd)
            scale = prod(map(factorial, alpha)) / factorial(self.order)
            d = self.direction[lookup[alpha]] * scale
            if not isinstance(d, Node) and d == 0:
                continue
            # Contract each slot with A, collapsing onto multi-indices
            for target in numpy.ndindex(shape):
                term = reduce(mul, (A[target[k], index[k]]
                                    for k in range(self.order)), d)
                m = lookup[_index_alpha(target, sd)]
                direction[m] = direction[m] + term
        return self.with_direction(direction)

    def evaluate(self, fiat_element: FiniteElement) -> numpy.ndarray:
        r"""Apply this functional to the nodal basis of a FIAT element.

        This is the generalized Vandermonde computation: the restriction
        of a functional :math:`\\ell` to the polynomial space satisfies
        :math:`\\pi \\ell = \\sum_j \\ell(\\psi_j)\\, \\pi n_j`.  Only
        valid for functionals with numeric direction.

        Parameters
        ----------
        fiat_element :
            The FIAT element providing the nodal basis.

        Returns
        -------
        numpy.ndarray
            The vector of values of this functional on the nodal basis.

        """
        sd = fiat_element.get_reference_element().get_spatial_dimension()
        tab = fiat_element.tabulate(self.order, self.points)
        if self.rank > 0:
            T = tab[(0,) * sd]
            T = T.reshape(T.shape[0], -1, len(self.points))
            return numpy.einsum("jcq,qc->j", T, self.weights)
        if self.order == 0:
            return tab[(0,) * sd] @ self.weights
        alphas = multiindices(sd, self.order)
        return self.direction @ numpy.array([tab[alpha] @ self.weights
                                             for alpha in alphas])


def _index_alpha(index: tuple, sd: int) -> tuple:
    """Convert a tensor index into a derivative multi-index."""
    alpha = [0] * sd
    for k in index:
        alpha[k] += 1
    return tuple(alpha)
