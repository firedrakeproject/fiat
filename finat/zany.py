r"""Automatic basis transformations for physically mapped elements.

This module automates the transformation theory of Kirby (2017),
Aznaran-Kirby-Farrell (2022), and Brubeck & Kirby (2025).  Given a FIAT
element whose degrees of freedom are not preserved under push-forward
-- Morley, Hermite, Argyris, Bell (affine pullback), and Mardal-Tai-
Winther, Johnson-Mercier, Guzman-Neilan ((double) contravariant Piola
pullback) -- :class:`ZanyPhysicallyMappedElement` derives the basis
transformation directly from the FIAT dual basis, with a single
mechanism regardless of the pullback:

* Each physical node's row of the generalized Vandermonde matrix
  :math:`B_{ij} = \ell_i(\hat\psi_j)` -- its evaluation against the
  *reference* nodal basis :math:`\hat\psi_j`, left untransplanted --
  is computed directly by
  :meth:`finat.functional.PhysicallyMappedFunctional.evaluate`, with no
  geometric frame required there.  All of the geometry instead lives in
  the physical node itself, built once from the reference node's own
  recipe: a scalar derivative node's direction is either a fixed
  Cartesian tensor (vertex/interior derivative jets) or the cofactor
  image of the reference facet normal (facet derivative dofs), built by
  :meth:`_physical_direction`; a Piola-mapped value node's weight
  profile is pushed forward by :meth:`_physical_weights` instead --
  the cofactor away from a facet, or a normal/tangential frame
  decomposition on one, with the tangential part carrying the
  contragredient correction of the reciprocal basis FIAT builds it on.
* But :math:`B` relates physical nodes to reference basis functions,
  not to reference nodes, so the actual basis transformation
  :math:`V = B^{-1}` still has to be assembled: a physical node's row
  of :math:`B` is nonzero only on its own entity and on entities of
  strictly lower topological dimension, so entities are visited in
  increasing dimension and each one's small diagonal block of
  :math:`B` is inverted to eliminate the already-known contribution of
  the lower-dimensional ones.

Degrees of freedom are represented symbolically by
:class:`finat.functional.PhysicallyMappedFunctional` and processed
generically, without dispatching over FIAT functional types or over
the kind of pullback.
"""

from functools import reduce
from operator import add

import numpy

from gem import Literal, ListTensor, Zero, as_gem
from finat.functional import PhysicallyMappedFunctional
from finat.physically_mapped import PhysicallyMappedElement, adjugate


class ZanyPhysicallyMappedElement(PhysicallyMappedElement):
    r"""Mixin deriving the basis transformation of a FIAT element whose
    dual basis is not preserved under push-forward, for either an
    affine (scalar) or a (double) contravariant Piola pullback.

    Every reference node is, by construction, one of: a value moment
    (push-forward invariant, whether scalar or Piola-mapped), a scalar
    directional-derivative moment along a vertex/interior Cartesian
    axis (Hermite/Argyris/Bell jets) or a facet normal (Morley/Argyris
    /Bell facet dofs), or a Piola-mapped value moment (Mardal-Tai-
    Winther, Johnson-Mercier, Guzman-Neilan).  The corresponding
    physical node reuses the same points, with only its direction
    (scalar case, :meth:`_physical_direction`) or weight profile (Piola
    case, :meth:`_physical_weights`) replaced by their physical
    counterpart; the reference nodal basis itself is never transplanted.
    Evaluating that physical node against the *full* reference nodal
    basis (no geometric frame needed there) gives a row of :math:`B_{ij}
    = \ell_i(\hat\psi_j)`, the generalized Vandermonde matrix; but
    :math:`V = B^{-1}` is what relates the reference nodes to the
    push-forward of the physical ones, so :math:`B` still needs
    inverting.  By construction, a physical node's row of :math:`B` is
    nonzero only on its own entity and on entities of strictly lower
    topological dimension (already resolved), so the entities are
    still visited in increasing dimension, and each entity's own small
    diagonal block of :math:`B` is solved (:func:`_solve`) and used to
    eliminate the already-known contribution of the lower-dimensional
    entities.
    """

    #: Numerical tolerance used throughout automatic basis transformation
    #: to detect vanishing coefficients.
    tol = 1e-12

    #: If False, physical facet moments are plain integrals rather than
    #: the measure-intrinsic integral averages of the reference nodes.
    #: Only meaningful for scalar (affine pullback) facet derivative dofs.
    avg = True

    #: FIAT mapping strings this class knows how to transform.
    _supported_mappings = frozenset({
        "affine", "contravariant piola", "double contravariant piola"})

    def _check_mapping(self, fiat_element):
        """Verify that this class knows how to transform this element's pullback.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not supported.
        """
        mappings = set(fiat_element.mapping())
        if len(mappings) != 1:
            raise NotImplementedError(
                f"{type(self).__name__} does not support mixed pullbacks {mappings}.")
        mapping, = mappings
        if mapping not in self._supported_mappings:
            raise NotImplementedError(
                f"{type(self).__name__} does not support the {mapping!r} pullback.")

    def basis_transformation(self, coordinate_mapping) -> ListTensor:
        fiat_element = self._element
        self._check_mapping(fiat_element)

        ref_el = fiat_element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        bary, = ref_el.make_points(sd, 0, sd + 1)
        J = coordinate_mapping.jacobian_at(bary)
        Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                          dtype=object)
        K = adjugate(Jnp).T

        nodes = fiat_element.dual_basis()
        entity_ids = fiat_element.entity_dofs()
        entity_support = fiat_element.entity_closure_dofs()
        ndof = fiat_element.space_dimension()
        B = numpy.full((ndof, ndof), Zero(), dtype=object)

        # TODO implement PhysicallyMappedDualSet (set of PhysicallyMappedFunctional)
        # Should implement PhysicallyMappedDualSet.evaluate_dual(fiat_element) to compute B directly,
        # minimizing the number of calls to fiat_element.tabulate. The same pattern is in FIAT.DualSet.toriesz

        for dim in entity_ids:
            for entity in entity_ids[dim]:
                support = entity_support[dim][entity]
                for i in entity_ids[dim][entity]:

                    # TODO ell =  PhysicallyMappedFunctional.from_fiat(nodes[i], J=J)
                    # move _physical_direction and _physical_weights into PhysicallyMappedFunctional
                    ell = PhysicallyMappedFunctional.from_fiat(nodes[i])
                    if ell.order > 0:
                        direction = self._physical_direction(ell, dim, sd, entity, ref_el, K)
                        ell = ell.with_direction(direction, J=J)
                    elif ell.rank > 0:
                        weights = self._physical_weights(ell, dim, sd, entity, ref_el, K, Jnp)
                        ell = type(ell)(ell.points, weights, rank=ell.rank, J=J)

                    row = ell.evaluate(fiat_element)
                    B[i, support] = row[support]

        B = numpy.vectorize(as_gem)(B)

        print(B)
        V = numpy.full(B.shape, Zero(), dtype=object)
        V[range(ndof), range(ndof)] = Literal(1)
        V[:, ndof:] = -B[:, ndof:]

        processed = []
        for dim in sorted(entity_ids):
            for entity in sorted(entity_ids[dim]):
                rows = entity_ids[dim][entity]
                Bii = B[numpy.ix_(rows, rows)]
                V[rows, :] -= B[numpy.ix_(rows, processed)] @ V[processed, :]
                V[rows, :] = _solve(Bii, V[rows, :])
                processed.extend(rows)

        _rescale_derivative_dofs(V, fiat_element, coordinate_mapping)
        ndof = self.space_dimension()
        return ListTensor(V[:, :ndof].T)

    def _physical_direction(self, ell: PhysicallyMappedFunctional, dim: int, sd: int,
                            entity: int, ref_el, K: numpy.ndarray) -> numpy.ndarray:
        r"""Build the physical direction tensor of a scalar derivative node.

        :arg ell: The reference functional, with its numeric direction
            recovered by :meth:`PhysicallyMappedFunctional.from_fiat`.
        :arg dim: Topological dimension of the node's entity.
        :arg sd: Spatial dimension of the cell.
        :arg entity: The entity number.
        :arg ref_el: The reference cell.
        :arg K: GEM cofactor matrix :math:`\operatorname{adj}(J)^T`.
        :returns: The GEM direction tensor of the physical node.
        """
        if dim != sd - 1:
            # A vertex or interior derivative jet is a fixed Cartesian
            # tensor: its meaning does not depend on cell geometry.
            return numpy.array([Literal(d) for d in ell.direction], dtype=object)

        # A facet derivative node differentiates along the reference
        # normal; the physical normal is its cofactor image K@direction
        # (the standard transformation law for the normal of a
        # hyperplane under an affine map), normalized to unit length to
        # match FIAT's own facet dof convention (compute_normal).
        Kd = K @ ell.direction
        if self.avg or len(ell.points) == 1:
            norm = (Kd @ Kd) ** 0.5
            return Kd / norm
        # A plain integral (not a measure-intrinsic average) scales the
        # unit-normal moment back up by the reference facet measure,
        # since the physical and reference facet measures differ by
        # exactly the norm of Kd that was just divided out.
        vol = ref_el.volume_of_subcomplex(sd - 1, entity)
        return Kd * vol

    def _physical_weights(self, ell: PhysicallyMappedFunctional, dim: int, sd: int,
                          entity: int, ref_el, K: numpy.ndarray,
                          Jnp: numpy.ndarray) -> numpy.ndarray:
        r"""Build the physical weight profile of a Piola-mapped value node.

        The reference nodal basis is left untransformed
        (:meth:`~finat.functional.PhysicallyMappedFunctional.evaluate`
        does not touch it for a value moment); instead the moment's own
        test function is pushed forward here, so that the two together
        reproduce the physical node's action exactly.  Away from a
        facet the test function is a fixed physical direction with no
        frame to resolve, so it pushes forward by the cofactor, exactly
        like a point evaluation of the trial space.  On a facet, the
        test function is decomposed in the reference normal/tangential
        frame; the normal part pushes forward the same way, but the
        tangential part is built by FIAT on the *reciprocal* basis
        (cross products of the frame) and so transforms contragrediently,
        picking up the in-plane correction
        :math:`S = \operatorname{adj}(G)\,\hat G_t/\det\hat G_t` of the
        change of tangent Gram matrices, :math:`G_{ij} = (J\hat t_i)
        \cdot(J\hat t_j)`, :math:`\hat G_t = \hat t_i\cdot\hat t_j`; in
        2D the tangent "plane" is one-dimensional and :math:`S = 1`.

        :arg ell: The reference functional, with its numeric weight
            profile recovered by :meth:`PhysicallyMappedFunctional.from_fiat`.
        :arg dim: Topological dimension of the node's entity.
        :arg sd: Spatial dimension of the cell.
        :arg entity: The entity number.
        :arg ref_el: The reference cell.
        :arg K: GEM cofactor matrix :math:`\operatorname{adj}(J)^T`.
        :arg Jnp: GEM cell Jacobian, as a square object array.
        :returns: The physical weight profile, the same shape as ``ell.weights``.
        """
        if dim != sd - 1:
            M = K
        else:
            that = ref_el.compute_tangents(sd - 1, entity)
            nhat = ref_el.compute_scaled_normal(entity)
            Ghat = numpy.column_stack([nhat, *that])
            Ghatinv = numpy.linalg.inv(Ghat)
            Aphys = Jnp @ that.T

            G = numpy.array([[Jnp @ t1 @ (Jnp @ t2) for t2 in that] for t1 in that])
            Ghat_t = that @ that.T
            S = adjugate(G) @ Ghat_t / numpy.linalg.det(Ghat_t)

            M = (numpy.outer(K @ nhat, Ghatinv[0, :])
                 + Aphys @ S @ Ghatinv[1:, :])

        return _transform_weights(ell.weights, M, ell.rank)


def _transform_weights(weights: numpy.ndarray, M: numpy.ndarray, rank: int) -> numpy.ndarray:
    r"""Contract every value axis of a per-point weight profile with the same matrix.

    :arg weights: Weight profile, shape ``(npoints, sd**rank)``.
    :arg M: The matrix to contract into each of the ``rank`` value axes.
    :arg rank: The value rank.
    :returns: The transformed profile, the same shape as ``weights``.
    """
    npoints = weights.shape[0]
    sd = M.shape[0]
    W = numpy.reshape(weights, (npoints,) + (sd,) * rank).astype(object)
    ndim = W.ndim
    for i in range(rank):
        axis = i + 1
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        W = W.transpose(perm)
        W = numpy.tensordot(W, M, axes=(-1, 1))
        W = W.transpose(perm)
    return W.reshape(npoints, -1)


def _solve(A: numpy.ndarray, rhs: numpy.ndarray) -> numpy.ndarray:
    r"""Solve :math:`AX = \text{rhs}` by symbolic Gauss-Jordan elimination.

    A block of :math:`B` need not be small in a way that is visible
    structurally: entries that are zero for every geometry can fail to
    reduce to literal ``gem.Zero()`` when they come from a deep
    symbolic construction (e.g. a Piola tensor-value weight profile),
    so sparsity-detecting approaches --
    :func:`~finat.physically_mapped.inverse`'s connected components, or
    peeling off rows whose remaining entries structurally check as
    ``gem.Zero()`` -- can fail to find structure that is only true
    numerically, and fall back to a cofactor-expansion inverse or a
    joint solve that is combinatorial in the block size.  Gauss-Jordan
    elimination, :math:`O(n^3)`, needs no such detection.  No pivoting
    is done: :math:`A`'s diagonal entries are generically nonzero by
    unisolvence.

    :arg A: Square object array.
    :arg rhs: Object array with the same number of rows as ``A``.
    :returns: :math:`X` such that :math:`AX = \text{rhs}`.
    """
    n = A.shape[0]
    C = numpy.concatenate([A, rhs], axis=1).astype(object)
    for k in range(n):
        C[k, :] = C[k, :] / C[k, k]
        for i in range(n):
            if i != k and not isinstance(C[i, k], Zero):
                # The array must be the left operand: a GEM scalar's
                # own __mul__ absorbs a whole numpy array into one
                # tensor-valued node instead of distributing over it.
                C[i, :] -= C[k, :] * C[i, k]
    return C[:, n:]


def _rescale_derivative_dofs(V, fiat_element, coordinate_mapping):
    r"""Rescale derivative degrees of freedom by the cell size.

    Each physical node of derivative order :math:`m` is redefined with a
    factor :math:`h^{-m}`, where :math:`h` averages the cell size over
    the vertices of its entity.  This is the FInAT convention keeping
    the mass matrix well-conditioned; it is consistent across cells
    because the scaling only depends on shared entities.

    :arg V: Object array being assembled; columns are rescaled in place.
    :arg fiat_element: The FIAT element defined on the reference cell.
    :arg coordinate_mapping: Object providing the physical geometry as
        GEM expressions.
    """
    # cell_size may be a GEM expression or a numpy array of numbers
    h = coordinate_mapping.cell_size()
    top = fiat_element.get_reference_element().get_topology()
    nodes = fiat_element.dual_basis()
    entity_ids = fiat_element.entity_dofs()
    for dim in entity_ids:
        for entity in entity_ids[dim]:
            verts = top[dim][entity]
            havg = reduce(add, (h[v] for v in verts)) / len(verts)
            for i in entity_ids[dim][entity]:
                order = nodes[i].max_deriv_order
                if order > 0:
                    V[:, i] = V[:, i] * havg**(-order)
