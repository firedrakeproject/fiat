r"""Automatic basis transformations for physically mapped elements.

This module automates the transformation theory of Kirby (2017) and
Brubeck & Kirby (2025).  Given a FIAT element whose degrees of freedom
are not preserved under push-forward, each physical node's row of the
generalized Vandermonde matrix :math:`B_{ij} = \ell_i(\hat\psi_j)` --
its evaluation against the reference nodal basis, transplanted to
physical space by the affine cell map :math:`x = J\hat{x} + v_0` -- is
computed directly by
:meth:`finat.functional.PhysicallyMappedFunctional.evaluate`, with no
geometric frame required.  But :math:`B` relates physical nodes to
reference basis functions, not to reference nodes, so the actual basis
transformation :math:`V = B^{-1}` still has to be assembled: a
physical node's row of :math:`B` is nonzero only on its own entity and
on entities of strictly lower topological dimension, so entities are
visited in increasing dimension and each one's small diagonal block of
:math:`B` is inverted to eliminate the already-known contribution of
the lower-dimensional ones.

Degrees of freedom are represented symbolically by
:class:`finat.functional.PhysicallyMappedFunctional` and processed
generically, without dispatching over FIAT functional types.  Two
mixins specialize :class:`ZanyPhysicallyMappedElement` to a pullback:

* :class:`ScalarPhysicallyMappedElement`, for scalar elements with an
  affine (identity) pullback -- Morley, Hermite, Argyris, Bell.  Every
  reference node's own recipe (points, weights, a numeric direction
  tensor) is reused for the physical node, with only the direction
  replaced by its physical counterpart: unchanged (a fixed Cartesian
  tensor) for vertex/interior derivative jets, or the cofactor image of
  the reference facet normal for facet derivative dofs.  No geometric
  frame is needed to build :math:`B`, only the block inversion above.

* :class:`PiolaPhysicallyMappedElement`, for vector- or tensor-valued
  elements under the (double) contravariant Piola pullback --
  Mardal-Tai-Winther, Johnson-Mercier, Guzman-Neilan.  This retains the
  entity-by-entity assembly loop of Kirby (2017) and Brubeck & Kirby
  (2025): the scaled facet normal is the cofactor image of the
  reference one, so pure normal-component moments are invariant, while
  scaled tangents map by the Jacobian.
"""

from functools import reduce
from operator import add

import numpy

from FIAT.finite_element import FiniteElement
from gem import Literal, ListTensor, Node, Power, Zero
from finat.functional import PhysicallyMappedFunctional
from finat.physically_mapped import PhysicallyMappedElement, adjugate, determinant, identity, inverse


class ZanyPhysicallyMappedElement(PhysicallyMappedElement):
    r"""Mixin holding what :class:`ScalarPhysicallyMappedElement` and
    :class:`PiolaPhysicallyMappedElement` share: a numerical tolerance
    and the requirement to validate the FIAT element's pullback before
    deriving its basis transformation.
    """

    #: Numerical tolerance used throughout automatic basis transformation
    #: to detect vanishing coefficients.
    tol = 1e-12

    def _check_mapping(self, fiat_element):
        """Verify that this class knows how to transform this element's pullback.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement automatic basis transformation.")


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


class ScalarPhysicallyMappedElement(ZanyPhysicallyMappedElement):
    r"""Mixin deriving the basis transformation for a scalar element
    with an affine (identity) pullback.

    Every reference node is, by construction, either a value moment
    (push-forward invariant) or a directional-derivative moment along
    either a vertex/interior Cartesian axis (Hermite/Argyris/Bell
    jets) or a facet normal (Morley/Argyris/Bell facet dofs); FIAT
    never mixes the two within one functional.  The corresponding
    physical node reuses the same points, weights, and derivative
    order, with only the direction replaced by its physical
    counterpart (:meth:`_physical_direction`): a Cartesian jet
    direction is geometry-independent, while a facet normal direction
    pushes forward by the cofactor matrix, the standard transformation
    law for the normal of a hyperplane under an affine map.  Evaluating
    that physical node against the *full* reference nodal basis (no
    geometric frame needed) gives a row of :math:`B_{ij} =
    \ell_i(\hat\psi_j)`, the generalized Vandermonde matrix; but
    :math:`V = B^{-1}` is what relates the reference nodes to the
    push-forward of the physical ones, so :math:`B` still needs
    inverting.  By construction, a physical node's row of :math:`B` is
    nonzero only on its own entity and on entities of strictly lower
    topological dimension (already resolved), so the entities are
    still visited in increasing dimension, and each entity's own small
    diagonal block of :math:`B` is inverted (:func:`~finat.
    physically_mapped.inverse`) and used to eliminate the
    already-known contribution of the lower-dimensional entities.
    """

    #: If False, physical facet moments are plain integrals rather than
    #: the measure-intrinsic integral averages of the reference nodes.
    avg = True

    def _check_mapping(self, fiat_element):
        mappings = set(fiat_element.mapping())
        if mappings != {"affine"}:
            raise NotImplementedError(
                f"{type(self).__name__} expects an affine pullback, not {mappings}.")

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
        ndof = fiat_element.space_dimension()
        B = numpy.empty((ndof, ndof), dtype=object)
        V = numpy.empty((ndof, ndof), dtype=object)

        processed = []
        for dim in entity_ids:
            for entity in entity_ids[dim]:
                rows = entity_ids[dim][entity]
                for i in rows:
                    ell = PhysicallyMappedFunctional.from_fiat(nodes[i])
                    if ell.order > 0:
                        direction = self._physical_direction(ell, dim, sd, entity, ref_el, K)
                        ell = ell.with_direction(direction, J=J)
                    row = ell.evaluate(fiat_element)
                    B[i, :] = [x if isinstance(x, Node) else Literal(x) for x in row]

                # B[rows, :] @ V = I restricted to rows: eliminate the
                # (already-known) contribution of lower-dimensional
                # entities and solve the entity's own diagonal block.
                target = numpy.full((len(rows), ndof), Zero(), dtype=object)
                for a, i in enumerate(rows):
                    target[a, i] = Literal(1.0)
                if processed:
                    target = target - B[numpy.ix_(rows, processed)] @ V[processed, :]
                Dinv = inverse(B[numpy.ix_(rows, rows)])
                V[rows, :] = Dinv @ target
                processed.extend(rows)

        _rescale_derivative_dofs(V, fiat_element, coordinate_mapping)
        ndof = self.space_dimension()
        return ListTensor(V[:, :ndof].T)

    def _physical_direction(self, ell: PhysicallyMappedFunctional, dim: int, sd: int,
                            entity: int, ref_el, K: numpy.ndarray) -> numpy.ndarray:
        r"""Build the physical direction tensor of a derivative node.

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
            norm = Power(reduce(add, (Kd[m] * Kd[m] for m in range(sd))), Literal(0.5))
            return Kd / norm
        # A plain integral (not a measure-intrinsic average) scales the
        # unit-normal moment back up by the reference facet measure,
        # since the physical and reference facet measures differ by
        # exactly the norm of Kd that was just divided out.
        vol = ref_el.volume_of_subcomplex(sd - 1, entity)
        return Kd * vol


class PiolaPhysicallyMappedElement(ZanyPhysicallyMappedElement):
    r"""Mixin deriving the basis transformation for a vector- or
    tensor-valued element under the (double) contravariant Piola
    pullback.

    Following the factorization :math:`V = E V^c D` of Kirby (2017) and
    Brubeck & Kirby (2025), the matrix :math:`V` relating the reference
    nodes to the push-forwards of the physical nodes is assembled one
    topological entity at a time, in increasing dimension, so that the
    completion of a node on an entity can always be resolved against
    the already-assembled rows of lower-dimensional entities.  The
    roles of the normal and tangential directions are mirrored with
    respect to the scalar case: interior value moments are Piola
    invariant by construction; value moments on a facet are resolved in
    the frame of the scaled facet normal (the cofactor image of the
    reference one) and the mapped reference tangents; point evaluations
    elsewhere have no geometric frame to expand in and instead act as
    their own completion group.
    """

    def basis_transformation(self, coordinate_mapping) -> ListTensor:
        fiat_element = self._element
        self._check_mapping(fiat_element)

        ref_el = fiat_element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        bary, = ref_el.make_points(sd, 0, sd + 1)
        J = coordinate_mapping.jacobian_at(bary)

        nodes = fiat_element.dual_basis()
        V = identity(fiat_element.space_dimension())

        processed = set()
        entity_ids = fiat_element.entity_dofs()
        for dim in sorted(entity_ids):
            for entity in sorted(entity_ids[dim]):
                group = {i: PhysicallyMappedFunctional.from_fiat(nodes[i])
                         for i in entity_ids[dim][entity]}
                invariant = self._invariant_dofs(group, dim, sd)
                processed.update(invariant)
                group = {i: ell for i, ell in group.items() if i not in invariant}
                if not group:
                    continue
                if dim == sd - 1:
                    self._facet_dof_rows(V, group, fiat_element, entity, J, processed)
                else:
                    self._point_dof_rows(V, group, fiat_element, J, processed)

        _rescale_derivative_dofs(V, fiat_element, coordinate_mapping)
        ndof = self.space_dimension()
        return ListTensor(V[:, :ndof].T)

    def _check_mapping(self, fiat_element):
        mappings = set(fiat_element.mapping())
        if mappings not in ({"contravariant piola"}, {"double contravariant piola"}):
            raise NotImplementedError(
                f"{type(self).__name__} expects a (double) contravariant "
                f"Piola pullback, not {mappings}.")

    def _invariant_dofs(self, group, dim, sd):
        # Interior moments are Piola invariant by construction
        return {i for i, ell in group.items() if ell.order == 0 and dim == sd}

    def _facet_dof_rows(self, V, group, fiat_element, entity, J, processed):
        _check_piola_group(group)
        _piola_facet_rows(V, group, fiat_element, entity, J, processed, self.tol)

    def _point_dof_rows(self, V, group, fiat_element, J, processed):
        _check_piola_group(group)
        _piola_point_rows(V, group, J, processed, self.tol)


def _check_piola_group(group: dict) -> None:
    """Validate that a group of non-invariant Piola-mapped nodes are all
    value moments with a nonzero value rank.

    :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional.
    :raises NotImplementedError: If a node is a scalar weight or carries
        a derivative, which the theory does not yet cover.
    """
    if any(ell.rank == 0 or ell.order > 0 for ell in group.values()):
        raise NotImplementedError("Cannot yet Piola-transform this node group.")


def _piola_facet_rows(V: numpy.ndarray, group: dict,
                      fiat_element: FiniteElement, entity: int, J: Node,
                      processed: set, tol: float) -> None:
    r"""Assemble the rows of V for facet moments of Piola-mapped values.

    This mirrors the scalar case with the roles of the normal and
    tangential directions exchanged: under the contravariant Piola map
    the scaled facet normal is the image of the reference one under the
    cofactor matrix :math:`K = \operatorname{adj}(J)^T`, so pure normal
    moments are invariant, while the scaled tangents map by :math:`J`.
    Distinct nodes on a facet can share the same tangential directions
    (e.g. two RT-type facet dofs in 3D) and are only told apart by how
    their weight varies from point to point, so a node is identified by
    its full per-point profile of frame coordinates, not by direction
    alone; the pulled-back reference node mixes the coordinates through
    the frame expansion of :math:`K\hat{t}_k`, the tangential profiles
    are matched within the group, and the residual normal profile is
    eliminated numerically through already assembled rows of V.

    :arg V: Object array being assembled.
    :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
        for the value moments on this facet.
    :arg fiat_element: The FIAT element.
    :arg entity: The facet number.
    :arg J: GEM expression for the cell Jacobian.
    :arg processed: Indices of the already assembled rows; updated in place.
    :arg tol: Tolerance for detecting zeros in the numeric coefficients.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    that = ref_el.compute_tangents(sd - 1, entity)
    nhat = ref_el.compute_scaled_normal(entity)
    Ghat = numpy.column_stack([nhat, *that])
    Ghatinv = numpy.linalg.inv(Ghat)

    Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                      dtype=object)
    K = adjugate(Jnp).T

    # Reference frame coordinate profiles, shared with the physical nodes:
    # each node's own quadrature weights, decomposed into (normal,
    # tangential) frame coordinates at each of its points.  This is what
    # distinguishes nodes with the same tangential directions from one
    # another (see the point-dependence note in the docstring above).
    coords = {}
    for i, ell in group.items():
        C = ell.weights.reshape(-1, *(sd,) * ell.rank)
        for _ in range(ell.rank):
            C = numpy.tensordot(C, Ghatinv, axes=(1, 1))
        coords[i] = C.reshape(len(ell.points), -1)
        # Pure normal moments are Piola invariant
        if numpy.allclose(coords[i][:, 1:], 0, atol=tol):
            processed.add(i)

    group = {i: ell for i, ell in group.items() if i not in processed}
    if not group:
        return
    rank, = {ell.rank for ell in group.values()}
    points, = {ell.points for ell in group.values()}

    # Frame coordinates of the mapped frame image of the reference frame:
    # the normal is invariant and the pulled tangents are expanded by a
    # symbolic solve in the mapped frame [K nhat | J that_k]
    A = numpy.column_stack([K @ nhat, *(Jnp @ t for t in that)])
    adjA = adjugate(A)
    detA = determinant(A)
    Y = numpy.full((sd, sd), Zero(), dtype=object)
    Y[0, 0] = Literal(1.0)
    for k, t in enumerate(that):
        Y[:, k + 1] = (adjA @ (K @ t)) / detA

    # Physical tangential components are built on the reciprocal basis
    # (cross products of the frame), so they carry the in-plane
    # contravariant transformation S = adj(G Ghat^{-1})^T of the change
    # of tangent Gram matrices.  Absorb S^{-1} into the coordinate
    # mixing so that the physical profiles keep the reference
    # coordinates; in 2D the tangent plane is one-dimensional and S = 1.
    G = numpy.array([[Jnp @ t1 @ (Jnp @ t2) for t2 in that] for t1 in that])
    Ghat_t = that @ that.T
    Sinv = (numpy.linalg.inv(Ghat_t) @ G) * \
        (numpy.linalg.det(Ghat_t) / determinant(G))
    Y[1:, :] = Sinv @ Y[1:, :]

    # Numeric matching of the tangential coordinate profiles in the group.
    # B has full row rank by unisolvence, so the Gram matrix B @ B.T is
    # square and invertible; a rank deficiency (a genuine bug) surfaces
    # as a LinAlgError here rather than a silent least-squares fit.
    B = numpy.array([coords[j][:, 1:].ravel() for j in group])
    Binv = numpy.linalg.inv(B @ B.T) @ B
    Binv[abs(Binv) < tol] = 0

    # Numeric elimination of the normal profile: one pure normal moment
    # per quadrature point, evaluated on the nodal basis
    ndir = numpy.ones(())
    for _ in range(rank):
        ndir = numpy.multiply.outer(ndir, nhat)
    T = fiat_element.tabulate(0, points)[(0,) * sd]
    L = numpy.einsum("jcq,c->jq", T.reshape(T.shape[0], -1, len(points)),
                     ndir.ravel())
    L[abs(L) < tol] = 0

    for i, ell in group.items():
        # Pull back the coordinate profile, contracting each slot with Y
        P = coords[i].reshape(-1, *(sd,) * rank)
        for _ in range(rank):
            P = numpy.tensordot(P, Y, axes=(1, 1))
        P = P.reshape(len(points), -1)

        row = numpy.full(V.shape[1], Zero(), dtype=object)
        c = Binv @ P[:, 1:].ravel()
        for cj, j in zip(c, group):
            row[j] = cj
        # Residual normal profile after removing the group contribution
        residual = P[:, 0] - c @ numpy.array([coords[j][:, 0] for j in group])
        for q in range(len(points)):
            for m in numpy.flatnonzero(L[:, q]):
                if m not in processed:
                    raise NotImplementedError(
                        f"Completion of node {i} couples to node {m}, "
                        "which has not been transformed yet.")
                row = row + V[m, :] * (residual[q] * L[m, q])
        V[i, :] = row
        processed.add(i)


def _piola_point_rows(V: numpy.ndarray, group: dict, J: Node,
                      processed: set, tol: float) -> None:
    r"""Assemble the rows of V for point values of Piola-mapped fields.

    Away from facets, physical point evaluations keep the reference
    (Cartesian) components, which pull back through the cofactor matrix
    :math:`K = \operatorname{adj}(J)^T` of the contravariant Piola map,
    so the group of components at each point acts as its own
    completion.

    :arg V: Object array being assembled.
    :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
        for the value nodes on this entity.
    :arg J: GEM expression for the cell Jacobian.
    :arg processed: Indices of the already assembled rows; updated in place.
    :arg tol: Tolerance for detecting zeros in the numeric coefficients.
    """
    sd = J.shape[0]
    Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                      dtype=object)
    K = adjugate(Jnp).T

    subgroups = {}
    for i, ell in group.items():
        if len(ell.points) != 1 or ell.rank != 1:
            raise NotImplementedError(
                "Only single-point vector evaluations are handled.")
        subgroups.setdefault(ell.points, {})[i] = ell

    for sub in subgroups.values():
        directions = numpy.array([ell.weights[0] for ell in sub.values()])
        if directions.shape[0] != directions.shape[1]:
            raise NotImplementedError(
                "Directions do not span the vector components.")
        Dinv = numpy.linalg.inv(directions.T)
        for i, ell in sub.items():
            Kd = K @ ell.weights[0]
            for col, j in enumerate(sub):
                x = Dinv[col]
                nz = numpy.flatnonzero(abs(x) > tol)
                V[i, j] = reduce(add, (Kd[m] * x[m] for m in nz)) if len(nz) else Zero()
            processed.add(i)
