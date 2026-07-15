r"""Automatic basis transformations for physically mapped elements.

This module automates the transformation theory of Kirby (2017) and
Brubeck & Kirby (2025).  Given a FIAT element whose degrees of freedom
are not preserved under push-forward, :class:`ZanyPhysicallyMappedElement`
constructs the matrix :math:`V` relating the reference nodes to the
push-forwards of the physical nodes, so that the physical basis
functions are obtained as :math:`M F^*(\hat\Psi)` with :math:`M = V^T`.

Degrees of freedom are represented symbolically by
:class:`finat.functional.PhysicallyMappedFunctional` and processed
generically, without dispatching over FIAT functional types.
:class:`ZanyPhysicallyMappedElement` implements the entity-by-entity
assembly loop once, calling four hooks that carry all mapping-specific
knowledge; this module supplies the two mixins implementing them:

* :class:`ScalarPhysicallyMappedElement`, for scalar elements with an
  affine (identity) pullback -- Morley, Hermite, Argyris, Bell.  Each
  reference node is pulled back to the physical cell by the chain rule
  and expanded in the frame of the physical facet normal and the mapped
  reference tangents (:class:`FacetFrame`); the tangential components
  are derivatives along *mapped* reference tangents and therefore
  coincide with reference functionals, whose expansion in the element's
  own nodes is a purely numeric generalized Vandermonde row.

* :class:`PiolaPhysicallyMappedElement`, for vector- or tensor-valued
  elements under the (double) contravariant Piola pullback --
  Mardal-Tai-Winther, Johnson-Mercier, Guzman-Neilan.  The roles of the
  normal and tangential directions are mirrored (:class:`PiolaFacetFrame`):
  the scaled facet normal is the cofactor image of the reference one, so
  pure normal-component moments are invariant, while scaled tangents map
  by the Jacobian.

In the language of the theory, the frame expansion in either mixin
realizes :math:`E V^c`, and the numeric elimination of the tangential
(respectively normal) completion realizes :math:`D`.
"""

from abc import abstractmethod
from functools import reduce
from operator import add

import numpy

from FIAT.finite_element import FiniteElement
from gem import Literal, ListTensor, Node, Zero
from finat.functional import PhysicallyMappedFunctional
from finat.physically_mapped import PhysicallyMappedElement, adjugate, determinant, identity, inverse


def generalized_cross(tangents) -> numpy.ndarray:
    r"""Generalized cross product of d-1 vectors in d dimensions.

    :arg tangents: A (d-1, d) array of vectors, with numeric or GEM entries.
    :returns: The vector :math:`C` such that :math:`C \cdot w =
        \det([t_1; \dots; t_{d-1}; w])` for all :math:`w`; it is
        orthogonal to every :math:`t_k`.
    """
    A = numpy.asarray(tangents)
    d = A.shape[1]
    cols = numpy.ones(d, dtype=bool)
    C = []
    for i in range(d):
        cols[i] = False
        C.append((-1) ** (d - 1 + i) * determinant(A[:, cols]))
        cols[i] = True
    return numpy.asarray(C)


class FacetFrame:
    r"""Normal/tangential frame of a facet and its push-forward.

    The reference frame consists of the FIAT facet normal
    :math:`\hat{n}` and the scaled facet tangents :math:`\hat{t}_k`;
    the physical frame consists of the physical facet normal and the
    mapped tangents :math:`J\hat{t}_k`.  Because FIAT normals are
    computed from the tangents by the same formula on the reference and
    physical cells, the physical normal is :math:`\kappa\, C / \|C\|`
    with :math:`C` the generalized cross product of the mapped tangents
    and :math:`\kappa` a cell-independent constant recovered from the
    reference data.

    :arg fiat_element: The FIAT element, providing the reference cell.
    :arg entity: The facet number.
    :arg J: GEM expression for the cell Jacobian.
    """

    def __init__(self, fiat_element: FiniteElement, entity: int, J: Node):
        ref_el = fiat_element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        self.tangents = ref_el.compute_tangents(sd - 1, entity)
        self.normal = ref_el.compute_normal(entity)

        Chat = generalized_cross(self.tangents)
        kappa = self.normal @ Chat / numpy.linalg.norm(Chat)

        self.mapped_tangents = [J @ Literal(t) for t in self.tangents]
        C = generalized_cross([[Jt[i] for i in range(sd)]
                               for Jt in self.mapped_tangents])
        A = numpy.empty((sd, sd), dtype=object)
        A[:, 0] = C
        for k, Jt in enumerate(self.mapped_tangents):
            A[:, k + 1] = [Jt[i] for i in range(sd)]
        self._adjA = adjugate(A)
        self._detA = determinant(A)

        normC = (C @ C) ** 0.5
        self.normal_scale = normC / kappa
        vol = ref_el.volume_of_subcomplex(sd - 1, entity)
        self.measure = normC * (vol / numpy.linalg.norm(Chat))

    def reference_coefficients(self, direction: numpy.ndarray) -> numpy.ndarray:
        r"""Expand a numeric direction in the reference frame.

        :arg direction: A numeric direction vector.
        :returns: Coefficients ``(a, b_1, ..., b_{d-1})`` such that the
            direction equals :math:`a\hat{n} + \sum_k b_k \hat{t}_k`.
        """
        A = numpy.column_stack([self.normal, *self.tangents])
        return numpy.linalg.solve(A, direction)

    def decompose(self, direction: Node) -> list:
        r"""Expand a GEM direction in the un-normalized physical frame.

        :arg direction: A GEM direction vector.
        :returns: GEM coefficients ``(x_0, x_1, ..., x_{d-1})`` such that
            the direction equals :math:`x_0 C + \sum_k x_k J\hat{t}_k`.
        """
        sd = self._adjA.shape[0]
        return [reduce(add, (self._adjA[m, i] * direction[i]
                             for i in range(sd))) / self._detA
                for m in range(sd)]


class PiolaFacetFrame:
    r"""Normal/tangential frame of a facet for the (double) contravariant
    Piola pullback, and its expansion under push-forward.

    The roles of the normal and tangential directions are mirrored with
    respect to :class:`FacetFrame`: the reference frame's scaled normal
    :math:`\hat{n}` maps by the cofactor matrix :math:`K =
    \operatorname{adj}(J)^T`, while its tangents :math:`\hat{t}_k` map by
    :math:`J` directly.  ``Y`` is the (symbolic) matrix expanding the
    pulled-back frame image :math:`[K\hat{n}\;|\;K\hat{t}_k]` in the
    physical frame :math:`[K\hat{n}\;|\;J\hat{t}_k]`; the mapped tangents
    are built on the reciprocal basis in dimension > 2, so their
    coordinates carry an extra in-plane contravariant correction
    :math:`S^{-1}` folded into ``Y``'s tangential rows.

    :arg fiat_element: The FIAT element, providing the reference cell.
    :arg entity: The facet number.
    :arg J: GEM expression for the cell Jacobian.
    """

    def __init__(self, fiat_element: FiniteElement, entity: int, J: Node):
        ref_el = fiat_element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        self.tangents = ref_el.compute_tangents(sd - 1, entity)
        self.normal = ref_el.compute_scaled_normal(entity)

        Ghat = numpy.column_stack([self.normal, *self.tangents])
        self.Ghatinv = numpy.linalg.inv(Ghat)

        Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                          dtype=object)
        Knp = adjugate(Jnp).T

        # Frame coordinates of the mapped frame image of the reference frame:
        # the normal is invariant and the pulled tangents are expanded by a
        # symbolic solve in the mapped frame [K nhat | J that_k]
        Kn = self.normal @ Knp.T
        Jt = self.tangents @ Jnp.T
        A = numpy.column_stack([Kn, *Jt])

        Y = numpy.full((sd, sd), Zero(), dtype=object)
        Y[0, 0] = Literal(1.0)
        Y[:, 1:] = inverse(A) @ (Knp @ self.tangents.T)

        # Physical tangential components are built on the reciprocal basis
        # (cross products of the frame), so they carry the in-plane
        # contravariant transformation S = adj(G Ghat^{-1})^T of the change
        # of tangent Gram matrices.  Absorb S^{-1} into the coordinate
        # mixing so that the physical profiles keep the reference
        # coordinates; in 2D the tangent plane is one-dimensional and S = 1.
        G = Jt @ Jt.T
        Ghat_t = self.tangents @ self.tangents.T
        Sinv = (numpy.linalg.inv(Ghat_t) @ G) * \
            (numpy.linalg.det(Ghat_t) / determinant(G))
        Y[1:, :] = Sinv @ Y[1:, :]
        self.Y = Y


def _weight_ratio(wi: numpy.ndarray, wj: numpy.ndarray, tol: float) -> float:
    """Return the scalar s with wi == s * wj, if it exists."""
    s = wi @ wj / (wj @ wj)
    if not numpy.allclose(wi, s * wj, atol=tol * numpy.linalg.norm(wi)):
        raise NotImplementedError("Weights are not parallel.")
    return s


class ZanyPhysicallyMappedElement(PhysicallyMappedElement):
    r"""Mixin implementing the entity-by-entity assembly loop shared by
    :class:`ScalarPhysicallyMappedElement` and
    :class:`PiolaPhysicallyMappedElement`.

    Following the factorization :math:`V = E V^c D` of Kirby (2017) and
    Brubeck & Kirby (2025), the matrix :math:`V` relating the reference
    nodes to the push-forwards of the physical nodes is assembled one
    topological entity at a time, in increasing dimension, so that the
    completion of a node on an entity can always be resolved against
    the already-assembled rows of lower-dimensional entities.

    On each entity, nodes that are already push-forward invariant
    (:meth:`invariant_dofs`) contribute an identity row for free,
    since :math:`V` starts out as the identity.  The rest are assembled
    by :meth:`facet_dof_rows` (on a codimension-1 entity) or
    :meth:`point_dof_rows` (elsewhere); these hooks, together with
    :meth:`_check_mapping`, encode the mapping-specific (affine or
    Piola) part of the theory, and :meth:`basis_transformation` itself
    contains no knowledge of which mapping is in play.
    """

    #: Numerical tolerance used throughout automatic basis transformation
    #: to detect vanishing coefficients.
    tol = 1e-12

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
                invariant = self.invariant_dofs(group, dim, sd)
                processed.update(invariant)
                group = {i: ell for i, ell in group.items() if i not in invariant}
                if not group:
                    continue
                if dim == sd - 1:
                    self.facet_dof_rows(V, group, fiat_element, entity, J, processed, tol=self.tol)
                else:
                    self.point_dof_rows(V, group, fiat_element, entity, J, processed, tol=self.tol)

        _rescale_derivative_dofs(V, fiat_element, coordinate_mapping)
        ndof = self.space_dimension()
        return ListTensor(V[:, :ndof].T)

    def _check_mapping(self, fiat_element):
        """Verify that this class knows how to transform this element's pullback.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not supported.
        """
        pass

    @abstractmethod
    def invariant_dofs(self, group, dim, sd):
        """Select the nodes of an entity that are already push-forward invariant.

        :arg group: Dict mapping node index to :class:`PhysicallyMappedFunctional`
            for the reference nodes associated with one entity.
        :arg dim: Topological dimension of the entity.
        :arg sd: Spatial dimension of the cell.
        :returns: The subset of ``group`` keys whose row of :math:`V` is
            the identity row.
        """
        pass

    @abstractmethod
    def facet_dof_rows(self, V, group, fiat_element, entity, J, processed, tol):
        """Assemble the rows of V for the non-invariant nodes on a facet.

        :arg V: Object array being assembled; rows are set in place.
        :arg group: Dict mapping node index to :class:`PhysicallyMappedFunctional`
            for the non-invariant reference nodes on this facet.
        :arg fiat_element: The FIAT element defined on the reference cell.
        :arg entity: The facet number.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows of ``V``;
            updated in place.
        :arg tol: Tolerance for detecting zeros in the numeric coefficients.
        """
        pass

    @abstractmethod
    def point_dof_rows(self, V, group, fiat_element, entity, J, processed, tol):
        """Assemble the rows of V for the non-invariant nodes away from a facet.

        :arg V: Object array being assembled; rows are set in place.
        :arg group: Dict mapping node index to :class:`PhysicallyMappedFunctional`
            for the non-invariant reference nodes on this entity.
        :arg fiat_element: The FIAT element defined on the reference cell.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows of ``V``;
            updated in place.
        :arg tol: Tolerance for detecting zeros in the numeric coefficients.
        """
        pass


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

    Push-forward invariance and completion follow directly from the
    chain rule :math:`\nabla(\hat\psi\circ F) = J^T\hat\nabla\hat\psi
    \circ F`: point values pull back unchanged; derivative nodes on a
    facet are resolved in the normal/tangential frame of
    :class:`FacetFrame`; derivative nodes elsewhere (vertex jets) have
    no geometric frame to expand in and instead act as their own
    completion group.
    """

    #: If False, physical facet moments are plain integrals rather than
    #: the measure-intrinsic integral averages of the reference nodes.
    avg = True

    def _check_mapping(self, fiat_element):
        mappings = set(fiat_element.mapping())
        if mappings != {"affine"}:
            raise NotImplementedError(
                f"{type(self).__name__} expects an affine pullback, not {mappings}.")

    def invariant_dofs(self, group, dim, sd):
        # Point values pull back exactly; only derivative nodes need work.
        return {i for i, ell in group.items() if ell.order == 0}

    def facet_dof_rows(self, V: numpy.ndarray, group: dict, fiat_element: FiniteElement,
                       entity: int, J: Node, processed: set, tol: float) -> None:
        r"""Assemble the rows of V for derivative nodes on a facet.

        Physical facet nodes take their normal component along the physical
        facet normal and their tangential components along the mapped
        reference tangents.  The pulled-back reference node is expanded in
        this frame, and the tangential remainders, being derivatives along
        mapped reference tangents, coincide with reference functionals that
        are eliminated numerically through already assembled rows of V.

        :arg V: Object array being assembled.
        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
            for the derivative nodes on this facet.
        :arg fiat_element: The FIAT element.
        :arg entity: The facet number.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows; updated in place.
        :arg tol: Tolerance for detecting zeros in the numeric coefficients.
        :arg avg: If False, physical facet moments are plain integrals rather
            than integral averages, and their columns are rescaled by the
            physical facet measure.
        """
        frame = FacetFrame(fiat_element, entity, J)
        for i, ell in group.items():
            # Split the direction into normal and tangential parts
            a, *beta = frame.reference_coefficients(ell.direction)
            if abs(a) < tol:
                # Mapped tangential derivatives are invariant
                processed.add(i)
                continue

            # Expand the pulled-back node in the physical frame
            x = frame.decompose(ell.pullback(J).direction)
            c = x[0] * frame.normal_scale / a
            if not self.avg and len(ell.points) > 1:
                # the physical moment is a plain integral, not an average
                c = c / frame.measure

            V[i, i] = c
            for k, that in enumerate(frame.tangents):
                r = x[k + 1] - c * beta[k]
                coefficients = ell.with_direction(that).evaluate(fiat_element)
                coefficients[abs(coefficients) < tol] = 0
                for j in numpy.flatnonzero(coefficients):
                    if j not in processed:
                        raise NotImplementedError(
                            f"Completion of node {i} couples to node {j}, "
                            "which has not been transformed yet.")
                    V[i] += V[j] * (r * coefficients[j])
            processed.add(i)

    def point_dof_rows(self, V: numpy.ndarray, group: dict, fiat_element: FiniteElement,
                       entity: int, J: Node, processed: set, tol: float) -> None:
        r"""Assemble the rows of V for derivative nodes away from facets.

        Away from facets there is no geometric frame, and physical nodes
        keep the reference (Cartesian) directions, so the group must span
        all directions and acts as its own completion: this is the
        affine-interpolation equivalent case, and each pulled-back node is
        expanded within the group.

        :arg V: Object array being assembled.
        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
            for the derivative nodes on this entity.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows; updated in place.
        :arg tol: Tolerance for detecting zeros in the numeric coefficients.
        """
        suborders = {}
        for i, ell in group.items():
            suborders.setdefault(ell.order, {})[i] = ell

        for sub in suborders.values():
            directions = numpy.array([ell.direction for ell in sub.values()])
            if len(set(ell.points for ell in sub.values())) > 1:
                raise NotImplementedError("Group nodes at different points.")
            if directions.shape[0] != directions.shape[1]:
                raise NotImplementedError(
                    "Directions do not span the derivative jet.")

            # coefficients of the direction basis expansion of each multi-index
            Dinv = numpy.linalg.inv(directions.T)
            for i, ell in sub.items():
                Jd = ell.pullback(J).direction
                for col, (j, ellj) in enumerate(sub.items()):
                    s = _weight_ratio(ell.weights, ellj.weights, tol)
                    x = s * Dinv[col]
                    nz = numpy.flatnonzero(abs(x) > tol)
                    V[i, j] = reduce(add, (Jd[m] * x[m] for m in nz)) if len(nz) else Zero()
                processed.add(i)


class PiolaPhysicallyMappedElement(ZanyPhysicallyMappedElement):
    r"""Mixin deriving the basis transformation for a vector- or
    tensor-valued element under the (double) contravariant Piola
    pullback.

    The roles of the normal and tangential directions are mirrored with
    respect to the scalar case: interior value moments are Piola
    invariant by construction; value moments on a facet are resolved in
    the frame of the scaled facet normal (the cofactor image of the
    reference one) and the mapped reference tangents; point evaluations
    elsewhere have no geometric frame to expand in and instead act as
    their own completion group.
    """

    @staticmethod
    def _check_piola_group(group: dict) -> None:
        """Validate that a group of non-invariant Piola-mapped nodes are all
        value moments with a nonzero value rank.

        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional.
        :raises NotImplementedError: If a node is a scalar weight or carries
            a derivative, which the theory does not yet cover.
        """
        if any(ell.rank == 0 or ell.order > 0 for ell in group.values()):
            raise NotImplementedError("Cannot yet Piola-transform this node group.")

    @staticmethod
    def _divergence_rows(V: numpy.ndarray, group: dict, J: Node, processed: set) -> dict:
        r"""Assemble the rows of V for divergence nodes and strip them from the group.

        The (double) contravariant Piola pullback commutes with the
        divergence up to the Jacobian determinant, independently of the
        entity the node sits on, so each divergence node is its own,
        trivially invariant group.

        :arg V: Object array being assembled; rows are set in place.
        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
            for the nodes on this entity.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows; updated in place.
        :returns: The remaining, non-divergence nodes of ``group``.
        """
        divs = {i: ell for i, ell in group.items() if ell.divergence}
        if divs:
            sd = J.shape[0]
            Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                              dtype=object)
            detJ = determinant(Jnp)
            for i, ell in divs.items():
                V[i, i] = detJ * ell.weights[0]
                processed.add(i)
        return {i: ell for i, ell in group.items() if i not in divs}

    @staticmethod
    def _is_cartesian_point_group(group: dict) -> bool:
        """Recognize a group of Cartesian point-value nodes sharing one point.

        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional.
        :returns: True if every node is a rank-1, order-0 node at the same
            single point, spanning exactly as many components as the group
            has members -- the same structural pattern :meth:`_piola_point_rows`
            (and :meth:`ScalarPhysicallyMappedElement.point_dof_rows`) expect,
            regardless of the entity dimension the group sits on.
        """
        points = {ell.points for ell in group.values()}
        return (len(points) == 1 and len(next(iter(points))) == 1
                and all(ell.order == 0 and ell.rank == 1 for ell in group.values()))

    @staticmethod
    def _piola_point_rows(V: numpy.ndarray, group: dict, J: Node,
                          processed: set, tol: float) -> None:
        r"""Assemble the rows of V for Cartesian point values of Piola-mapped fields.

        Physical point evaluations keep the reference (Cartesian) components,
        which pull back through the cofactor matrix :math:`K =
        \operatorname{adj}(J)^T` of the contravariant Piola map, so the group
        of components sharing a point acts as its own completion.  This is
        the same treatment regardless of which entity the point sits on: a
        single-point, rank-1 node group spanning the Cartesian components is
        not a facet moment (whose weights are genuine, usually multi-point
        quadrature averages aligned with the facet normal/tangential frame)
        even when it happens to sit on a codimension-1 entity, e.g. the edge
        dofs of :class:`~finat.alfeld_sorokina.AlfeldSorokina`.

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

    def _check_mapping(self, fiat_element):
        mappings = set(fiat_element.mapping())
        if mappings not in ({"contravariant piola"}, {"double contravariant piola"}):
            raise NotImplementedError(
                f"{type(self).__name__} expects a (double) contravariant "
                f"Piola pullback, not {mappings}.")

    def invariant_dofs(self, group, dim, sd):
        # Interior moments are Piola invariant by construction
        return {i for i, ell in group.items() if ell.order == 0 and dim == sd}

    def facet_dof_rows(self, V: numpy.ndarray, group: dict,
                       fiat_element: FiniteElement, entity: int, J: Node,
                       processed: set, tol: float) -> None:
        r"""Assemble the rows of V for facet moments of Piola-mapped values.

        This mirrors :func:`_scalar_facet_rows` with the roles of the normal
        and tangential directions exchanged: under the contravariant Piola
        map the scaled facet normal is the image of the reference one under
        the cofactor matrix :math:`K = \operatorname{adj}(J)^T`, so pure
        normal moments are invariant, while the scaled tangents map by
        :math:`J`.  Distinct nodes on a facet can share the same tangential
        directions (e.g. two RT-type facet dofs in 3D) and are only told
        apart by how their weight varies from point to point, so a node is
        identified by its full per-point profile of frame coordinates, not
        by direction alone; the pulled-back reference node mixes the
        coordinates through the frame expansion of :math:`K\hat{t}_k`, the
        tangential profiles are matched within the group, and the residual
        normal profile is eliminated numerically through already assembled
        rows of V.

        :arg V: Object array being assembled.
        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
            for the value moments on this facet.
        :arg fiat_element: The FIAT element.
        :arg entity: The facet number.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows; updated in place.
        :arg tol: Tolerance for detecting zeros in the numeric coefficients.
        """
        group = self._divergence_rows(V, group, J, processed)
        if not group:
            return
        if self._is_cartesian_point_group(group):
            # Not a genuine facet moment (see _piola_point_rows): e.g. the
            # edge dofs of AlfeldSorokina are plain Cartesian point values
            # that happen to sit on a codimension-1 entity.
            PiolaPhysicallyMappedElement._check_piola_group(group)
            self._piola_point_rows(V, group, J, processed, tol)
            return
        PiolaPhysicallyMappedElement._check_piola_group(group)
        sd = J.shape[0]
        frame = PiolaFacetFrame(fiat_element, entity, J)
        nhat = frame.normal
        Y = frame.Y

        # Reference frame coordinate profiles, shared with the physical nodes:
        # each node's own quadrature weights, decomposed into (normal,
        # tangential) frame coordinates at each of its points.  This is what
        # distinguishes nodes with the same tangential directions from one
        # another (see the point-dependence note in the docstring above).
        coords = {}
        for i, ell in group.items():
            C = ell.weights.reshape(-1, *(sd,) * ell.rank)
            for _ in range(ell.rank):
                C = numpy.tensordot(C, frame.Ghatinv, axes=(1, 1))
            coords[i] = C.reshape(len(ell.points), -1)
            # Pure normal moments are Piola invariant
            if numpy.allclose(coords[i][:, 1:], 0, atol=tol):
                processed.add(i)

        group = {i: ell for i, ell in group.items() if i not in processed}
        if not group:
            return
        rank, = {ell.rank for ell in group.values()}
        points, = {ell.points for ell in group.values()}

        # Numeric matching of the tangential coordinate profiles in the group.
        # B has full row rank by unisolvence, so the Gram matrix B @ B.T is
        # square and invertible; a rank deficiency (a genuine bug) surfaces
        # as a LinAlgError here rather than a silent least-squares fit.
        B = numpy.array([coords[j][:, 1:].ravel() for j in group])
        Binv = numpy.linalg.inv(B @ B.T) @ B
        Binv[abs(Binv) < tol] = 0

        # Numeric elimination of the residual normal profile against every
        # basis function of the element (not just this facet's own normal
        # dofs, since a completing dof may live on a different point set,
        # e.g. Guzman-Neilan's mixed-order facet dofs). The quadrature-point
        # axis is contracted purely numerically here, one reference-frame
        # multi-index at a time, so that whether a coupling is present is
        # decided from a plain numeric array (exact zero where a profile has
        # no support at these points) rather than by inspecting the type of
        # a symbolic GEM expression -- a sum of symbolic terms that is
        # identically zero for all J need not reduce to a literal
        # gem.Zero() node, so testing that structurally would either miss
        # real cancellations or (as here) spuriously keep terms whose
        # numeric coefficient actually vanishes.
        ndir = numpy.ones(())
        for _ in range(rank):
            ndir = numpy.multiply.outer(ndir, nhat)
        T = fiat_element.tabulate(0, points)[(0,) * sd]
        L = numpy.einsum("jcq,c->jq", T.reshape(T.shape[0], -1, len(points)),
                         ndir.ravel())
        L[abs(L) < tol] = 0
        # Lmap[i][m, r] = sum_q L[m, q] * coords[i][q, r]: numeric coupling
        # of every basis function to each frame coordinate of node i's own
        # profile, contracted over the shared quadrature points.
        Lmap = {i: L @ coords[i] for i in group}
        for i in Lmap:
            Lmap[i][abs(Lmap[i]) < tol] = 0

        for i, ell in group.items():
            # Pull back the coordinate profile, contracting each slot with Y
            P = coords[i].reshape(-1, *(sd,) * rank)
            for _ in range(rank):
                P = numpy.tensordot(P, Y, axes=(1, 1))
            P = P.reshape(len(points), -1)

            c = Binv @ P[:, 1:].ravel()
            V[i, list(group)] = c

            # Couple the residual normal profile to every basis function,
            # one reference-frame multi-index (or group member) at a time:
            # each numeric column of Lmap decides its own sparsity, and the
            # small symbolic frame-mixing coefficient only scales terms
            # that already survived the numeric test.
            terms = []
            for index in numpy.ndindex(*(sd,) * rank):
                flat = numpy.ravel_multi_index(index, (sd,) * rank)
                coef = Literal(1.0)
                for r in index:
                    coef = coef * Y[0, r]
                terms.append((Lmap[i][:, flat], coef))
            for k, j in enumerate(group):
                terms.append((-Lmap[j][:, 0], c[k]))

            for vec, coef in terms:
                for m in numpy.flatnonzero(vec):
                    if m not in processed:
                        raise NotImplementedError(
                            f"Completion of node {i} couples to node {m}, "
                            "which has not been transformed yet.")
                    V[i] += V[m] * (vec[m] * coef)
            processed.add(i)

    def point_dof_rows(self, V: numpy.ndarray, group: dict,
                       fiat_element: FiniteElement, entity: int, J: Node,
                       processed: set, tol: float) -> None:
        r"""Assemble the rows of V for point values of Piola-mapped fields.

        This mirrors :func:`_scalar_point_rows`: away from facets, physical
        point evaluations keep the reference (Cartesian) components, which
        pull back through the cofactor matrix :math:`K = \operatorname{adj}(J)^T`
        of the contravariant Piola map, so the group of components at each
        point acts as its own completion.

        :arg V: Object array being assembled.
        :arg group: Mapping from node index to symbolic PhysicallyMappedFunctional
            for the value nodes on this entity.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows; updated in place.
        :arg tol: Tolerance for detecting zeros in the numeric coefficients.
        """
        group = self._divergence_rows(V, group, J, processed)
        if not group:
            return
        PiolaPhysicallyMappedElement._check_piola_group(group)
        self._piola_point_rows(V, group, J, processed, tol)
