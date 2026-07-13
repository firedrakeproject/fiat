from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import reduce
from operator import add

import gem
import numpy

from finat.citations import cite
from finat.functional import PhysicallyMappedFunctional


class NeedsCoordinateMappingElement(metaclass=ABCMeta):
    """Abstract class for elements that require physical information
    either to map or construct their basis functions."""

    def dual_transformation(self, Q, coordinate_mapping=None):
        raise NotImplementedError(f"Dual evaluation for {type(self).__name__} is not implemented.")


class MappedTabulation(Mapping):
    """A lazy tabulation dict that applies the basis transformation only
    on the requested derivatives.

    :arg M: a gem.ListTensor with the basis transformation matrix.
    :arg ref_tabulation: a dict of tabulations on the reference cell.
    :kwarg indices: an optional list of restriction indices on the basis functions.
    """
    def __init__(self, M, ref_tabulation, indices=None):
        self.M = M
        self.ref_tabulation = ref_tabulation
        if indices is None:
            indices = list(range(M.shape[0]))
        self.indices = indices
        # we expect M to be sparse with O(1) nonzeros per row
        # for each row, get the column index of each nonzero entry
        csr = [[j for j in range(M.shape[1]) if not isinstance(M.array[i, j], gem.Zero)]
               for i in indices]
        self.csr = csr
        self._tabulation_cache = {}

    def matvec(self, table):
        # basis recombination using hand-rolled sparse-dense matrix multiplication
        ii = gem.indices(len(table.shape)-1)
        phi = [gem.Indexed(table, (j, *ii)) for j in range(self.M.shape[1])]
        # the sum approach is faster than calling numpy.dot or gem.IndexSum
        exprs = [gem.ComponentTensor(gem.Sum(*(self.M.array[i, j] * phi[j] for j in js)), ii)
                 for i, js in zip(self.indices, self.csr)]

        result = gem.ListTensor(exprs)
        result, = gem.optimise.unroll_indexsum((result,), lambda index: True)
        # result = gem.optimise.aggressive_unroll(self.M @ table)
        return result

    def __getitem__(self, alpha):
        try:
            return self._tabulation_cache[alpha]
        except KeyError:
            result = self.matvec(self.ref_tabulation[alpha])
            return self._tabulation_cache.setdefault(alpha, result)

    def __iter__(self):
        return iter(self.ref_tabulation)

    def __len__(self):
        return len(self.ref_tabulation)


class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions.

    Concrete elements either implement :meth:`basis_transformation`
    entirely by hand, or derive it automatically by mixing in
    :class:`~finat.zany.ScalarPhysicallyMappedElement` or
    :class:`~finat.zany.PiolaPhysicallyMappedElement`, which supply the
    four hooks below and inherit the entity-by-entity assembly loop
    implemented here.
    """

    #: Numerical tolerance used throughout automatic basis transformation
    #: to detect vanishing coefficients.
    tol = 1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cite("Kirby2018zany")
        cite("Kirby2019zany")
        self.restriction_indices = None

    def basis_transformation(self, coordinate_mapping):
        r"""Assemble the basis transformation :math:`M = V^T`.

        Following the factorization :math:`V = E V^c D` of Kirby (2017)
        and Brubeck & Kirby (2025), the matrix :math:`V` relating the
        reference nodes to the push-forwards of the physical nodes is
        assembled one topological entity at a time, in increasing
        dimension, so that the completion of a node on an entity can
        always be resolved against the already-assembled rows of
        lower-dimensional entities.

        On each entity, nodes that are already push-forward invariant
        (:meth:`_invariant_dofs`) contribute an identity row for free,
        since :math:`V` starts out as the identity.  The rest are
        assembled by :meth:`_facet_dof_rows` (on a codimension-1
        entity) or :meth:`_point_dof_rows` (elsewhere); these three
        hooks, together with :meth:`_check_mapping`, encode the
        mapping-specific (affine or Piola) part of the theory, and this
        method contains no knowledge of which mapping is in play.

        :arg coordinate_mapping: Object providing physical geometry.
        """
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
        return gem.ListTensor(V[:, :ndof].T)

    def _check_mapping(self, fiat_element):
        """Verify that this class knows how to transform this element's pullback.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement automatic basis transformation.")

    def _invariant_dofs(self, group, dim, sd):
        """Select the nodes of an entity that are already push-forward invariant.

        :arg group: Dict mapping node index to :class:`PhysicallyMappedFunctional`
            for the reference nodes associated with one entity.
        :arg dim: Topological dimension of the entity.
        :arg sd: Spatial dimension of the cell.
        :returns: The subset of ``group`` keys whose row of :math:`V` is
            the identity row.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement automatic basis transformation.")

    def _facet_dof_rows(self, V, group, fiat_element, entity, J, processed):
        """Assemble the rows of V for the non-invariant nodes on a facet.

        :arg V: Object array being assembled; rows are set in place.
        :arg group: Dict mapping node index to :class:`PhysicallyMappedFunctional`
            for the non-invariant reference nodes on this facet.
        :arg fiat_element: The FIAT element defined on the reference cell.
        :arg entity: The facet number.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows of ``V``;
            updated in place.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement automatic basis transformation.")

    def _point_dof_rows(self, V, group, fiat_element, J, processed):
        """Assemble the rows of V for the non-invariant nodes away from a facet.

        :arg V: Object array being assembled; rows are set in place.
        :arg group: Dict mapping node index to :class:`PhysicallyMappedFunctional`
            for the non-invariant reference nodes on this entity.
        :arg fiat_element: The FIAT element defined on the reference cell.
        :arg J: GEM expression for the cell Jacobian.
        :arg processed: Indices of the already assembled rows of ``V``;
            updated in place.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement automatic basis transformation.")

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        M = self.basis_transformation(coordinate_mapping)
        return MappedTabulation(M, ref_tabulation, indices=self.restriction_indices)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        result = super().basis_evaluation(order, ps, entity=entity)
        return self.map_tabulation(result, coordinate_mapping)

    def dual_transformation(self, Q, coordinate_mapping=None):
        M = self.basis_transformation(coordinate_mapping)

        M = M.array
        if M.shape[1] > M.shape[0]:
            M = M[:, :M.shape[0]]

        M_dual = inverse(M.T)
        if self.restriction_indices is not None:
            indices = self.restriction_indices
            M_dual = M_dual[numpy.ix_(indices, indices)]
        M_dual = gem.ListTensor(M_dual)

        key = None
        return MappedTabulation(M_dual, {key: Q})[key]


class DirectlyDefinedElement(NeedsCoordinateMappingElement):
    """Base class for directly defined elements such as direct
    serendipity that bypass a coordinate mapping."""
    pass


class PhysicalGeometry(metaclass=ABCMeta):

    @abstractmethod
    def cell_size(self):
        """The cell size at each vertex.

        :returns: A GEM expression for the cell size, shape (nvertex, ).
        """

    @abstractmethod
    def jacobian_at(self, point):
        """The jacobian of the physical coordinates at a point.

        :arg point: The point in reference space (on the cell) to
             evaluate the Jacobian.
        :returns: A GEM expression for the Jacobian, shape (gdim, tdim).
        """

    @abstractmethod
    def detJ_at(self, point):
        """The determinant of the jacobian of the physical coordinates at a point.

        :arg point: The point in reference space to evaluate the Jacobian determinant.
        :returns: A GEM expression for the Jacobian determinant.
        """

    @abstractmethod
    def reference_normals(self):
        """The (unit) reference cell normals for each facet.

        :returns: A GEM expression for the normal to each
           facet (numbered according to FIAT conventions), shape
           (nfacet, tdim).
        """

    @abstractmethod
    def physical_normals(self):
        """The (unit) physical cell normals for each facet.

        :returns: A GEM expression for the normal to each
           facet (numbered according to FIAT conventions).  These are
           all computed by a clockwise rotation of the physical
           tangents, shape (nfacet, gdim).
        """

    @abstractmethod
    def physical_tangents(self):
        """The (unit) physical cell tangents on each facet.

        :returns: A GEM expression for the tangent to each
           facet (numbered according to FIAT conventions).  These
           always point from low to high numbered local vertex, shape
           (nfacet, gdim).
        """

    @abstractmethod
    def physical_edge_lengths(self):
        """The length of each edge of the physical cell.

        :returns: A GEM expression for the length of each
           edge (numbered according to FIAT conventions), shape
           (nfacet, ).
        """

    @abstractmethod
    def physical_points(self, point_set, entity=None):
        """Maps reference element points to GEM for the physical coordinates

        :arg point_set: A point_set on the reference cell to push forward to physical space.
        :arg entity: Reference cell entity on which the point set is
                     defined (for example if it is a point set on a facet).
        :returns: a GEM expression for the physical locations of the
                  points, shape (gdim, ) with free indices of the point_set.
        """

    @abstractmethod
    def physical_vertices(self):
        """Physical locations of the cell vertices.

        :returns: a GEM expression for the physical vertices, shape
                (gdim, )."""


zero = gem.Zero()
one = gem.Literal(1.0)


def identity(*shape):
    V = numpy.eye(*shape, dtype=object)
    for multiindex in numpy.ndindex(V.shape):
        V[multiindex] = zero if V[multiindex] == 0 else one
    return V


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


def determinant(A):
    """Returns the determinant of A"""
    n = A.shape[0]
    if n == 0:
        return 1
    elif n == 1:
        return A[0, 0]
    elif n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    else:
        detA = A[0, 0] * determinant(A[1:, 1:])
        cols = numpy.ones(A.shape[1], dtype=bool)
        for j in range(1, n):
            cols[j] = False
            detA += (-1)**j * A[0, j] * determinant(A[1:][:, cols])
            cols[j] = True
        return detA


def adjugate(A):
    """Returns the adjugate matrix of A"""
    A = numpy.asarray(A)
    C = numpy.zeros_like(A)
    rows = numpy.ones(A.shape[0], dtype=bool)
    cols = numpy.ones(A.shape[1], dtype=bool)
    for i in range(A.shape[0]):
        rows[i] = False
        for j in range(A.shape[1]):
            cols[j] = False
            C[j, i] = (-1)**(i+j)*determinant(A[rows, :][:, cols])
            cols[j] = True
        rows[i] = True
    return C


def inverse(A):
    """Returns the inverse of A.

    Exploits block-diagonal structure with repeated blocks.
    """
    m, n = A.shape
    if m != n:
        raise ValueError("A must be square.")
    M = A.copy()
    cache = {}
    candidates = set(range(m))
    while len(candidates) > 0:
        # Extract a connected component
        seed = {min(candidates)}
        while True:
            ids = set(seed)
            for i in seed:
                ids.update(j for j in candidates if not isinstance(M[j, i], gem.Zero))
                ids.update(j for j in candidates if not isinstance(M[i, j], gem.Zero))
            if len(ids) == len(seed):
                break
            seed = ids
        candidates -= ids
        ids = list(ids)
        Mii = M[numpy.ix_(ids, ids)]

        # Have we already done this?
        key = gem.ListTensor(Mii)
        try:
            Minv = cache[key]
        except KeyError:
            Minv = adjugate(Mii) / determinant(Mii)
            cache[key] = Minv

        M[numpy.ix_(ids, ids)] = Minv
    return M
