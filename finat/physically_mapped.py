from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from numbers import Number

import gem
import numpy

from gem.optimise import canonical_bound_indices, factorise_scalar_sums

from finat.citations import cite
from finat.finiteelementbase import broadcast_tensor


zero = gem.Zero()
one = gem.Literal(1.0)


class NeedsCoordinateMappingElement(metaclass=ABCMeta):
    """Abstract class for elements that require physical information
    either to map or construct their basis functions."""

    def dual_transformation(self, Q, coordinate_mapping=None):
        raise NotImplementedError(f"Dual evaluation for {type(self).__name__} is not implemented.")


class MappedTabulation(Mapping):
    """Apply entity blocks of a basis transformation to reference tabulations.

    Parameters
    ----------
    transformation : gem.Node
        Matrix mapping reference basis functions to physical basis functions.
        Its nonzero blocks couple physical and reference entity functionals.
    ref_tabulation : Mapping
        Reference tabulations indexed by derivative order.
    ref_entity_dofs : Mapping
        Reference degrees of freedom on each entity, including constraint
        functions removed from the physical element.
    restriction_indices : sequence of int, optional
        Indices of the physical basis functions retained by a restriction.
        Their order defines the ordering of the restricted basis.

    Notes
    -----
    Basis functions with common reference entity support form dense
    transformation blocks. This retains diagonal jet maps and their
    couplings to higher-dimensional entities, including constraint functions.
    Every reduction has a fixed extent and contains only nonzero entries.
    """

    def __init__(self, transformation, ref_tabulation, ref_entity_dofs,
                 restriction_indices=None):
        self.ref_tabulation = ref_tabulation
        self._tabulation_cache = {}
        if restriction_indices is None:
            restriction_indices = range(transformation.shape[0])
        matrix = transformation.array
        groups = defaultdict(list)
        for dof, basis_index in enumerate(restriction_indices):
            support = []
            for dim, entities in ref_entity_dofs.items():
                for entity, dofs in entities.items():
                    columns = []
                    for column in dofs:
                        value = matrix[basis_index, column]
                        if (isinstance(value, gem.Zero)
                                or isinstance(value, Number) and value == 0):
                            continue
                        columns.append(column)
                    if columns:
                        support.append(((dim, entity), tuple(columns)))
            if support:
                groups[tuple(support)].append((dof, basis_index))
        self._blocks = []
        self._dimension = len(restriction_indices)
        for support, entries in groups.items():
            columns = tuple(column for _, dofs in support for column in dofs)
            coefficients = gem.ListTensor([
                [factorise_scalar_sums(gem.as_gem(matrix[basis_index, c])) for c in columns]
                for _, basis_index in entries])
            self._blocks.append((coefficients, columns, tuple(dof for dof, _ in entries)))

    def __getitem__(self, alpha):
        if alpha not in self._tabulation_cache:
            table = self.ref_tabulation[alpha]
            tail = gem.indices(len(table.shape) - 1)
            values = [zero] * self._dimension
            for coefficients, columns, positions in self._blocks:
                r, k = gem.indices(2)
                columns = gem.Literal(columns, dtype=gem.uint_type)
                reference = gem.VariableIndex(gem.Indexed(columns, (k,)))
                mapped = gem.IndexSum(gem.Product(
                    gem.Indexed(coefficients, (r, k)),
                    gem.Indexed(table, (reference, *tail))), (k,))
                block = gem.ComponentTensor(mapped, (r,))
                for local, position in enumerate(positions):
                    values[position] = gem.Indexed(block, (local,))
            r = gem.Index(extent=self._dimension)
            mapped = gem.Indexed(gem.ListTensor(values), (r,))
            self._tabulation_cache[alpha] = canonical_bound_indices(
                broadcast_tensor(mapped, (r, *tail)))
        return self._tabulation_cache[alpha]

    def __iter__(self):
        return iter(self.ref_tabulation)

    def __len__(self):
        return len(self.ref_tabulation)


class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cite("Kirby2018zany")
        cite("Kirby2019zany")
        self.restriction_indices = None

    @abstractmethod
    def basis_transformation(self, coordinate_mapping):
        """Transformation matrix for the basis functions.

        :arg coordinate_mapping: Object providing physical geometry."""
        pass

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        M = self.basis_transformation(coordinate_mapping)
        return MappedTabulation(M, ref_tabulation, self._element.entity_dofs(),
                                restriction_indices=self.restriction_indices)

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
        return MappedTabulation(M_dual, {key: Q}, self.entity_dofs())[key]


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


def identity(*shape):
    V = numpy.eye(*shape, dtype=object)
    for multiindex in numpy.ndindex(V.shape):
        V[multiindex] = zero if V[multiindex] == 0 else one
    return V


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
