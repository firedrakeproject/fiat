from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Mapping
from numbers import Number

import gem
import numpy

from finat.citations import cite


zero = gem.Zero()
one = gem.Literal(1.0)


def _as_basis_entry(value: object) -> gem.Node:
    """Normalize a basis-transformation matrix entry.

    Parameters
    ----------
    value
        Scalar numerical or GEM matrix entry.

    Returns
    -------
    gem.Node
        Scalar GEM entry with numerical zero and one represented
        symbolically.
    """
    if isinstance(value, Number):
        if value == 0:
            return zero
        if value == 1:
            return one
    return gem.as_gem(value)


class NeedsCoordinateMappingElement(metaclass=ABCMeta):
    """Abstract class for elements that require physical information
    either to map or construct their basis functions."""

    def dual_transformation(self, Q, coordinate_mapping=None):
        raise NotImplementedError(f"Dual evaluation for {type(self).__name__} is not implemented.")


class MappedTabulation(Mapping):
    """Apply a sparse basis transformation to reference tabulations.

    Parameters
    ----------
    M : gem.ListTensor
        Basis-transformation matrix.
    ref_tabulation : Mapping
        Reference tabulations indexed by derivative order.
    indices : iterable of int, optional
        Rows retained by an element restriction.

    Notes
    -----
    The transformation is represented as ``M = U + W``, where ``U`` holds
    unit entries and ``W`` the remaining nonzeros.  Each part is a ragged
    row contraction.  Exact row lengths avoid branches and padded-zero
    arithmetic, while separating ``U`` avoids multiplication by one.  The
    resulting GEM expression still exposes the transformation as a finite
    element linear map before quadrature contraction.
    """

    def __init__(
            self, M: gem.ListTensor, ref_tabulation: Mapping,
            indices: Iterable[int] | None = None) -> None:
        self.ref_tabulation = ref_tabulation
        if indices is None:
            indices = range(M.shape[0])
        self.indices = tuple(indices)

        nonzero_rows = []
        for source_row in self.indices:
            row = []
            for column in range(M.shape[1]):
                value = _as_basis_entry(M.array[source_row, column])
                if not isinstance(value, gem.Zero):
                    row.append((column, value))
            nonzero_rows.append(row)
        groups = []
        nrows = len(self.indices)
        for unit in (True, False):
            rows = [
                [(column, value) for column, value in row
                 if (value == one) == unit]
                for row in nonzero_rows
            ]
            lengths = numpy.asarray(tuple(map(len, rows)), dtype=gem.uint_type)
            width = int(lengths.max(initial=0))
            if width == 0:
                continue
            columns = numpy.zeros((nrows, width), dtype=gem.uint_type)
            data = None if unit \
                else numpy.full((nrows, width), zero, dtype=object)
            for row, entries in enumerate(rows):
                columns[row, :len(entries)] = tuple(
                    column for column, _ in entries)
                if data is not None:
                    data[row, :len(entries)] = tuple(
                        gem.as_gem(value) for _, value in entries)
            groups.append((
                lengths,
                gem.Literal(columns, dtype=gem.uint_type),
                None if unit else gem.ListTensor(data),
            ))
        self._sparse_maps = tuple(groups)
        self._tabulation_cache = {}

    def matvec(self, table: gem.Node) -> gem.Node:
        """Transform one reference tabulation.

        Parameters
        ----------
        table
            Reference tabulation with the basis axis first.

        Returns
        -------
        gem.Node
            Tabulation whose first axis is the transformed basis axis.
        """
        row = gem.Index(extent=len(self.indices))
        tail = gem.indices(len(table.shape) - 1)
        terms = []
        for lengths, column_table, coefficients in self._sparse_maps:
            entry = gem.RaggedIndex(
                extent=column_table.shape[1],
                parents=(row,), lengths=lengths)
            column = gem.VariableIndex(
                gem.Indexed(column_table, (row, entry)))
            basis = gem.Indexed(table, (column, *tail))
            if coefficients is not None:
                basis = gem.Product(
                    gem.Indexed(coefficients, (row, entry)), basis)
            terms.append(gem.IndexSum(basis, (entry,)))

        mapped = gem.Sum(*terms)
        return gem.ComponentTensor(mapped, (row, *tail))

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
