r"""Physically mapped elements and the automatic basis transformation.

The physical basis functions are :math:`M F^*(\hat\Psi)`, with
:math:`M = V^T`, :math:`V = B^{-1}`, and :math:`B_{ij} = n_i(\hat\psi_j
\circ F^{-1})` the Vandermonde matrix of the physical nodes :math:`n_i`
on the pulled-back reference nodal basis :math:`\hat\psi_j` (Kirby 2017,
Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Mapping
from functools import cached_property, reduce
from operator import add

import gem
import numpy
from gem.optimise import factorise_scalar_sums

from finat.citations import cite
from finat.functional import Functional, one, zero


class NeedsCoordinateMappingElement(metaclass=ABCMeta):
    """Abstract class for elements that require physical information
    either to map or construct their basis functions."""

    def dual_transformation(self, Q, coordinate_mapping=None):
        raise NotImplementedError(f"Dual evaluation for {type(self).__name__} is not implemented.")


class MappedTabulation(Mapping):
    """A lazy tabulation dict that applies the basis transformation only
    on the requested derivatives.

    Rows are padded to a common number of entries, so that a loop over the
    basis index has an affine iteration domain.  Constant tables select the
    reference column and one of the distinct symbolic coefficients, which
    shares equal entries without materialising the matrix entry by entry.

    :arg M: a gem.ListTensor with the basis transformation matrix.
    :arg ref_tabulation: a dict of tabulations on the reference cell.
    :kwarg indices: an optional list of restriction indices on the basis functions.
    """

    def __init__(
            self, M: gem.ListTensor, ref_tabulation: Mapping,
            indices: Iterable[int] | None = None) -> None:
        self.ref_tabulation = ref_tabulation
        if indices is None:
            indices = range(M.shape[0])
        self.indices = tuple(indices)
        self._space_dim = len(self.indices)
        self._value_dim = M.shape[1]

        nonzero_rows = []
        for source_row in self.indices:
            row = []
            for column in range(M.shape[1]):
                value = M.array[source_row, column]
                if not isinstance(value, gem.Zero):
                    row.append((column, value))
            nonzero_rows.append(row)
        width = max((len(row) for row in nonzero_rows), default=0)
        nrows = len(self.indices)
        columns = numpy.zeros((nrows, width), dtype=gem.uint_type)
        data = numpy.full((nrows, width), zero, dtype=object)
        for index, row in enumerate(nonzero_rows):
            columns[index, :len(row)] = tuple(column for column, _ in row)
            data[index, :len(row)] = tuple(
                factorise_scalar_sums(gem.as_gem(value)) for _, value in row)
        self._width = width
        self._columns = gem.Literal(columns, dtype=gem.uint_type)
        values = []
        value_numbers = {}
        value_indices = numpy.empty(data.shape, dtype=gem.uint_type)
        for multiindex, value in numpy.ndenumerate(data):
            try:
                number = value_numbers[value]
            except KeyError:
                number = len(values)
                value_numbers[value] = number
                values.append(value)
            value_indices[multiindex] = number
        self._value_indices = gem.Literal(value_indices, dtype=gem.uint_type)
        self._values = gem.ListTensor(values)
        self._tabulation_cache = {}

    @cached_property
    def _reference_index(self) -> gem.Index:
        """Contraction over the reference basis, shared by all tabulations."""
        return gem.Index(extent=self._value_dim)

    @cached_property
    def _row_index(self) -> gem.Index:
        """Contraction over a padded row, shared by all tabulations."""
        return gem.Index(extent=self._width)

    def _entry(self, r: gem.Index, a: gem.Index) -> gem.Node:
        """Entry ``M[r, a]`` of the basis transformation.

        :arg r: index over the rows retained by the element
        :arg a: index over the reference basis
        :returns: a sum over the padded row of an interned entry against a
                  Delta selecting its column, so that contracting either axis
                  of ``M`` is ordinary GEM algebra
        """
        k = self._row_index
        entry = gem.Indexed(
            self._values,
            (gem.VariableIndex(gem.Indexed(self._value_indices, (r, k))),))
        column = gem.VariableIndex(gem.Indexed(self._columns, (r, k)))
        return gem.IndexSum(gem.Product(entry, gem.Delta(column, a)), (k,))

    def matmul(self, table: gem.Node) -> gem.Node:
        """Apply the basis transformation to a reference tabulation."""
        r = gem.Index(extent=self._space_dim)
        a = self._reference_index
        tail = gem.indices(len(table.shape) - 1)
        mapped = gem.IndexSum(
            gem.Product(self._entry(r, a), gem.Indexed(table, (a, *tail))), (a,))
        return gem.ComponentTensor(mapped, (r, *tail))

    def __getitem__(self, alpha):
        try:
            return self._tabulation_cache[alpha]
        except KeyError:
            result = self.matmul(self.ref_tabulation[alpha])
            return self._tabulation_cache.setdefault(alpha, result)

    def __iter__(self):
        return iter(self.ref_tabulation)

    def __len__(self):
        return len(self.ref_tabulation)


class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions.

    :meth:`basis_transformation` derives the transformation from the
    FIAT dual basis by the duality formulation of the module docstring.
    Elements override :attr:`avg` and :meth:`dof_scale` to record the
    normalization conventions of their physical nodes.
    """

    #: Relative tolerance below which entries of the numeric reference
    #: tabulations are dropped from the symbolic rows.
    tol = 1e-10

    #: If False, physical scalar facet moments are plain integrals rather
    #: than the measure-intrinsic integral averages of the reference nodes.
    avg = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cite("Kirby2018zany")
        cite("Kirby2019zany")
        self.restriction_indices = None

    def basis_transformation(self, coordinate_mapping):
        r"""Transformation matrix for the basis functions.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A ``gem.ListTensor`` of shape ``(space_dimension(),
            nbf)``, with the trailing constraint columns of an extended
            element truncated.
        """
        V = self.physical_vandermonde(coordinate_mapping).inverse()
        for i, scale in enumerate(self.dof_scales(coordinate_mapping)):
            if scale is not None:
                V[:, i] = V[:, i] * scale
        ndof = self.space_dimension()
        return gem.ListTensor(V[:, :ndof].T)

    def dual_transformation(self, Q, coordinate_mapping=None):
        r"""Transforms reference dual evaluation into physical dual evaluation.

        The physical nodes of :math:`f` are :math:`n_i(f) = \sum_j B_{ij}
        \hat n_j(F^* f)`, so the reference weights are transformed by
        the physical Vandermonde matrix :math:`B`, restricted to the
        exposed dofs.

        :arg Q: The reference dual evaluation gem weight tensor
            mapping quadrature points to reference degrees of freedom.
        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: The physical dual evaluation gem weight tensor.
        """
        B = self.physical_vandermonde(coordinate_mapping).matrix()
        for i, scale in enumerate(self.dof_scales(coordinate_mapping)):
            if scale is not None:
                B[i] = B[i] / scale
        indices = self.restriction_indices
        if indices is None:
            indices = list(range(self.space_dimension()))
        B = gem.ListTensor(B[numpy.ix_(indices, indices)])

        key = None
        return MappedTabulation(B, {key: Q})[key]

    def physical_vandermonde(self, coordinate_mapping):
        """The physical Vandermonde matrix of the element.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: The :class:`PhysicalVandermondeMatrix`.
        """
        ref_el = self._element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        bary, = ref_el.make_points(sd, 0, sd + 1)
        if sd == 1:
            # The tangent derivative on a one-dimensional manifold carries
            # the orientation in the signed Jacobian determinant.
            J = gem.ListTensor([[coordinate_mapping.detJ_at(bary)]])
        else:
            J = coordinate_mapping.jacobian_at(bary)
        jacobian = Jacobian(J)
        return PhysicalVandermondeMatrix(self._element, self.functionals(), jacobian, self.tol, avg=self.avg)

    def functionals(self):
        """The degrees of freedom as coefficient tensors.

        Constraint functionals of an extended element that cannot be
        parsed are not exposed as physical dofs: their physical
        counterparts are the push-forwards of the reference ones (Kirby
        2017, section 5), with identity rows.

        :returns: A list with a :class:`~finat.functional.Functional` per
            dof, or None for a dof keeping an identity row.
        """
        fiat_element = self._element
        fiat_nodes = fiat_element.dual_basis()
        mappings = fiat_element.mapping()
        ndof = self.space_dimension()
        functionals = [None] * len(fiat_nodes)
        for dim, entities in fiat_element.entity_dofs().items():
            for entity, dofs in entities.items():
                for i in dofs:
                    try:
                        functionals[i] = Functional(fiat_nodes[i], (dim, entity), mappings[i])
                    except NotImplementedError:
                        if i < ndof:
                            raise
        return functionals

    def dof_scales(self, coordinate_mapping):
        r"""The rescaling of the physical degrees of freedom by powers of the cell size.

        Each dof is rescaled by :meth:`dof_scale` evaluated with the cell
        size averaged over the vertices of the dof's entity.  This is
        the FInAT convention keeping the mass matrix well-conditioned; it
        is consistent across cells because the scaling only depends on
        shared entities.  The columns of :math:`V` are multiplied by the
        scaling, and the rows of :math:`B` are divided by it.

        :arg coordinate_mapping: Object providing the physical geometry as
            GEM expressions.
        :returns: A list with the GEM scaling factor of each dof of the
            FIAT element, or ``None`` for no rescaling.
        """
        # cell_size may be a GEM expression or a numpy array of numbers
        h = coordinate_mapping.cell_size()
        fiat_element = self._element
        top = fiat_element.get_reference_element().get_topology()
        nodes = fiat_element.dual_basis()
        entity_ids = fiat_element.entity_dofs()
        scales = [None] * len(nodes)
        for dim in entity_ids:
            for entity in entity_ids[dim]:
                verts = top[dim][entity]
                havg = reduce(add, (h[v] for v in verts)) / len(verts)
                for i in entity_ids[dim][entity]:
                    scales[i] = self.dof_scale(nodes[i], dim, havg)
        return scales

    def dof_scale(self, node, dim, havg):
        r"""Return the conditioning rescaling factor of one physical dof.

        The default convention redefines each physical node of derivative
        order :math:`m > 0` with a factor :math:`h^{-m}`; elements whose
        hand-coded transformations established a different convention
        (e.g. the :math:`h^{-2}` vertex values of Hu-Zhang) override this
        method.

        :arg node: The FIAT functional of the dof.
        :arg dim: Topological dimension of the entity the dof sits on.
        :arg havg: GEM scalar for the cell size averaged over the vertices
            of the dof's entity.
        :returns: The GEM scaling factor, or ``None`` for no rescaling.
        """
        order = node.max_deriv_order
        return havg**(-order) if order > 0 else None

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        M = self.basis_transformation(coordinate_mapping)
        return MappedTabulation(M, ref_tabulation, indices=self.restriction_indices)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        result = super().basis_evaluation(order, ps, entity=entity)
        return self.map_tabulation(result, coordinate_mapping)


class DirectlyDefinedElement(NeedsCoordinateMappingElement):
    """Base class for directly defined elements such as direct
    serendipity that bypass a coordinate mapping."""
    pass


class PhysicalGeometry(metaclass=ABCMeta):
    """Geometry callbacks for physical cells."""

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
        :returns: A GEM expression for the Jacobian, shape ``(gdim, tdim)``.
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


def connected_components(A):
    """Returns the connected components of the sparsity pattern of A.

    :arg A: A square object array of GEM scalars.
    :returns: A list of sorted lists of indices.
    """
    m, n = A.shape
    if m != n:
        raise ValueError("A must be square.")
    components = []
    candidates = set(range(m))
    while len(candidates) > 0:
        seed = {min(candidates)}
        while True:
            ids = set(seed)
            for i in seed:
                ids.update(j for j in candidates if not isinstance(A[j, i], gem.Zero))
                ids.update(j for j in candidates if not isinstance(A[i, j], gem.Zero))
            if len(ids) == len(seed):
                break
            seed = ids
        candidates -= ids
        components.append(sorted(ids))
    return components


def solve(A, B, cache=None):
    """Solves A X = B for a matrix A of GEM scalars.

    Each connected component of A is solved through its adjugate and
    determinant, which are cached so that identical blocks (e.g. the
    vertex jets of every vertex) are only inverted once.

    :arg A: A square object array of GEM scalars.
    :arg B: An object array with the right-hand sides as columns.
    :arg cache: An optional dict of adjugate and determinant pairs.
    :returns: X as an object array.
    """
    if cache is None:
        cache = {}
    X = numpy.full(B.shape, zero, dtype=object)
    for ids in connected_components(A):
        Aii = A[numpy.ix_(ids, ids)]
        if len(ids) == 1:
            d = Aii[0, 0]
            X[ids] = B[ids] if d == one else B[ids] / d
            continue
        key = gem.ListTensor(Aii)
        try:
            adj, det = cache[key]
        except KeyError:
            adj, det = cache.setdefault(key, (adjugate(Aii), determinant(Aii)))
        X[ids] = (adj @ B[ids]) / det
    return X


def pseudoinverse(A):
    """Returns the Moore-Penrose pseudoinverse of a full-rank matrix.

    :arg A: An object array of GEM scalars with at least as many rows as
        columns.
    :returns: The Moore-Penrose pseudoinverse, formed as
        ``solve(A.T @ A, A.T)``.
    """
    m, n = A.shape
    if m < n:
        raise ValueError("A must have at least as many rows as columns.")
    return solve(A.T @ A, A.T)


class Jacobian:
    r"""The cell Jacobian and its derived matrices, as object arrays of GEM scalars.

    Following FInAT, :math:`J = \partial x/\partial\hat{x}` maps
    reference to physical coordinates; the papers' Jacobian is its
    inverse.

    For a rectangular Jacobian, :attr:`detJ` is the positive metric volume
    factor :math:`\sqrt{\det(J^T J)}`.  The adjugate and cofactor matrix are
    only defined here for square Jacobians; :func:`pseudoinverse` handles
    full-rank rectangular Jacobians.

    :arg J: A GEM expression for the Jacobian, shape ``(gdim, tdim)``.
    """
    def __init__(self, J):
        if len(J.shape) != 2:
            raise ValueError(f"The Jacobian must be a matrix, got shape {J.shape}.")
        gdim, tdim = J.shape
        self.J = numpy.array([[J[i, k] for k in range(tdim)] for i in range(gdim)], dtype=object)
        if gdim == tdim:
            self.detJ = determinant(self.J)
            self.adjJ = adjugate(self.J)
            #: the cofactor matrix, which maps normals to normals
            self.K = self.adjJ.T
        else:
            JTJ = self.J.T @ self.J
            self.detJ = determinant(JTJ)**0.5
            self.adjJ = None
            self.K = self.detJ * pseudoinverse(self.J).T


class PhysicalVandermondeMatrix:
    r"""The generalized Vandermonde matrix of the physical nodes.

    :math:`B_{ij} = n_i(\hat\psi_j\circ F^{-1})` pairs the physical
    nodes with the pulled-back reference nodal basis.  Every row is
    supported on the dofs of the node's own entity and of the entities
    in its closure, since the trace of a basis function on an entity is
    determined by the dofs on the closure of that entity (which is what
    makes the element conforming), so :math:`B` is block lower
    triangular in the entity order.

    :arg fiat_element: The FIAT element.
    :arg functionals: A :class:`~finat.functional.Functional` per dof, or
        None for a dof whose physical node is the push-forward of the
        reference one.
    :arg jacobian: The :class:`Jacobian`.
    :arg tol: Relative tolerance below which numeric couplings are dropped.
    :arg avg: Whether physical facet moments are integral averages, as
        the reference nodes are; if not, they carry the physical facet
        measure.
    """
    def __init__(self, fiat_element, functionals, jacobian, tol, avg=True):
        self.fiat_element = fiat_element
        self.functionals = functionals
        self.jacobian = jacobian
        self.tol = tol
        self.avg = avg

    def rows(self):
        r"""Assembles the rows of :math:`B` that are not rows of the identity.

        Each row is kept in numerator/denominator form, and is checked
        to be supported on the closure of its entity.

        :returns: A dict mapping the dof to its ``(numerator, denominator)``
            pair of an object array of GEM scalars and a GEM scalar.
        """
        fiat_element = self.fiat_element
        nbf = fiat_element.space_dimension()
        nodes = {i: node for i, node in enumerate(self.functionals)
                 if node is not None and not node.is_invariant(self.jacobian, self.tol)}
        if not nodes:
            return {}
        # tabulate once at the points of all the nodes, as DualSet.to_riesz does
        points = sorted({pt for node in nodes.values() for pt in node.points})
        order = max(node.order for node in nodes.values())
        tabulation = fiat_element.tabulate(order, points)
        columns = {pt: q for q, pt in enumerate(points)}

        closure_dofs = fiat_element.entity_closure_dofs()
        rows = {}
        for dim, entities in fiat_element.entity_dofs().items():
            for entity, dofs in entities.items():
                allowed = set(closure_dofs[dim][entity])
                for i in dofs:
                    if i not in nodes:
                        continue
                    T = nodes[i].arrange_tabulation(tabulation, columns)
                    numerator, denominator = nodes[i].action(T, self.jacobian, self.tol, avg=self.avg)
                    outside = [j for j in range(nbf) if j not in allowed
                               and not isinstance(numerator[j], gem.Zero)]
                    if outside:
                        raise NotImplementedError(
                            f"Dof {i} on entity {(dim, entity)} couples to dofs "
                            f"{outside} outside the closure of its entity.")
                    rows[i] = (numerator, denominator)
        return rows

    def matrix(self):
        r"""Computes :math:`B`.

        :returns: :math:`B` as an object array of GEM scalars.
        """
        B = identity(self.fiat_element.space_dimension())
        for i, (numerator, denominator) in self.rows().items():
            B[i] = numerator / denominator
        return B

    def inverse(self):
        r"""Computes :math:`V = B^{-1}` by block back-substitution over the entities.

        Entities are visited in order of increasing dimension, and for
        each entity :math:`e` with closure dofs :math:`c`

        .. math:: V_e = B_{ee}^{-1}(I_e - B_{ec} V_c).

        Starting from the identity, only the rows of entities carrying
        transformed nodes are replaced, and the rows of :math:`B` are
        kept in numerator/denominator form so that only polynomial
        blocks are inverted.

        :returns: :math:`V` as an object array of GEM scalars.
        """
        fiat_element = self.fiat_element
        nbf = fiat_element.space_dimension()
        entity_dofs = fiat_element.entity_dofs()
        closure_dofs = fiat_element.entity_closure_dofs()
        rows = self.rows()
        V = identity(nbf)
        cache = {}
        for dim in sorted(entity_dofs):
            for entity, block in entity_dofs[dim].items():
                if not any(i in rows for i in block):
                    continue
                closure = [j for j in closure_dofs[dim][entity] if j not in block]
                # B_ee V_e = D_e - B_ec V_c, with B = diag(1/D) numerators
                n = len(block)
                Bee = numpy.full((n, n), zero, dtype=object)
                rhs = numpy.full((n, nbf), zero, dtype=object)
                for k, i in enumerate(block):
                    if i not in rows:
                        Bee[k, k] = one
                        rhs[k, i] = one
                        continue
                    numerator, denominator = rows[i]
                    Bee[k] = numerator[block]
                    rhs[k, i] = denominator
                    for j in closure:
                        if not isinstance(numerator[j], gem.Zero):
                            # keep the object array on the left, lest GEM
                            # broadcast the scalar into a vector node
                            rhs[k] = rhs[k] - V[j] * numerator[j]
                V[block] = solve(Bee, rhs, cache)
        return V
