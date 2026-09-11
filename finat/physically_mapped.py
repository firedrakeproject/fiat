r"""Physically mapped elements and the automatic basis transformation.

The transformation matrix of a physically mapped element is obtained by
duality (Kirby 2017, Brubeck & Kirby 2025).  With :math:`\hat\psi_j` the
reference nodal basis, :math:`F` the cell map and :math:`n_i` the
physical node,

.. math:: B_{ij} = n_i(\hat\psi_j \circ F^{-1}), \qquad V = B^{-1},

and the physical basis functions are :math:`M F^*(\hat\Psi)` with
:math:`M = V^T`.  This is a second Vandermonde inversion on top of
FIAT's: FIAT expresses the reference nodal basis in the expansion set
of the polynomial space (:class:`ReferenceNodalBasis`), and FInAT
expresses the physical nodal basis in the pulled-back reference nodal
basis (:class:`PhysicalVandermonde`).

A physical node is its reference node with the directions of its
coefficient tensor replaced (:class:`PhysicalNode`): FIAT builds the
physical dual set with the same points and weights as on the reference
cell, and only the directions change.  Point data keeps its Cartesian
directions, facet nodes are framed on the physical facet
(:class:`FlagFrame`), and interior moments are push-forward invariant.
Pulling the physical directions back to the reference cell, index by
index of the coefficient tensor (:class:`DirectionPullback`), turns the
physical node into a functional on reference functions, whose values on
the reference nodal basis are numeric tabulations contracted with
matrices polynomial in the Jacobian, up to a scalar denominator.

:math:`B` is block lower triangular in the entity order, because a node
on an entity only couples to the dofs of that entity and of its
closure, and it is inverted by block back-substitution over the
entities, one small symbolic solve per entity.  Rows of invariant nodes
are rows of the identity and are never assembled.  The dual evaluation
needs no inversion: the physical nodes of a function are :math:`B`
applied to the reference nodes of its pullback.
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import reduce
from itertools import product
from math import prod
from operator import add, mul

import gem
import numpy

from finat.citations import cite
from finat.functional import DERIVATIVE, DIVERGENCE, FunctionalData


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
        :returns: The :class:`PhysicalVandermonde`.
        """
        basis = ReferenceNodalBasis(self._element)
        sd = basis.ref_el.get_spatial_dimension()
        bary, = basis.ref_el.make_points(sd, 0, sd + 1)
        jacobian = Jacobian(coordinate_mapping.jacobian_at(bary))
        nodes = self.physical_nodes(basis, jacobian)
        return PhysicalVandermonde(basis, nodes, self.tol)

    def physical_nodes(self, basis, jacobian):
        """The physical degrees of freedom.

        Constraint functionals of an extended element that cannot be
        parsed are not exposed as physical dofs: their physical
        counterparts are the push-forwards of the reference ones (Kirby
        2017, section 5), with identity rows.

        :arg basis: The :class:`ReferenceNodalBasis`.
        :arg jacobian: The :class:`Jacobian`.
        :returns: A list with a :class:`PhysicalNode` per dof, or None
            for a dof keeping an identity row.
        """
        fiat_element = basis.fiat_element
        fiat_nodes = fiat_element.dual_basis()
        mappings = fiat_element.mapping()
        ndof = self.space_dimension()
        nodes = [None] * len(fiat_nodes)
        for dim in basis.entity_dofs:
            for entity, dofs in basis.entity_dofs[dim].items():
                for i in dofs:
                    try:
                        node = ReferenceNode.from_fiat(fiat_nodes[i], mappings[i], (dim, entity), basis.ref_el)
                    except NotImplementedError:
                        if i < ndof:
                            raise
                        continue
                    nodes[i] = physical_node(node, basis.ref_el, jacobian, avg=self.avg)
        return nodes

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


def as_gem(v):
    """Converts a number to a GEM constant, preserving structural zeros."""
    if v == 0:
        return zero
    if v == 1:
        return one
    return gem.Literal(v)


def as_gem_array(A):
    """Converts a numeric array to an object array of GEM constants."""
    A = numpy.asarray(A)
    out = numpy.full(A.shape, zero, dtype=object)
    for index in numpy.ndindex(A.shape):
        out[index] = as_gem(float(A[index]))
    return out


class Jacobian:
    r"""The cell Jacobian and its derived matrices, as object arrays of GEM scalars.

    Following FInAT, :math:`J = \partial x/\partial\hat{x}` maps
    reference to physical coordinates; the papers' Jacobian is its
    inverse.

    :arg J: A GEM expression for the Jacobian, shape ``(sd, sd)``.
    """
    def __init__(self, J):
        sd = J.shape[0]
        self.J = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)], dtype=object)
        self.detJ = determinant(self.J)
        self.adjJ = adjugate(self.J)
        #: the cofactor matrix, which maps normals to normals
        self.K = self.adjJ.T


class ReferenceNodalBasis:
    """The nodal basis of a FIAT element on the reference cell.

    :arg fiat_element: The FIAT element.
    """
    def __init__(self, fiat_element):
        self.fiat_element = fiat_element
        self.ref_el = fiat_element.get_reference_element()
        # FIAT may list a dof on more than one entity (e.g. the edge
        # moments of Arnold-Winther reappear in its interior list); the
        # lowest-dimensional entity owns the dof.
        seen = set()
        self.entity_dofs = {}
        for dim, entities in sorted(fiat_element.entity_dofs().items()):
            self.entity_dofs[dim] = {}
            for entity, dofs in sorted(entities.items()):
                dofs = [i for i in dofs if i not in seen]
                seen.update(dofs)
                self.entity_dofs[dim][entity] = dofs
        self._tabulations = {}

    def closure_dofs(self, dim, entity):
        """The dofs on the sub-entities of an entity.

        :arg dim: The dimension of the entity.
        :arg entity: The entity number.
        :returns: The indices of the dofs owned by entities of lower
            dimension in the closure of the entity.
        """
        top = self.ref_el.get_topology()
        verts = set(top[dim][entity])
        return [i for d in self.entity_dofs if d < dim
                for e in self.entity_dofs[d] if set(top[d][e]) <= verts
                for i in self.entity_dofs[d][e]]

    def tabulate(self, mappings, points):
        """Tabulates the nodal basis with the axes of a coefficient tensor.

        :arg mappings: The index mappings of a
            :class:`~finat.functional.FunctionalData`.
        :arg points: The points, a tuple of tuples.
        :returns: An array of shape ``(nbf, *shape, len(points))`` with
            one axis per index of the coefficient tensor: the value
            components, then the derivatives, then the divergence (an
            axis of length one).
        """
        key = (mappings, points)
        try:
            return self._tabulations[key]
        except KeyError:
            pass
        sd = self.ref_el.get_spatial_dimension()
        order = sum(mapping in (DERIVATIVE, DIVERGENCE) for mapping in mappings)
        tab = self.fiat_element.tabulate(order, points)
        values = tab[(0,) * sd]
        T = numpy.zeros(values.shape[:-1] + (sd,) * order + values.shape[-1:])
        prefix = (slice(None),) * (values.ndim - 1)
        for index in numpy.ndindex((sd,) * order):
            alpha = [0] * sd
            for k in index:
                alpha[k] += 1
            T[prefix + index] = tab[tuple(alpha)]
        if DIVERGENCE in mappings:
            # contract the last value axis with the derivative axis
            T = numpy.trace(T, axis1=values.ndim - 2, axis2=-2)[..., None, :]
        return self._tabulations.setdefault(key, T)


def support_entity(ref_el, points, tol=1e-12):
    """The smallest entity whose closure contains all the points.

    :arg ref_el: The reference cell.
    :arg points: The points.
    :arg tol: Tolerance below which a barycentric coordinate vanishes.
    :returns: The pair ``(dim, entity)``.
    """
    bary = ref_el.compute_barycentric_coordinates(numpy.asarray(points))
    verts = set(numpy.flatnonzero(numpy.abs(bary).max(axis=0) > tol))
    top = ref_el.get_topology()
    for dim in sorted(top):
        for entity in sorted(top[dim]):
            if verts <= set(top[dim][entity]):
                return dim, entity


class ReferenceNode:
    """A reference degree of freedom and its place on the cell.

    :arg functional: The :class:`~finat.functional.FunctionalData`.
    :arg owner: The ``(dim, entity)`` FIAT lists the dof under.
    :arg support: The ``(dim, entity)`` of the smallest entity whose
        closure contains the points of the functional.
    """
    def __init__(self, functional, owner, support):
        self.functional = functional
        self.owner = owner
        self.support = support

    @classmethod
    def from_fiat(cls, node, mapping, owner, ref_el):
        """Parses a FIAT functional.

        :arg node: The FIAT :class:`~FIAT.functional.Functional`.
        :arg mapping: The FIAT mapping of the basis functions.
        :arg owner: The ``(dim, entity)`` FIAT lists the dof under.
        :arg ref_el: The reference cell.
        :returns: The :class:`ReferenceNode`.
        """
        functional = FunctionalData.from_fiat(node, mapping)
        return cls(functional, owner, support_entity(ref_el, functional.points))


def physical_convention(node, sd):
    """The convention by which FIAT builds the physical counterpart of a node.

    Instantiated on a physical cell, FIAT dual sets follow one of three
    conventions, which cannot be read off the reference functional
    alone:

    * ``"cartesian"``: point evaluations keep their Cartesian
      components, values and derivatives alike (vertex jets, the vertex
      divergences of Alfeld-Sorokina, the vertex and interior point
      values of Guzman-Neilan);
    * ``"frame"``: nodes on a facet refer to the physical facet normal
      and tangents (normal derivatives and their moments, normal and
      tangential moments of Piola-mapped fields, the facet point
      values of Hu-Zhang);
    * ``"invariant"``: interior moments are taken against pulled-back
      test functions, so the physical node is the push-forward of the
      reference one; point values of scalar fields are invariant too.

    Vector point values on a facet are Cartesian, not framed: they are
    the C0 data of vector-valued H1 elements (the edge midpoint values
    of Alfeld-Sorokina).

    :arg node: The :class:`ReferenceNode`.
    :arg sd: The spatial dimension.
    :returns: One of ``"cartesian"``, ``"frame"`` or ``"invariant"``.
    """
    dim = node.owner[0]
    functional = node.functional
    single = len(functional.points) == 1
    if functional.order == 0 and functional.rank == 0:
        return "invariant"
    if dim == sd - 1 and not (single and functional.order == 0 and functional.rank == 1):
        return "frame"
    if dim == sd and not single and functional.order == 0:
        return "invariant"
    return "cartesian"


class DirectionPullback:
    r"""The pullback of the physical directions of one index of a node.

    A physical node has coefficient tensors :math:`\Phi\hat{W}`, the
    reference ones with the directions of one index replaced by their
    physical counterparts, and the pullback of the basis functions acts
    on the same index with a matrix :math:`A`: :math:`J^{-T}` for a
    derivative or covariant component (the chain rule), :math:`J/\det J`
    for a contravariant component, :math:`1/\det J` for a divergence,
    and the identity for an affine component.  Applying the physical
    node to pulled-back basis functions is therefore the same as
    applying, to reference functions, the reference node with that
    index of its coefficients contracted with :math:`G = A^T\Phi`,
    which pulls the physical directions back to the reference cell.
    It is stored as :math:`G = P + \sum_t g_t C_t / d`, where :math:`P`
    is the numeric projector onto the reference directions whose
    physical counterparts pull back to themselves, the :math:`g_t` are
    GEM scalars polynomial in :math:`J`, and the :math:`C_t` are
    numeric matrices aligned with the frame of the node, so that the
    cancellations of the theory happen numerically.

    :arg invariant: The projector :math:`P`, a numeric square array.
    :arg terms: A list of pairs ``(g, C)``.
    :arg denominator: :math:`d`, a GEM scalar.
    """
    def __init__(self, invariant, terms, denominator):
        self.invariant = numpy.asarray(invariant)
        self.terms = list(terms)
        self.denominator = denominator

    def numerator_terms(self):
        """The terms of :math:`Gd`, the invariant projector included."""
        terms = list(self.terms)
        if self.invariant.any():
            terms.insert(0, (self.denominator, self.invariant))
        return terms


def identity_pullback(extent):
    """The pullback of an index whose physical directions pull back to themselves."""
    return DirectionPullback(numpy.eye(extent), [], one)


def cartesian_pullback(mapping, jacobian):
    """The pullback of an index keeping its Cartesian directions.

    :arg mapping: The mapping of the index.
    :arg jacobian: The :class:`Jacobian`.
    :returns: The :class:`DirectionPullback`.
    """
    sd = jacobian.J.shape[0]
    if mapping == "affine":
        return identity_pullback(sd)
    if mapping == DIVERGENCE:
        return DirectionPullback(numpy.zeros((1, 1)), [(one, numpy.ones((1, 1)))], jacobian.detJ)
    Q = jacobian.J.T if mapping == "contravariant piola" else jacobian.adjJ
    terms = []
    for a, b in numpy.ndindex(sd, sd):
        E = numpy.zeros((sd, sd))
        E[a, b] = 1
        terms.append((Q[a, b], E))
    return DirectionPullback(numpy.zeros((sd, sd)), terms, jacobian.detJ)


class FlagFrame:
    r"""The physical directions of a node framed on a facet.

    The frame is attached to the flag of entities support :math:`\subseteq`
    facet :math:`\subset` cell.  Tangents of the support entity map by
    :math:`J`, and each unit normal of the flag (the facet normal within
    the cell, and the normal to the support within the facet when the
    support is an edge of a face) maps to the physical unit normal,
    which FIAT computes from the mapped tangents by the same
    cross-product formulas as on the reference cell, so no orientation
    logic is needed.  With :math:`K = \operatorname{adj}(J)^T` the
    facet normal is :math:`\nu = K\hat{n}/|K\hat{n}|`, and the normal
    to an edge within a face is :math:`\nu_e = Jw/|Jw|` with
    :math:`w = ((J^TJ)\hat{t})\times\hat{n}`, by the identity
    :math:`Ja\times Kb = J((J^TJ)a\times b)`.  The pulled-back normals
    are expanded in the orthonormal reference frame of the flag, so
    that every term of a pullback pairs a symbolic coefficient with a
    reference direction along which the tabulation is nodal.

    :arg ref_el: The reference cell.
    :arg facet: The facet number.
    :arg support: The ``(dim, entity)`` supporting the node.
    :arg jacobian: The :class:`Jacobian`.
    """
    def __init__(self, ref_el, facet, support, jacobian):
        self.ref_el = ref_el
        self.facet = facet
        self.jacobian = jacobian
        sd = ref_el.get_spatial_dimension()
        J, adjJ, detJ, K = jacobian.J, jacobian.adjJ, jacobian.detJ, jacobian.K
        # orthonormal tangents of the facet and of the support entity
        tangents = ref_el.compute_tangents(sd - 1, facet)
        self.facet_tangents = list(numpy.linalg.qr(tangents.T)[0].T)
        n = ref_el.compute_normal(facet)
        n = n / numpy.linalg.norm(n)
        Kn = K @ n
        # unit normals of the flag as (nhat, u, d, frame), with
        # J^{-1} nu = u / d expanded in the reference directions frame
        self.normals = []
        if support[0] == sd - 2 >= 1:
            t = ref_el.compute_edge_tangent(support[1])
            m = numpy.cross(t, n)
            m = m / numpy.linalg.norm(m)
            self.tangents = [t / numpy.linalg.norm(t)]
            w = numpy.cross(J.T @ (J @ t), n)
            Jw = J @ w
            self.normals.append((m, w, (Jw @ Jw)**0.5, self.tangents + [m]))
        else:
            self.tangents = self.facet_tangents
        frame = self.tangents + [m for m, *_ in self.normals] + [n]
        self.normals.append((n, adjJ @ Kn, detJ * (Kn @ Kn)**0.5, frame))

    def pullback(self, mapping):
        """The :class:`DirectionPullback` of an index of a node in this frame."""
        if mapping == "contravariant piola":
            return self.contravariant_pullback()
        if mapping == "affine":
            return identity_pullback(self.ref_el.get_spatial_dimension())
        if mapping == DIVERGENCE:
            return cartesian_pullback(mapping, self.jacobian)
        return self.derivative_pullback()

    def derivative_pullback(self):
        r"""The pullback of a derivative or covariant component index.

        Tangents are invariant and each unit normal :math:`\hat{n}_k`
        maps to :math:`J^{-1}\nu_k = u_k/d_k`.
        """
        sd = self.ref_el.get_spatial_dimension()
        P = numpy.eye(sd) - sum(numpy.outer(n, n) for n, *_ in self.normals)
        dens = [d for _, _, d, _ in self.normals]
        terms = []
        for k, (n, u, _, frame) in enumerate(self.normals):
            scale = reduce(mul, dens[:k] + dens[k+1:], one)
            for f in frame:
                terms.append(((u @ as_gem_array(f)) * scale, numpy.outer(f, n)))
        return DirectionPullback(P, terms, reduce(mul, dens))

    def contravariant_pullback(self):
        r"""The pullback of a contravariant component index.

        The scaled normal :math:`\hat\nu` is invariant, since
        :math:`K\hat\nu` is the physical scaled normal, while a
        tangential direction :math:`\hat\tau` maps to :math:`s\hat\tau
        + r\hat\nu` with :math:`s = |K\hat\nu|^2/(\det J\,|\hat\nu|^2)`
        and :math:`r = -(K\hat\tau)\cdot(K\hat\nu)/(\det J\,|\hat\nu|^2)`:
        FIAT takes the physical tangential components against the
        cofactor image of the reference tangents projected onto the
        physical facet.
        """
        K, detJ = self.jacobian.K, self.jacobian.detJ
        sd = self.ref_el.get_spatial_dimension()
        nu = self.ref_el.compute_scaled_normal(self.facet)
        nn = nu @ nu
        Knu = K @ nu
        KtKnu = K.T @ Knu
        P = numpy.outer(nu, nu) / nn
        terms = [(Knu @ Knu, numpy.eye(sd) - P)]
        for tau in self.facet_tangents:
            terms.append((KtKnu @ as_gem_array(-tau), numpy.outer(nu, tau)))
        return DirectionPullback(P, terms, detJ * nn)

    def measure(self):
        """The measure of the physical facet."""
        sd = self.ref_el.get_spatial_dimension()
        nu = self.ref_el.compute_scaled_normal(self.facet)
        Knu = self.jacobian.K @ nu
        vol = self.ref_el.volume_of_subcomplex(sd - 1, self.facet)
        return (vol / numpy.linalg.norm(nu)) * (Knu @ Knu)**0.5


class PhysicalNode:
    """A physical degree of freedom: a reference node with its directions replaced.

    :arg node: The :class:`ReferenceNode`.
    :arg pullbacks: A :class:`DirectionPullback` per index of the coefficient tensor.
    :arg scale: An optional GEM scalar multiplying the coefficients.
    """
    def __init__(self, node, pullbacks, scale=None):
        self.node = node
        self.pullbacks = tuple(pullbacks)
        self.scale = scale

    def is_invariant(self, tol):
        """Whether the physical node is the push-forward of the reference node.

        :arg tol: Relative tolerance on the coefficients.
        """
        W = self.node.functional.coefficients
        for k, pullback in enumerate(self.pullbacks):
            P = pullback.invariant
            R = numpy.tensordot(W, numpy.eye(P.shape[0]) - P, axes=(1 + k, 1))
            if numpy.abs(R).max() > tol * numpy.abs(W).max():
                return False
        return True

    def evaluate(self, basis, tol):
        """The values of the physical node on the pulled-back nodal basis.

        The row is a sum over products of one term per pullback.  The
        numeric matrices of the terms are contracted with the reference
        tabulation, quadrature sum included, before the GEM coefficient
        multiplies the result, so that whether a coupling is present is
        decided numerically in the frame of the node and never by
        inspecting a GEM expression.

        :arg basis: The :class:`ReferenceNodalBasis`.
        :arg tol: Relative tolerance below which numeric entries of the
            tabulation are dropped.
        :returns: The pair ``(numerator, denominator)`` of an object
            array of GEM scalars and a GEM scalar, the row of the
            physical Vandermonde matrix being their ratio.
        """
        functional = self.node.functional
        T = basis.tabulate(functional.mappings, functional.points)
        nbf = T.shape[0]
        # N[I', I, j] pairs the reference coefficients with indices I'
        # against the tabulated basis with indices I
        N = numpy.tensordot(functional.coefficients, T, axes=(0, -1))
        size = prod(functional.coefficients.shape[1:])
        N = N.reshape(size, nbf, size).transpose(0, 2, 1).reshape(size * size, nbf)
        scale = tol * numpy.abs(N).max()
        numerator = numpy.full(nbf, zero, dtype=object)
        for choice in product(*(pullback.numerator_terms() for pullback in self.pullbacks)):
            coefficient = reduce(mul, (g for g, _ in choice))
            C = reduce(numpy.kron, [C for _, C in choice])
            row = C.T.ravel() @ N
            for j in numpy.flatnonzero(numpy.abs(row) > scale * numpy.abs(C).max()):
                numerator[j] = numerator[j] + coefficient * row[j]
        if self.scale is not None:
            numerator = numerator * self.scale
        denominator = reduce(mul, (pullback.denominator for pullback in self.pullbacks))
        return numerator, denominator


def physical_node(node, ref_el, jacobian, avg=True):
    """Builds the physical counterpart of a reference node.

    :arg node: The :class:`ReferenceNode`.
    :arg ref_el: The reference cell.
    :arg jacobian: The :class:`Jacobian`.
    :arg avg: Whether physical facet moments are integral averages, as
        the reference nodes are; if not, they carry the physical facet
        measure.
    :returns: The :class:`PhysicalNode`.
    """
    sd = ref_el.get_spatial_dimension()
    functional = node.functional
    convention = physical_convention(node, sd)
    scale = None
    if convention == "invariant":
        pullbacks = [identity_pullback(sd) for mapping in functional.mappings]
    elif convention == "cartesian":
        pullbacks = [cartesian_pullback(mapping, jacobian) for mapping in functional.mappings]
    else:
        frame = FlagFrame(ref_el, node.owner[1], node.support, jacobian)
        pullbacks = [frame.pullback(mapping) for mapping in functional.mappings]
        if not avg and len(functional.points) > 1:
            scale = frame.measure()
    return PhysicalNode(node, pullbacks, scale)


class PhysicalVandermonde:
    r"""The generalized Vandermonde matrix of the physical nodes.

    :math:`B_{ij} = n_i(\hat\psi_j\circ F^{-1})` pairs the physical
    nodes with the pulled-back reference nodal basis.  Every row is
    supported on the dofs of the node's own entity and of the entities
    in its closure, since the trace of a basis function on an entity is
    determined by the dofs on the closure of that entity (which is what
    makes the element conforming), so :math:`B` is block lower
    triangular in the entity order.

    :arg basis: The :class:`ReferenceNodalBasis`.
    :arg nodes: A :class:`PhysicalNode` per dof, or None for a dof
        whose physical node is the push-forward of the reference one.
    :arg tol: Relative tolerance below which numeric couplings are dropped.
    """
    def __init__(self, basis, nodes, tol):
        self.basis = basis
        self.nodes = nodes
        self.tol = tol

    def rows(self):
        r"""Assembles the rows of :math:`B` that are not rows of the identity.

        Each row is kept in numerator/denominator form, and is checked
        to be supported on the closure of its entity.

        :returns: A dict mapping the dof to its ``(numerator, denominator)``
            pair of an object array of GEM scalars and a GEM scalar.
        """
        basis = self.basis
        nbf = basis.fiat_element.space_dimension()
        rows = {}
        for dim in basis.entity_dofs:
            for entity, block in basis.entity_dofs[dim].items():
                allowed = set(block) | set(basis.closure_dofs(dim, entity))
                for i in block:
                    node = self.nodes[i]
                    if node is None or node.is_invariant(self.tol):
                        continue
                    numerator, denominator = node.evaluate(basis, self.tol)
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
        nbf = self.basis.fiat_element.space_dimension()
        B = identity(nbf)
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
        basis = self.basis
        nbf = basis.fiat_element.space_dimension()
        rows = self.rows()
        V = identity(nbf)
        cache = {}
        for dim in basis.entity_dofs:
            for entity, block in basis.entity_dofs[dim].items():
                if not any(i in rows for i in block):
                    continue
                closure = basis.closure_dofs(dim, entity)
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
