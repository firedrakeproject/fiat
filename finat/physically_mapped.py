r"""Physically mapped elements and the automatic basis transformation.

The transformation matrix of a physically mapped element is obtained by
duality alone (Kirby 2017, Brubeck & Kirby 2025).  With
:math:`\hat\psi_j` the reference nodal basis, :math:`F` the cell map and
:math:`n_i` the physical node,

.. math:: B_{ij} = n_i(\hat\psi_j \circ F^{-1}), \qquad V = B^{-1},

and the physical basis functions are :math:`M F^*(\hat\Psi)` with
:math:`M = V^T`.  Two ingredients make the rows of :math:`B` computable
without frame algebra, for any element whose dual basis parses into
:class:`~finat.functional.PhysicallyMappedFunctional`:

* **Physical nodes by per-slot maps.**  The physical node shares the
  points and weights of its reference partner; only the directional
  data changes, one tensor slot at a time.  A derivative slot of a
  facet node maps its unit-normal component to the unit physical normal
  :math:`K\hat{n}/|K\hat{n}|` (:math:`K = \operatorname{adj}(J)^T` the
  cofactor matrix, which maps normals to normals) and its tangential
  complement by :math:`J` (mapped tangents); away from facets
  derivative slots keep their reference (Cartesian) directions.  A
  contravariant value slot of a facet moment maps its scaled-normal
  component by :math:`K` (the cofactor lemma :math:`K\hat\nu^s =
  \nu^s` is exact) and its tangential complement by the reciprocal of
  the scalar tangential push-forward; Cartesian point data keeps its
  weights, interior moments are invariant by convention, and
  divergence nodes contract to :math:`\det J` times the identity.

* **The adjoint acts on the tabulation.**  Dually to the push-forward
  law :math:`d = J^{\otimes m}\hat{d}`, each derivative slot of the
  *numeric* reference tabulation carries :math:`J^{-T}` and each value
  slot :math:`J/\det J`; transposing these onto the direction gives a
  single effective per-slot matrix, so a row of :math:`B` is a numeric
  generalized Vandermonde pairing with symbolic per-slot coefficients.

Crucially, :math:`B` is never inverted densely, and its sparsity is
*inferred from numerical tabulations*: :math:`B` is assembled
numerically at a few generic sample Jacobians, entries that vanish at
every sample are structural zeros (the support law: dof :math:`j`
couples to dof :math:`i` only on the closure of :math:`i`'s entity, the
same property that makes the element conforming), and entries that
agree at every sample are constants, so push-forward-invariant rows
never touch symbolic algebra.  The coupling graph is block
triangularized by its strongly connected components and :math:`V` is
computed by sparse block back-substitution, one small
adjugate/determinant solve per diagonal block, with fill-in confined to
the closures.
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import reduce
from operator import add, mul

import gem
import numpy

from FIAT.finite_element import FiniteElement

from finat.citations import cite
from finat.functional import PhysicallyMappedFunctional, multiindices


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

    :meth:`basis_transformation` derives the transformation generically
    from the FIAT dual basis by the duality formulation of the module
    docstring; the mixins of :mod:`finat.zany` supply the
    family-specific conventions (expected pullback, moment
    normalization, dof rescaling), and elements whose dual basis the
    generic engine cannot parse override it entirely by hand.
    """

    #: Numerical tolerance used to drop negligible terms of the numeric
    #: tabulations from the symbolic rows.
    tol = 1e-12

    #: Row-relative tolerance deciding, from the sample assemblies, which
    #: entries of the transformation are structural zeros or constants.
    pattern_tol = 1e-10

    #: If False, physical scalar facet moments are plain integrals rather
    #: than the measure-intrinsic integral averages of the reference nodes.
    avg = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cite("Kirby2018zany")
        cite("Kirby2019zany")
        self.restriction_indices = None

    def basis_transformation(self, coordinate_mapping) -> gem.ListTensor:
        r"""Transformation matrix for the basis functions.

        Assembles the generalized Vandermonde matrix :math:`B_{ij} =
        n_i(\hat\psi_j\circ F^{-1})` and computes :math:`M = B^{-T}`
        by sparse block back-substitution.  The sparsity and the
        constant entries of :math:`B` are inferred from numeric
        assemblies at sample Jacobians; only the remaining entries are
        built symbolically, in numerator/denominator form, and each
        diagonal block of the strongly-connected-component
        triangularization is solved through its adjugate.

        Parameters
        ----------
        coordinate_mapping :
            Object providing the physical geometry as GEM expressions.

        Returns
        -------
        gem.ListTensor
            The transformation matrix, shape ``(space_dimension(),
            nbf)``, with the trailing constraint columns of an extended
            element truncated.

        """
        fiat_element = self._element
        self._check_mapping(fiat_element)
        ref_el = fiat_element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        bary, = ref_el.make_points(sd, 0, sd + 1)
        J = _materialize_jacobian(coordinate_mapping.jacobian_at(bary))

        nodes = fiat_element.dual_basis()
        mappings = fiat_element.mapping()
        entity_ids = fiat_element.entity_dofs()
        nbf = fiat_element.space_dimension()
        ndof = self.space_dimension()

        # Parse the dual basis.  FIAT may list a dof on more than one
        # entity (e.g. the edge moments of Arnold-Winther reappear in its
        # interior list); the lowest-dimensional entity owns the dof.
        # Unparseable constraint functionals of an extended element are
        # never exposed as physical dofs: their physical counterparts are
        # defined as the pullbacks of the reference ones (Kirby 2017,
        # section 5), keeping identity rows, and the corresponding
        # columns are truncated below.
        ells = {}
        owners = {}
        for dim in sorted(entity_ids):
            for entity in sorted(entity_ids[dim]):
                for i in entity_ids[dim][entity]:
                    if i in owners:
                        continue
                    owners[i] = (dim, entity)
                    try:
                        ells[i] = self._functional_from_node(
                            nodes[i], i, mappings[i])
                    except NotImplementedError:
                        if i < ndof:
                            raise
                        ells[i] = None

        # Infer the sparsity and the constant entries numerically
        tabs = {}
        samples = [self._assemble_numeric(fiat_element, ells, owners, Jk, tabs)
                   for Jk in _sample_jacobians(sd)]
        pattern, constant, Bconst = _infer_pattern(samples, self.pattern_tol)

        # Build the non-constant rows symbolically, in
        # numerator/denominator form, on their own pattern columns only
        numer = numpy.empty(nbf, dtype=object)
        denom = numpy.empty(nbf, dtype=object)
        for i in range(nbf):
            cols = numpy.flatnonzero(pattern[i])
            if ells[i] is None or constant[i, cols].all():
                numer[i] = numpy.array([_as_gem(v) for v in Bconst[i, cols]],
                                       dtype=object)
                denom[i] = one
            else:
                dim, entity = owners[i]
                numer[i], denom[i] = _node_row(
                    fiat_element, ells[i], dim, entity, J,
                    self.avg, self.tol, tabs, cols=cols)

        # Sparse block back-substitution over the strongly connected
        # components of the coupling graph, in dependency order:
        # V_S = B_SS^{-1} (I_S - B_Sc V_c), with B = diag(1/denom) numer,
        # so the denominators only enter the right-hand side.
        V = numpy.full((nbf, nbf), zero, dtype=object)
        inv_cache = {}
        for block in _scc_blocks(pattern):
            inside = set(block)
            rhs = numpy.full((len(block), nbf), zero, dtype=object)
            BSS = numpy.full((len(block), len(block)), zero, dtype=object)
            index = {i: k for k, i in enumerate(block)}
            for k, i in enumerate(block):
                rhs[k, i] = denom[i]
                for c, j in enumerate(numpy.flatnonzero(pattern[i])):
                    if j in inside:
                        BSS[k, index[j]] = numer[i][c]
                    elif not isinstance(numer[i][c], gem.Zero):
                        # numpy elementwise product: the object array must
                        # be the left operand, lest GEM broadcast a vector
                        rhs[k] = rhs[k] - V[j] * numer[i][c]
            V[block] = _solve_block(BSS, rhs, inv_cache)

        self._rescale_dofs(V, fiat_element, coordinate_mapping)
        return gem.ListTensor(V[:, :ndof].T)

    def _assemble_numeric(self, fiat_element: FiniteElement, ells: dict,
                          owners: dict, J: numpy.ndarray, tabs: dict) -> numpy.ndarray:
        """Assemble B numerically at one sample Jacobian.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :arg ells: Map from dof index to parsed
            :class:`~finat.functional.PhysicallyMappedFunctional`, or
            None for a constraint functional keeping an identity row.
        :arg owners: Map from dof index to its owning ``(dim, entity)``.
        :arg J: Numeric sample Jacobian.
        :arg tabs: Cache of reference tabulations, shared across samples.
        :returns: The matrix B as a numpy array.
        """
        B = numpy.eye(fiat_element.space_dimension())
        for i, ell in ells.items():
            if ell is None:
                continue
            dim, entity = owners[i]
            res = _node_row(fiat_element, ell, dim, entity, J,
                            self.avg, self.tol, tabs)
            if res is not None:
                row, den = res
                B[i] = row / den
        return B

    def _functional_from_node(self, node, index: int, mapping: str):
        """Build the symbolic functional of one dof of the dual basis.

        By default the point locations, derivative order and direction are
        recovered numerically from the FIAT functional. Elements whose dofs
        are known symbolically may override this to supply them exactly.

        :arg node: The FIAT functional of the dof.
        :arg index: Its index in the dual basis.
        :arg mapping: The FIAT mapping string of the basis functions this
            functional is dual to.
        :returns: The corresponding
            :class:`~finat.functional.PhysicallyMappedFunctional`.
        """
        return PhysicallyMappedFunctional.from_fiat(node, mapping=mapping)

    def _check_mapping(self, fiat_element: FiniteElement) -> None:
        """Verify that this class knows how to transform this element's pullback.

        The generic engine dispatches on the mapping of each dof; this
        hook lets the family mixins of :mod:`finat.zany` reject FIAT
        elements outside their conventions up front.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not supported.
        """
        pass

    def _rescale_dofs(self, V: numpy.ndarray, fiat_element: FiniteElement,
                      coordinate_mapping) -> None:
        r"""Rescale the physical degrees of freedom by powers of the cell size.

        Each column of ``V`` is multiplied by :meth:`dof_scale` evaluated
        with the cell size averaged over the vertices of the dof's entity.
        This is the FInAT convention keeping the mass matrix
        well-conditioned; it is consistent across cells because the
        scaling only depends on shared entities.

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
                    scale = self.dof_scale(nodes[i], dim, havg)
                    if scale is not None:
                        V[:, i] = V[:, i] * scale

    def dof_scale(self, node, dim: int, havg):
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


def _materialize_jacobian(J: gem.Node) -> numpy.ndarray:
    """Materialize a GEM Jacobian node as a numpy object array of GEM scalars.

    :arg J: GEM expression for the cell Jacobian, shape ``(sd, sd)``.
    :returns: A ``(sd, sd)`` numpy object array with the same entries as
        ``J``, usable with the numeric-or-symbolic linear algebra of this
        module (:func:`determinant`, :func:`adjugate`), which index a GEM
        node's entries directly rather than through numpy fancy indexing.
    """
    sd = J.shape[0]
    return numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)], dtype=object)


def _as_gem(v: float) -> gem.Node:
    """Convert a number to a GEM constant, preserving structural zeros.

    :arg v: The number.
    :returns: ``Zero`` for an exact zero, so that downstream products
        and sums fold, the shared ``one`` for an exact one, and a
        ``Literal`` otherwise.
    """
    if v == 0:
        return zero
    if v == 1:
        return one
    return gem.Literal(v)


def _as_gem_array(A) -> numpy.ndarray:
    """Convert a numeric array to an object array of GEM constants.

    :arg A: The numeric array.
    :returns: An object array of the same shape, entrywise :func:`_as_gem`.
    """
    A = numpy.asarray(A)
    out = numpy.full(A.shape, zero, dtype=object)
    for index in numpy.ndindex(A.shape):
        out[index] = _as_gem(float(A[index]))
    return out


def _sample_jacobians(sd: int, nsamples: int = 3) -> list:
    """Deterministic generic sample Jacobians for sparsity inference.

    :arg sd: The spatial dimension.
    :arg nsamples: How many samples; the last is orientation-reversing.
    :returns: A list of well-conditioned ``(sd, sd)`` numeric Jacobians
        with no special alignments, at which entries of the
        transformation that vanish (or agree) at every sample may be
        declared structurally zero (or constant).
    """
    samples = []
    t = 0.0
    while len(samples) < nsamples:
        t += 1.0
        A = numpy.eye(sd) + 0.4 * numpy.cos(
            t + 7.3 * numpy.arange(1.0, sd * sd + 1.0)).reshape(sd, sd)
        det = numpy.linalg.det(A)
        if abs(det) < 0.3:
            continue
        if len(samples) == nsamples - 1 and det * numpy.linalg.det(samples[0]) > 0:
            A = A[::-1] if sd > 1 else -A
        samples.append(A)
    return samples


def _infer_pattern(samples: list, tol: float) -> tuple:
    """Infer the sparsity and the constant entries of B from samples.

    :arg samples: Numeric assemblies of B at the sample Jacobians.
    :arg tol: Row-relative tolerance below which an entry is a
        structural zero, and spread below which it is a constant.
    :returns: A tuple ``(pattern, constant, Bconst)`` of the boolean
        coupling pattern (with full diagonal), the boolean mask of
        J-independent entries, and their values (averaged over the
        samples, snapped to exact integers where they round).
    """
    stack = numpy.stack(samples)
    scale = numpy.abs(stack).max(axis=(0, 2))
    pattern = (numpy.abs(stack) > tol * scale[None, :, None]).any(axis=0)
    numpy.fill_diagonal(pattern, True)
    Bconst = stack.mean(axis=0)
    spread = stack.max(axis=0) - stack.min(axis=0)
    constant = spread <= tol * numpy.maximum(1.0, numpy.abs(Bconst))
    snapped = numpy.round(Bconst)
    near = constant & (numpy.abs(Bconst - snapped)
                       <= tol * numpy.maximum(1.0, numpy.abs(Bconst)))
    Bconst[near] = snapped[near]
    return pattern, constant, Bconst


def _scc_blocks(pattern: numpy.ndarray) -> list:
    """Strongly connected components of the coupling graph, in dependency order.

    Tarjan's algorithm on the digraph with an edge ``i -> j`` whenever
    ``pattern[i, j]``: each emitted component only depends on (has
    pattern entries in) previously emitted ones, so emission order is
    elimination order, and the block triangular structure of B is a
    consequence of the inferred sparsity rather than an assumption.

    :arg pattern: Boolean coupling pattern of B.
    :returns: List of sorted lists of dof indices, one per component.
    """
    n = pattern.shape[0]
    adjacency = [numpy.flatnonzero(pattern[i]) for i in range(n)]
    index = [-1] * n
    low = [0] * n
    onstack = [False] * n
    stack = []
    blocks = []
    counter = [0]

    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack[v] = True
        for w in adjacency[v]:
            if w == v:
                continue
            if index[w] == -1:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif onstack[w]:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                onstack[w] = False
                component.append(w)
                if w == v:
                    break
            blocks.append(sorted(component))

    for v in range(n):
        if index[v] == -1:
            strongconnect(v)
    return blocks


def _solve_block(B: numpy.ndarray, rhs: numpy.ndarray, cache: dict) -> numpy.ndarray:
    """Solve one diagonal block through its adjugate.

    :arg B: Object array with the (GEM) block of B numerators.
    :arg rhs: Object array with the right-hand side rows.
    :arg cache: Cache of adjugate/determinant pairs, shared across
        blocks so that identical blocks (e.g. the vertex jets of every
        vertex) are only inverted once.
    :returns: The rows of V for this block.
    """
    if B.shape[0] == 1:
        d = B[0, 0]
        return rhs if d == one else rhs / d
    key = gem.ListTensor(B)
    try:
        adj, det = cache[key]
    except KeyError:
        adj, det = cache.setdefault(key, (adjugate(B), determinant(B)))
    return (adj @ rhs) / det


def _tabulate(fiat_element: FiniteElement, order: int, points: tuple,
              tabs: dict) -> dict:
    """Cached reference tabulation of the nodal basis.

    :arg fiat_element: The FIAT element.
    :arg order: The derivative order.
    :arg points: The points, as a hashable tuple.
    :arg tabs: The cache.
    :returns: The FIAT tabulation dict.
    """
    key = (order, points)
    try:
        return tabs[key]
    except KeyError:
        return tabs.setdefault(key, fiat_element.tabulate(order, points))


def _slot_map(ref_el, entity: int, J: numpy.ndarray,
              derivative: bool, facet: bool) -> tuple:
    r"""Effective per-slot matrix pairing a physical node with the reference tabulation.

    Transposing the per-slot action of the push-forward onto the
    physical direction data gives a single matrix per slot:
    :math:`G = J^{-1}\Phi` for a derivative slot with physical
    direction map :math:`\Phi`, and :math:`G = (J/\det J)^T\Phi` for a
    contravariant value slot with physical weight map :math:`\Phi`.
    Away from a facet :math:`\Phi` is the identity (Cartesian data);
    on a facet it maps the normal component by the cofactor law and the
    tangential complement by the mapped tangents, as in the module
    docstring.  The map is returned in numerator/denominator form so
    that the numerators stay free of symbolic division.

    :arg ref_el: The reference cell.
    :arg entity: The facet number (ignored unless ``facet``).
    :arg J: The cell Jacobian: numeric array, or object array of GEM scalars.
    :arg derivative: Whether the slot is a derivative or a value slot.
    :arg facet: Whether the node data is in the facet frame rather than
        Cartesian.
    :returns: The pair ``(numerator, denominator)`` with
        ``G = numerator / denominator``.
    """
    detJ = determinant(J)
    adjJ = adjugate(J)
    if not facet:
        return (adjJ, detJ) if derivative else (J.T, detJ)
    sd = ref_el.get_spatial_dimension()
    symbolic = J.dtype == object
    lit = _as_gem_array if symbolic else numpy.asarray
    K = adjJ.T
    if derivative:
        n = ref_el.compute_normal(entity)
        nn = n @ n
        Kn = K @ n
        q = (Kn @ Kn) ** 0.5
        Gnum = (numpy.outer(adjJ @ Kn, lit(n * (nn ** 0.5 / nn)))
                + lit(numpy.eye(sd) - numpy.outer(n, n) / nn) * (q * detJ))
        return Gnum, detJ * q
    nu = ref_el.compute_scaled_normal(entity)
    nn = nu @ nu
    Knu = K @ nu
    qq = Knu @ Knu
    IP = lit((numpy.eye(sd) - numpy.outer(nu, nu) / nn) / nn)
    Gnum = (lit(numpy.outer(nu, nu) / nn) * (detJ * detJ)
            + ((J.T @ K) * qq - numpy.outer(lit(nu), K.T @ Knu) * detJ) @ IP)
    return Gnum, detJ * detJ


def _sparse_combination(coeffs, N: numpy.ndarray, tol: float,
                        symbolic: bool) -> numpy.ndarray:
    """Combine numeric tabulation rows with (possibly symbolic) coefficients.

    Computes ``row[j] = sum_k coeffs[k] * N[k, j]``, dropping negligible
    numeric entries so that whether a coupling is present is decided
    from the numeric array, never by inspecting a symbolic expression.

    :arg coeffs: The coefficients, numeric or GEM scalars.
    :arg N: The numeric tabulation pairing matrix.
    :arg tol: Relative tolerance below which entries of ``N`` are dropped.
    :arg symbolic: Whether to accumulate GEM expressions.
    :returns: The combined row.
    """
    row = (numpy.full(N.shape[1], zero, dtype=object) if symbolic
           else numpy.zeros(N.shape[1]))
    scale = numpy.abs(N).max()
    if scale == 0:
        return row
    for k, c in enumerate(coeffs):
        if isinstance(c, gem.Zero) or (not isinstance(c, gem.Node) and c == 0):
            continue
        for j in numpy.flatnonzero(numpy.abs(N[k]) > tol * scale):
            row[j] = row[j] + c * N[k, j]
    return row


def _node_row(fiat_element: FiniteElement, ell: PhysicallyMappedFunctional,
              dim: int, entity: int, J: numpy.ndarray, avg: bool, tol: float,
              tabs: dict, cols=None):
    r"""One row of the generalized Vandermonde matrix B.

    The row is the pairing of the physical node data with the reference
    tabulation carrying the adjoint of the push-forward, assembled with
    the single effective per-slot matrix of :func:`_slot_map`;
    divergence nodes commute with the Piola pullback up to
    :math:`\det J`, and nodes whose physical counterpart is the
    push-forward of the reference one by convention (point values,
    interior moments) return None for an identity row.

    :arg fiat_element: The FIAT element.
    :arg ell: The parsed reference node.
    :arg dim: The dimension of the entity the node sits on.
    :arg entity: The entity number.
    :arg J: The cell Jacobian: numeric array, or object array of GEM scalars.
    :arg avg: Whether physical scalar facet moments are integral averages.
    :arg tol: Relative tolerance below which numeric couplings are dropped.
    :arg cols: Optional column indices to restrict the row to.
    :returns: The pair ``(numerator_row, denominator)`` with
        ``B[i] = numerator_row / denominator``, or None for a row of
        the identity.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    symbolic = J.dtype == object
    if cols is None:
        cols = slice(None)

    if ell.divergence:
        tab = _tabulate(fiat_element, 1, ell.points, tabs)
        alphas = [tuple(int(k == c) for k in range(sd)) for c in range(sd)]
        div = sum(tab[alphas[c]].reshape(tab[alphas[c]].shape[0], sd, -1)[:, c, :]
                  for c in range(sd))
        row = div @ ell.weights
        scale = numpy.abs(row).max()
        row[numpy.abs(row) <= tol * scale] = 0.0
        row = row[cols]
        if symbolic:
            row = numpy.array([_as_gem(v) for v in row], dtype=object)
        return row, determinant(J)

    if ell.rank:
        if ell.mapping not in ("contravariant piola", "double contravariant piola"):
            raise NotImplementedError(
                f"Cannot transform value nodes with a {ell.mapping} pullback.")
        if ell.order:
            raise NotImplementedError(
                "Cannot transform nodes mixing values and derivatives.")
        npts = len(ell.points)
        # Single-point rank-1 nodes are Cartesian point data, not facet
        # moments, even when they sit on a codimension-1 entity (e.g. the
        # edge midpoint values of Alfeld-Sorokina); single-point rank-2
        # nodes remain frame dofs (e.g. the edge dofs of the Hu-Zhang
        # point variant).
        point_data = npts == 1 and (ell.rank == 1 or dim != sd - 1)
        if dim == sd and not point_data:
            # Interior moments are Piola invariant by convention: their
            # physical test functions are themselves Piola-mapped.
            return None
        facet = dim == sd - 1 and not point_data
        Gnum, den = _slot_map(ref_el, entity, J, derivative=False, facet=facet)
        tab = _tabulate(fiat_element, 0, ell.points, tabs)[(0,) * sd]
        T = tab.reshape(tab.shape[0], sd**ell.rank, npts)
        N = numpy.einsum("jpq,qc->pcj", T, ell.weights)
        N = N.reshape(sd**ell.rank * sd**ell.rank, -1)
        Gprod = reduce(numpy.kron, [Gnum] * ell.rank)
        row = _sparse_combination(Gprod.ravel(), N[:, cols], tol, symbolic)
        return row, reduce(mul, [den] * ell.rank)

    if ell.order == 0:
        # Point values pull back exactly
        return None
    facet = dim == sd - 1
    Gnum, den = _slot_map(ref_el, entity, J, derivative=True, facet=facet)
    direction = ell.pullback(Gnum).direction
    tab = _tabulate(fiat_element, ell.order, ell.points, tabs)
    N = numpy.stack([tab[alpha] @ ell.weights
                     for alpha in multiindices(sd, ell.order)])
    row = _sparse_combination(direction, N[:, cols], tol, symbolic)
    if not avg and len(ell.points) > 1 and facet:
        # The reference weights are measure-intrinsic (integral averages),
        # so a plain physical integral carries the physical facet measure.
        nu = ref_el.compute_scaled_normal(entity)
        Knu = adjugate(J).T @ nu
        measure = (ref_el.volume_of_subcomplex(sd - 1, entity)
                   / (nu @ nu) ** 0.5) * (Knu @ Knu) ** 0.5
        row = row * measure
    return row, reduce(mul, [den] * ell.order)
