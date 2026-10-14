r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import cached_property, reduce
from operator import add

import gem
import numpy

from finat.citations import cite
from finat.functional import (DERIVATIVE, DIVERGENCE, Functional, PhysicalEntityFrame,
                              cartesian_pullback, identity_pullback, to_riesz, one, zero)


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

    #: False selects plain integrals for physical scalar facet moments.
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

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        M = self.basis_transformation(coordinate_mapping)
        return MappedTabulation(M, ref_tabulation, indices=self.restriction_indices)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        result = super().basis_evaluation(order, ps, entity=entity)
        return self.map_tabulation(result, coordinate_mapping)

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
        J = numpy.array([[J[i, k] for k in range(J.shape[1])]
                         for i in range(J.shape[0])], dtype=object)
        return PhysicalVandermondeMatrix(self._element, self.functionals, self.reference_riesz,
                                         J, self.tol, avg=self.avg)

    @cached_property
    def functionals(self):
        """The degrees of freedom as coefficient tensors.

        Unparsed constraint nodes retain identity rows in the extended
        element (Kirby 2017, section 5).  Every exposed dof must be parsed.

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

    @cached_property
    def reference_riesz(self):
        """Numeric reference pairings for the physical Vandermonde rows.

        :returns: A dict mapping dofs to their Riesz representations.
        """
        return to_riesz(self.functionals, self._element)

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
        functionals = self.functionals
        entity_ids = fiat_element.entity_dofs()
        scales = [None] * len(functionals)
        for dim in entity_ids:
            for entity in entity_ids[dim]:
                verts = top[dim][entity]
                havg = reduce(add, (h[v] for v in verts)) / len(verts)
                for i in entity_ids[dim][entity]:
                    if functionals[i] is not None:
                        scales[i] = self.dof_scale(functionals[i], havg)
        return scales

    def dof_scale(self, functional, havg):
        r"""Return the conditioning rescaling factor of one physical dof.

        The physical node of a pulled back reference basis function
        dilates as :math:`h^{-p}`, and the dof is rescaled by :math:`h^{-p}`
        to make it dimensionless.  The exponent :math:`p` is inferred from
        the axes of the node:

        * a derivative axis carries :math:`J^{-T}`, of exponent :math:`1`;
        * a covariant component axis carries :math:`J^{-T}`, of exponent
          :math:`1`, a contravariant component axis carries
          :math:`J/\det J`, of exponent :math:`d-1`, and a divergence axis
          carries :math:`1/\det J`, of exponent :math:`d`, where :math:`d`
          is the spatial dimension.  These Piola weights only affect nodes
          that are Cartesian components;
        * a node in the physical frame of a facet absorbs the Piola
          weights of its components, since its normals and tangents are
          the images of the reference ones, except that the tangential
          components of a contravariant axis have exponent :math:`2-d`;
        * a node that is the push-forward of a reference node has exponent
          zero.

        Unless :attr:`avg` is set, a node that integrates over its entity
        of dimension :math:`k` is a plain integral rather than an average,
        which contributes the entity measure :math:`h^k`.

        :arg functional: The :class:`~finat.functional.Functional` of the dof.
        :arg havg: GEM scalar for the cell size averaged over the vertices
            of the dof's entity.
        :returns: The GEM scaling factor, or ``None`` for no rescaling.
        """
        sd = functional.ref_el.get_spatial_dimension()
        weights = {DERIVATIVE: 1}
        if functional.transformation == "cartesian":
            weights.update({"covariant piola": 1, "contravariant piola": sd - 1, DIVERGENCE: sd})
        elif functional.transformation == "invariant":
            weights = {}
        exponent = sum(weights.get(mapping, 0) for mapping in functional.mappings)
        if functional.transformation == "frame":
            # The normal component of a contravariant axis is invariant, and
            # the tangential components pull back with degree sd - 2.
            nu = functional.ref_el.compute_scaled_normal(functional.entity[1])
            normal = numpy.outer(nu, nu) / (nu @ nu)
            W = functional.coefficients
            for k, mapping in enumerate(functional.mappings):
                if mapping == "contravariant piola":
                    R = numpy.tensordot(W, numpy.eye(sd) - normal, axes=(1 + k, 1))
                    if numpy.abs(R).max() > self.tol * numpy.abs(W).max():
                        exponent += 2 - sd
            if not self.avg and len(functional.points) > 1:
                exponent -= functional.entity[0]
        return havg**-exponent if exponent else None


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
    neighbors = [{j for j in range(m) if not isinstance(A[i, j], gem.Zero)
                  or not isinstance(A[j, i], gem.Zero)} for i in range(m)]
    while candidates:
        ids = set()
        pending = [min(candidates)]
        while pending:
            i = pending.pop()
            if i not in ids:
                ids.add(i)
                pending.extend(neighbors[i] - ids)
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
            adj, det = cache[key] = adjugate_and_determinant(Aii)
        X[ids] = (adj @ B[ids]) / det
    return X


def pseudoinverse(A):
    """Returns the Moore-Penrose pseudoinverse of a full-column-rank matrix.

    :arg A: An object array of GEM scalars with at least as many rows as
        columns.
    :returns: The Moore-Penrose pseudoinverse, formed as
        ``solve(A.T @ A, A.T)``.
    """
    m, n = A.shape
    if m < n:
        raise ValueError("A must have at least as many rows as columns.")
    return solve(A.T @ A, A.T)


class PhysicalVandermondeMatrix:
    r"""The generalized Vandermonde matrix of the physical nodes.

    :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` pairs physical nodes with
    mapped reference basis functions.  Closure compatibility makes :math:`B`
    block lower triangular over entities.  Row assembly checks this property;
    it requires more than conformity for nodes involving normal derivatives.

    :arg fiat_element: The FIAT element.
    :arg functionals: A :class:`~finat.functional.Functional` per dof, or
        None for a dof whose physical node is the push-forward of the
        reference one.
    :arg reference_riesz: Numeric reference pairings from :func:`~finat.functional.to_riesz`.
    :arg J: The Jacobian as an object array of GEM scalars.
    :arg tol: Relative tolerance below which numeric couplings are dropped.
    :arg avg: Whether physical facet moments are integral averages, as
        the reference nodes are; if not, they carry the physical facet
        measure.
    """
    def __init__(self, fiat_element, functionals, reference_riesz, J, tol, avg=True):
        self.fiat_element = fiat_element
        self.functionals = functionals
        self.J = J
        self.adjJ, self.detJ = adjugate_and_determinant(J)
        self.tol = tol
        self.avg = avg
        self.reference_riesz = reference_riesz

    @cached_property
    def rows(self):
        r"""Assembles the rows of :math:`B` that are not rows of the identity.

        Each row is kept in numerator/denominator form, and is checked
        to be supported on the closure of its entity.

        :returns: A dict mapping the dof to its ``(numerator, denominator)``
            pair of an object array of GEM scalars and a GEM scalar.
        """
        fiat_element = self.fiat_element
        nbf = fiat_element.space_dimension()
        sd = self.J.shape[1]
        closure_dofs = fiat_element.entity_closure_dofs()
        cartesian = {}
        frames = {}
        rows = {}
        for i, node in enumerate(self.functionals):
            if node is None or node.transformation == "invariant":
                continue
            if node.transformation == "cartesian":
                for mapping in node.mappings:
                    if mapping not in cartesian:
                        cartesian[mapping] = cartesian_pullback(mapping, self.J, self.adjJ, self.detJ)
                on_edge = 0 < node.entity[0] < sd - 1
                pullbacks = [identity_pullback(sd) if on_edge and mapping == DERIVATIVE
                             else cartesian[mapping] for mapping in node.mappings]
                measure = one
            else:
                key = (node.entity[1], node.support)
                if key not in frames:
                    frames[key] = PhysicalEntityFrame(node.ref_el, *key, self.J, self.adjJ, self.detJ)
                frame = frames[key]
                pullbacks = [frame.pullback(mapping) for mapping in node.mappings]
                measure = frame.measure if not self.avg and len(node.points) > 1 else one
            if node.is_invariant(pullbacks, self.tol):
                continue
            numerator, denominator = node.action(self.reference_riesz[i], pullbacks, self.tol)
            numerator = numerator * measure
            dim, entity = node.entity
            allowed = set(closure_dofs[dim][entity])
            outside = [j for j in range(nbf) if j not in allowed
                       and not isinstance(numerator[j], gem.Zero)]
            if outside:
                raise NotImplementedError(
                    f"Dof {i} on entity {node.entity} couples to dofs "
                    f"{outside} outside the closure of its entity.")
            rows[i] = (numerator, denominator)
        return rows

    def matrix(self):
        r"""Computes :math:`B`.

        :returns: :math:`B` as an object array of GEM scalars.
        """
        B = identity(self.fiat_element.space_dimension())
        for i, (numerator, denominator) in self.rows.items():
            B[i] = numerator / denominator
        return B

    def inverse(self):
        r"""Computes :math:`V = B^{-1}` by block back-substitution over the entities.

        Entities are visited in order of increasing dimension, and for
        each entity :math:`e` with closure dofs :math:`c`

        .. math:: V_e = B_{ee}^{-1}(I_e - B_{ec} V_c).

        Only entities with transformed nodes are solved.  Row denominators
        move to the right-hand side, keeping divisions out of the blocks.

        :returns: :math:`V` as an object array of GEM scalars.
        """
        fiat_element = self.fiat_element
        nbf = fiat_element.space_dimension()
        entity_dofs = fiat_element.entity_dofs()
        closure_dofs = fiat_element.entity_closure_dofs()
        rows = self.rows
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
                            # Array first: GEM scalar multiplication otherwise broadcasts.
                            rhs[k] = rhs[k] - V[j] * numerator[j]
                V[block] = solve(Bee, rhs, cache)
        return V


def adjugate_and_determinant(J):
    r"""Return the determinant-scaled inverse and volume factor of a Jacobian.

    Square matrices use the signed determinant and adjugate.  For a
    rectangular matrix the volume factor is :math:`\sqrt{\det(J^T J)}`
    and the scaled inverse is :math:`\det J\,J^+`.

    :arg J: Full-column-rank Jacobian as an object array of GEM scalars.
    :returns: The scaled inverse (numerator) and volume factor (denominator).
        The transpose of the scaled inverse is the cofactor matrix.
    """
    gdim, tdim = J.shape
    if gdim == tdim:
        return adjugate(J), determinant(J)
    detJ = determinant(J.T @ J)**0.5
    return pseudoinverse(J) * detJ, detJ
