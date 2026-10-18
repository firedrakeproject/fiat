r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property, reduce
from itertools import chain, product
from operator import mul

import gem
import numpy

from FIAT.functional import IntegralMoment
from finat.citations import cite

zero = gem.Zero()
one = gem.Literal(1.0)


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
    """
    def __init__(self, M, ref_tabulation):
        self.M = M
        self.ref_tabulation = ref_tabulation
        # we expect M to be sparse with O(1) nonzeros per row
        # for each row, get the column index of each nonzero entry
        csr = [[j for j in range(M.shape[1]) if not isinstance(M.array[i, j], gem.Zero)]
               for i in range(M.shape[0])]
        self.csr = csr
        self._tabulation_cache = {}

    def matvec(self, table):
        # basis recombination using hand-rolled sparse-dense matrix multiplication
        ii = gem.indices(len(table.shape)-1)
        phi = [gem.Indexed(table, (j, *ii)) for j in range(self.M.shape[1])]
        # the sum approach is faster than calling numpy.dot or gem.IndexSum
        exprs = [gem.ComponentTensor(gem.Sum(*(self.M.array[i, j] * phi[j] for j in js)), ii)
                 for i, js in enumerate(self.csr)]

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
    basis functions."""

    #: Relative tolerance below which entries of the numeric reference
    #: tabulations are dropped from the symbolic rows.
    tol = 1e-10

    #: False selects plain integrals for physical scalar facet moments.
    avg = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cite("Kirby2018zany")
        cite("Kirby2019zany")

    def basis_transformation(self, coordinate_mapping):
        r"""Transformation matrix for the basis functions.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A ``gem.ListTensor`` with exposed physical dofs as rows
            and the complete reference basis as columns.
        """
        V = self._inverse_physical_vandermonde(coordinate_mapping)
        return gem.ListTensor(V[:, self._physical_indices].T)

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
        B = self._physical_vandermonde(coordinate_mapping)
        indices = self._physical_indices
        B = gem.ListTensor(B[numpy.ix_(indices, indices)])

        key = None
        return MappedTabulation(B, {key: Q})[key]

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        M = self.basis_transformation(coordinate_mapping)
        return MappedTabulation(M, ref_tabulation)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        result = super().basis_evaluation(order, ps, entity=entity)
        return self.map_tabulation(result, coordinate_mapping)

    @cached_property
    def node_mapping(self):
        """The physical convention of each node: ``cartesian``, ``frame`` or ``invariant``.

        Scalar values and interior moments are invariant. Other facet
        moments, and facet nodes with derivatives or matrix values, are
        posed on the facet frame. All other nodes map as Cartesian tensors.
        """
        dual = self._element.get_dual_set()
        sd = self.cell.get_spatial_dimension()
        mappings = [None] * len(dual.nodes)
        for dim, entities in dual.get_entity_ids().items():
            for dofs in entities.values():
                for i in dofs:
                    node = dual.nodes[i]
                    order = node.max_deriv_order
                    rank = len(node.target_shape)
                    if order == rank == 0 or (dim == sd and isinstance(node, IntegralMoment)):
                        mappings[i] = "invariant"
                    elif dim == sd - 1 and (isinstance(node, IntegralMoment) or order > 0 or rank > 1):
                        mappings[i] = "frame"
                    else:
                        mappings[i] = "cartesian"
        return tuple(mappings)

    @cached_property
    def _physical_indices(self):
        """The reference dofs exposed as physical dofs, in local order."""
        return sorted(i for entities in self.entity_dofs().values()
                      for dofs in entities.values() for i in dofs)

    @cached_property
    def _inverse_dual_coeffs(self):
        r"""For :math:`n=Cn_0`, the Vandermonde transforms as :math:`B=CB_0C^{-1}`."""
        return numpy.linalg.inv(self._element.get_dual_set().get_coeffs())

    @cached_property
    def _weights(self):
        """The weight tensor of each node, see :attr:`FIAT.functional.Functional.weight_tensors`."""
        weights = []
        for node in self._element.get_dual_set().nodes:
            if len(node.weight_tensors) > 1:
                raise NotImplementedError(f"{type(node).__name__} mixes derivative orders.")
            _, W = node.weight_tensors[node.max_deriv_order]
            weights.append(W)
        return weights

    @cached_property
    def _axis_mappings(self):
        """The mapping of each value and derivative axis of the weights of each node."""
        axis_mappings = []
        for node, mapping in zip(self._element.get_dual_set().nodes, self._element.mapping()):
            rank = len(node.target_shape)
            words = mapping.split()
            if words == ["affine"]:
                value_mappings = (mapping,) * rank
            elif words[0] == "double":
                value_mappings = (f"{words[1]} piola",) * 2
            else:
                value_mappings = tuple(f"{word} piola" for word in words[:-1])
            if len(value_mappings) != rank:
                raise ValueError(f"A {mapping} functional must have rank {len(value_mappings)}, not {rank}.")
            axis_mappings.append(value_mappings + ("derivative",) * node.max_deriv_order)
        return axis_mappings

    @cached_property
    def _axis_groups(self):
        """Group the axes of the weights of each physical node by their pullback.

        On Cartesian nodes, a contravariant axis paired with a covariant axis
        through the identity is pulled back by duality, and derivatives along
        an edge are fixed by its trace.
        """
        dual = self._element.get_dual_set()
        sd = self.cell.get_spatial_dimension()
        groups = [None] * len(dual.nodes)
        for dim, entities in dual.get_entity_ids().items():
            for entity, dofs in entities.items():
                for i in dofs:
                    if self.node_mapping[i] == "invariant":
                        continue
                    W = self._weights[i]
                    axis_mappings = self._axis_mappings[i]
                    cartesian = self.node_mapping[i] == "cartesian"
                    on_edge = cartesian and 0 < dim < sd - 1
                    groups[i] = {}
                    if cartesian:
                        covectors = {"covariant piola"} if on_edge else {"covariant piola", "derivative"}
                        for k, vector in enumerate(axis_mappings):
                            for l, covector in enumerate(axis_mappings):
                                if (vector == "contravariant piola" and covector in covectors
                                        and not any(k in axes or l in axes for axes in groups[i])
                                        and _in_range(W, (k, l), _duality_projector(sd), self.tol)):
                                    groups[i][k, l] = "duality"
                    for k, mapping in enumerate(axis_mappings):
                        if any(k in axes for axes in groups[i]):
                            continue
                        if on_edge and mapping == "derivative":
                            t = self.cell.compute_edge_tangent(entity)
                            if not _in_range(W, (k,), numpy.outer(t, t) / (t @ t), self.tol):
                                raise NotImplementedError("No physical convention for derivatives normal to an edge")
                            mapping = "affine"
                        groups[i][k,] = mapping
        return groups

    @cached_property
    def _uncontracted_riesz(self):
        r"""Uncontracted Riesz representation of each node in the source nodal basis.

        :math:`R_{i,a,\beta,b,\gamma,j}=\sum_q W_{i,a,\beta,q}(\partial^\gamma\psi^0_j)_b(x_q)`,
        where :math:`\psi^0` is the nodal basis of the nodes before
        recombination. A pullback :math:`G` contracts it into the physical
        Vandermonde row :math:`B_{ij}=\sum G_{b\gamma,a\beta}R_{i,a,\beta,b,\gamma,j}`.
        """
        dual = self._element.get_dual_set()
        coeffs = self._element.get_coeffs()
        C = dual.get_coeffs()
        if C is not None:
            # The source nodal basis is C^T times the basis dual to C n_0.
            coeffs = numpy.tensordot(C.T, coeffs, axes=(1, 0))
        riesz = dual.to_uncontracted_riesz(self._element.get_nodal_basis())
        result = [None] * len(dual.nodes)
        for i, (node, W) in enumerate(zip(dual.nodes, self._weights)):
            if self.node_mapping[i] == "invariant":
                continue
            rank = len(node.target_shape)
            R = numpy.tensordot(riesz[node.max_deriv_order][i], coeffs, axes=(-1, -1))
            # (a, beta, gamma, j, b) -> (a, beta, b, gamma, j)
            result[i] = numpy.moveaxis(R, range(R.ndim - rank, R.ndim), range(W.ndim - 1, W.ndim - 1 + rank))
        return result

    @cached_property
    def _entity_support_ids(self):
        """Partition each entity_ids[dim][entity] by the smallest entity containing the points of each node."""
        dual = self._element.get_dual_set()
        top = dual.ref_el.get_topology()
        support_ids = {}
        for dim, entities in dual.get_entity_ids().items():
            support_ids[dim] = {}
            for entity, dofs in entities.items():
                supports = defaultdict(list)
                for i in dofs:
                    node = dual.nodes[i]
                    bary = dual.ref_el.compute_barycentric_coordinates(numpy.array([*node.pt_dict, *node.deriv_dict]))
                    # A nonzero barycentric coordinate requires its vertex in the entity.
                    verts = set(numpy.flatnonzero(numpy.abs(bary).max(axis=0) > 1e-12))
                    support = next((d, e) for d in sorted(top) for e in sorted(top[d]) if verts <= set(top[d][e]))
                    supports[support].append(i)
                support_ids[dim][entity] = dict(supports)
        return support_ids

    @cached_property
    def _scaling_exponents(self):
        r"""The exponents :math:`p_i` such that each physical node dilates as :math:`h^{-p_i}`.

        The avg flag selects physical averages for framed integral moments.
        """
        dual = self._element.get_dual_set()
        sd = self.cell.get_spatial_dimension()
        exponents = numpy.zeros(len(dual.nodes), dtype=int)
        for dim, entities in dual.get_entity_ids().items():
            for entity, dofs in entities.items():
                for i in dofs:
                    axis_mappings = self._axis_mappings[i]
                    if self.node_mapping[i] == "cartesian":
                        weights = {"derivative": 1, "covariant piola": 1, "contravariant piola": sd - 1}
                        exponents[i] = sum(weights.get(mapping, 0) for mapping in axis_mappings)
                    elif self.node_mapping[i] == "frame":
                        exponents[i] = axis_mappings.count("derivative")
                        nu = self.cell.compute_scaled_normal(entity)
                        for k, mapping in enumerate(axis_mappings):
                            # Tangential components of a contravariant axis
                            if mapping == "contravariant piola" and not _in_range(
                                    self._weights[i], (k,), numpy.outer(nu, nu) / (nu @ nu), self.tol):
                                exponents[i] += 2 - sd
                        if isinstance(dual.nodes[i], IntegralMoment) and not self.avg:
                            exponents[i] -= dim
        return exponents

    def _dof_scales(self, coordinate_mapping):
        h = coordinate_mapping.cell_size()
        dual = self._element.get_dual_set()
        topology = self.cell.get_topology()
        exponents = self._scaling_exponents
        for dim, entities in dual.get_entity_ids().items():
            for entity, dofs in entities.items():
                if not any(exponents[i] for i in dofs):
                    continue
                vertices = topology[dim][entity]
                havg = sum(h[v] for v in vertices) / len(vertices)
                for i in dofs:
                    if exponents[i]:
                        yield i, havg**-exponents[i]

    def _physical_rows(self, coordinate_mapping):
        """Assemble nonidentity Vandermonde rows in numerator/denominator form.

        Contract the pullback of each node with its uncontracted Riesz
        representation, then check that each physical node couples only to
        its entity's closure.
        """
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd + 1)
        if sd == 1:
            # The signed determinant carries the orientation of the tangent.
            J = gem.ListTensor([[coordinate_mapping.detJ_at(bary)]])
        else:
            J = coordinate_mapping.jacobian_at(bary)
        J = numpy.array([[J[i, k] for k in range(J.shape[1])] for i in range(J.shape[0])], dtype=object)
        adjJ, detJ = adjugate_and_determinant(J)
        cartesian = _cartesian_pullbacks(J, adjJ, detJ)

        dual = self._element.get_dual_set()
        closures = self._element.entity_closure_dofs()
        rows = {}
        for dim, entities in self._entity_support_ids.items():
            for entity, supports in entities.items():
                for support, dofs in supports.items():
                    if any(self.node_mapping[i] == "frame" for i in dofs):
                        frame = _frame_pullbacks(self.cell, entity, support, J, adjJ, detJ)
                    for i in dofs:
                        if self.node_mapping[i] == "invariant":
                            continue
                        pullbacks = frame if self.node_mapping[i] == "frame" else cartesian
                        measure = one
                        if self.node_mapping[i] == "frame" and not self.avg and isinstance(dual.nodes[i], IntegralMoment):
                            # The physical facet measure induced by the cofactor map
                            nu = self.cell.compute_scaled_normal(entity)
                            Knu = adjJ.T @ nu
                            measure = self.cell.volume_of_subcomplex(sd - 1, entity) / numpy.linalg.norm(nu) * (Knu @ Knu)**0.5
                        row = _physical_row(self._uncontracted_riesz[i], self._weights[i],
                                            {axes: pullbacks[mapping] for axes, mapping in self._axis_groups[i].items()},
                                            measure, self.tol)
                        if row is None:
                            continue
                        outside = [j for j, entry in enumerate(row[0])
                                   if j not in closures[dim][entity] and not isinstance(entry, gem.Zero)]
                        if outside:
                            raise NotImplementedError(f"Dof {i} on entity {(dim, entity)} couples outside its closure: {outside}")
                        rows[i] = row
        return rows

    def _physical_vandermonde(self, coordinate_mapping):
        """The physical Vandermonde matrix :math:`B` over all reference dofs."""
        B = identity(self._element.space_dimension())
        for i, (numerator, denominator) in self._physical_rows(coordinate_mapping).items():
            B[i] = numerator / denominator
        for i, scale in self._dof_scales(coordinate_mapping):
            B[i] = B[i] / scale
        C = self._element.get_dual_set().get_coeffs()
        if C is not None:
            B = C @ B @ self._inverse_dual_coeffs
        return B

    def _inverse_physical_vandermonde(self, coordinate_mapping):
        """:math:`B^{-1}` over all reference dofs.

        Solve entity blocks in closure order, retaining completion functions.
        """
        rows = self._physical_rows(coordinate_mapping)
        nbf = self._element.space_dimension()
        closures = self._element.entity_closure_dofs()
        entity_dofs = self._element.entity_dofs()
        V = identity(nbf)
        cache = {}
        for dim, entities in sorted(closures.items()):
            for entity, closure_dofs in entities.items():
                block = entity_dofs[dim][entity]
                if not any(i in rows for i in block):
                    continue
                closure = [j for j in closure_dofs if j not in block]
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
                            rhs[k] = rhs[k] - V[j] * numerator[j]
                V[block] = solve(Bee, rhs, cache)
        for i, scale in self._dof_scales(coordinate_mapping):
            V[:, i] = V[:, i] * scale
        C = self._element.get_dual_set().get_coeffs()
        if C is not None:
            V = C @ V @ self._inverse_dual_coeffs
        return V


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


class Pullback:
    r"""The pullback :math:`G=(sP+\sum_t g_t C_t)/d` of a group of tensor axes.

    The terms :math:`C_t` vanish on the range of the projector :math:`P`,
    where :math:`G` only scales by :math:`s/d`. By default :math:`s=d`.
    """

    def __init__(self, projector, terms=(), denominator=one, scale=None):
        self.projector = numpy.asarray(projector)
        self.terms = list(terms)
        self.denominator = denominator
        self.scale = denominator if scale is None else scale

    def numerator(self, in_range=False):
        """The numerator terms of G, restricted to the range of P if in_range."""
        terms = [] if in_range else self.terms
        if self.projector.any():
            terms = [(self.scale, self.projector), *terms]
        return terms


def _cartesian_pullbacks(J, adjJ, detJ):
    """Pull back each axis mapping in Cartesian components."""
    sd = J.shape[1]
    units = numpy.eye(sd * sd).reshape(sd, sd, sd, sd)
    contravariant = Pullback(numpy.zeros((sd, sd)), [(J[b, a], units[a, b]) for a, b in numpy.ndindex(sd, sd)], detJ)
    covariant = Pullback(numpy.zeros((sd, sd)), [(adjJ[a, b], units[a, b]) for a, b in numpy.ndindex(sd, sd)], detJ)
    # The pairing of a contravariant axis with a covariant axis only scales by 1/detJ.
    P = _duality_projector(sd)
    complement = numpy.eye(sd * sd) - P
    duality = Pullback(P, [(g * h, numpy.kron(C, D) @ complement)
                           for (g, C), (h, D) in product(contravariant.terms, covariant.terms)],
                       detJ * detJ, scale=detJ)
    return {"affine": Pullback(numpy.eye(sd)), "contravariant piola": contravariant,
            "covariant piola": covariant, "derivative": covariant, "duality": duality}


def _duality_projector(sd):
    """Projector onto the identity, the pairing of a contravariant axis with a covariant axis."""
    delta = numpy.eye(sd).ravel()
    return numpy.outer(delta, delta) / sd


def _frame_pullbacks(ref_el, facet, support, J, adjJ, detJ):
    """Pull back the facet's physical normal and tangential directions."""
    sd = ref_el.get_spatial_dimension()
    tangents = list(numpy.linalg.qr(ref_el.compute_tangents(sd - 1, facet).T)[0].T)
    nu = ref_el.compute_scaled_normal(facet)
    K = adjJ.T
    n = nu / numpy.linalg.norm(nu)
    Kn = K @ n
    # Each normal satisfies J^{-1} n_phys = u/d in the given reference frame.
    normals = []
    if support[0] == sd - 2 >= 1:
        t = ref_el.compute_edge_tangent(support[1])
        m = numpy.cross(t, n)
        m /= numpy.linalg.norm(m)
        edge_tangents = [t / numpy.linalg.norm(t)]
        w = numpy.cross(J.T @ (J @ t), n)
        Jw = J @ w
        normals.append((m, w, (Jw @ Jw)**0.5, edge_tangents + [m]))
    else:
        edge_tangents = tangents
    frame = edge_tangents + [m for m, *_ in normals] + [n]
    normals.append((n, adjJ @ Kn, detJ * (Kn @ Kn)**0.5, frame))
    P = numpy.eye(sd) - sum(numpy.outer(n, n) for n, *_ in normals)
    dens = [d for _, _, d, _ in normals]
    terms = []
    for k, (n, u, _, frame) in enumerate(normals):
        scale = reduce(mul, dens[:k] + dens[k+1:], one)
        terms.extend(((u @ f) * scale, numpy.outer(f, n)) for f in frame)
    covariant = Pullback(P, terms, reduce(mul, dens))
    nn = nu @ nu
    Knu = K @ nu
    KtKnu = K.T @ Knu
    P = numpy.outer(nu, nu) / nn
    terms = [(Knu @ Knu, numpy.eye(sd) - P)]
    terms.extend((KtKnu @ -tau, numpy.outer(nu, tau)) for tau in tangents)
    contravariant = Pullback(P, terms, detJ * nn)
    return {"affine": Pullback(numpy.eye(sd)), "covariant piola": covariant,
            "derivative": covariant, "contravariant piola": contravariant}


def _in_range(weights, axes, projector, tol):
    """Whether the weights lie in the range of the projector on the given axes."""
    W = numpy.moveaxis(weights, axes, range(len(axes))).reshape(projector.shape[1], -1)
    return numpy.abs(W - projector @ W).max() <= tol * numpy.abs(weights).max()


def _physical_row(riesz, weights, pullbacks, measure, tol):
    r"""Contract :math:`B_j=\sum_{a,b}G_{ba}R_{abj}` in numerator/denominator form.

    Each group of axes contributes a Kronecker factor of G. Returns None
    if the node is invariant.
    """
    numerators = {}
    invariant = measure == one
    for axes, pullback in pullbacks.items():
        in_range = _in_range(weights, axes, pullback.projector, tol)
        invariant = invariant and in_range and pullback.scale == pullback.denominator
        numerators[axes] = pullback.numerator(in_range)
    if invariant:
        return None

    n = (riesz.ndim - 1) // 2
    axes = list(chain.from_iterable(pullbacks))
    R = riesz.transpose(*axes, *(n + k for k in axes), riesz.ndim - 1).reshape(-1, riesz.shape[-1])
    numerator = numpy.full(R.shape[1], zero, dtype=object)
    cutoff = tol * numpy.abs(R).max()
    for terms in product(*numerators.values()):
        coefficients, matrices = zip(*terms) if terms else ((), ())
        C = reduce(numpy.kron, matrices) if matrices else numpy.ones((1, 1))
        reference_row = C.T.ravel() @ R
        nonzeros = numpy.flatnonzero(numpy.abs(reference_row) > cutoff * numpy.abs(C).max())
        coefficient = gem.Product(*coefficients)
        for j in nonzeros:
            numerator[j] += coefficient * reference_row[j]
    return numerator * measure, gem.Product(*(pullback.denominator for pullback in pullbacks.values()))
