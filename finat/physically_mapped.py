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


class PhysicallyMappedFunctional:
    r"""A FIAT node and the data that maps it to the physical cell.

    :attr:`pullback_mappings` assigns a mapping to each group of axes of
    :attr:`weights`: ``affine``, ``covariant piola``, ``contravariant
    piola``, or ``L2 piola`` for a contravariant axis paired with a
    covariant axis through the identity.  These mappings act on Cartesian
    components if :attr:`support_facet` is None, and on components in the
    frame of :attr:`support_facet` otherwise.

    :arg node: The FIAT node.
    :arg ref_el: The reference cell.
    :arg dim: The dimension of the entity that supports the node.
    :arg entity: The entity that supports the node.
    :arg mapping: The FIAT mapping of the node.
    :arg avg: If False, an integral moment with a :attr:`support_facet` is
        a plain integral over its entity instead of an average.
    :arg tol: Relative tolerance for the range tests on the weights, and
        below which entries of the Riesz representation are dropped.
    """

    def __init__(self, node, ref_el, dim, entity, mapping, avg, tol):
        self.node = node
        self.ref_el = ref_el
        self.dim = dim
        self.entity = entity
        self.mapping = mapping
        self.avg = avg
        self.tol = tol

    @cached_property
    def weights(self):
        """The weights of the node, see :meth:`FIAT.functional.Functional.get_order_tensors`."""
        node = self.node
        if len(node.orders) != 1:
            raise NotImplementedError(f"{type(node).__name__} mixes derivative orders.")
        _, W = node.get_order_tensors(node.orders[0])
        return W

    @cached_property
    def axis_mappings(self):
        """The mapping of each value and derivative axis of the weights."""
        rank = len(self.node.target_shape)
        words = self.mapping.split()
        if words == ["affine"]:
            value_mappings = (self.mapping,) * rank
        elif words[0] == "double":
            value_mappings = (f"{words[1]} piola",) * 2
        else:
            value_mappings = tuple(f"{word} piola" for word in words[:-1])
        if len(value_mappings) != rank:
            raise ValueError(f"A {self.mapping} functional must have rank {len(value_mappings)}, not {rank}.")
        return value_mappings + ("covariant piola",) * self.node.max_deriv_order

    @cached_property
    def support_facet(self):
        r"""The facet in whose frame :attr:`pullback_mappings` act, or None for Cartesian components.

        For integral moments on a facet, and for nodes on a facet with
        derivatives or matrix values, this is the facet itself, with the
        frame of its tangents and normal.  For an integral moment on an edge
        with derivatives normal to the edge, this is the facet :math:`f`
        whose plane contains those derivatives, with the frame
        :math:`(t, t \times n_f, n_f)` of the edge tangent :math:`t` and
        the normal :math:`n_f`.  Otherwise it is None.
        """
        ref_el = self.ref_el
        sd = ref_el.get_spatial_dimension()
        node = self.node
        moment = isinstance(node, IntegralMoment)
        if self.dim == sd - 1 and (moment or node.max_deriv_order > 0 or len(node.target_shape) > 1):
            return self.entity
        if moment and 0 < self.dim < sd - 1:
            covectors = [k for k, mapping in enumerate(self.axis_mappings) if mapping == "covariant piola"]
            t = ref_el.compute_edge_tangent(self.entity)
            if not all(in_range_of(self.weights, numpy.outer(t, t) / (t @ t), (k,), self.tol) for k in covectors):
                top = ref_el.get_topology()
                vertices = set(top[self.dim][self.entity])
                for facet in sorted(top[sd - 1]):
                    nu = ref_el.compute_scaled_normal(facet)
                    P = numpy.eye(sd) - numpy.outer(nu, nu) / (nu @ nu)
                    if (vertices <= set(top[sd - 1][facet])
                            and all(in_range_of(self.weights, P, (k,), self.tol) for k in covectors)):
                        return facet
                raise NotImplementedError(f"No facet contains the derivatives of {type(node).__name__}.")
        return None

    @cached_property
    def pullback_mappings(self):
        """A dict from each group of axes of :attr:`weights` that map together to its mapping."""
        # vandermonde_row takes the Kronecker product of the weight transformations
        # in the order of this dict.
        sd = self.ref_el.get_spatial_dimension()
        W = self.weights
        axis_mappings = self.axis_mappings
        # Interior moments map every axis as affine.
        if self.dim == sd and isinstance(self.node, IntegralMoment):
            return {(k,): "affine" for k in range(len(axis_mappings))}
        # In the frame of a support facet, each axis keeps its own mapping.
        if self.support_facet is not None:
            return {(k,): mapping for k, mapping in enumerate(axis_mappings)}

        # A contravariant axis contracted with a covariant axis through the
        # identity is a trace, and the pair maps as L2 piola.
        groups = {}
        for k, vector in enumerate(axis_mappings):
            for l, covector in enumerate(axis_mappings):
                if (vector == "contravariant piola" and covector == "covariant piola"
                        and not any(k in axes or l in axes for axes in groups)
                        and in_range_of(W, pairing_projector(sd), (k, l), self.tol)):
                    groups[k, l] = "L2 piola"
        # A derivative along the edge maps as affine.
        # Every other axis keeps its own mapping.
        on_edge = 0 < self.dim < sd - 1
        for k, mapping in enumerate(axis_mappings):
            if any(k in axes for axes in groups):
                continue
            if on_edge and mapping == "covariant piola":
                t = self.ref_el.compute_edge_tangent(self.entity)
                if not in_range_of(W, numpy.outer(t, t) / (t @ t), (k,), self.tol):
                    raise NotImplementedError(f"Cannot map derivatives normal to an edge of {type(self.node).__name__}.")
                mapping = "affine"
            groups[k,] = mapping
        return groups

    @cached_property
    def scaling_exponent(self):
        """The exponent :math:`p` such that the physical node dilates as :math:`h^{-p}`.

        Each group in :attr:`pullback_mappings` contributes its scaling.  If
        :attr:`avg` is False, an integral moment with a :attr:`support_facet`
        is a plain integral over its entity, which contributes :math:`-dim`.
        """
        sd = self.ref_el.get_spatial_dimension()
        if self.support_facet is None:
            weights = {"covariant piola": 1, "contravariant piola": sd - 1, "L2 piola": sd}
            return sum(weights.get(mapping, 0) for mapping in self.pullback_mappings.values())
        nu = self.ref_el.compute_scaled_normal(self.support_facet)
        exponent = 0
        for axes, mapping in self.pullback_mappings.items():
            if mapping == "covariant piola":
                exponent += 1
            elif mapping == "contravariant piola" and not in_range_of(
                    self.weights, numpy.outer(nu, nu) / (nu @ nu), axes, self.tol):
                # Tangential components of a contravariant axis
                exponent += 2 - sd
        if isinstance(self.node, IntegralMoment) and not self.avg:
            exponent -= self.dim
        return exponent

    def vandermonde_row(self, transformations, D, J, h):
        r"""The row of the physical Vandermonde matrix for this node.

        The row is :math:`B_j = h^p\,|e|\,(GW):D_j`, where :math:`W` are
        the :attr:`weights`, :math:`D` are the derivatives of the basis,
        :math:`G` is the Kronecker product of the weight transformations of
        :attr:`pullback_mappings`, :math:`p` is :attr:`scaling_exponent`, and
        :math:`|e|` is the physical measure of the entity if :attr:`avg` is
        False and there is a :attr:`support_facet`, and 1 otherwise.

        :arg transformations: The :class:`WeightTransformation` of each mapping,
            in Cartesian components or in the frame of :attr:`support_facet`.
        :arg D: The derivatives of the basis at the points of the node, with
            the axes of the weights followed by the basis.
        :arg J: The Jacobian as an object array of GEM scalars.
        :arg h: The cell size at each vertex.
        :returns: The numerator and denominator of the Vandermonde row.
        """
        measure = one
        if self.support_facet is not None and isinstance(self.node, IntegralMoment) and not self.avg:
            # The physical measure of the support, from the Gram determinant of its tangents
            T = self.ref_el.compute_tangents(self.dim, self.entity)
            JT = J @ T.T
            volume = self.ref_el.volume_of_subcomplex(self.dim, self.entity)
            measure = volume / numpy.sqrt(numpy.linalg.det(T @ T.T)) * determinant(JT.T @ JT)**0.5

        G = reduce(WeightTransformation.kron, map(transformations.get, self.pullback_mappings.values()),
                   WeightTransformation([one], [numpy.eye(1)]))
        axes = list(chain.from_iterable(self.pullback_mappings))
        numerator = G.to_riesz(self.weights, axes, D, self.tol)
        numerator *= measure
        denominator = G.denominator

        p = self.scaling_exponent
        if p:
            vertices = self.ref_el.get_topology()[self.dim][self.entity]
            havg = sum(h[v] for v in vertices) / len(vertices)
            denominator *= havg**-p
        return numerator, denominator


class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions."""

    #: Relative tolerance for the range tests on the weights of each node,
    #: and below which entries of the Riesz representation are dropped.
    tol = 1e-10

    #: If False, an integral moment with a support facet is a plain
    #: integral over its entity instead of an average.
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
    def _physical_indices(self):
        """The reference dofs exposed as physical dofs, in local order.

        These are the dofs of :meth:`entity_dofs`, which omits the
        constraint nodes that only complete the transformation.
        """
        return sorted(i for entities in self.entity_dofs().values()
                      for dofs in entities.values() for i in dofs)

    @cached_property
    def _physical_functionals(self):
        """The physical counterpart of each node of the FIAT dual set."""
        dual = self._element.get_dual_set()
        mappings = self._element.mapping()
        functionals = [None] * len(dual.nodes)
        for dim, entities in dual.get_entity_ids().items():
            for entity, dofs in entities.items():
                for i in dofs:
                    functionals[i] = PhysicallyMappedFunctional(dual.nodes[i], self.cell, dim, entity,
                                                                mappings[i], self.avg, self.tol)
        return tuple(functionals)

    @cached_property
    def _reference_derivatives(self):
        r"""The derivatives of the source nodal basis at the points of the nodes.

        A dict from each derivative order k and point x to the tensor
        :math:`D_{b\gamma j}=(\partial^\gamma\psi^0_j)_b(x)` with k
        derivative axes :math:`\gamma`, where :math:`\psi^0` is the nodal
        basis of the nodes before recombination.  The basis is tabulated one
        time for each order.
        """
        sd = self.cell.get_spatial_dimension()
        dual = self._element.get_dual_set()
        points = defaultdict(set)
        for node in dual.nodes:
            order = node.max_deriv_order
            node_points, _ = node.get_order_tensors(order)
            points[order].update(node_points)

        C = dual.get_coeffs()
        derivatives = {}
        for order in points:
            pts = sorted(points[order])
            tabulation = self._element.tabulate(order, pts)
            values = tabulation[(0,) * sd]
            rank = values.ndim - 2
            D = numpy.zeros((sd,) * order + values.shape)
            for index in numpy.ndindex(D.shape[:order]):
                D[index] = tabulation[tuple(map(index.count, range(sd)))]
            # (gamma, j, b, x) -> (x, b, gamma, j)
            D = D.transpose(-1, *range(order + 1, order + 1 + rank), *range(order), order)
            if C is not None:
                # The source nodal basis is C^T times the basis dual to C n_0.
                D = D @ C
            derivatives[order] = dict(zip(pts, D))
        return derivatives

    def _vandermonde_rows(self, coordinate_mapping):
        """The rows of the physical Vandermonde matrix that differ from the identity.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A dict from each node number to the numerator and
            denominator of its :meth:`PhysicallyMappedFunctional.vandermonde_row`.
        :raises NotImplementedError: If a row couples to a dof outside the
            closure of the entity of the node.
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
        h = coordinate_mapping.cell_size()

        cartesian = cartesian_transformations(J, adjJ, detJ)
        frames = {}
        derivatives = self._reference_derivatives

        closures = self._element.entity_closure_dofs()
        rows = {}
        for i, functional in enumerate(self._physical_functionals):
            if functional.support_facet is not None:
                key = (functional.support_facet, (functional.dim, functional.entity))
                if key not in frames:
                    frames[key] = frame_transformations(self.cell, *key, J, adjJ, detJ)
                transformations = frames[key]
            else:
                transformations = cartesian
            order = functional.node.max_deriv_order
            points, _ = functional.node.get_order_tensors(order)
            D = numpy.stack([derivatives[order][x] for x in points], axis=-2)
            numerator, denominator = functional.vandermonde_row(transformations, D, J, h)
            nonzeros = [j for j, entry in enumerate(numerator) if not isinstance(entry, gem.Zero)]
            if nonzeros == [i] and numerator[i] == denominator:
                continue
            closure = closures[functional.dim][functional.entity]
            outside = [j for j in nonzeros if j not in closure]
            if outside:
                raise NotImplementedError(f"Dof {i} on entity {(functional.dim, functional.entity)} "
                                          f"couples outside its closure: {outside}")
            rows[i] = numerator, denominator
        return rows

    def _recombine(self, A):
        r"""For :math:`n=Cn_0`, a matrix over the nodes transforms as :math:`CAC^{-1}`."""
        C = self._element.get_dual_set().get_coeffs()
        if C is None:
            return A
        return C @ A @ numpy.linalg.inv(C)

    def _physical_vandermonde(self, coordinate_mapping):
        """The physical Vandermonde matrix :math:`B` over all reference dofs."""
        B = identity(self._element.space_dimension())
        for i, (numerator, denominator) in self._vandermonde_rows(coordinate_mapping).items():
            B[i] = numerator / denominator
        return self._recombine(B)

    def _inverse_physical_vandermonde(self, coordinate_mapping):
        """:math:`B^{-1}` over all reference dofs.

        Solve entity blocks in closure order, retaining completion functions.
        """
        rows = self._vandermonde_rows(coordinate_mapping)
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
        return self._recombine(V)


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


class WeightTransformation:
    r"""A matrix :math:`G=\frac{1}{d}\sum_t g_t C_t` that transforms the weights on a group of tensor axes.

    The coefficients :math:`g_t` and the denominator :math:`d` are GEM
    scalars, and the matrices :math:`C_t` are constant, so that
    contractions with constant tensors cancel numerically.

    :arg coefficients: The coefficients :math:`g_t`.
    :arg matrices: The matrices :math:`C_t`.
    :arg denominator: The denominator :math:`d`.
    """

    def __init__(self, coefficients, matrices, denominator=one):
        self.coefficients = tuple(coefficients)
        self.matrices = numpy.asarray(matrices, dtype=float)
        self.denominator = denominator

    @classmethod
    def from_matrix(cls, A, denominator=one):
        """The transformation :math:`A/d` of a matrix of GEM scalars, expanded in its entries."""
        n = len(A)
        units = numpy.eye(n * n).reshape(n, n, n, n)
        entries = [(A[a, b], units[a, b]) for a, b in numpy.ndindex(A.shape) if not isinstance(A[a, b], gem.Zero)]
        return cls(*zip(*entries), denominator)

    def __add__(self, other):
        d, e = self.denominator, other.denominator
        return WeightTransformation([g * e for g in self.coefficients] + [g * d for g in other.coefficients],
                                    numpy.concatenate([self.matrices, other.matrices]), d * e)

    def kron(self, other):
        """The Kronecker product, which transforms the axes of self followed by the axes of other."""
        return WeightTransformation([g * h for g, h in product(self.coefficients, other.coefficients)],
                                    [numpy.kron(C, D) for C, D in product(self.matrices, other.matrices)],
                                    self.denominator * other.denominator)

    def eigenvalue(self, W, tol):
        r"""The numerator :math:`\lambda d` such that :math:`GW=\lambda W`, or None.

        This holds when each :math:`C_tW=c_tW`, and then
        :math:`\lambda d=\sum_t c_t g_t`.

        :arg W: The weights as a matrix, with the transformed axes as rows.
        :arg tol: Relative tolerance on :math:`C_tW-c_tW`.
        """
        CW = self.matrices @ W
        c = numpy.tensordot(CW, W, axes=2) / numpy.tensordot(W, W, axes=2)
        residual = numpy.abs(CW - c[:, None, None] * W).max(axis=(1, 2))
        if (residual > tol * numpy.abs(self.matrices).max(axis=(1, 2)) * numpy.abs(W).max()).any():
            return None
        c = round_to_int(c, tol)
        return sum((g * ct for g, ct in zip(self.coefficients, c) if ct), zero)

    def to_riesz(self, W, axes, D, tol):
        r"""The numerator of the Riesz representation :math:`\sum_t g_t\,(C_tW):D` of the transformed node.

        If :math:`GW=\lambda W`, this is :math:`\lambda\,W:D`.

        :arg W: The weights of the node.
        :arg axes: The axes of W that G transforms, in the order of its
            Kronecker factors.
        :arg D: The derivatives of the basis at the points of the node, with
            the axes of W followed by the basis.
        :arg tol: Tolerance relative to W and D below which terms are dropped.
        """
        # The transformed axes of W as rows
        transposed = numpy.moveaxis(W, axes, range(len(axes)))
        X = transposed.reshape(len(self.matrices[0]), -1)
        coefficients = self.coefficients
        matrices = self.matrices
        eigenvalue = self.eigenvalue(X, tol)
        if eigenvalue is not None:
            coefficients = [eigenvalue]
            matrices = numpy.eye(len(X))[None]
        # The weights C_t W of each term, with the axes of W restored
        CW = (matrices @ X).reshape(len(matrices), *transposed.shape)
        CW = numpy.moveaxis(CW, range(1, len(axes) + 1), [1 + k for k in axes])
        rows = numpy.tensordot(CW, D, axes=W.ndim)
        cutoffs = tol * numpy.abs(W).max() * numpy.abs(D).max() * numpy.abs(matrices).max(axis=(1, 2))
        numerator = numpy.full(D.shape[-1], zero, dtype=object)
        for g, row, cutoff in zip(coefficients, rows, cutoffs):
            row = round_to_int(row, cutoff)
            for j in numpy.flatnonzero(numpy.abs(row) > cutoff):
                numerator[j] += g * row[j]
        return numerator


def round_to_int(x, tol):
    """Round the entries of x that lie within tol of an integer."""
    rounded = numpy.round(x)
    return numpy.where(numpy.abs(x - rounded) <= tol, rounded, x)


def cartesian_transformations(J, adjJ, detJ):
    """The weight transformation of each mapping in Cartesian components."""
    sd = J.shape[1]
    contravariant = WeightTransformation.from_matrix(J.T, detJ)
    covariant = WeightTransformation.from_matrix(adjJ, detJ)
    # The pairing of a contravariant axis with a covariant axis scales the
    # identity by 1/detJ, so it is expanded in the identity and its complement.
    pairing = contravariant.kron(covariant)
    P = pairing_projector(sd)
    pairing = WeightTransformation([detJ, *pairing.coefficients], [P, *(pairing.matrices @ (numpy.eye(sd * sd) - P))],
                                   pairing.denominator)
    return {"affine": WeightTransformation([one], [numpy.eye(sd)]), "contravariant piola": contravariant,
            "covariant piola": covariant, "L2 piola": pairing}


def pairing_projector(sd):
    r"""Orthogonal projector onto the span of the flattened identity :math:`\delta_{ab}`.

    Weights on a contravariant axis :math:`a` and a covariant axis
    :math:`b` lie in its range when they only contract :math:`a` with
    :math:`b`, as in the divergence :math:`\delta_{ab}\partial_b u_a`.
    """
    delta = numpy.eye(sd).ravel()
    return numpy.outer(delta, delta) / sd


def frame_transformations(ref_el, facet, support, J, adjJ, detJ):
    r"""The weight transformation of each mapping in the frame of the facet.

    On an edge support, the frame is :math:`(t, t \times n, n)` for the
    edge tangent :math:`t` and the facet normal :math:`n`.
    """
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
    # Covariant weights are unchanged along the tangents, and each normal
    # component maps through the frame.
    covariant = WeightTransformation([one], [numpy.eye(sd) - sum(numpy.outer(n, n) for n, *_ in normals)])
    for n, u, d, frame in normals:
        covariant += WeightTransformation([u @ f for f in frame], [numpy.outer(f, n) for f in frame], d)
    # Contravariant weights keep their normal component, and their
    # tangential components map through the frame.
    nn = nu @ nu
    Knu = K @ nu
    KtKnu = K.T @ Knu
    P = numpy.outer(nu, nu) / nn
    d = detJ * nn
    contravariant = WeightTransformation([d, Knu @ Knu, *(KtKnu @ -tau for tau in tangents)],
                                         [P, numpy.eye(sd) - P, *(numpy.outer(nu, tau) for tau in tangents)], d)
    return {"affine": WeightTransformation([one], [numpy.eye(sd)]), "covariant piola": covariant,
            "contravariant piola": contravariant}


def in_range_of(weights, projector, axes, tol):
    """Whether the weights lie in the range of the projector acting on the given axes."""
    shape = tuple(weights.shape[k] for k in axes)
    complement = numpy.eye(projector.shape[0]) - projector
    residual = numpy.tensordot(complement.reshape(shape + shape), weights, axes=(range(len(axes), 2 * len(axes)), axes))
    return numpy.abs(residual).max() <= tol * numpy.abs(weights).max()
