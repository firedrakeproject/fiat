r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from functools import cached_property, partial, reduce
from itertools import chain, product
from numbers import Number
from operator import add, mul

import gem
import numpy

from gem.optimise import canonical_bound_indices, factorise_scalar_sums

from FIAT.functional import IntegralMoment
from finat.citations import cite
from finat.finiteelementbase import broadcast_tensor

zero = gem.Zero()
one = gem.Literal(1.0)

#: The pullback of the weights of a node and its inverse, as
#: :class:`LinearCombination` objects.
Pullback = namedtuple("Pullback", ["forward", "inverse"])


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


class PhysicalTabulation(Mapping):
    """A lazy tabulation dict that applies the basis transformation only
    on the requested derivatives.

    :arg matvec: a callable that applies the basis transformation to a
        reference tabulation, with the basis functions as the first axis.
    :arg ref_tabulation: a dict of tabulations on the reference cell.
    """
    def __init__(self, matvec, ref_tabulation):
        self.matvec = matvec
        self.ref_tabulation = ref_tabulation
        self._tabulation_cache = {}

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
    components if :attr:`frame` is None, and on components in the frame of
    the entity :attr:`frame` otherwise.

    :arg node: The FIAT node.
    :arg cell: The reference cell.
    :arg dim: The dimension of the entity that supports the node.
    :arg entity: The entity that supports the node.
    :arg mapping: The FIAT mapping of the node.
    :arg avg: If False, an integral moment with a :attr:`frame` is a plain
        integral over its entity instead of an average.
    :arg tol: Relative tolerance for the range tests on the weights, and
        below which entries of the Riesz representation are dropped.
    """

    def __init__(self, node, cell, dim, entity, mapping, avg, tol):
        self.node = node
        self.cell = cell
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
    def frame(self):
        r"""The entity ``(dim, entity)`` in whose frame :attr:`pullback_mappings` act, or None for Cartesian components.

        For integral moments on a facet, and for nodes on a facet with
        derivatives or matrix values, this is the facet, with the frame of
        its tangents and normal.  For an integral moment on an edge of a
        tetrahedron with derivatives normal to the edge, this is the edge,
        with the frame :math:`(t, t \times n_1, t \times n_2)` of the edge
        tangent :math:`t` and the normals to :math:`t` within the two facets
        that contain the edge.  Otherwise it is None.
        """
        cell = self.cell
        sd = cell.get_spatial_dimension()
        node = self.node
        moment = isinstance(node, IntegralMoment)
        if self.dim == sd - 1 and (moment or node.max_deriv_order > 0 or len(node.target_shape) > 1):
            return (self.dim, self.entity)
        if moment and 0 < self.dim < sd - 1:
            covectors = [k for k, mapping in enumerate(self.axis_mappings) if mapping == "covariant piola"]
            Q = tangential_projector(cell.compute_tangents(self.dim, self.entity))
            if not all(in_range_of(self.weights, Q, (k,), self.tol) for k in covectors):
                return (self.dim, self.entity)
        return None

    @cached_property
    def pullback_mappings(self):
        """A dict from each group of axes of :attr:`weights` that map together to its mapping."""
        # pullback takes the Kronecker product of the pullbacks of the groups
        # in the order of this dict.
        sd = self.cell.get_spatial_dimension()
        W = self.weights
        axis_mappings = self.axis_mappings
        # Interior moments map every axis as affine.
        if self.dim == sd and isinstance(self.node, IntegralMoment):
            return {(k,): "affine" for k in range(len(axis_mappings))}
        # In a frame, each axis keeps its own mapping.
        if self.frame is not None:
            return {(k,): mapping for k, mapping in enumerate(axis_mappings)}

        # A contravariant axis contracted with a covariant axis through the
        # identity is a trace, and the pair maps as L2 piola.
        delta = numpy.eye(sd).ravel()
        trace = numpy.outer(delta, delta) / sd
        groups = {}
        for k, vector in enumerate(axis_mappings):
            for l, covector in enumerate(axis_mappings):
                if (vector == "contravariant piola" and covector == "covariant piola"
                        and not any(k in axes or l in axes for axes in groups)
                        and in_range_of(W, trace, (k, l), self.tol)):
                    groups[k, l] = "L2 piola"
        # A derivative along the edge maps as affine.
        # Every other axis keeps its own mapping.
        on_edge = 0 < self.dim < sd - 1
        for k, mapping in enumerate(axis_mappings):
            if any(k in axes for axes in groups):
                continue
            if on_edge and mapping == "covariant piola":
                Q = tangential_projector(self.cell.compute_tangents(self.dim, self.entity))
                if not in_range_of(W, Q, (k,), self.tol):
                    raise NotImplementedError(f"Cannot map derivatives normal to an edge of {type(self.node).__name__}.")
                mapping = "affine"
            groups[k,] = mapping
        return groups

    @cached_property
    def scaling_exponent(self):
        """The exponent :math:`p` such that the physical node dilates as :math:`h^{-p}`.

        Each group in :attr:`pullback_mappings` contributes its scaling.  If
        :attr:`avg` is False, an integral moment with a :attr:`frame` is a
        plain integral over its entity, which contributes :math:`-dim`.
        """
        sd = self.cell.get_spatial_dimension()
        if self.frame is None:
            exponents = {"covariant piola": 1, "contravariant piola": sd - 1, "L2 piola": sd}
            return sum(exponents.get(mapping, 0) for mapping in self.pullback_mappings.values())
        exponent = 0
        for axes, mapping in self.pullback_mappings.items():
            if mapping == "covariant piola":
                exponent += 1
            elif mapping == "contravariant piola" and not in_range_of(
                    self.weights, facet_normal_projector(self.cell, self.entity), axes, self.tol):
                # Tangential components of a contravariant axis
                exponent += 2 - sd
        if isinstance(self.node, IntegralMoment) and not self.avg:
            exponent -= self.dim
        return exponent

    @cached_property
    def plain_integral(self):
        """Whether the node is a plain integral over the entity of its :attr:`frame` instead of an average."""
        return self.frame is not None and isinstance(self.node, IntegralMoment) and not self.avg

    @cached_property
    def pullback_key(self):
        """The :attr:`frame`, :attr:`plain_integral`, and the :attr:`pullback_mappings`."""
        return (self.frame, self.plain_integral, tuple(self.pullback_mappings.items()))

    def pullback(self, J, adjJ, detJ, metric, adj_metric):
        """The Kronecker product of the :class:`Pullback` of each group of :attr:`pullback_mappings`.

        If the node is a :attr:`plain_integral`, the first group also carries
        the physical measure of the entity of the :attr:`frame`.

        :arg J: The Jacobian as an object array of GEM scalars.
        :arg adjJ: The adjugate of the Jacobian.
        :arg detJ: The determinant of the Jacobian.
        :arg metric: The metric :math:`J^TJ`.
        :arg adj_metric: The adjugate of the metric.
        :returns: A :class:`Pullback`.
        """
        if self.frame is None:
            pullbacks = cartesian_pullbacks(J, adjJ, detJ)
        else:
            pullbacks = frame_pullbacks(self.cell, self.frame, metric, adj_metric, detJ)
        mappings = list(self.pullback_mappings.values())
        factors = [pullbacks[mapping] for mapping in mappings]
        unit = LinearCombination([one], [numpy.eye(1)])
        if self.plain_integral:
            integral = frame_pullbacks(self.cell, self.frame, metric, adj_metric, detJ, avg=False)
            factors[:1] = [integral[mapping] for mapping in mappings[:1]] or [integral[None]]
        forward = reduce(LinearCombination.kron, (factor.forward for factor in factors), unit)
        inverse = reduce(LinearCombination.kron, (factor.inverse for factor in factors), unit)
        return Pullback(forward, inverse)

    def scaling(self, h):
        r"""The factor :math:`h^p` of the row of the physical Vandermonde matrix :math:`B=SB_0`.

        Here :math:`p` is :attr:`scaling_exponent`.

        :arg h: The cell size at each vertex.
        :returns: A GEM scalar.
        """
        p = self.scaling_exponent
        if not p:
            return one
        vertices = self.cell.get_topology()[self.dim][self.entity]
        havg = sum(h[v] for v in vertices) / len(vertices)
        dilation = reduce(mul, [havg] * abs(p))
        return one / dilation if p < 0 else gem.as_gem(dilation)


class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions."""

    #: Relative tolerance for the range tests on the weights of each node,
    #: and below which entries of the Riesz representation are dropped.
    tol = 1e-10

    #: If False, an integral moment with a frame is a plain
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
        V = self._vandermonde_matrix(coordinate_mapping).inverse_array()
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
        B = self._vandermonde_matrix(coordinate_mapping).array()
        indices = self._physical_indices
        B = gem.ListTensor(B[numpy.ix_(indices, indices)])
        return sparse_matvec(B, Q)

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        B = self._vandermonde_matrix(coordinate_mapping)
        return PhysicalTabulation(partial(self._physical_tabulation, B), ref_tabulation)

    def _physical_tabulation(self, B, table):
        r"""The physical tabulation :math:`B^{-T}\phi` on the exposed dofs.

        :arg B: The :class:`VandermondeMatrix`.
        :arg table: A reference tabulation with the basis functions as the first axis.
        """
        ii = gem.indices(len(table.shape)-1)
        phi = [gem.Indexed(table, (j, *ii)) for j in range(table.shape[0])]
        chi = B.transpose_solve(phi)
        exprs = [gem.ComponentTensor(chi[i], ii) for i in self._physical_indices]
        result, = gem.optimise.unroll_indexsum((gem.ListTensor(exprs),), lambda index: True)
        return canonical_bound_indices(result)

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
    def _physical_dual_set(self):
        """The :class:`PhysicallyMappedDualSet` of the FIAT element."""
        return PhysicallyMappedDualSet(self._element.get_dual_set(), self._element.mapping(), self.avg, self.tol)

    @cached_property
    def _reference_tabulation(self):
        """The :meth:`PhysicallyMappedDualSet.tabulate` of the nodal basis of the FIAT element."""
        return self._physical_dual_set.tabulate(self._element.get_nodal_basis())

    def _vandermonde_matrix(self, coordinate_mapping):
        """The :class:`VandermondeMatrix` on the physical cell."""
        return VandermondeMatrix(self._physical_dual_set, self._reference_tabulation, coordinate_mapping)


class PhysicallyMappedDualSet:
    """The nodes of a FIAT dual set, as :class:`PhysicallyMappedFunctional` objects.

    :arg dual: The FIAT dual set.
    :arg mappings: The FIAT mapping of each node.
    :arg avg: If False, an integral moment with a frame is a plain
        integral over its entity instead of an average.
    :arg tol: Relative tolerance for the range tests on the weights, and
        below which entries of the Riesz representation are dropped.
    """

    def __init__(self, dual, mappings, avg, tol):
        cell = dual.get_reference_element()
        nodes = dual.get_nodes()
        functionals = [None] * len(nodes)
        for dim, entities in dual.get_entity_ids().items():
            for entity, dofs in entities.items():
                for i in dofs:
                    functionals[i] = PhysicallyMappedFunctional(nodes[i], cell, dim, entity,
                                                                mappings[i], avg, tol)
        self.cell = cell
        self.tol = tol
        self.functionals = tuple(functionals)
        self.coeffs = dual.get_coeffs()
        self.entity_ids = dual.get_entity_ids()
        self.entity_closure_ids = dual.get_entity_closure_ids()

    @cached_property
    def groups(self):
        """A dict from each entity ``(dim, entity)`` and :attr:`~PhysicallyMappedFunctional.pullback_key` to its nodes."""
        groups = defaultdict(list)
        for i, functional in enumerate(self.functionals):
            groups[functional.dim, functional.entity, functional.pullback_key].append(i)
        return dict(groups)

    def tabulate(self, poly_set):
        r"""The tabulation of a nodal basis at the points of the nodes.

        :arg poly_set: The nodal basis :math:`\psi`, dual to the nodes.
        :returns: A dict from each derivative order k to a tuple of a dict from each
        point :math:`x_q` to its index q, and the tensor
        :math:`D_{jb\gamma q}=(\partial^\gamma\psi^0_j)_b(x_q)` with k
        derivative axes :math:`\gamma`, where :math:`\psi^0` is the nodal
        basis of the nodes before recombination.  The basis is tabulated one
        time for each order.
        """
        sd = self.cell.get_spatial_dimension()
        num_members = poly_set.get_num_members()
        tshape = poly_set.get_shape()
        points = defaultdict(set)
        for functional in self.functionals:
            order = functional.node.max_deriv_order
            node_points, _ = functional.node.get_order_tensors(order)
            points[order].update(node_points)

        derivatives = {}
        for order in points:
            pts = sorted(points[order])
            tabulation = poly_set.tabulate(pts, order)
            dshape = (sd,) * order
            D = numpy.zeros((num_members, *tshape, *dshape, len(pts)))
            for index in numpy.ndindex(dshape):
                D[..., *index, :] = tabulation[tuple(map(index.count, range(sd)))]
            if self.coeffs is not None:
                # n = C n_0, so the basis dual to n_0 is psi^0 = C^T psi.
                D = numpy.tensordot(self.coeffs, D, axes=(0, 0))
            point_index = {pt: q for q, pt in enumerate(pts)}
            derivatives[order] = (point_index, D)
        return derivatives

    def to_riesz(self, nodes, G, tabulation, columns):
        r"""The Riesz representations :math:`(GW_i):D_j` of nodes with the same :attr:`~PhysicallyMappedFunctional.pullback_mappings`.

        Here :math:`W_i` are the weights of node i and :math:`D_j` are the
        derivatives of the basis function j at its points.  The nodes with
        the same derivative order and points contract in a single product,
        as in :meth:`FIAT.dual_set.DualSet.to_riesz`.

        :arg nodes: The nodes i.
        :arg G: The pullback of the weights, a :class:`LinearCombination` on their axes.
        :arg tabulation: The :meth:`tabulate` of the basis.
        :arg columns: The basis functions j.
        :returns: A :class:`LinearCombination` of constant matrices over
            the nodes and the basis functions.
        """
        axes = list(chain.from_iterable(self.functionals[nodes[0]].pullback_mappings))
        rank = len(axes)
        point_groups = defaultdict(list)
        for k, i in enumerate(nodes):
            node = self.functionals[i].node
            order = node.max_deriv_order
            points, _ = node.get_order_tensors(order)
            point_groups[order, points].append(k)

        R = numpy.zeros((len(G.matrices), len(nodes), len(columns)))
        for (order, points), positions in point_groups.items():
            point_index, D = tabulation[order]
            D = D[columns][..., [point_index[x] for x in points]]
            W = numpy.stack([self.functionals[nodes[k]].weights for k in positions])
            # Contract the weights with the derivatives at the points, and
            # order the axes of the groups as in the Kronecker product
            A = numpy.tensordot(W, D, axes=(-1, -1))
            A = A.transpose(*(rank + 2 + k for k in axes), *(1 + k for k in axes), 0, rank + 1)
            n = W[0].size // W.shape[-1]
            R[:, positions] = G.contract(A.reshape(n, n, len(positions), -1), self.tol).matrices
        return LinearCombination(G.coefficients, R)


#: The rows of :math:`B_0` of the nodes of an entity that share a pullback,
#: as a :class:`LinearCombination` of constant matrices, their rows of the
#: inverse of the diagonal block of :math:`B_0` on the entity, as an object
#: array of GEM scalars, and the scaling of each row of :math:`B=SB_0`.
RowGroup = namedtuple("RowGroup", ["nodes", "entries", "inverse", "scales"])


class VandermondeMatrix:
    r"""The physical Vandermonde matrix :math:`B=SB_0` on a physical cell.

    The rows of :math:`B_0` of the nodes of an entity that share a pullback
    are :math:`\sum_t g_tR_t`, with the same GEM scalars :math:`g_t` from the
    geometry, and constant matrices :math:`R_t` from the reference cell, see
    :class:`RowGroup`.  The blocks of the back-substitution in
    :meth:`transpose_solve` are expanded one time from the
    :meth:`~LinearCombination.collect_terms`, in :attr:`couplings` and in the
    inverses of the :class:`RowGroup` objects.  Their entries do not depend
    on the points of a tabulation, so the back-substitution multiplies each
    entry with a tabulated value.

    :arg dual: The :class:`PhysicallyMappedDualSet`.
    :arg tabulation: The :meth:`PhysicallyMappedDualSet.tabulate` of the
        reference basis.
    :arg coordinate_mapping: Object providing the physical geometry
        as GEM expressions.
    :raises NotImplementedError: If a row couples to a dof outside the
        closure of the entity of the node.
    """

    def __init__(self, dual, tabulation, coordinate_mapping):
        sd = dual.cell.get_spatial_dimension()
        bary, = dual.cell.make_points(sd, 0, sd + 1)
        if sd == 1:
            # The signed determinant carries the orientation of the tangent.
            J = gem.ListTensor([[coordinate_mapping.detJ_at(bary)]])
        else:
            J = coordinate_mapping.jacobian_at(bary)
        J = numpy.array([[J[i, k] for k in range(J.shape[1])] for i in range(J.shape[0])], dtype=object)
        adjJ, detJ = adjugate_and_determinant(J)
        metric, adj_metric = metric_and_adjugate(J)
        h = coordinate_mapping.cell_size()

        columns = list(range(len(dual.functionals)))
        pullbacks = {}
        groups = defaultdict(list)
        for (dim, entity, key), nodes in dual.groups.items():
            # Assemble the rows by applying the pulled-back nodes to the reference basis
            if key not in pullbacks:
                pullbacks[key] = dual.functionals[nodes[0]].pullback(J, adjJ, detJ, metric, adj_metric)
            pullback = pullbacks[key]
            entries = dual.to_riesz(nodes, pullback.forward, tabulation, columns)

            # The rows may only couple to the dofs in the closure of their entity
            closure = dual.entity_closure_ids[dim][entity]
            outside = [j for j in numpy.flatnonzero(entries.matrices.any(axis=(0, 1))) if j not in closure]
            if outside:
                raise NotImplementedError(f"The pullback of dofs {nodes} on entity {(dim, entity)} "
                                          f"needs dofs {outside} outside its closure, so it is not local to the entity.")

            block = dual.entity_ids[dim][entity]
            inverse = dual.to_riesz(nodes, pullback.inverse, tabulation, block).collect_terms().expand()
            scales = [dual.functionals[i].scaling(h) for i in nodes]
            groups[dim, entity].append(RowGroup(nodes, entries, inverse, scales))

        # The inverse of the diagonal block of B on each entity, with
        # B_ee^{-1} = B0_ee^{-1} S_e^{-1}
        scale = {i: s for row_groups in groups.values() for group in row_groups
                 for i, s in zip(group.nodes, group.scales)}
        inverses = {}
        for (dim, entity), row_groups in groups.items():
            block = dual.entity_ids[dim][entity]
            inverse = numpy.full((len(block), len(block)), zero, dtype=object)
            for group in row_groups:
                inverse[[block.index(i) for i in group.nodes]] = group.inverse
            reciprocals = [one if scale[j] == one else gem.Division(one, scale[j]) for j in block]
            inverses[dim, entity] = inverse * numpy.array(reciprocals, dtype=object)

        # The negated rows of B outside each diagonal block, on the entity
        # blocks of the support of the rows.  Since chi_f = rhs_f B_ff^{-1},
        # a coupling multiplies rhs_f by B_ff^{-1} B_fe instead of chi_f by
        # B_fe if that has no more nonzeros.
        owners = {j: (dim, entity) for dim, entities in dual.entity_ids.items()
                  for entity, dofs in entities.items() for j in dofs}
        couplings = defaultdict(list)
        for (dim, entity), row_groups in groups.items():
            block = dual.entity_ids[dim][entity]
            for group in row_groups:
                support = {owners[j] for j in numpy.flatnonzero(group.entries.matrices.any(axis=(0, 1)))}
                support.discard((dim, entity))
                for other_dim, other_entity in sorted(support):
                    R = group.entries.matrices[:, :, dual.entity_ids[other_dim][other_entity]]
                    coupling = LinearCombination(group.entries.coefficients, -R).collect_terms().expand()
                    coupling = coupling * numpy.array(group.scales, dtype=object)[:, None]
                    composed = sparse_matmul(inverses[dim, entity][:, [block.index(i) for i in group.nodes]], coupling)
                    if count_nonzeros(composed) <= count_nonzeros(coupling):
                        couplings[other_dim, other_entity].append((block, composed, True))
                    else:
                        couplings[other_dim, other_entity].append((group.nodes, coupling, False))
        self.dual = dual
        #: A dict from each entity to its :class:`RowGroup` objects.
        self.groups = dict(groups)
        #: A dict from each entity to the inverse of the diagonal block of
        #: :math:`B` on its dofs, as an object array of GEM scalars.
        self.inverses = inverses
        #: A dict from each entity e to the triples of dofs of another entity
        #: f, a matrix of GEM scalars from those dofs to the dofs of e, and
        #: whether it acts on the right-hand side :math:`\phi_f - \sum
        #: B_{gf}^T\chi_g` of f, as :math:`-B_{ff}^{-1}B_{fe}`, or else on
        #: :math:`\chi_f`, as :math:`-B_{fe}`.
        self.couplings = dict(couplings)

    def array(self):
        """The physical Vandermonde matrix :math:`B` over all reference dofs, as an object array of GEM scalars."""
        n = len(self.dual.functionals)
        B = numpy.full((n, n), zero, dtype=object)
        for groups in self.groups.values():
            for group in groups:
                B[group.nodes] = group.entries.expand() * numpy.array(group.scales, dtype=object)[:, None]
        C = self.dual.coeffs
        if C is None:
            return B
        return C @ B @ numpy.linalg.inv(C)

    def inverse_array(self):
        r""":math:`B^{-1}` over all reference dofs, retaining completion functions.

        The row j is :math:`B^{-T}e_j` from :meth:`transpose_solve`.
        """
        n = len(self.dual.functionals)
        return numpy.array([self.transpose_solve([one if i == j else zero for i in range(n)])
                            for j in range(n)], dtype=object)

    def transpose_solve(self, phi):
        r"""The GEM scalars :math:`\chi=B^{-T}\phi`, without forming :math:`B^{-1}`.

        Back-substitute :math:`B^T\chi=\phi` over the entity blocks in
        decreasing dimension,
        :math:`\chi_e = B_{ee}^{-T}(\phi_e - \sum_f B_{fe}^T\chi_f)` over the
        entities f whose closure contains e, using only the nonzero entries
        of :math:`B` and of the inverses of its diagonal blocks.  With
        :math:`B=SB_0`, these blocks carry the scaling :math:`S`, so it
        multiplies entries once per cell instead of :math:`\chi` at each
        point.

        :arg phi: The GEM scalars :math:`\phi`, one for each reference dof.
        :returns: A list of GEM scalars, one for each reference dof.
        """
        C = self.dual.coeffs
        if C is not None:
            # For n = C n_0, B^{-T} = C^{-T} B'^{-T} C^T, with B' on the nodes n_0
            phi = sparse_vecmat(phi, literal_array(C))

        entity_ids = self.dual.entity_ids
        chi = [None] * len(phi)
        rhs = [None] * len(phi)
        for dim in sorted(entity_ids, reverse=True):
            for entity, dofs in entity_ids[dim].items():
                if not dofs:
                    continue
                rhs_block = [phi[j] for j in dofs]
                for nodes, coupling, on_rhs in self.couplings.get((dim, entity), []):
                    y = sparse_vecmat([(rhs if on_rhs else chi)[i] for i in nodes], coupling)
                    rhs_block = [gem.Sum(a, b) for a, b in zip(rhs_block, y)]

                chi_block = sparse_vecmat(rhs_block, self.inverses[dim, entity])
                for j, r, value in zip(dofs, rhs_block, chi_block):
                    rhs[j] = r
                    chi[j] = value

        if C is not None:
            chi = sparse_vecmat(chi, literal_array(numpy.linalg.inv(C)))
        return chi


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


def sparse_matvec(M, table):
    """The product of a gem.ListTensor M, with O(1) nonzeros per row, and a table."""
    ii = gem.indices(len(table.shape)-1)
    phi = [gem.Indexed(table, (j, *ii)) for j in range(M.shape[1])]
    # the sum approach is faster than calling numpy.dot or gem.IndexSum
    exprs = [gem.ComponentTensor(gem.Sum(*(M.array[i, j] * phi[j] for j in range(M.shape[1])
                                           if not isinstance(M.array[i, j], gem.Zero))), ii)
             for i in range(M.shape[0])]
    result, = gem.optimise.unroll_indexsum((gem.ListTensor(exprs),), lambda index: True)
    return result


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
        return (adjugate(J), determinant(J))
    JTJ = J.T @ J
    detJ = sqrt(determinant(JTJ))
    return (adjugate(JTJ) @ J.T / detJ, detJ)


def metric_and_adjugate(J):
    r"""Return the metric :math:`G=J^TJ` and its adjugate.

    :arg J: Full-column-rank Jacobian as an object array of GEM scalars.
    :returns: The symmetric matrices :math:`G` and :math:`\operatorname{adj}(G)`.
    """
    G = symmetric(J.T @ J)
    return (G, symmetric(adjugate(G)))


def sqrt(x):
    """The square root of a GEM scalar, which compiles to a call of sqrt instead of pow."""
    x = gem.as_gem(x)
    if isinstance(x, gem.gem.Constant):
        return gem.Literal(numpy.sqrt(x.value))
    return gem.MathFunction("sqrt", x)


def symmetric(A):
    """The symmetric matrix with the upper triangle of A."""
    A = A.copy()
    i, j = numpy.tril_indices(len(A), -1)
    A[i, j] = A[j, i]
    return A


def literal_array(A):
    """The constant array A as an object array of GEM literals, with GEM zeros for its zero entries."""
    A_gem = numpy.full(A.shape, zero, dtype=object)
    nonzero = A != 0
    A_gem[nonzero] = [gem.Literal(a) for a in A[nonzero]]
    return A_gem


def sparse_matmul(A, B):
    """The product of matrices of GEM scalars, which skips the zero entries."""
    columns = [[j for j, b in enumerate(row) if not isinstance(b, gem.Zero)] for row in B]
    C = numpy.full((A.shape[0], B.shape[1]), zero, dtype=object)
    for i, k in numpy.ndindex(A.shape):
        if not isinstance(A[i, k], gem.Zero):
            for j in columns[k]:
                C[i, j] = gem.Sum(C[i, j], gem.Product(A[i, k], B[k, j]))
    return C


def count_nonzeros(A):
    """The number of entries of an object array of GEM scalars that are not GEM zeros."""
    return sum(not isinstance(a, gem.Zero) for a in A.flat)


def sparse_vecmat(x, A):
    """The product :math:`x^TA` of a vector and a matrix of GEM scalars, which skips the zero entries.

    :returns: A list of GEM scalars, one for each column of A.
    """
    y = [zero] * A.shape[1]
    for xi, row in zip(x, A):
        if not isinstance(xi, gem.Zero):
            for j, entry in enumerate(row):
                if not isinstance(entry, gem.Zero):
                    y[j] = gem.Sum(y[j], gem.Product(entry, xi))
    return y


class LinearCombination:
    r"""A matrix :math:`G=\sum_t g_t C_t` of GEM scalars :math:`g_t` and constant matrices :math:`C_t`.

    :arg coefficients: The coefficients :math:`g_t`.
    :arg matrices: The matrices :math:`C_t`.
    """

    def __init__(self, coefficients, matrices):
        self.coefficients = numpy.asarray(coefficients, dtype=object)
        self.matrices = numpy.asarray(matrices, dtype=float)

    @classmethod
    def from_matrix(cls, A):
        """The linear combination of the entries of a matrix of GEM scalars."""
        n = len(A)
        units = numpy.eye(n * n).reshape(n, n, n, n)
        entries = [(A[a, b], units[a, b]) for a, b in numpy.ndindex(A.shape) if not isinstance(A[a, b], gem.Zero)]
        return cls(*zip(*entries))

    @classmethod
    def outer(cls, u, v):
        """The outer product of a vector u of GEM scalars and a constant vector v."""
        return cls(u, [numpy.outer(e, v) for e in numpy.eye(len(u))])

    @property
    def T(self):
        """The transpose."""
        return LinearCombination(self.coefficients, self.matrices.transpose(0, 2, 1))

    def __add__(self, other):
        return LinearCombination([*self.coefficients, *other.coefficients], [*self.matrices, *other.matrices])

    def scaled(self, c):
        """The product with a GEM scalar."""
        return LinearCombination(self.coefficients * c, self.matrices)

    def kron(self, other):
        """The Kronecker product."""
        return LinearCombination(numpy.multiply.outer(self.coefficients, other.coefficients).ravel(),
                                 [numpy.kron(C, D) for C, D in product(self.matrices, other.matrices)])

    def contract(self, A, tol):
        r"""The contraction :math:`G:A=\sum_t g_tR_t` with a constant tensor, where :math:`R_t=C_t:A`.

        :arg A: A constant tensor whose first two axes match :math:`G`,
            followed by an axis of nodes and an axis of basis functions.
        :arg tol: Tolerance relative to each node of A and each :math:`C_t`
            below which entries of :math:`R` are set to zero.
        :returns: The :class:`LinearCombination` of the :math:`R_t`, with
            the axes of nodes and basis functions.
        """
        R = numpy.tensordot(self.matrices, A, axes=2)
        cutoffs = tol * numpy.abs(self.matrices).max(axis=(1, 2))[:, None, None] * numpy.abs(A).max(axis=(0, 1, 3))[:, None]
        R = round_to_int(R, cutoffs)
        R[numpy.abs(R) <= cutoffs] = 0
        return LinearCombination(self.coefficients, R)

    def expand(self):
        r"""The GEM scalars :math:`\sum_t g_tC_t`, as an object array of the shape of :math:`C_t`."""
        C = literal_array(self.matrices.reshape(len(self.matrices), -1))
        return sparse_matmul(self.coefficients[None, :], C).reshape(self.matrices.shape[1:])

    def collect_terms(self):
        r"""Collect the terms with proportional :math:`C_t` into one term, and drop the terms with :math:`C_t=0`.

        The terms :math:`g_sC_s+g_tC_t` with :math:`C_t=\lambda C_s` collect
        into :math:`(g_s+\lambda g_t)C_s`, as in
        :func:`gem.refactorise.collect_monomials`.  Each collected
        :math:`C_s` is scaled to have its first nonzero entry equal to 1.
        """
        terms = {}
        for g, C in zip(self.coefficients, self.matrices):
            nonzero = numpy.flatnonzero(C)
            if len(nonzero) == 0:
                continue
            pivot = C.flat[nonzero[0]]
            C = C / pivot
            key = tuple(numpy.round(C.ravel(), 12))
            if key in terms:
                terms[key] = (gem.Sum(terms[key][0], gem.Literal(pivot) * g), C)
            else:
                terms[key] = (gem.Literal(pivot) * g, C)
        coefficients = [g for g, _ in terms.values()]
        matrices = [C for _, C in terms.values()]
        return LinearCombination(coefficients, numpy.reshape(matrices, (len(matrices), *self.matrices.shape[1:])))


def round_to_int(x, tol):
    """Round the entries of x that lie within tol of an integer."""
    rounded = numpy.round(x)
    return numpy.where(numpy.abs(x - rounded) <= tol, rounded, x)


def cartesian_pullbacks(J, adjJ, detJ):
    """The :class:`Pullback` of each mapping in Cartesian components."""
    sd = J.shape[1]
    affine = LinearCombination([one], [numpy.eye(sd)])
    contravariant = LinearCombination.from_matrix(J.T / detJ)
    contravariant_inverse = LinearCombination.from_matrix(adjJ.T)
    covariant = LinearCombination.from_matrix(adjJ / detJ)
    covariant_inverse = LinearCombination.from_matrix(J)
    # The identity on a contravariant axis and a covariant axis pulls back
    # as (J^T/detJ) K^T = I/detJ, so the pair scales its weights by 1/detJ.
    pairing = LinearCombination([one / detJ], [numpy.eye(sd * sd)])
    pairing_inverse = LinearCombination([detJ], [numpy.eye(sd * sd)])
    return {"affine": Pullback(affine, affine),
            "contravariant piola": Pullback(contravariant, contravariant_inverse),
            "covariant piola": Pullback(covariant, covariant_inverse),
            "L2 piola": Pullback(pairing, pairing_inverse)}


def frame_pullbacks(cell, frame, metric, adj_metric, detJ, avg=True):
    """The :class:`Pullback` of each mapping in the frame of an entity.

    :arg cell: The reference cell.
    :arg frame: The entity ``(dim, entity)``, a facet or an edge of a tetrahedron.
    :arg metric: The metric :math:`J^TJ`.
    :arg adj_metric: The adjugate of the metric.
    :arg detJ: The determinant of the Jacobian.
    :arg avg: If False, the pullbacks are of plain integrals over the
        entity instead of averages, so they carry its physical measure, and
        the key None maps to the measure on no axes.
    """
    dim, entity = frame
    sd = cell.get_spatial_dimension()
    if dim == sd - 1:
        pullbacks, measure = facet_pullbacks(cell, entity, metric, adj_metric, detJ, avg)
    else:
        pullbacks, measure = edge_pullbacks(cell, entity, metric, avg)
    affine = Pullback(LinearCombination([measure], [numpy.eye(sd)]), LinearCombination([one / measure], [numpy.eye(sd)]))
    pullbacks = {"affine": affine, **pullbacks}
    if not avg:
        pullbacks[None] = Pullback(LinearCombination([measure], [numpy.eye(1)]), LinearCombination([one / measure], [numpy.eye(1)]))
    return pullbacks


def facet_pullbacks(cell, facet, metric, adj_metric, detJ, avg=True):
    r"""The :class:`Pullback` of each Piola mapping in the frame of a facet.

    The frame consists of the tangents of the facet and its scaled normal
    :math:`\nu`.  The metric is :math:`G=J^TJ`.

    :returns: The dict of pullbacks, and the physical measure of the facet
        if not ``avg``, or 1 otherwise, which multiplies the pullbacks.
    """
    sd = cell.get_spatial_dimension()
    tangents = cell.compute_tangents(sd - 1, facet)
    nu = cell.compute_scaled_normal(facet)
    nn = nu @ nu
    # The physical normal is the cofactor normal x = adj(J)^T nu = J w, with
    # w = adj(G) nu / detJ and |x|^2 = nu^T adj(G) nu.
    z = adj_metric @ nu
    xx = z @ nu
    norm = sqrt(xx)
    measure = one
    if not avg:
        # The measure is |f| = |f_ref| |x| / |nu|, so the unit normal x / |x|
        # times the measure has no square root.
        volume = cell.volume_of_subcomplex(sd - 1, facet)
        measure = volume * norm / numpy.sqrt(nn)
        norm = gem.Literal(numpy.sqrt(nn) / volume)
    covariant = covariant_frame_pullback(metric, tangents, [nu], [z / detJ], [norm], measure)

    # Contravariant weights keep their normal component, and their
    # tangential components map through the dual tangents of the facet.
    zt = [z @ tau for tau in tangents]
    dual_tangents = numpy.linalg.solve(tangents @ tangents.T, tangents)
    matrices = [facet_normal_projector(cell, facet), tangential_projector(tangents),
                *(numpy.outer(nu, s) for s in dual_tangents)]
    contravariant = LinearCombination([one, xx / (detJ * nn), *(-c / (detJ * nn) for c in zt)], matrices)
    contravariant_inverse = LinearCombination([one, detJ * nn / xx, *(c / xx for c in zt)], matrices)
    contravariant = Pullback(contravariant.scaled(measure), contravariant_inverse.scaled(one / measure))
    return {"covariant piola": covariant, "contravariant piola": contravariant}, measure


def edge_pullbacks(cell, edge, metric, avg=True):
    r"""The :class:`Pullback` of the covariant mapping in the frame of an edge of a tetrahedron.

    The frame consists of the edge tangent :math:`t` and the normals
    :math:`t \times \nu_i` to :math:`t` within the two facets that contain
    the edge, with scaled normals :math:`\nu_i`.  The metric is :math:`G=J^TJ`.

    :returns: The dict of pullbacks, and the physical measure of the edge
        if not ``avg``, or 1 otherwise, which multiplies the pullbacks.
    """
    sd = cell.get_spatial_dimension()
    t = cell.compute_edge_tangent(edge)
    nus = [cell.compute_scaled_normal(f) for f in cell.get_connectivity()[(1, sd - 1)][edge]]
    # The physical normal to the edge within facet i is J w_i, where w_i lies
    # in the plane of the facet and J w_i is orthogonal to J t.
    Gt = metric @ t
    w = [numpy.cross(Gt, nu) for nu in nus]
    measure = one
    if not avg:
        measure = cell.volume_of_subcomplex(1, edge) * sqrt(t @ Gt) / numpy.linalg.norm(t)
    norms = [sqrt(wi @ metric @ wi) / measure for wi in w]
    covariant = covariant_frame_pullback(metric, t[None], [numpy.cross(t, nu) for nu in nus], w, norms, measure)
    return {"covariant piola": covariant}, measure


def covariant_frame_pullback(metric, tangents, normals, w, norms, measure=one):
    r"""The :class:`Pullback` of the covariant mapping in the frame of an entity.

    The frame :math:`F` consists of the tangents of the entity and its
    normals :math:`n_i`.  Its pullback :math:`A` from the physical cell
    consists of the same tangents and the pullbacks :math:`w_i|n_i|/|Jw_i|`
    of the unit physical normals, scaled by :math:`|n_i|`, all times a
    measure :math:`m`.  Covariant weights map by :math:`AF^{-1}`, and back
    by :math:`FA^{-1}`.

    :arg metric: The metric :math:`G=J^TJ`.
    :arg tangents: The tangents as rows.
    :arg normals: The normals :math:`n_i`.
    :arg w: The pullbacks :math:`w_i` of the physical normals, with
        :math:`\tau^TGw_i = 0` for each tangent :math:`\tau`.
    :arg norms: The lengths :math:`|Jw_i|/m` of the physical normals over the measure.
    :arg measure: The measure :math:`m`.
    """
    normals = numpy.asarray(normals)
    lengths = numpy.linalg.norm(normals, axis=1)
    # The columns of A for the normals
    A = [wi * (length / norm) for wi, norm, length in zip(w, norms, lengths)]
    # The dual basis of the frame, as rows
    dual = numpy.linalg.inv(numpy.concatenate([tangents, normals]).T)
    tangents_dual, normals_dual = dual[:len(tangents)], dual[len(tangents):]
    forward = reduce(add, (LinearCombination.outer(a, n_dual) for a, n_dual in zip(A, normals_dual)),
                     LinearCombination([measure], [tangents.T @ tangents_dual]))

    # The rows of A^{-1} for the tangents are (T G T^T)^{-1} T G, which vanish
    # on the w_i.  The rows for the normals vanish on the tangents, so they
    # combine the dual normals through the inverse of N_ij = n_i^* . A_j.
    GT = metric @ tangents.T
    g = symmetric(GT.T @ tangents.T)
    components = adjugate(g) @ GT.T / (determinant(g) * measure)
    N = numpy.array([[n_dual @ a for a in A] for n_dual in normals_dual])
    inverse = reduce(add, (LinearCombination.outer(row, tau).T for tau, row in zip(tangents, components)),
                     LinearCombination((adjugate(N) / determinant(N)).ravel(),
                                       [numpy.outer(n, n_dual) for n in normals for n_dual in normals_dual]))
    return Pullback(forward, inverse)


def facet_normal_projector(cell, facet):
    """The projector onto the normal of a facet."""
    nu = cell.compute_scaled_normal(facet)
    return numpy.outer(nu, nu) / (nu @ nu)


def tangential_projector(tangents):
    r"""The projector :math:`t^T\operatorname{adj}(g)\,t/\det g` onto the span of the tangents t.

    :arg tangents: The tangents as rows, with the Gram matrix :math:`g=tt^T`.
    """
    g = tangents @ tangents.T
    return tangents.T @ adjugate(g) @ tangents / determinant(g)


def in_range_of(weights, projector, axes, tol):
    """Whether the weights lie in the range of the projector acting on the given axes."""
    shape = tuple(weights.shape[k] for k in axes)
    complement = numpy.eye(projector.shape[0]) - projector
    residual = numpy.tensordot(complement.reshape(shape + shape), weights, axes=(range(len(axes), 2 * len(axes)), axes))
    return numpy.abs(residual).max() <= tol * numpy.abs(weights).max()
