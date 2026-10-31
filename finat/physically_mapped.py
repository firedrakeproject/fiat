r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from functools import cached_property, reduce
from itertools import chain, product
from operator import add

import gem
import numpy

from FIAT.functional import IntegralMoment
from finat.citations import cite

zero = gem.Zero()
one = gem.Literal(1.0)

#: A weight transformation and its inverse.
Pullback = namedtuple("Pullback", ["forward", "inverse"])

#: The row of :math:`B_0` of a node, the row of the inverse of the diagonal
#: block of :math:`B_0`, and the scaling of the row of :math:`B=SB_0`.
VandermondeRow = namedtuple("VandermondeRow", ["entries", "inverse", "scale"])


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
    piola``, ``L2 piola`` for a contravariant axis paired with a covariant
    axis through the identity, or ``contravariant inner product`` and
    ``covariant inner product`` for an axis paired with a test function that
    maps in the same way.  These mappings act on Cartesian components if
    :attr:`frame` is None, and on components in the frame of the entity
    :attr:`frame` otherwise.

    :arg node: The FIAT node.
    :arg cell: The reference cell.
    :arg dim: The dimension of the entity that supports the node.
    :arg entity: The entity that supports the node.
    :arg mapping: The FIAT mapping of the node.
    :arg avg: If False, an integral moment with a :attr:`frame` is a plain
        integral over its entity instead of an average.
    :arg tol: Relative tolerance for the range tests on the weights, and
        below which entries of the Riesz representation are dropped.
    :arg test_mapping: The FIAT mapping of the test function of the node, or
        None if the dual set does not declare it.
    """

    def __init__(self, node, cell, dim, entity, mapping, avg, tol, test_mapping=None):
        self.node = node
        self.cell = cell
        self.dim = dim
        self.entity = entity
        self.mapping = mapping
        self.avg = avg
        self.tol = tol
        self.test_mapping = test_mapping

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
        mappings = value_mappings(self.mapping, rank)
        if len(mappings) != rank:
            raise ValueError(f"A {self.mapping} functional must have rank {len(mappings)}, not {rank}.")
        return mappings + ("covariant piola",) * self.node.max_deriv_order

    @cached_property
    def inner_product_axes(self):
        """The leading axes of the weights that pair with a test function that maps in the same way."""
        if self.test_mapping is None:
            return ()
        tests = value_mappings(self.test_mapping, len(self.node.target_shape))
        return tuple(k for k, test in enumerate(tests) if "piola" in test and test == self.axis_mappings[k])

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
        # pullback takes the Kronecker product of the weight transformations in
        # the order of this dict.
        sd = self.cell.get_spatial_dimension()
        W = self.weights
        # An axis paired with a test function that maps in the same way is an inner product.
        inner_products = {"contravariant piola": "contravariant inner product",
                          "covariant piola": "covariant inner product"}
        axis_mappings = list(self.axis_mappings)
        for k in self.inner_product_axes:
            axis_mappings[k] = inner_products[axis_mappings[k]]
        # Other interior moments map every axis as affine.
        if self.dim == sd and isinstance(self.node, IntegralMoment) and not self.inner_product_axes:
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
            exponents = {"covariant piola": 1, "contravariant piola": sd - 1, "L2 piola": sd,
                         "covariant inner product": 2, "contravariant inner product": 2 * sd - 2}
            return sum(exponents.get(mapping, 0) for mapping in self.pullback_mappings.values())
        exponent = 0
        for axes, mapping in self.pullback_mappings.items():
            if mapping == "covariant piola":
                exponent += 1
            elif mapping == "covariant inner product":
                exponent += 2
            elif mapping == "contravariant inner product":
                exponent += 2 * sd - 2
            elif mapping == "contravariant piola" and not in_range_of(
                    self.weights, facet_normal_projector(self.cell, self.entity), axes, self.tol):
                # Tangential components of a contravariant axis
                exponent += 2 - sd
        if isinstance(self.node, IntegralMoment) and not self.avg:
            exponent -= self.dim
        return exponent

    @cached_property
    def pullback_key(self):
        """The :attr:`frame` and the :attr:`pullback_mappings`."""
        return (self.frame, tuple(self.pullback_mappings.values()))

    def pullback(self, J, adjJ, detJ, metric, adj_metric):
        """The Kronecker product of the weight transformations of :attr:`pullback_mappings`.

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
        pullbacks.update(inner_product_pullbacks(metric, adj_metric, detJ))
        mappings = self.pullback_mappings.values()
        unit = LinearCombination([one], [numpy.eye(1)])
        forward = reduce(LinearCombination.kron, (pullbacks[mapping].forward for mapping in mappings), unit)
        inverse = reduce(LinearCombination.kron, (pullbacks[mapping].inverse for mapping in mappings), unit)
        return Pullback(forward, inverse)

    def to_riesz(self, G, D):
        r"""The Riesz representation :math:`(GW):D_j` of the node on a basis, where :math:`W` are the :attr:`weights`.

        For the :meth:`pullback` G and the reference basis, this is the row of
        :math:`B_0`, and for its inverse and the basis of the entity of the
        node, the row of the inverse of the diagonal block if G maps
        :math:`\ker c` into itself.

        :arg G: The pullback of the weights, a :class:`LinearCombination` on their axes.
        :arg D: The derivatives of the basis at the points of the node, with
            the basis followed by the axes of the weights.
        :returns: An object array of GEM scalars, one for each basis function.
        """
        # Contract the weights with the derivatives at the points, and order
        # the axes of the groups as in the Kronecker product
        axes = list(chain.from_iterable(self.pullback_mappings))
        rank = len(axes)
        A = numpy.tensordot(self.weights, D, axes=(-1, -1))
        A = A.transpose(*(rank + 1 + k for k in axes), *axes, rank)
        n = self.weights.size // self.weights.shape[-1]
        return G.contract(A.reshape(n, n, -1), self.tol)

    def scaling(self, J, h):
        r"""The factor :math:`h^p\,|e|` of the row of the physical Vandermonde matrix :math:`B=SB_0`.

        Here :math:`p` is :attr:`scaling_exponent`, and :math:`|e|` is the
        physical measure of the entity if :attr:`avg` is False and there is
        a :attr:`frame`, and 1 otherwise.

        :arg J: The Jacobian as an object array of GEM scalars.
        :arg h: The cell size at each vertex.
        :returns: A GEM scalar.
        """
        measure = one
        if self.frame is not None and isinstance(self.node, IntegralMoment) and not self.avg:
            # The physical measure of the support, from the Gram determinant of its tangents
            T = self.cell.compute_tangents(self.dim, self.entity)
            JT = J @ T.T
            volume = self.cell.volume_of_subcomplex(self.dim, self.entity)
            measure = volume * (determinant(JT.T @ JT) / determinant(T @ T.T))**0.5

        dilation = one
        p = self.scaling_exponent
        if p:
            vertices = self.cell.get_topology()[self.dim][self.entity]
            havg = sum(h[v] for v in vertices) / len(vertices)
            dilation = havg**p
        return measure * dilation


class EntityPullback:
    r"""The pullback of an entity's nodes and the inverse of its diagonal block.

    Nodes with the same inner product mapping use the Schur complement of
    the inverse pullback on their weights and an automatically tabulated
    completion.  The other nodes may have different mappings.  Their rows
    are inverted first, then the coupling to them is included in the
    inverse of the whole entity block.

    :arg functionals: A dict from the entity's dofs to their physical functionals.
    :arg tabulation: The reference tabulation, indexed by derivative order.
    :arg closure: The dofs on the closure of the entity.
    """

    def __init__(self, functionals, tabulation, closure):
        self.functionals = functionals
        self.dofs = list(functionals)
        self.inner = [i for i, functional in functionals.items() if functional.inner_product_axes]
        self.closure = closure
        self.tol = next(iter(functionals.values())).tol
        self.tabulation = {}
        for i, functional in functionals.items():
            order = functional.node.max_deriv_order
            points = functional.node.get_order_tensors(order)[0]
            point_index = tabulation[order][0]
            D = tabulation[order][1]
            self.tabulation[i] = D[..., [point_index[x] for x in points]]

    @cached_property
    def completion(self):
        r"""Directions in :math:`\ker c` for the inner product nodes, indexed by closure dofs."""
        if not self.inner:
            return {}
        others = [j for j in self.closure if j not in self.inner]
        if not others:
            return {}
        candidates = []
        for i in self.inner:
            functional = self.functionals[i]
            k, = functional.inner_product_axes
            sd = functional.cell.get_spatial_dimension()
            D = self.tabulation[i]
            for A in numpy.eye(sd * sd).reshape(sd * sd, sd, sd):
                E = numpy.moveaxis(numpy.tensordot(A, functional.weights, axes=(1, k)), 0, k)
                candidates.append((i, A, numpy.tensordot(D[self.closure], E, axes=E.ndim)))
        rows = numpy.array([row for _, _, row in candidates])
        rows = rows[:, [self.closure.index(j) for j in others]]
        tol = self.tol * numpy.abs(rows).max()
        basis = {}
        completion = {}
        # Elimination selects independent directions and coordinates together.
        for m, (i, A, c) in enumerate(candidates):
            row = rows[m].copy()
            for j, direction in basis.items():
                row -= row[j] * direction
            j = numpy.abs(row).argmax()
            if abs(row[j]) > tol:
                basis[j] = row / row[j]
                completion[others[j]] = (i, A, c[[self.closure.index(k) for k in self.inner]])
        return completion

    def __call__(self, pullbacks):
        r"""Tabulate the forward rows and :math:`B_{ee}^{-1}`."""
        forward = []
        inverse = numpy.full((len(self.dofs), len(self.dofs)), zero, dtype=object)
        for k, (i, functional) in enumerate(self.functionals.items()):
            pullback = pullbacks[functional.pullback_key]
            D = self.tabulation[i]
            forward.append(functional.to_riesz(pullback.forward, D))
            if i not in self.inner:
                inverse[k] = functional.to_riesz(pullback.inverse, D[self.dofs])
        forward = numpy.array(forward)
        if not self.inner:
            return Pullback(forward, inverse)

        indices = [self.dofs.index(i) for i in self.inner]
        if any(not isinstance(forward[k, j], gem.Zero)
               for k, i in enumerate(self.dofs) if i not in self.inner for j in self.inner):
            raise NotImplementedError("The other nodes of an entity couple to its inner product nodes.")
        columns = self.inner + list(self.completion)
        weights = {}
        for i in self.inner:
            functional = self.functionals[i]
            H = pullbacks[functional.pullback_key].inverse
            weights[i] = functional.to_riesz(H, self.tabulation[i][columns])
        kernel = []
        for i, A, c in self.completion.values():
            functional = self.functionals[i]
            H = pullbacks[functional.pullback_key].inverse
            factors = [A if axes == functional.inner_product_axes else numpy.eye(numpy.prod([len(A)] * len(axes)))
                       for axes in functional.pullback_mappings]
            HA = LinearCombination(H.coefficients, H.matrices @ reduce(numpy.kron, factors))
            row = functional.to_riesz(HA, self.tabulation[i][columns])
            kernel.append(row - sum(weights[j] * cj for j, cj in zip(self.inner, c)))
        weights = numpy.array([weights[i] for i in self.inner])
        n = len(self.inner)
        diagonal = weights[:, :n]
        if kernel:
            # The inverse on these nodes is H_ww - H_wk H_kk^{-1} H_kw.
            kernel = numpy.array(kernel)
            H_wk = weights[:, n:]
            H_kw = kernel[:, :n]
            H_kk = kernel[:, n:]
            diagonal -= sparse_matmul(H_wk, sparse_matmul(adjugate(H_kk), H_kw)) / determinant(H_kk)
        rhs = identity(len(self.dofs))[indices] - sparse_matmul(forward[indices][:, self.dofs], inverse)
        inverse[indices] = sparse_matmul(diagonal, rhs)
        return Pullback(forward, inverse)


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
        test_mappings = dual.get_mapping() or [None] * len(dual.nodes)
        functionals = [None] * len(dual.nodes)
        for dim, entities in dual.get_entity_ids().items():
            for entity, dofs in entities.items():
                for i in dofs:
                    functionals[i] = PhysicallyMappedFunctional(dual.nodes[i], self.cell, dim, entity,
                                                                mappings[i], self.avg, self.tol, test_mappings[i])
        return tuple(functionals)

    @cached_property
    def _entity_pullbacks(self):
        """The reference data for each entity's pullback."""
        closure_dofs = self._element.entity_closure_dofs()
        return {(dim, entity): EntityPullback({i: self._physical_functionals[i] for i in dofs},
                                              self._reference_tabulation, closure_dofs[dim][entity])
                for dim, entities in self._element.entity_dofs().items()
                for entity, dofs in entities.items() if dofs}

    @cached_property
    def _reference_tabulation(self):
        r"""The tabulation of the source nodal basis at the points of the nodes.

        A dict from each derivative order k to a tuple of a dict from each
        point :math:`x_q` to its index q, and the tensor
        :math:`D_{jb\gamma q}=(\partial^\gamma\psi^0_j)_b(x_q)` with k
        derivative axes :math:`\gamma`, where :math:`\psi^0` is the nodal
        basis of the nodes before recombination.  The basis is tabulated one
        time for each order.
        """
        sd = self.cell.get_spatial_dimension()
        num_members = self._element.space_dimension()
        tshape = self._element.value_shape()
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
            dshape = (sd,) * order
            D = numpy.zeros((num_members, *tshape, *dshape, len(pts)))
            for index in numpy.ndindex(dshape):
                D[..., *index, :] = tabulation[tuple(map(index.count, range(sd)))]
            if C is not None:
                # n = C n_0, so the basis dual to n_0 is psi^0 = C^T psi.
                D = numpy.tensordot(C, D, axes=(0, 0))
            point_index = {pt: q for q, pt in enumerate(pts)}
            derivatives[order] = (point_index, D)
        return derivatives

    def _vandermonde_rows(self, coordinate_mapping):
        r"""The rows of the physical Vandermonde matrix :math:`B=SB_0` that differ from the identity.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A dict from each node number to its :class:`VandermondeRow`.
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
        metric, adj_metric = metric_and_adjugate(J)
        h = coordinate_mapping.cell_size()

        pullbacks = {}
        for functional in self._physical_functionals:
            key = functional.pullback_key
            if key not in pullbacks:
                pullbacks[key] = functional.pullback(J, adjJ, detJ, metric, adj_metric)

        rows = {}
        for entity_pullback in self._entity_pullbacks.values():
            pullback = entity_pullback(pullbacks)
            for k, (i, functional) in enumerate(entity_pullback.functionals.items()):
                row = pullback.forward[k]
                scale = functional.scaling(J, h)
                nonzeros = [j for j, entry in enumerate(row) if not isinstance(entry, gem.Zero)]
                if nonzeros == [i] and row[i] == one and scale == one:
                    continue

                # The row may only couple to the dofs in the closure of its entity
                outside = [j for j in nonzeros if j not in entity_pullback.closure]
                if outside:
                    raise NotImplementedError(f"The pullback of dof {i} on entity {(functional.dim, functional.entity)} "
                                              f"needs dofs {outside} outside its closure, so it is not local to the entity.")
                rows[i] = VandermondeRow(row, pullback.inverse[k], scale)
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
        for i, row in self._vandermonde_rows(coordinate_mapping).items():
            B[i] = row.entries * row.scale
        return self._recombine(B)

    def _inverse_physical_vandermonde(self, coordinate_mapping):
        r""":math:`B^{-1}=B_0^{-1}S^{-1}` over all reference dofs.

        Back-substitute the entity blocks of :math:`B_0` in closure order,
        retaining completion functions.  The rows of each block e are
        :math:`B_{ee}^{-1}(I_e - B_{ec}V_c)` over the rest c of its closure.
        """
        rows = self._vandermonde_rows(coordinate_mapping)
        closure_dofs = self._element.entity_closure_dofs()
        entity_dofs = self._element.entity_dofs()
        I = identity(self._element.space_dimension())
        V = identity(self._element.space_dimension())
        for dim, entities in sorted(closure_dofs.items()):
            for entity in entities:
                block = entity_dofs[dim][entity]
                if not any(i in rows for i in block):
                    continue
                closure = [j for j in closure_dofs[dim][entity] if j not in block]
                inverse = identity(len(block))
                offdiag = numpy.full((len(block), len(closure)), zero, dtype=object)
                for k, i in enumerate(block):
                    if i in rows:
                        inverse[k] = rows[i].inverse
                        offdiag[k] = rows[i].entries[closure]
                rhs = I[block] - sparse_matmul(offdiag, V[closure])
                V[block] = sparse_matmul(inverse, rhs)
        for i, row in rows.items():
            V[:, i] = V[:, i] / row.scale
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


def value_mappings(mapping, rank):
    """The mapping of each value axis for a FIAT mapping of the given rank."""
    words = mapping.split()
    if words == ["affine"]:
        return (mapping,) * rank
    if words[0] == "double":
        return (f"{words[1]} piola",) * 2
    return tuple(f"{word} piola" for word in words[:-1])


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
    detJ = determinant(JTJ)**0.5
    return (adjugate(JTJ) @ J.T / detJ, detJ)


def metric_and_adjugate(J):
    r"""Return the metric :math:`G=J^TJ` and its adjugate, which equals :math:`\operatorname{adj}(J)\operatorname{adj}(J)^T`.

    :arg J: Full-column-rank Jacobian as an object array of GEM scalars.
    :returns: The symmetric matrices :math:`G` and :math:`\operatorname{adj}(G)`.
    """
    G = symmetric(J.T @ J)
    return G, symmetric(adjugate(G))


def symmetric(A):
    """The symmetric matrix with the upper triangle of A."""
    A = A.copy()
    i, j = numpy.tril_indices(len(A), -1)
    A[i, j] = A[j, i]
    return A


def sparse_matmul(A, B):
    """The product of matrices of GEM scalars, which skips the zero entries."""
    columns = [[j for j, b in enumerate(row) if not isinstance(b, gem.Zero)] for row in B]
    C = numpy.full((A.shape[0], B.shape[1]), zero, dtype=object)
    for i, k in numpy.ndindex(A.shape):
        if not isinstance(A[i, k], gem.Zero):
            for j in columns[k]:
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


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

    def kron(self, other):
        """The Kronecker product."""
        return LinearCombination(numpy.multiply.outer(self.coefficients, other.coefficients).ravel(),
                                 [numpy.kron(C, D) for C, D in product(self.matrices, other.matrices)])

    def contract(self, A, tol):
        r"""The contraction :math:`G:A=g^TR` with a constant tensor, where :math:`R_t=C_t:A`.

        :arg A: A constant tensor whose first two axes match :math:`G`.
        :arg tol: Tolerance relative to A and each :math:`C_t` below which
            entries of :math:`R` are dropped.
        :returns: An object array of GEM scalars with the remaining axes of A.
        """
        R = numpy.tensordot(self.matrices, A, axes=2)
        R = R.reshape(len(R), -1)
        cutoffs = tol * numpy.abs(A).max() * numpy.abs(self.matrices).max(axis=(1, 2))[:, None]
        R = round_to_int(R, cutoffs)
        nonzero = numpy.abs(R) > cutoffs
        R_gem = numpy.full(R.shape, zero, dtype=object)
        R_gem[nonzero] = [gem.Literal(r) for r in R[nonzero]]
        return sparse_matmul(self.coefficients[None, :], R_gem).reshape(A.shape[2:])


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


def inner_product_pullbacks(metric, adj_metric, detJ):
    r"""The :class:`Pullback` of each mapping of an axis paired with a test function that maps in the same way.

    For the Piola map F, the pair pulls back as :math:`F^TF`, which is
    :math:`G/\det J^2` for the contravariant map and :math:`G^{-1}` for
    the covariant map, with the metric :math:`G=J^TJ`.
    """
    contravariant = LinearCombination.from_matrix(metric / detJ**2)
    contravariant_inverse = LinearCombination.from_matrix(adj_metric)
    covariant = LinearCombination.from_matrix(adj_metric / detJ**2)
    covariant_inverse = LinearCombination.from_matrix(metric)
    return {"contravariant inner product": Pullback(contravariant, contravariant_inverse),
            "covariant inner product": Pullback(covariant, covariant_inverse)}


def frame_pullbacks(cell, frame, metric, adj_metric, detJ):
    """The :class:`Pullback` of each mapping in the frame of an entity.

    :arg frame: The entity ``(dim, entity)``, a facet or an edge of a tetrahedron.
    :arg metric: The metric :math:`J^TJ`.
    :arg adj_metric: The adjugate of the metric.
    :arg detJ: The determinant of the Jacobian.
    """
    dim, entity = frame
    sd = cell.get_spatial_dimension()
    if dim == sd - 1:
        pullbacks = facet_pullbacks(cell, entity, metric, adj_metric, detJ)
    else:
        pullbacks = edge_pullbacks(cell, entity, metric)
    affine = LinearCombination([one], [numpy.eye(sd)])
    return {"affine": Pullback(affine, affine), **pullbacks}


def facet_pullbacks(cell, facet, metric, adj_metric, detJ):
    r"""The :class:`Pullback` of each Piola mapping in the frame of a facet.

    The frame consists of the tangents of the facet and its scaled normal
    :math:`\nu`.  The metric is :math:`G=J^TJ`.
    """
    sd = cell.get_spatial_dimension()
    tangents = cell.compute_tangents(sd - 1, facet)
    nu = cell.compute_scaled_normal(facet)
    # The physical normal is the cofactor normal x = adj(J)^T nu = J w, with
    # w = adj(G) nu / detJ and |x|^2 = nu^T adj(G) nu.
    z = adj_metric @ nu
    xx = z @ nu
    covariant = covariant_frame_pullback(metric, tangents, [nu], [z / detJ], [xx**0.5])

    # Contravariant weights keep their normal component, and their
    # tangential components map through the dual tangents of the facet.
    nn = nu @ nu
    zt = [z @ tau for tau in tangents]
    dual_tangents = numpy.linalg.solve(tangents @ tangents.T, tangents)
    matrices = [facet_normal_projector(cell, facet), tangential_projector(tangents),
                *(numpy.outer(nu, s) for s in dual_tangents)]
    contravariant = LinearCombination([one, xx / (detJ * nn), *(-c / (detJ * nn) for c in zt)], matrices)
    contravariant_inverse = LinearCombination([one, detJ * nn / xx, *(c / xx for c in zt)], matrices)
    return {"covariant piola": covariant,
            "contravariant piola": Pullback(contravariant, contravariant_inverse)}


def edge_pullbacks(cell, edge, metric):
    r"""The :class:`Pullback` of the covariant mapping in the frame of an edge of a tetrahedron.

    The frame consists of the edge tangent :math:`t` and the normals
    :math:`t \times \nu_i` to :math:`t` within the two facets that contain
    the edge, with scaled normals :math:`\nu_i`.  The metric is :math:`G=J^TJ`.
    """
    sd = cell.get_spatial_dimension()
    t = cell.compute_edge_tangent(edge)
    nus = [cell.compute_scaled_normal(f) for f in cell.get_connectivity()[(1, sd - 1)][edge]]
    # The physical normal to the edge within facet i is J w_i, where w_i lies
    # in the plane of the facet and J w_i is orthogonal to J t.
    Gt = metric @ t
    w = [numpy.cross(Gt, nu) for nu in nus]
    norms = [(wi @ metric @ wi)**0.5 for wi in w]
    covariant = covariant_frame_pullback(metric, t[None], [numpy.cross(t, nu) for nu in nus], w, norms)
    return {"covariant piola": covariant}


def covariant_frame_pullback(metric, tangents, normals, w, norms):
    r"""The :class:`Pullback` of the covariant mapping in the frame of an entity.

    The frame :math:`F` consists of the tangents of the entity and its
    normals :math:`n_i`.  Its pullback :math:`A` from the physical cell
    consists of the same tangents and the pullbacks :math:`w_i|n_i|/|Jw_i|`
    of the unit physical normals, scaled by :math:`|n_i|`.  Covariant
    weights map by :math:`AF^{-1}`, and back by :math:`FA^{-1}`.

    :arg metric: The metric :math:`G=J^TJ`.
    :arg tangents: The tangents as rows.
    :arg normals: The normals :math:`n_i`.
    :arg w: The pullbacks :math:`w_i` of the physical normals, with
        :math:`\tau^TGw_i = 0` for each tangent :math:`\tau`.
    :arg norms: The lengths :math:`|Jw_i|` of the physical normals.
    """
    normals = numpy.asarray(normals)
    lengths = numpy.linalg.norm(normals, axis=1)
    # The columns of A for the normals
    A = [wi * (length / norm) for wi, norm, length in zip(w, norms, lengths)]
    # The dual basis of the frame, as rows
    dual = numpy.linalg.inv(numpy.concatenate([tangents, normals]).T)
    tangents_dual, normals_dual = dual[:len(tangents)], dual[len(tangents):]
    forward = reduce(add, (LinearCombination.outer(a, n_dual) for a, n_dual in zip(A, normals_dual)),
                     LinearCombination([one], [tangents.T @ tangents_dual]))

    # The rows of A^{-1} for the tangents are (T G T^T)^{-1} T G, which vanish
    # on the w_i.  The rows for the normals vanish on the tangents, so they
    # combine the dual normals through the inverse of N_ij = n_i^* . A_j.
    GT = metric @ tangents.T
    g = symmetric(GT.T @ tangents.T)
    components = adjugate(g) @ GT.T / determinant(g)
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
