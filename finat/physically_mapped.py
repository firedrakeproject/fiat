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
    :arg ref_el: The reference cell.
    :arg dim: The dimension of the entity that supports the node.
    :arg entity: The entity that supports the node.
    :arg mapping: The FIAT mapping of the node.
    :arg avg: If False, an integral moment with a :attr:`frame` is a plain
        integral over its entity instead of an average.
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
        ref_el = self.ref_el
        sd = ref_el.get_spatial_dimension()
        node = self.node
        moment = isinstance(node, IntegralMoment)
        if self.dim == sd - 1 and (moment or node.max_deriv_order > 0 or len(node.target_shape) > 1):
            return (self.dim, self.entity)
        if moment and 0 < self.dim < sd - 1:
            covectors = [k for k, mapping in enumerate(self.axis_mappings) if mapping == "covariant piola"]
            t = ref_el.compute_edge_tangent(self.entity)
            if not all(in_range_of(self.weights, numpy.outer(t, t) / (t @ t), (k,), self.tol) for k in covectors):
                return (self.dim, self.entity)
        return None

    @cached_property
    def pullback_mappings(self):
        """A dict from each group of axes of :attr:`weights` that map together to its mapping."""
        # pullback takes the Kronecker product of the weight transformations in
        # the order of this dict.
        sd = self.ref_el.get_spatial_dimension()
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
        :attr:`avg` is False, an integral moment with a :attr:`frame` is a
        plain integral over its entity, which contributes :math:`-dim`.
        """
        sd = self.ref_el.get_spatial_dimension()
        if self.frame is None:
            weights = {"covariant piola": 1, "contravariant piola": sd - 1, "L2 piola": sd}
            return sum(weights.get(mapping, 0) for mapping in self.pullback_mappings.values())
        exponent = 0
        for axes, mapping in self.pullback_mappings.items():
            if mapping == "covariant piola":
                exponent += 1
            elif mapping == "contravariant piola" and not in_range_of(
                    self.weights, facet_normal_projector(self.ref_el, self.entity), axes, self.tol):
                # Tangential components of a contravariant axis
                exponent += 2 - sd
        if isinstance(self.node, IntegralMoment) and not self.avg:
            exponent -= self.dim
        return exponent

    @cached_property
    def pullback_key(self):
        """The :attr:`frame` and the :attr:`pullback_mappings`."""
        return (self.frame, tuple(self.pullback_mappings.values()))

    def pullback(self, J, adjJ, detJ):
        """The Kronecker product of the weight transformations of :attr:`pullback_mappings`.

        :arg J: The Jacobian as an object array of GEM scalars.
        :arg adjJ: The adjugate of the Jacobian.
        :arg detJ: The determinant of the Jacobian.
        :returns: A :class:`Pullback`.
        """
        if self.frame is None:
            transformations = cartesian_transformations(J, adjJ, detJ)
        else:
            transformations = frame_transformations(self.ref_el, self.frame, J, adjJ, detJ)
        pullbacks = [transformations[mapping] for mapping in self.pullback_mappings.values()]
        unit = LinearCombination([one], [numpy.eye(1)])
        forward = reduce(LinearCombination.kron, (pullback.forward for pullback in pullbacks), unit)
        inverse = reduce(LinearCombination.kron, (pullback.inverse for pullback in pullbacks), unit)
        return Pullback(forward, inverse)

    def vandermonde_row(self, G, D):
        r"""The row :math:`(GW):D_j` of the Vandermonde matrix, where :math:`W` are the :attr:`weights`.

        For the :meth:`pullback` G and the reference basis, this is the row of
        :math:`B_0`, and for its inverse and the basis of the entity of the
        node, the row of the inverse of the diagonal block.

        :arg G: The pullback of the weights, a :class:`LinearCombination` on their axes.
        :arg D: The derivatives of the basis at the points of the node, with
            the axes of the weights followed by the basis.
        :returns: An object array of GEM scalars, one for each basis function.
        """
        # Contract the weights with the derivatives at the points, and order
        # the axes of the groups as in the Kronecker product
        axes = list(chain.from_iterable(self.pullback_mappings))
        rank = len(axes)
        A = numpy.tensordot(self.weights, D, axes=(-1, -2))
        A = A.transpose(*(rank + k for k in axes), *axes, -1)
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
            T = self.ref_el.compute_tangents(self.dim, self.entity)
            JT = J @ T.T
            volume = self.ref_el.volume_of_subcomplex(self.dim, self.entity)
            measure = volume * (determinant(JT.T @ JT) / determinant(T @ T.T))**0.5

        dilation = one
        p = self.scaling_exponent
        if p:
            vertices = self.ref_el.get_topology()[self.dim][self.entity]
            havg = sum(h[v] for v in vertices) / len(vertices)
            dilation = havg**p
        return measure * dilation


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

        return sparse_matvec(B, Q)

    def map_tabulation(self, ref_tabulation, coordinate_mapping):
        assert coordinate_mapping is not None
        rows = self._vandermonde_rows(coordinate_mapping)
        return MappedTabulation(partial(self._inverse_matvec, rows), ref_tabulation)

    def _inverse_matvec(self, rows, table):
        r"""The physical tabulation :math:`B^{-T}\phi` on the exposed dofs, without forming :math:`B^{-1}`.

        With :math:`B=SB_0`, back-substitute :math:`B_0^T\chi=\phi` over the
        entity blocks in decreasing dimension,
        :math:`\chi_e = B_{ee}^{-T}(\phi_e - \sum_f B_{fe}^T\chi_f)` over the
        entities f whose closure contains e, using only the nonzero entries
        of :math:`B_0` and of the inverses of its diagonal blocks.  Then the
        physical basis is :math:`S^{-1}\chi`.

        :arg rows: The :meth:`_vandermonde_rows`.
        :arg table: A reference tabulation with the basis functions as the first axis.
        :returns: The physical tabulation of the exposed dofs.
        """
        entity_dofs = self._element.entity_dofs()
        ii = gem.indices(len(table.shape)-1)
        phi = [gem.Indexed(table, (j, *ii)) for j in range(table.shape[0])]

        # The entries of B_0 outside the diagonal blocks, by column
        block = {i: dofs for entities in entity_dofs.values() for dofs in entities.values() for i in dofs}
        couplings = defaultdict(list)
        for i, row in rows.items():
            for j, entry in enumerate(row.entries):
                if not isinstance(entry, gem.Zero) and j not in block[i]:
                    couplings[j].append((i, entry))

        chi = [None] * len(phi)
        for dim in sorted(entity_dofs, reverse=True):
            for dofs in entity_dofs[dim].values():
                rhs = [gem.Sum(phi[j], *(-(entry * chi[i]) for i, entry in couplings[j])) for j in dofs]
                inverse = identity(len(dofs))
                for k, i in enumerate(dofs):
                    if i in rows:
                        inverse[k] = rows[i].inverse
                for k, j in enumerate(dofs):
                    chi[j] = gem.Sum(*(inverse[m, k] * rhs[m] for m in range(len(dofs))
                                       if not isinstance(inverse[m, k], gem.Zero)))

        exprs = [gem.ComponentTensor(chi[i] / rows[i].scale if i in rows else chi[i], ii)
                 for i in self._physical_indices]
        result, = gem.optimise.unroll_indexsum((gem.ListTensor(exprs),), lambda index: True)
        return result

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
        h = coordinate_mapping.cell_size()

        closures = self._element.entity_closure_dofs()
        entity_dofs = self._element.entity_dofs()
        derivatives = self._reference_derivatives
        pullbacks = {}
        rows = {}
        for i, functional in enumerate(self._physical_functionals):
            key = functional.pullback_key
            if key not in pullbacks:
                pullbacks[key] = functional.pullback(J, adjJ, detJ)
            order = functional.node.max_deriv_order
            points, _ = functional.node.get_order_tensors(order)
            D = numpy.stack([derivatives[order][x] for x in points], axis=-2)
            block = entity_dofs[functional.dim][functional.entity]
            row = functional.vandermonde_row(pullbacks[key].forward, D)
            inverse = functional.vandermonde_row(pullbacks[key].inverse, D[..., block])
            scale = functional.scaling(J, h)
            nonzeros = [j for j, entry in enumerate(row) if not isinstance(entry, gem.Zero)]
            if nonzeros == [i] and row[i] == one and scale == one:
                continue
            closure = closures[functional.dim][functional.entity]
            outside = [j for j in nonzeros if j not in closure]
            if outside:
                raise NotImplementedError(f"Dof {i} on entity {(functional.dim, functional.entity)} "
                                          f"couples outside its closure: {outside}")
            rows[i] = VandermondeRow(row, inverse, scale)
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
        nbf = self._element.space_dimension()
        closures = self._element.entity_closure_dofs()
        entity_dofs = self._element.entity_dofs()
        I = identity(nbf)
        V = identity(nbf)
        for dim, entities in sorted(closures.items()):
            for entity, closure_dofs in entities.items():
                block = entity_dofs[dim][entity]
                if not any(i in rows for i in block):
                    continue
                closure = [j for j in closure_dofs if j not in block]
                inverse = identity(len(block))
                coupling = numpy.full((len(block), len(closure)), zero, dtype=object)
                for k, i in enumerate(block):
                    if i in rows:
                        inverse[k] = rows[i].inverse
                        coupling[k] = rows[i].entries[closure]
                rhs = I[block] - sparse_matmul(coupling, V[closure])
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


def sparse_matmul(A, B):
    """The product of matrices of GEM scalars, which skips the zero entries."""
    columns = [[j for j, b in enumerate(row) if not isinstance(b, gem.Zero)] for row in B]
    C = numpy.full((A.shape[0], B.shape[1]), zero, dtype=object)
    for i, k in numpy.ndindex(A.shape):
        if not isinstance(A[i, k], gem.Zero):
            for j in columns[k]:
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


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


def cartesian_transformations(J, adjJ, detJ):
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


def frame_transformations(ref_el, frame, J, adjJ, detJ):
    """The :class:`Pullback` of each mapping in the frame of an entity.

    :arg frame: The entity ``(dim, entity)``, a facet or an edge of a tetrahedron.
    """
    dim, entity = frame
    if dim == ref_el.get_spatial_dimension() - 1:
        return facet_transformations(ref_el, entity, J, adjJ, detJ)
    return edge_transformations(ref_el, entity, J, adjJ, detJ)


def facet_transformations(ref_el, facet, J, adjJ, detJ):
    r"""The :class:`Pullback` of each mapping in the frame of a facet.

    The frame consists of the tangents of the facet and its scaled normal :math:`\nu`.
    """
    sd = ref_el.get_spatial_dimension()
    affine = LinearCombination([one], [numpy.eye(sd)])
    tangents = ref_el.compute_tangents(sd - 1, facet)
    nu = ref_el.compute_scaled_normal(facet)
    # The cofactor normal
    x = adjJ.T @ nu
    length = ((x @ x) * (nu @ nu))**0.5
    # Covariant weights are unchanged along the tangents, and the normal maps
    # to the pullback of the unit physical normal.  The inverse takes the
    # components along the pushed forward tangents and the unit physical normal.
    Jt = J @ tangents.T
    gJ = Jt.T @ Jt
    components = adjugate(gJ) @ Jt.T @ J / determinant(gJ)
    covariant = LinearCombination([*(adjJ @ x / (detJ * length)), one],
                                  [*(numpy.outer(e, nu) for e in numpy.eye(sd)), tangential_projector(tangents)])
    covariant_inverse = LinearCombination([detJ / length, *components.ravel()],
                                          [numpy.outer(nu, nu), *(numpy.outer(tau, e) for tau in tangents
                                                                  for e in numpy.eye(sd))])

    # Contravariant weights keep their normal component, and their
    # tangential components map through the dual tangents of the facet.
    g = tangents @ tangents.T
    dual_tangents = adjugate(g) @ tangents / determinant(g)
    P = facet_normal_projector(ref_el, facet)
    Q = tangential_projector(tangents)
    z = adjJ @ x
    contravariant = LinearCombination([one, (x @ x) / (detJ * (nu @ nu)),
                                       *(z @ -tau / (detJ * (nu @ nu)) for tau in tangents)],
                                      [P, Q, *(numpy.outer(nu, s) for s in dual_tangents)])
    contravariant_inverse = LinearCombination([one, detJ * (nu @ nu) / (x @ x), *(z @ tau / (x @ x) for tau in tangents)],
                                              [P, Q, *(numpy.outer(nu, s) for s in dual_tangents)])
    return {"affine": Pullback(affine, affine),
            "covariant piola": Pullback(covariant, covariant_inverse),
            "contravariant piola": Pullback(contravariant, contravariant_inverse)}


def edge_transformations(ref_el, edge, J, adjJ, detJ):
    r"""The :class:`Pullback` of each mapping in the frame of an edge of a tetrahedron.

    The frame consists of the edge tangent :math:`t` and the normals
    :math:`m_i = t \times \nu_i` to :math:`t` within the two facets that
    contain the edge, with scaled normals :math:`\nu_i`.
    """
    sd = ref_el.get_spatial_dimension()
    affine = LinearCombination([one], [numpy.eye(sd)])
    top = ref_el.get_topology()
    t = ref_el.compute_edge_tangent(edge)
    facets = [f for f in sorted(top[sd - 1]) if set(top[1][edge]) <= set(top[sd - 1][f])]
    m = [numpy.cross(t, ref_el.compute_scaled_normal(f)) for f in facets]
    # The dual basis of the frame, as rows
    F = numpy.array([t, *m]).T
    t_dual, *m_dual = adjugate(F) / determinant(F)
    # The physical normal to the edge within facet i is J w_i, where w_i lies
    # in the plane of the facet and J w_i is orthogonal to J t.
    Jt = J @ t
    Gt = J.T @ Jt
    w = [numpy.cross(Gt, ref_el.compute_scaled_normal(f)) for f in facets]
    Jw = [J @ wi for wi in w]
    norms = [(Jwi @ Jwi)**0.5 for Jwi in Jw]
    lengths = [(mi @ mi)**0.5 for mi in m]
    # Covariant weights are unchanged along t, and each m_i maps to the
    # pullback of the unit physical normal within facet i, scaled by |m_i|.
    covariant = LinearCombination([one], [numpy.outer(t, t_dual)])
    for wi, norm, length, mi_dual in zip(w, norms, lengths, m_dual):
        covariant += LinearCombination(wi * (length / norm), [numpy.outer(e, mi_dual) for e in numpy.eye(sd)])
    # The inverse takes the component along J t, and the components along the
    # unit physical normals N_i, which are orthogonal to J t, through the
    # inverse of their Gram matrix [[1, c], [c, 1]].
    N = [Jwi / norm for Jwi, norm in zip(Jw, norms)]
    c = N[0] @ N[1]
    K = numpy.array([[one, -c], [-c, one]], dtype=object) / (1 - c * c)
    covariant_inverse = LinearCombination(Gt / (Jt @ Jt), [numpy.outer(t, e) for e in numpy.eye(sd)])
    for i, (mi, length) in enumerate(zip(m, lengths)):
        row = sum((J.T @ N[j]) * K[i, j] for j in range(2)) / length
        covariant_inverse += LinearCombination(row, [numpy.outer(mi, e) for e in numpy.eye(sd)])
    return {"affine": Pullback(affine, affine),
            "covariant piola": Pullback(covariant, covariant_inverse)}


def facet_normal_projector(ref_el, facet):
    """The projector onto the normal of a facet."""
    nu = ref_el.compute_scaled_normal(facet)
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
