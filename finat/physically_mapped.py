r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Mapping
from functools import cached_property, reduce, singledispatch
from itertools import chain

import gem
import numpy
from gem.node import Memoizer, reuse_if_untouched
from gem.optimise import (cancel_inverses, cancel_reciprocals, common_denominator, fold_constants,
                          remove_componenttensors, sum_factorise, traverse_product, traverse_sum,
                          unroll_indexsum)
from gem.refactorise import ATOMIC, COMPOUND, OTHER

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

    Each axis of :attr:`weights` holds components in a physical frame: the
    Cartesian frame if :attr:`support_facet` is None, and otherwise the
    frame of the support entity within :attr:`support_facet`.  The frame
    pushes the reference tangents forward by the Jacobian, and replaces
    the reference normals by the physical normals.

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
        r"""The facet that holds the frame of the weights, or None for the Cartesian frame.

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
    def interior(self):
        """Whether the node is an interior moment, whose weights stay in the reference frame."""
        return self.dim == self.ref_el.get_spatial_dimension() and isinstance(self.node, IntegralMoment)

    @cached_property
    def scaling_exponent(self):
        """The exponent :math:`p` such that the physical node dilates as :math:`h^{-p}`.

        In the Cartesian frame, a covariant axis contributes 1, and a
        contravariant axis contributes :math:`sd-1`.  In the frame of a
        :attr:`support_facet`, a covariant axis contributes 1, and a
        contravariant axis with tangential components contributes
        :math:`2-sd`.  If :attr:`avg` is False, an integral moment with a
        :attr:`support_facet` is a plain integral over its entity, which
        contributes :math:`-dim`.
        """
        sd = self.ref_el.get_spatial_dimension()
        if self.interior:
            return 0
        if self.support_facet is None:
            on_edge = 0 < self.dim < sd - 1
            weights = {"covariant piola": 0 if on_edge else 1, "contravariant piola": sd - 1}
            return sum(weights.get(mapping, 0) for mapping in self.axis_mappings)
        nu = self.ref_el.compute_scaled_normal(self.support_facet)
        exponent = 0
        for k, mapping in enumerate(self.axis_mappings):
            if mapping == "covariant piola":
                exponent += 1
            elif mapping == "contravariant piola" and not in_range_of(
                    self.weights, numpy.outer(nu, nu) / (nu @ nu), (k,), self.tol):
                exponent += 2 - sd
        if isinstance(self.node, IntegralMoment) and not self.avg:
            exponent -= self.dim
        return exponent

    @cached_property
    def tangents(self):
        r"""The reference tangents :math:`t_i` of :attr:`support_facet`, as columns.

        On an edge support, these are the edge tangent :math:`t` and
        :math:`t \times \nu`, where :math:`\nu` is the scaled normal of
        :attr:`support_facet`.
        """
        ref_el = self.ref_el
        sd = ref_el.get_spatial_dimension()
        if self.dim == sd - 2 >= 1:
            t = ref_el.compute_edge_tangent(self.entity)
            return numpy.stack([t, numpy.cross(t, ref_el.compute_scaled_normal(self.support_facet))], axis=1)
        return ref_el.compute_tangents(sd - 1, self.support_facet).T

    @cached_property
    def gram(self):
        r"""Opaque placeholders for the Gram matrix :math:`G_{ij} = Jt_i \cdot Jt_j` of the pushed forward
        :attr:`tangents`, its adjugate, its determinant, and the products :math:`J\nu \cdot Jt_i` with
        the pushed forward scaled normal."""
        r = 0 if self.support_facet is None else self.tangents.shape[1]
        name = f"{self.dim}_{self.entity}_{self.support_facet}"
        return (gem.Variable(f"G_{name}", (r, r)), gem.Variable(f"adjG_{name}", (r, r)),
                gem.Variable(f"detG_{name}", ()), gem.Variable(f"Gnu_{name}", (r,)))

    def gram_replacements(self, J):
        """The :attr:`gram` placeholders in terms of the Jacobian J."""
        if self.support_facet is None or not self.tangents.size:
            return {}
        Jt = entries(J) @ self.tangents
        G = Jt.T @ Jt
        Jnu = entries(J) @ self.ref_el.compute_scaled_normal(self.support_facet)
        return dict(zip(self.gram, (as_tensor(G), as_tensor(adjugate(G)), determinant(G), as_tensor(Jnu @ Jt))))

    def pullback(self, k, b, a, J, K, detJ):
        r"""The matrix :math:`G_{ba}` that pulls back axis k of the weights.

        A physical node with weights :math:`w` on axis k acts on the
        reference function as the weights :math:`Gw`.

        :arg k: The axis of the weights.
        :arg b: The index of the reference axis.
        :arg a: The index of the physical axis.
        :arg J: The Jacobian.
        :arg K: The inverse of the Jacobian.
        :arg detJ: The determinant of the Jacobian.
        :returns: A GEM scalar with free indices b and a.
        """
        ref_el = self.ref_el
        sd = ref_el.get_spatial_dimension()
        mapping = self.axis_mappings[k]
        c, e, l, m, p, q = gem.indices(6)
        if mapping == "affine" or self.interior:
            return gem.Delta(b, a)

        if self.support_facet is None:
            if mapping == "contravariant piola":
                return J[a, b] / detJ
            if 0 < self.dim < sd - 1:
                # A derivative along an edge pushes forward by J.
                t = ref_el.compute_edge_tangent(self.entity)
                if not in_range_of(self.weights, numpy.outer(t, t) / (t @ t), (k,), self.tol):
                    raise NotImplementedError(f"Cannot map derivatives normal to an edge of {type(self.node).__name__}.")
                return gem.IndexSum(K[b, c] * J[c, a], (c,))
            return K[b, a]

        # The reference frame of the tangents t and the scaled normal nu, with the
        # Gram matrix g of the tangents
        G, adjG, detG, Gnu = self.gram
        tangents = self.tangents
        nu = ref_el.compute_scaled_normal(self.support_facet)
        g = tangents.T @ tangents
        adjg = adjugate(g)
        detg = determinant(g)
        t = gem.Literal(tangents)
        n = gem.Literal(nu)
        if mapping == "contravariant piola":
            # The normal maps to the cofactor normal.  The tangents map to the
            # reciprocal frame of the pushed forward tangents, scaled by detG/detg.
            return (gem.IndexSum(J[c, b] * K[e, c] * n[e], (c, e)) * n[a] / (nu @ nu)
                    + gem.IndexSum(t[b, l] * gem.Literal(adjg)[l, m] * G[m, p] * adjG[p, q] * t[a, q],
                                   (l, m, p, q)) / (detg**2 * detJ)
                    + n[b] * gem.IndexSum(Gnu[m] * adjG[m, p] * t[a, p], (m, p)) / ((nu @ nu) * detg * detJ))

        # The tangents push forward by J, and each normal maps to the unit
        # physical normal.
        if self.dim == sd - 2 >= 1:
            # The normal to the edge within the facet is the reciprocal frame of the
            # pushed forward edge tangent.
            t0, t1 = tangents.T
            P = numpy.outer(t0, t0) / (t0 @ t0)
            norm = (gem.IndexSum(adjG[l, 1] * G[l, m] * adjG[m, 1], (l, m)) * (t1 @ t1))**0.5
            edge_normal = gem.IndexSum(K[b, c] * J[c, e] * t[e, l] * adjG[l, 1], (c, e, l)) * t[a, 1] / norm
        else:
            P = tangents @ adjg @ tangents.T / detg
            edge_normal = gem.Zero()
        tangential = gem.IndexSum(K[b, c] * J[c, e] * gem.Literal(P)[e, a], (c, e)) if tangents.size else gem.Zero()
        # The cofactor normal
        x = (detJ * entries(K)).T @ nu
        return (tangential + edge_normal
                + detJ * gem.IndexSum(K[b, c] * K[e, c] * n[e], (c, e)) * n[a] / ((x @ x) * (nu @ nu))**0.5)

    def vandermonde_row(self, riesz, j, J, K, detJ):
        r"""The row of the physical Vandermonde matrix for this node.

        The row is :math:`B_j = \sum_{a,b}G_{ba}R_{abj}`, where :math:`G` is
        the tensor product of the :meth:`pullback` of each axis, and
        :math:`R` is the uncontracted Riesz representation.

        :arg riesz: See :attr:`PhysicallyMappedElement._uncontracted_riesz`.
        :arg j: The index of the reference basis.
        :arg J: The Jacobian.
        :arg K: The inverse of the Jacobian.
        :arg detJ: The determinant of the Jacobian.
        :returns: A GEM scalar with the free index j.
        """
        rank = len(self.axis_mappings)
        a = gem.indices(rank)
        b = gem.indices(rank)
        pullbacks = [self.pullback(k, b[k], a[k], J, K, detJ) for k in range(rank)]
        return gem.IndexSum(gem.Product(*pullbacks, gem.Literal(riesz)[(*a, *b, j)]), a + b)

    def scaling(self, J, h):
        r"""The factor :math:`h^p\,|e|` of the row, see :attr:`scaling_exponent`.

        :math:`|e|` is the physical measure of the entity if :attr:`avg` is
        False and there is a :attr:`support_facet`, and 1 otherwise.

        :arg J: The Jacobian as an object array of GEM scalars.
        :arg h: The cell size at each vertex.
        """
        result = one
        if self.support_facet is not None and isinstance(self.node, IntegralMoment) and not self.avg:
            # The physical measure of the support, from the Gram determinant of its tangents
            t = self.ref_el.compute_tangents(self.dim, self.entity).T
            Jt = J @ t
            volume = self.ref_el.volume_of_subcomplex(self.dim, self.entity)
            result = volume / numpy.sqrt(numpy.linalg.det(t.T @ t)) * determinant(Jt.T @ Jt)**0.5
        p = self.scaling_exponent
        if p:
            vertices = self.ref_el.get_topology()[self.dim][self.entity]
            havg = sum(h[v] for v in vertices) / len(vertices)
            result = result * havg**p
        return result


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
    def _uncontracted_riesz(self):
        r"""Uncontracted Riesz representation of each node in the source nodal basis.

        For node :math:`i`, the rank-3 tensor
        :math:`R_{a\beta b\gamma j}=\sum_q W_{i,a,\beta,q}(\partial^\gamma\psi^0_j)_b(x_q)`,
        where :math:`\psi^0` is the nodal basis of the nodes before
        recombination.  The axes are
        unflattened, with the axes of the node followed by those of the basis.
        """
        dual = self._element.get_dual_set()
        coeffs = self._element.get_coeffs()
        C = dual.get_coeffs()
        if C is not None:
            # The source nodal basis is C^T times the basis dual to C n_0.
            coeffs = numpy.tensordot(C.T, coeffs, axes=(1, 0))
        sd = self.cell.get_spatial_dimension()
        value_shape = coeffs.shape[1:-1]
        num_bf, num_exp = coeffs.shape[0], coeffs.shape[-1]
        coeffs = coeffs.reshape(num_bf, -1, num_exp)
        riesz = dual.to_uncontracted_riesz(self._element.get_nodal_basis())
        for order, R in riesz.items():
            R = R.reshape(len(R), -1, sd**order, num_exp)
            # (i, a beta, gamma, e) x (j, b, e) -> (i, a beta, gamma, j, b) -> (i, a beta, b, gamma, j)
            R = numpy.tensordot(R, coeffs, axes=(-1, -1)).transpose(0, 1, 4, 2, 3)
            shape = (*value_shape, *(sd,) * order)
            riesz[order] = R.reshape(len(R), *shape, *shape, num_bf)

        return [riesz[functional.node.max_deriv_order][i] for i, functional in enumerate(self._physical_functionals)]

    def _vandermonde_rows(self, coordinate_mapping):
        """The rows of the physical Vandermonde matrix that differ from the identity.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A dict from each node number to its
            :meth:`PhysicallyMappedFunctional.vandermonde_row`.
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
        J_ = entries(J)
        adjJ, detJ = adjugate_and_determinant(J_)
        h = coordinate_mapping.cell_size()

        # Opaque placeholders keep the Jacobian algebra syntactic until the rows are built.
        gdim, tdim = J.shape
        placeholders = (gem.Variable("J", (gdim, tdim)), gem.Variable("K", (tdim, gdim)), gem.Variable("detJ", ()))
        Jp, Kp, detJp = placeholders
        K = numpy.vectorize(lambda x: x * (one / detJ), otypes=[object])(adjJ)
        geometry = dict(zip(placeholders, (as_tensor(J_), as_tensor(K), detJ)))

        functionals = self._physical_functionals
        riesz = self._uncontracted_riesz
        inverses = {(Kp, Jp): one}
        if gdim == tdim:
            inverses[Jp, Kp] = one
        gram = {}
        for functional in functionals:
            G, adjG, detG, _ = functional.gram
            inverses[G, adjG] = inverses[adjG, G] = detG
            gram.update(functional.gram_replacements(Jp))
        geometry.update(zip(gram, substitute(list(gram.values()), geometry)))

        j = gem.Index()
        exprs = [functional.vandermonde_row(R, j, *placeholders) for functional, R in zip(functionals, riesz)]
        exprs = cancel_reciprocals(cancel_inverses(exprs, inverses, self.tol))
        tables = numpy.array(to_riesz(exprs, j, [self.tol * numpy.abs(R).max() for R in riesz]))
        tables = numpy.reshape(cancel_reciprocals(substitute(list(tables.ravel()), geometry)), tables.shape)

        closures = self._element.entity_closure_dofs()
        rows = {}
        for i, (functional, row) in enumerate(zip(functionals, tables)):
            row = row * functional.scaling(J_, h)
            nonzeros = [j for j, entry in enumerate(row) if not isinstance(entry, gem.Zero)]
            if nonzeros == [i] and row[i] == one:
                continue
            closure = closures[functional.dim][functional.entity]
            outside = [j for j in nonzeros if j not in closure]
            if outside:
                raise NotImplementedError(f"Dof {i} on entity {(functional.dim, functional.entity)} "
                                          f"couples outside its closure: {outside}")
            rows[i] = row
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
            B[i] = row
        return self._recombine(B)

    def _inverse_physical_vandermonde(self, coordinate_mapping):
        """:math:`B^{-1}` over all reference dofs.

        Solve entity blocks in closure order, retaining completion functions.
        Each row is scaled by its common denominator, so that the blocks
        are solved without divisions.
        """
        rows = {i: common_denominator(row) for i, row in self._vandermonde_rows(coordinate_mapping).items()}
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
                    numerator = numpy.array(numerator, dtype=object)
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


def round_to_int(x, tol):
    """Round the entries of x that lie within tol of an integer."""
    rounded = numpy.round(x)
    return numpy.where(numpy.abs(x - rounded) <= tol, rounded, x)


def entries(A):
    """The entries of a GEM matrix as an object array of GEM scalars."""
    return numpy.array([[A[i, k] for k in range(A.shape[1])] for i in range(A.shape[0])], dtype=object)


def as_tensor(A):
    """A GEM tensor from an object array of GEM scalars or numbers."""
    A = numpy.asarray(A)
    if A.dtype != object:
        return gem.Literal(A)
    return gem.ListTensor(numpy.vectorize(gem.as_gem, otypes=[object])(A))


@singledispatch
def _substitute(node, self):
    raise AssertionError(f"cannot handle type {type(node)}")


_substitute.register(gem.Node)(reuse_if_untouched)


@_substitute.register(gem.Variable)
def _substitute_variable(node, self):
    return self.replacements.get(node, node)


def substitute(expressions, replacements):
    """Replace variables in GEM expressions.

    :arg expressions: A list of GEM expressions.
    :arg replacements: A dict from each :class:`gem.Variable` to its replacement.
    """
    mapper = Memoizer(_substitute)
    mapper.replacements = replacements
    return list(map(mapper, expressions))


def symbolic_classifier():
    """A classifier of GEM expressions that keeps the constants apart.

    Expressions without a :class:`gem.Variable` are constants.  Sums,
    products, and contractions with constant tensors are broken up, and
    the other expressions of variables are atomic.
    """
    symbolic = {}
    tensors = {}

    def visit(node):
        stack = [node]
        while stack:
            n = stack[-1]
            pending = [c for c in n.children if c not in symbolic]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            symbolic[n] = isinstance(n, gem.Variable) or any(symbolic[c] for c in n.children)
            constant_tensor = isinstance(n, gem.Delta) or (
                isinstance(n, gem.Indexed) and isinstance(n.children[0], gem.gem.Constant) and n.free_indices)
            tensors[n] = constant_tensor or any(tensors[c] for c in n.children)

    def classify(node):
        if node not in symbolic:
            visit(node)
        if not symbolic[node]:
            return OTHER
        if isinstance(node, gem.IndexSum):
            return COMPOUND if tensors[node] else ATOMIC
        if isinstance(node, (gem.Product, gem.Sum)) or (isinstance(node, gem.Division) and node.children[0] != one):
            return COMPOUND
        return ATOMIC
    return classify


def to_riesz(expressions, j, tol):
    r"""The rows over the index j of GEM scalars of expressions in symbolic and constant factors.

    The constant factors of each term are folded into one tensor, and the
    term is unrolled over the indices that this tensor shares with the
    symbolic factors.  The constants that multiply the same symbolic factors
    are rounded to integers within the tolerance and dropped below it,
    both before and after they are summed.

    :arg expressions: A list of GEM scalars with the free index j.
    :arg j: The free index, which only appears in constant factors.
    :arg tol: The tolerance of each expression.
    :returns: A list of object arrays of GEM scalars over j.
    """
    classifier = symbolic_classifier()
    terms = []
    constants = []
    for owner, expression in enumerate(remove_componenttensors(expressions)):
        for term in traverse_sum(expression):
            sum_indices, factors = traverse_product(term, stop_at=lambda e: classifier(e) != COMPOUND)
            others = [f for f in factors if classifier(f) != OTHER]
            other_indices = set(chain.from_iterable(f.free_indices for f in others))
            constant = [f for f in factors if classifier(f) == OTHER]
            terms.append((owner, [i for i in sum_indices if i in other_indices], others))
            constants.append(gem.IndexSum(gem.Product(*constant), tuple(i for i in sum_indices if i not in other_indices)))

    # Fix the indices that the symbolic factors share with the constants
    keys = []
    fixed = []
    for (owner, sum_indices, others), rest in zip(terms, fold_constants(constants)):
        shared = tuple(i for i in sum_indices if i in rest.free_indices)
        local = [i for i in sum_indices if i not in rest.free_indices]
        # Contract the indices that only the symbolic factors carry
        contracted = [f for f in others if set(f.free_indices) & set(local)]
        factors = [f for f in others if not set(f.free_indices) & set(local)]
        if contracted:
            factors.append(sum_factorise(local, contracted))
        if isinstance(rest, gem.Zero):
            continue
        array, indices = (rest.children[0].array, rest.multiindex) if isinstance(rest, gem.Indexed) else (rest.array, ())
        array = numpy.transpose(array, [indices.index(i) for i in shared + (j,)])
        array = round_to_int(array, tol[owner])
        array[numpy.abs(array) <= tol[owner]] = 0
        for beta in numpy.ndindex(array.shape[:-1]):
            if array[beta].any():
                keys.append((owner, len(factors), array[beta]))
                for f in factors:
                    axes = [k for k, i in enumerate(shared) if i in f.free_indices]
                    fixed.append(gem.Indexed(gem.ComponentTensor(f, tuple(shared[k] for k in axes)),
                                             tuple(beta[k] for k in axes)))
    fixed = remove_componenttensors(unroll_indexsum(remove_componenttensors(fixed), predicate=lambda index: True))

    # Sum the constants of the same symbolic factors
    coefficients = [{} for _ in expressions]
    fixed = iter(fixed)
    for owner, n, c in keys:
        factors = tuple(next(fixed) for _ in range(n))
        key = frozenset(Counter(factors).items())
        factors, total = coefficients[owner].get(key, (factors, 0))
        coefficients[owner][key] = factors, total + c

    rows = []
    for coefficient, cutoff in zip(coefficients, tol):
        row = [[] for _ in range(j.extent)]
        for factors, c in coefficient.values():
            c = round_to_int(c, cutoff)
            g = gem.Product(*factors)
            for k in numpy.flatnonzero(numpy.abs(c) > cutoff):
                row[k].append(g * c[k])
        rows.append(numpy.array([reduce(gem.Sum, terms, zero) for terms in row], dtype=object))
    return rows


def in_range_of(weights, projector, axes, tol):
    """Whether the weights lie in the range of the projector acting on the given axes."""
    shape = tuple(weights.shape[k] for k in axes)
    complement = numpy.eye(projector.shape[0]) - projector
    residual = numpy.tensordot(complement.reshape(shape + shape), weights, axes=(range(len(axes), 2 * len(axes)), axes))
    return numpy.abs(residual).max() <= tol * numpy.abs(weights).max()
