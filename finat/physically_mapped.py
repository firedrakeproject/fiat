r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
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


class PhysicalFunctional:
    r"""A FIAT node and the data that maps it to the physical cell.

    :attr:`group_mappings` assigns a mapping to each group of axes of
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
        r"""The facet in whose frame :attr:`group_mappings` act, or None for Cartesian components.

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
            if not all(_in_range(self.weights, (k,), numpy.outer(t, t) / (t @ t), self.tol) for k in covectors):
                top = ref_el.get_topology()
                vertices = set(top[self.dim][self.entity])
                for facet in sorted(top[sd - 1]):
                    nu = ref_el.compute_scaled_normal(facet)
                    P = numpy.eye(sd) - numpy.outer(nu, nu) / (nu @ nu)
                    if (vertices <= set(top[sd - 1][facet])
                            and all(_in_range(self.weights, (k,), P, self.tol) for k in covectors)):
                        return facet
                raise NotImplementedError(f"No facet contains the derivatives of {type(node).__name__}.")
        return None

    @cached_property
    def group_mappings(self):
        """The mapping of each group of axes of :attr:`weights`.

        Interior moments map every axis as ``affine``.  With a
        :attr:`support_facet`, each axis keeps its :attr:`axis_mappings`
        entry.  Otherwise, a contravariant axis paired with a covariant axis
        through the identity maps as ``L2 piola``, and a covariant axis
        along an edge maps as ``affine``.
        """
        sd = self.ref_el.get_spatial_dimension()
        W = self.weights
        axis_mappings = self.axis_mappings
        if self.dim == sd and isinstance(self.node, IntegralMoment):
            return {(k,): "affine" for k in range(len(axis_mappings))}
        if self.support_facet is not None:
            return {(k,): mapping for k, mapping in enumerate(axis_mappings)}

        groups = {}
        for k, vector in enumerate(axis_mappings):
            for l, covector in enumerate(axis_mappings):
                if (vector == "contravariant piola" and covector == "covariant piola"
                        and not any(k in axes or l in axes for axes in groups)
                        and _in_range(W, (k, l), _pairing_projector(sd), self.tol)):
                    groups[k, l] = "L2 piola"
        on_edge = 0 < self.dim < sd - 1
        for k, mapping in enumerate(axis_mappings):
            if any(k in axes for axes in groups):
                continue
            if on_edge and mapping == "covariant piola":
                t = self.ref_el.compute_edge_tangent(self.entity)
                if not _in_range(W, (k,), numpy.outer(t, t) / (t @ t), self.tol):
                    raise NotImplementedError(f"Cannot map derivatives normal to an edge of {type(self.node).__name__}.")
                mapping = "affine"
            groups[k,] = mapping
        return groups

    @cached_property
    def scaling_exponent(self):
        """The exponent :math:`p` such that the physical node dilates as :math:`h^{-p}`.

        Each group in :attr:`group_mappings` contributes its scaling.  If
        :attr:`avg` is False, an integral moment with a :attr:`support_facet`
        is a plain integral over its entity, which contributes :math:`-dim`.
        """
        sd = self.ref_el.get_spatial_dimension()
        if self.support_facet is None:
            weights = {"covariant piola": 1, "contravariant piola": sd - 1, "L2 piola": sd}
            return sum(weights.get(mapping, 0) for mapping in self.group_mappings.values())
        nu = self.ref_el.compute_scaled_normal(self.support_facet)
        exponent = 0
        for axes, mapping in self.group_mappings.items():
            if mapping == "covariant piola":
                exponent += 1
            elif mapping == "contravariant piola" and not _in_range(
                    self.weights, axes, numpy.outer(nu, nu) / (nu @ nu), self.tol):
                # Tangential components of a contravariant axis
                exponent += 2 - sd
        if isinstance(self.node, IntegralMoment) and not self.avg:
            exponent -= self.dim
        return exponent

    def vandermonde_row(self, uncontracted_riesz, pullbacks, J, h):
        r"""The row of the physical Vandermonde matrix for this node.

        The row is :math:`B_j = h^p\,|e|\sum_{a,b}G_{ba}R_{abj}`, where
        :math:`G` is the Kronecker product of the pullbacks of
        :attr:`group_mappings`, :math:`p` is :attr:`scaling_exponent`, and
        :math:`|e|` is the physical measure of the entity if :attr:`avg` is
        False and there is a :attr:`support_facet`, and 1 otherwise.

        :arg uncontracted_riesz: :math:`R_{a,\beta,b,\gamma,j}=\sum_q W_{a,\beta,q}(\partial^\gamma\psi^0_j)_b(x_q)`,
            where :math:`W` are the weights of the node and :math:`\psi^0`
            is the nodal basis of the nodes before recombination.
        :arg pullbacks: The pullback of each mapping, in Cartesian components
            or in the frame of :attr:`support_facet`.
        :arg J: The Jacobian as an object array of GEM scalars.
        :arg h: The cell size at each vertex.
        :returns: The numerator and denominator of the Vandermonde row, or
            None if the row is the identity.
        """
        measure = one
        if self.support_facet is not None and isinstance(self.node, IntegralMoment) and not self.avg:
            # The physical measure of the support, from the Gram determinant of its tangents
            T = self.ref_el.compute_tangents(self.dim, self.entity)
            JT = J @ T.T
            volume = self.ref_el.volume_of_subcomplex(self.dim, self.entity)
            measure = volume / numpy.sqrt(numpy.linalg.det(T @ T.T)) * determinant(JT.T @ JT)**0.5

        numerators = {}
        is_identity = measure == one and self.scaling_exponent == 0
        for axes, mapping in self.group_mappings.items():
            pullback = pullbacks[mapping]
            in_range = _in_range(self.weights, axes, pullback.projector, self.tol)
            is_identity = is_identity and in_range and pullback.scale == pullback.denominator
            numerators[axes] = pullback.numerator(in_range)
        if is_identity:
            return None

        n = (uncontracted_riesz.ndim - 1) // 2
        axes = list(chain.from_iterable(self.group_mappings))
        R = uncontracted_riesz.transpose(*axes, *(n + k for k in axes), -1).reshape(-1, uncontracted_riesz.shape[-1])
        numerator = numpy.full(R.shape[1], zero, dtype=object)
        cutoff = self.tol * numpy.abs(R).max()
        for terms in product(*numerators.values()):
            coefficients, matrices = zip(*terms) if terms else ((), ())
            C = reduce(numpy.kron, matrices) if matrices else numpy.ones((1, 1))
            reference_row = C.T.ravel() @ R
            nonzeros = numpy.flatnonzero(numpy.abs(reference_row) > cutoff * numpy.abs(C).max())
            coefficient = gem.Product(*coefficients)
            for j in nonzeros:
                numerator[j] += coefficient * reference_row[j]
        numerator = numerator * measure
        denominator = gem.Product(*(pullbacks[mapping].denominator for mapping in self.group_mappings.values()))

        p = self.scaling_exponent
        if p is not None:
            vertices = self.ref_el.get_topology()[self.dim][self.entity]
            havg = sum(h[v] for v in vertices) / len(vertices)
            denominator = denominator * havg**-p
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
        """The reference dofs exposed as physical dofs, in local order."""
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
                    functionals[i] = PhysicalFunctional(dual.nodes[i], self.cell, dim, entity, mappings[i],
                                                        self.avg, self.tol)
        return tuple(functionals)

    @cached_property
    def _uncontracted_riesz(self):
        r"""Uncontracted Riesz representation of each node in the source nodal basis.

        :math:`R_{i,a,\beta,b,\gamma,j}=\sum_q W_{i,a,\beta,q}(\partial^\gamma\psi^0_j)_b(x_q)`,
        where :math:`\psi^0` is the nodal basis of the nodes before
        recombination.
        """
        dual = self._element.get_dual_set()
        coeffs = self._element.get_coeffs()
        C = dual.get_coeffs()
        if C is not None:
            # The source nodal basis is C^T times the basis dual to C n_0.
            coeffs = numpy.tensordot(C.T, coeffs, axes=(1, 0))
        riesz = dual.to_uncontracted_riesz(self._element.get_nodal_basis())
        result = []
        for i, node in enumerate(dual.nodes):
            rank = len(node.target_shape)
            order = node.max_deriv_order
            R = numpy.tensordot(riesz[order][i], coeffs, axes=(-1, -1))
            # (a, beta, gamma, j, b) -> (a, beta, b, gamma, j)
            result.append(numpy.moveaxis(R, range(R.ndim - rank, R.ndim), range(rank + order, 2 * rank + order)))
        return result

    def _vandermonde_rows(self, coordinate_mapping):
        """The rows of the physical Vandermonde matrix that differ from the identity.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A dict from each node number to the numerator and
            denominator of its :meth:`PhysicalFunctional.vandermonde_row`.
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

        cartesian = _cartesian_pullbacks(J, adjJ, detJ)
        frames = {}

        closures = self._element.entity_closure_dofs()
        rows = {}
        for i, functional in enumerate(self._physical_functionals):
            if functional.support_facet is not None:
                key = (functional.support_facet, (functional.dim, functional.entity))
                if key not in frames:
                    frames[key] = _frame_pullbacks(self.cell, *key, J, adjJ, detJ)
                pullbacks = frames[key]
            else:
                pullbacks = cartesian
            row = functional.vandermonde_row(self._uncontracted_riesz[i], pullbacks, J, h)
            if row is None:
                continue
            closure = closures[functional.dim][functional.entity]
            outside = [j for j, entry in enumerate(row[0]) if j not in closure and not isinstance(entry, gem.Zero)]
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
    """The pullback of each mapping in Cartesian components."""
    sd = J.shape[1]
    units = numpy.eye(sd * sd).reshape(sd, sd, sd, sd)
    contravariant = Pullback(numpy.zeros((sd, sd)), [(J[b, a], units[a, b]) for a, b in numpy.ndindex(sd, sd)], detJ)
    covariant = Pullback(numpy.zeros((sd, sd)), [(adjJ[a, b], units[a, b]) for a, b in numpy.ndindex(sd, sd)], detJ)
    # The pairing of a contravariant axis with a covariant axis only scales by 1/detJ.
    P = _pairing_projector(sd)
    complement = numpy.eye(sd * sd) - P
    pairing = Pullback(P, [(g * h, numpy.kron(C, D) @ complement)
                           for (g, C), (h, D) in product(contravariant.terms, covariant.terms)],
                       detJ * detJ, scale=detJ)
    return {"affine": Pullback(numpy.eye(sd)), "contravariant piola": contravariant,
            "covariant piola": covariant, "L2 piola": pairing}


def _pairing_projector(sd):
    """Projector onto the identity, the pairing of a contravariant axis with a covariant axis."""
    delta = numpy.eye(sd).ravel()
    return numpy.outer(delta, delta) / sd


def _frame_pullbacks(ref_el, facet, support, J, adjJ, detJ):
    r"""The pullback of each mapping in the frame of the facet.

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
            "contravariant piola": contravariant}


def _in_range(weights, axes, projector, tol):
    """Whether the weights lie in the range of the projector on the given axes."""
    W = numpy.moveaxis(weights, axes, range(len(axes))).reshape(projector.shape[1], -1)
    return numpy.abs(W - projector @ W).max() <= tol * numpy.abs(weights).max()
