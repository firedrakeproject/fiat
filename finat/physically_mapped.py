r"""Physically mapped elements and the automatic basis transformation.

Let :math:`\mathcal{P}` map reference functions to the physical cell,
including any Piola map.  The physical basis is :math:`M\mathcal{P}\hat\Psi`,
where :math:`B_{ij} = n_i(\mathcal{P}\hat\psi_j)` and :math:`M = B^{-T}`
(Kirby 2017, Brubeck & Kirby 2025).
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import cached_property, reduce
from itertools import product
from operator import mul
from math import prod

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
        self.restriction_indices = None

    def basis_transformation(self, coordinate_mapping):
        r"""Transformation matrix for the basis functions.

        :arg coordinate_mapping: Object providing the physical geometry
            as GEM expressions.
        :returns: A ``gem.ListTensor`` with exposed physical dofs as rows
            and the complete reference basis as columns.
        """
        V = self._inverse_physical_vandermonde(self._physical_rows(coordinate_mapping))
        for i, scale in self._dof_scales(coordinate_mapping):
            V[:, i] = V[:, i] * scale
        C = self._element.get_dual_set().get_coeffs()
        if C is not None:
            V = C @ V @ self._inverse_dual_coeffs
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
        B = identity(self._element.space_dimension())
        for i, (numerator, denominator) in self._physical_rows(coordinate_mapping).items():
            B[i] = numerator / denominator
        for i, scale in self._dof_scales(coordinate_mapping):
            B[i] = B[i] / scale
        C = self._element.get_dual_set().get_coeffs()
        if C is not None:
            B = C @ B @ self._inverse_dual_coeffs
        indices = self._physical_indices
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

    @cached_property
    def _physical_indices(self):
        if self.restriction_indices is not None:
            return self.restriction_indices
        return sorted(i for entities in self.entity_dofs().values()
                      for dofs in entities.values() for i in dofs)

    @cached_property
    def _inverse_dual_coeffs(self):
        r"""For :math:`n=Cn_0`, the Vandermonde transforms as :math:`B=CB_0C^{-1}`."""
        return numpy.linalg.inv(self._element.get_dual_set().get_coeffs())

    @cached_property
    def _dual_tensors(self):
        return self._element.get_dual_set().as_tensor(self._element.mapping(), self.tol)

    @cached_property
    def _reference_riesz(self):
        r"""Source pairings :math:`R_{i,a,b,j}=\sum_q W_{i,q,a}(D_i\hat\psi^0_j)(x_{i,q})_b`.

        The basis psi^0 is nodal to n_0. Indices a and b enumerate flattened
        coefficient and derivative tensors.
        """
        dual = self._element.get_dual_set()
        points, coefficients, axis_mappings = self._dual_tensors
        indices = [i for i, mapping in enumerate(dual.node_mapping) if mapping != "invariant"]
        if not indices:
            return {}
        sd = self.cell.get_spatial_dimension()
        order = [sum(axis in ("derivative", "divergence") for axis in axes) for axes in axis_mappings]
        evaluation_points = sorted({pt for i in indices for pt in points[i]})
        columns = {pt: q for q, pt in enumerate(evaluation_points)}
        tabulation = self._element.tabulate(max(order[i] for i in indices), evaluation_points)
        C = dual.get_coeffs()
        if C is not None:
            # The source nodal basis is C^T times the basis dual to C n_0.
            tabulation = {alpha: numpy.tensordot(C.T, table, axes=(1, 0))
                          for alpha, table in tabulation.items()}
        riesz = {}
        for i in indices:
            cols = [columns[pt] for pt in points[i]]
            values = tabulation[(0,) * sd][..., cols]
            T = numpy.zeros(values.shape[:-1] + (sd,) * order[i] + values.shape[-1:])
            prefix = (slice(None),) * (values.ndim - 1)
            for index in numpy.ndindex((sd,) * order[i]):
                alpha = tuple(numpy.bincount(index, minlength=sd))
                T[prefix + index] = tabulation[alpha][..., cols]
            if "divergence" in axis_mappings[i]:
                T = numpy.trace(T, axis1=values.ndim - 2, axis2=-2)[..., None, :]
            W = coefficients[i]
            size = prod(W.shape[1:])
            nbf = T.shape[0]
            R = numpy.einsum("qa,jbq->abj", W.reshape(-1, size), T.reshape(nbf, size, -1))
            riesz[i] = R.reshape(size * size, nbf)
        return riesz

    @cached_property
    def _scaling_exponents(self):
        _, coefficients, axis_mappings = self._dual_tensors
        return self._element.get_dual_set().get_scaling_exponents(coefficients, axis_mappings, self.avg, self.tol)

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

        Contract the selected pullbacks with FIAT's reference pairings, then
        check that each physical node couples only to its entity's closure.
        """
        pullbacks = PhysicalPullbacks(self.cell, coordinate_mapping)
        dual = self._element.get_dual_set()
        closures = self._element.entity_closure_dofs()
        _, coefficients, axis_mappings = self._dual_tensors
        riesz = self._reference_riesz
        rows = {}
        for dim, entities in closures.items():
            for entity, closure_dofs in entities.items():
                allowed = set(closure_dofs)
                for support, dofs in dual.entity_support_ids[dim][entity].items():
                    for i in dofs:
                        node_mapping = dual.node_mapping[i]
                        if node_mapping == "invariant":
                            continue
                        axis_pullbacks = pullbacks.select(node_mapping, axis_mappings[i], (dim, entity), support)
                        measure = one
                        if node_mapping == "frame" and not self.avg and isinstance(dual.nodes[i], IntegralMoment):
                            measure = pullbacks.measure(entity)
                        if measure == one and _is_invariant(coefficients[i], axis_pullbacks, self.tol):
                            continue
                        numerator, denominator = _physical_node_row(riesz[i], axis_pullbacks, self.tol)
                        numerator = numerator * measure
                        outside = [j for j, entry in enumerate(numerator)
                                   if j not in allowed and not isinstance(entry, gem.Zero)]
                        if outside:
                            raise NotImplementedError(f"Dof {i} on entity {(dim, entity)} couples outside its closure: {outside}")
                        rows[i] = numerator, denominator
        return rows

    def _inverse_physical_vandermonde(self, rows):
        """Solve entity blocks in closure order, retaining completion functions."""
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


class PhysicalPullbacks:
    """Select axis pullbacks, sharing geometry across nodes on the same cell."""

    def __init__(self, ref_el, coordinate_mapping):
        self.ref_el = ref_el
        self.sd = ref_el.get_spatial_dimension()
        bary, = ref_el.make_points(self.sd, 0, self.sd + 1)
        if self.sd == 1:
            # The signed determinant carries the orientation of the tangent.
            J = gem.ListTensor([[coordinate_mapping.detJ_at(bary)]])
        else:
            J = coordinate_mapping.jacobian_at(bary)
        self.J = numpy.array([[J[i, k] for k in range(J.shape[1])]
                              for i in range(J.shape[0])], dtype=object)
        self.adjJ, self.detJ = adjugate_and_determinant(self.J)
        self._cartesian = {}
        self._frames = {}
        self._measures = {}

    def select(self, node_mapping, axis_mappings, entity, support):
        """Select Cartesian or physical frame pullbacks for the node's tensor axes."""
        dim, entity_id = entity
        if node_mapping == "cartesian":
            pullbacks = []
            for mapping in axis_mappings:
                if 0 < dim < self.sd - 1 and mapping == "derivative":
                    # Edge tangent derivatives are already fixed by the trace.
                    pullbacks.append(_pullback(numpy.eye(self.sd)))
                else:
                    if mapping not in self._cartesian:
                        self._cartesian[mapping] = _cartesian_pullback(mapping, self.J, self.adjJ, self.detJ)
                    pullbacks.append(self._cartesian[mapping])
            return pullbacks
        if node_mapping == "frame":
            key = (entity_id, support)
            if key not in self._frames:
                self._frames[key] = _frame_pullbacks(self.ref_el, entity_id, support, self.J, self.adjJ, self.detJ)
            frame = self._frames[key]
            return [frame[mapping] for mapping in axis_mappings]
        raise NotImplementedError(f"No pullback for node mapping {node_mapping}")

    def measure(self, facet):
        """Return the physical facet measure induced by the cofactor map."""
        if facet not in self._measures:
            nu = self.ref_el.compute_scaled_normal(facet)
            Knu = self.adjJ.T @ nu
            reference_measure = self.ref_el.volume_of_subcomplex(self.sd - 1, facet)
            self._measures[facet] = (reference_measure / numpy.linalg.norm(nu)) * (Knu @ Knu)**0.5
        return self._measures[facet]


def _pullback(invariant, terms=(), denominator=one):
    r"""Represent :math:`G=P+\sum_t g_t C_t/d` by P, numerator terms, and d."""
    invariant = numpy.asarray(invariant)
    numerator = list(terms)
    if invariant.any():
        numerator.insert(0, (denominator, invariant))
    return invariant, numerator, denominator


def _cartesian_pullback(mapping, J, adjJ, detJ):
    sd = J.shape[1]
    if mapping == "affine":
        return _pullback(numpy.eye(sd))
    if mapping == "divergence":
        return _pullback(numpy.zeros((1, 1)), [(one, numpy.ones((1, 1)))], detJ)
    if mapping == "contravariant piola":
        Q = J.T
    elif mapping in {"covariant piola", "derivative"}:
        Q = adjJ
    else:
        raise NotImplementedError(f"No pullback for {mapping}")
    units = numpy.eye(sd * sd).reshape(sd, sd, sd, sd)
    return _pullback(numpy.zeros((sd, sd)), [(Q[a, b], units[a, b]) for a, b in numpy.ndindex(sd, sd)], detJ)


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
    covariant = _pullback(P, terms, reduce(mul, dens))
    nn = nu @ nu
    Knu = K @ nu
    KtKnu = K.T @ Knu
    P = numpy.outer(nu, nu) / nn
    terms = [(Knu @ Knu, numpy.eye(sd) - P)]
    terms.extend((KtKnu @ -tau, numpy.outer(nu, tau)) for tau in tangents)
    contravariant = _pullback(P, terms, detJ * nn)
    return {"affine": _pullback(numpy.eye(sd)), "covariant piola": covariant,
            "derivative": covariant, "contravariant piola": contravariant,
            "divergence": _cartesian_pullback("divergence", J, adjJ, detJ)}


def _is_invariant(coefficients, pullbacks, tol):
    """Check whether the node weights lie in every axis's invariant subspace."""
    cutoff = tol * numpy.abs(coefficients).max()
    for axis, (projector, _, _) in enumerate(pullbacks, start=1):
        # Axis zero indexes quadrature points; the remaining axes transform.
        complement = numpy.eye(projector.shape[0]) - projector
        residual = numpy.tensordot(coefficients, complement, axes=(axis, 1))
        if numpy.abs(residual).max() > cutoff:
            return False
    return True


def _physical_node_row(riesz, pullbacks, tol):
    r"""Contract :math:`B_j=\sum_{a,b}G_{ba}R_{abj}` in numerator/denominator form."""
    numerator = numpy.full(riesz.shape[1], zero, dtype=object)
    cutoff = tol * numpy.abs(riesz).max()
    for terms in product(*(terms for _, terms, _ in pullbacks)):
        coefficients, matrices = zip(*terms) if terms else ((), ())
        C = reduce(numpy.kron, matrices) if matrices else numpy.ones((1, 1))
        reference_row = C.T.ravel() @ riesz
        nonzeros = numpy.flatnonzero(numpy.abs(reference_row) > cutoff * numpy.abs(C).max())
        coefficient = gem.Product(*coefficients)
        for j in nonzeros:
            numerator[j] += coefficient * reference_row[j]
    return numerator, gem.Product(*(denominator for _, _, denominator in pullbacks))
