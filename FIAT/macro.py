from itertools import chain, combinations

import numpy

from FIAT import expansions, polynomial_set, reference_element
from FIAT.quadrature import FacetQuadratureRule, QuadratureRule
from FIAT.reference_element import (TRIANGLE, SimplicialComplex, lattice_iter,
                                    make_lattice)


def bary_to_xy(verts, bary, result=None):
    """Maps barycentric coordinates to physical points.

    :arg verts: A tuple of points.
    :arg bary: A row-stacked numpy array of barycentric coordinates.
    :arg result: A row-stacked numpy array of physical points.
    :returns: result
    """
    return numpy.dot(bary, verts, out=result)


def xy_to_bary(verts, pts, result=None):
    """Maps physical points to barycentric coordinates.

    :arg verts: A tuple of points.
    :arg pts: A row-stacked numpy array of physical points.
    :arg result: A row-stacked numpy array of barycentric coordinates.
    :returns: result
    """
    verts = numpy.asarray(verts)
    pts = numpy.asarray(pts)
    npts = pts.shape[0]
    sdim = verts.shape[1]

    mat = numpy.vstack((verts.T, numpy.ones((1, sdim+1))))
    b = numpy.vstack((pts.T, numpy.ones((1, npts))))
    foo = numpy.linalg.solve(mat, b)
    if result is None:
        return numpy.copy(foo.T)
    else:
        result[:, :] = foo[:, :].T
    return result


def facet_support(facet_coords, tol=1.e-12):
    """Returns the support of a facet.

    :arg facet_coords: An iterable of tuples (barycentric coordinates) describing the facet.
    :returns: A tuple of vertex ids where some coordinate is nonzero.
    """
    return tuple(sorted(set(i for x in facet_coords for (i, xi) in enumerate(x) if abs(xi) > tol)))


def invert_cell_topology(T):
    """Returns a dict of dicts mapping dimension x vertices to entity id."""
    return {dim: {T[dim][entity]: entity for entity in T[dim]} for dim in T}


def make_topology(sd, num_verts, edges):
    topology = {}
    topology[0] = {i: (i,) for i in range(num_verts)}
    topology[1] = dict(enumerate(sorted(edges)))

    # Get an adjacency list for each vertex
    adjacency = {v: set(chain.from_iterable(verts for verts in edges if v in verts))
                 for v in topology[0]}

    # Complete the higher dimensional facets by appending a vertex
    # adjacent to the vertices of codimension 1 facets
    for dim in range(1, sd):
        entities = []
        for entity in topology[dim]:
            facet = topology[dim][entity]
            facet_verts = set(facet)
            for v in range(min(facet)):
                if (facet_verts < adjacency[v]):
                    entities.append((v, *facet))

        topology[dim+1] = dict(enumerate(sorted(entities)))
    return topology


class SplitSimplicialComplex(SimplicialComplex):
    """Abstract class to implement a split on a Simplex.

    :arg parent: The parent Simplex to split.
    :arg vertices: The vertices of the simplicial complex.
    :arg topology: The topology of the simplicial complex.
    """
    def __init__(self, parent, vertices, topology):
        self._parent_complex = parent
        while parent.get_parent():
            parent = parent.get_parent()
        self._parent_simplex = parent

        bary = xy_to_bary(parent.get_vertices(), vertices)
        parent_top = parent.get_topology()
        parent_inv_top = invert_cell_topology(parent_top)

        # dict mapping child facets to their parent facet
        child_to_parent = {}
        # dict mapping parent facets to their children
        parent_to_children = {dim: {entity: [] for entity in parent_top[dim]} for dim in parent_top}
        for dim in topology:
            child_to_parent[dim] = {}
            for entity in topology[dim]:
                facet_ids = topology[dim][entity]
                facet_coords = bary[list(facet_ids), :]
                parent_verts = facet_support(facet_coords)
                parent_dim = len(parent_verts) - 1
                parent_entity = parent_inv_top[parent_dim][parent_verts]
                child_to_parent[dim][entity] = (parent_dim, parent_entity)
                parent_to_children[parent_dim][parent_entity].append((dim, entity))

        for dim in parent_to_children:
            for entity in parent_to_children[dim]:
                children = parent_to_children[dim][entity]
                if len(children) > 1:
                    # sort children lexicographically
                    pts = [tuple(numpy.average([vertices[i] for i in topology[cdim][centity]], 0))
                           for cdim, centity in children]
                    bary = parent.compute_barycentric_coordinates(pts, entity=(dim, entity))
                    order = numpy.lexsort(bary.T)
                    children = tuple(children[j] for j in order)
                else:
                    children = tuple(children)
                parent_to_children[dim][entity] = children

        self._child_to_parent = child_to_parent
        self._parent_to_children = parent_to_children

        sd = parent.get_spatial_dimension()
        inv_top = invert_cell_topology(topology)

        # dict mapping cells to their boundary facets for each dimension,
        # while respecting the ordering on the parent simplex
        connectivity = {cell: {dim: [] for dim in topology} for cell in topology[sd]}
        for cell in topology[sd]:
            cell_verts = topology[sd][cell]
            for dim in parent_top:
                for entity in parent_top[dim]:
                    ref_verts = parent_top[dim][entity]
                    global_verts = tuple(cell_verts[v] for v in ref_verts)
                    connectivity[cell][dim].append(inv_top[dim][global_verts])
        self._cell_connectivity = connectivity

        # dict mapping subentity dimension to interior facets
        interior_facets = {dim: [entity for entity in child_to_parent[dim]
                                 if child_to_parent[dim][entity][0] == sd]
                           for dim in sorted(child_to_parent)}
        self._interior_facets = interior_facets

        super().__init__(parent.shape, vertices, topology)

    def get_child_to_parent(self):
        """Maps split complex facet tuple to its parent entity tuple."""
        return self._child_to_parent

    def get_parent_to_children(self):
        """Maps parent facet tuple to a list of tuples of entites in the split complex."""
        return self._parent_to_children

    def get_cell_connectivity(self):
        """Connectitivity from cell in a complex to global facet ids and
        respects the entity numbering on the reference cell.

        N.B. cell_connectivity[cell][dim] has the same contents as
        self.connectivity[(sd, dim)][cell], except those are sorted.
        """
        return self._cell_connectivity

    def get_interior_facets(self, dimension):
        """Returns the list of entities of the given dimension that are
        supported on the parent's interior.

        :arg dimension: subentity dimension (integer)
        """
        return self._interior_facets[dimension]

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.get_parent().construct_subelement(dimension)

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)

    def is_macrocell(self):
        return True

    def get_parent(self):
        return self._parent_simplex

    def get_parent_complex(self):
        return self._parent_complex


class IsoSplit(SplitSimplicialComplex):
    """Splits simplex into the simplicial complex obtained by
    connecting points on a regular lattice.

    :arg ref_el: The parent Simplex to split.
    :kwarg degree: The number of subdivisions along each edge of the simplex.
    :kwarg variant: The point distribution variant.
    """
    def __init__(self, ref_el, degree=2, variant=None):
        self.degree = degree
        self.variant = variant
        # Place new vertices on a lattice
        sd = ref_el.get_spatial_dimension()
        new_verts = make_lattice(ref_el.vertices, degree, variant=variant)
        flat_index = {tuple(alpha): i for i, alpha in enumerate(lattice_iter(0, degree+1, sd))}

        # Loop through degree-1 vertices
        # Construct a P1 simplex by connecting edges between a vertex and
        # its neighbors obtained by shifting each coordinate up by 1
        edges = []
        for alpha in lattice_iter(0, degree, sd):
            simplex = []
            for beta in lattice_iter(0, 2, sd):
                v1 = flat_index[tuple(a+b for a, b in zip(alpha, beta))]
                edges.extend((v0, v1) for v0 in simplex)
                simplex.append(v1)

        if sd == 3:
            # Cut the octahedron
            # FIXME do this more generically
            assert degree == 2
            v0, v1 = flat_index[(1, 0, 0)], flat_index[(0, 1, 1)]
            edges.append(tuple(sorted((v0, v1))))

        new_topology = make_topology(sd, len(new_verts), edges)
        super().__init__(ref_el, tuple(new_verts), new_topology)

    def construct_subcomplex(self, dimension):
        """Constructs the reference subcomplex of the parent complex
        specified by subcomplex dimension.
        """
        if dimension == self.get_dimension():
            return self
        ref_el = self.construct_subelement(dimension)
        if dimension == 0:
            return ref_el
        else:
            # Iso on facets is Iso
            return IsoSplit(ref_el, self.degree, self.variant)


class PowellSabinSplit(SplitSimplicialComplex):
    """Splits a simplicial complex by connecting barycenters of subentities
    to the barycenter of the superentities of the given dimension or higher.
    """
    def __init__(self, ref_el, dimension=1):
        self.split_dimension = dimension
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        connectivity = ref_el.get_connectivity()
        new_verts = list(ref_el.get_vertices())
        dim = dimension - 1
        simplices = {dim: {entity: [top[dim][entity]] for entity in top[dim]}}

        for dim in range(dimension, sd+1):
            simplices[dim] = {}
            for entity in top[dim]:
                bary_id = len(new_verts)
                new_verts.extend(ref_el.make_points(dim, entity, dim+1))

                # Connect entity barycenter to every subsimplex on the entity
                simplices[dim][entity] = [(*s, bary_id)
                                          for child in connectivity[(dim, dim-1)][entity]
                                          for s in simplices[dim-1][child]]

        simplices = list(chain.from_iterable(simplices[sd].values()))
        new_topology = {}
        new_topology[0] = {i: (i,) for i in range(len(new_verts))}
        for dim in range(1, sd):
            facets = chain.from_iterable((combinations(s, dim+1) for s in simplices))
            if dim < self.split_dimension:
                # Preserve the numbering of the unsplit entities
                facets = chain(top[dim].values(), facets)
            unique_facets = dict.fromkeys(facets)
            new_topology[dim] = dict(enumerate(unique_facets))
        new_topology[sd] = dict(enumerate(simplices))

        parent = ref_el if dimension == sd else PowellSabinSplit(ref_el, dimension=dimension+1)
        super().__init__(parent, tuple(new_verts), new_topology)

    def construct_subcomplex(self, dimension):
        """Constructs the reference subcomplex of the parent complex
        specified by subcomplex dimension.
        """
        if dimension == self.get_dimension():
            return self
        parent = self.get_parent_complex()
        subcomplex = parent.construct_subcomplex(dimension)
        if dimension < self.split_dimension:
            return subcomplex
        else:
            # Powell-Sabin on facets is Powell-Sabin
            return PowellSabinSplit(subcomplex, dimension=self.split_dimension)


class AlfeldSplit(PowellSabinSplit):
    """Splits a simplicial complex by connecting cell vertices to their
    barycenter.
    """
    def __new__(cls, ref_el):
        try:
            return ref_el._split_cache[cls]
        except KeyError:
            self = super().__new__(cls)
            return ref_el._split_cache.setdefault(cls, self)

    def __init__(self, ref_el):
        sd = ref_el.get_spatial_dimension()
        super().__init__(ref_el, dimension=sd)


class WorseyFarinSplit(PowellSabinSplit):
    """Splits a simplicial complex by connecting cell and facet vertices to their
    barycenter. This reduces to Powell-Sabin on the triangle, and Alfeld on the interval.
    """
    def __new__(cls, ref_el):
        try:
            return ref_el._split_cache[cls]
        except KeyError:
            self = super().__new__(cls)
            return ref_el._split_cache.setdefault(cls, self)

    def __init__(self, ref_el):
        sd = ref_el.get_spatial_dimension()
        super().__init__(ref_el, dimension=sd-1)


class PowellSabin12Split(SplitSimplicialComplex):
    """Splits a triangle (only!) by connecting each vertex to the opposite
    edge midpoint and edge midpoints to each other.
    """
    def __init__(self, ref_el):
        assert ref_el.get_shape() == TRIANGLE
        verts = ref_el.get_vertices()
        new_verts = list(verts)
        new_verts.extend(
            map(tuple, bary_to_xy(verts,
                [(1/3, 1/3, 1/3),
                 (1/2, 1/2, 0),
                 (1/2, 0, 1/2),
                 (0, 1/2, 1/2),
                 (1/2, 1/4, 1/4),
                 (1/4, 1/2, 1/4),
                 (1/4, 1/4, 1/2)])))

        edges = [(0, 4), (0, 7), (0, 5),
                 (1, 4), (1, 8), (1, 6),
                 (2, 5), (2, 9), (2, 6),
                 (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
                 (4, 7), (4, 8), (5, 7), (5, 9), (6, 8), (6, 9)]

        parent = PowellSabinSplit(ref_el)
        new_topology = make_topology(2, len(new_verts), edges)
        super().__init__(parent, tuple(new_verts), new_topology)

    def construct_subcomplex(self, dimension):
        """Constructs the reference subcomplex of the parent cell subentity
        specified by subcomplex dimension.
        """
        if dimension == 2:
            return self
        elif dimension == 1:
            return AlfeldSplit(self.construct_subelement(1))
        elif dimension == 0:
            return self.construct_subelement(0)
        else:
            raise ValueError("Illegal dimension")


class MacroQuadratureRule(QuadratureRule):
    """Composite quadrature rule on parent facets that respects the splitting.

    :arg ref_el: A simplicial complex.
    :arg Q_ref: A QuadratureRule on the reference simplex.
    :args parent_facets: An iterable of facets of the same dimension as Q_ref,
                         defaults to all facets.

    :returns: A quadrature rule on the sub entities of the simplicial complex.
    """
    def __init__(self, ref_el, Q_ref, parent_facets=None):
        parent_dim = Q_ref.ref_el.get_spatial_dimension()
        if parent_facets is not None:
            parent_to_children = ref_el.get_parent_to_children()
            facets = []
            for parent_entity in parent_facets:
                children = parent_to_children[parent_dim][parent_entity]
                facets.extend(entity for dim, entity in children if dim == parent_dim)
        else:
            top = ref_el.get_topology()
            facets = top[parent_dim]

        pts = []
        wts = []
        for entity in facets:
            Q_cur = FacetQuadratureRule(ref_el, parent_dim, entity, Q_ref)
            pts.extend(Q_cur.pts)
            wts.extend(Q_cur.wts)
        pts = tuple(pts)
        wts = tuple(wts)
        super().__init__(ref_el, pts, wts)


class CkPolynomialSet(polynomial_set.PolynomialSet):
    """Constructs a C^k-continuous PolynomialSet on a simplicial complex.

    :arg ref_el: The simplicial complex.
    :arg degree: The polynomial degree.
    :kwarg order: The order of continuity across facets or a dict of dicts
        mapping dimension and entity_id to the order of continuity at each entity.
    :kwarg vorder: The order of super-smoothness at interior vertices.
    :kwarg shape: The value shape.
    :kwarg variant: The variant for the underlying ExpansionSet.
    :kwarg scale: The scale for the underlying ExpansionSet.
    """
    def __init__(self, ref_el, degree, order=1, vorder=None, shape=(), **kwargs):
        from FIAT.quadrature_schemes import create_quadrature

        if not isinstance(order, (int, dict)):
            raise TypeError(f"'order' must be either an int or dict, not {type(order).__name__}")

        sd = ref_el.get_spatial_dimension()
        if isinstance(order, int):
            order = {sd-1: dict.fromkeys(ref_el.get_interior_facets(sd-1), order)}
        if vorder is not None:
            order[0] = dict.fromkeys(ref_el.get_interior_facets(0), vorder)
        elif 0 not in order:
            order[0] = {}

        if not all(k in {0, sd-1} for k in order):
            raise NotImplementedError("Only face or vertex constraints have been implemented.")

        expansion_set = expansions.ExpansionSet(ref_el, **kwargs)
        k = 1 if expansion_set.continuity == "C0" else 0

        # Impose C^forder continuity across interior facets
        facet_el = ref_el.construct_subelement(sd-1)
        phi_deg = 0 if sd == 1 else degree - k
        phi = polynomial_set.ONPolynomialSet(facet_el, phi_deg)
        Q = create_quadrature(facet_el, 2 * phi_deg)
        qpts, qwts = Q.get_points(), Q.get_weights()
        phi_at_qpts = phi.tabulate(qpts)[(0,) * (sd-1)]
        weights = numpy.multiply(phi_at_qpts, qwts)

        rows = []
        for facet in order[sd-1]:
            forder = order[sd-1][facet]
            jumps = expansion_set.tabulate_normal_jumps(degree, qpts, facet, order=forder)
            for r in range(k, forder+1):
                num_wt = 1 if sd == 1 else expansions.polynomial_dimension(facet_el, degree-r)
                rows.append(numpy.tensordot(weights[:num_wt], jumps[r], axes=(-1, -1)))

        # Impose C^vorder super-smoothness at interior vertices
        # C^forder automatically gives C^{forder+dim-1} at the interior vertex
        verts = numpy.asarray(ref_el.get_vertices())
        for vorder in set(order[0].values()):
            vids = [i for i in order[0] if order[0][i] == vorder]
            facets = chain.from_iterable(ref_el.connectivity[(0, sd-1)][v] for v in vids)
            forder = min(order[sd-1][f] for f in facets)
            sorder = forder + sd - 1
            if vorder > sorder:
                jumps = expansion_set.tabulate_jumps(degree, verts[vids], order=vorder)
                rows.extend(numpy.vstack(jumps[r].T) for r in range(sorder+1, vorder+1))

        if len(rows) > 0:
            for row in rows:
                row *= 1 / max(numpy.max(abs(row)), 1)
            dual_mat = numpy.vstack(rows)
            coeffs = polynomial_set.spanning_basis(dual_mat, nullspace=True)
        else:
            coeffs = numpy.eye(expansion_set.get_num_members(degree))

        if shape != ():
            m, n = coeffs.shape
            ncomp = numpy.prod(shape)
            coeffs = numpy.kron(coeffs, numpy.eye(ncomp))
            coeffs = coeffs.reshape(m*ncomp, *shape, n)

        super().__init__(ref_el, degree, degree, expansion_set, coeffs)


def hdiv_conforming_coefficients(U, order=0):
    """Constrains a PolynomialSet to have vanishing normal jumps on the
       interior facets of a simplicial complex.

    :arg U: A PolynomialSet, it may be vector- or tensor-valued.
    :arg order: The maximum order of differentiation on the normal jump constraints.

    :returns: The coefficients of the constrained PolynomialSet.
    """
    from FIAT.quadrature_schemes import create_quadrature
    degree = U.degree
    ref_el = U.get_reference_element()
    coeffs = U.get_coeffs()
    shape = U.get_shape()
    expansion_set = U.get_expansion_set()
    k = 1 if expansion_set.continuity == "C0" else 0

    sd = ref_el.get_spatial_dimension()
    facet_el = ref_el.construct_subelement(sd-1)

    phi_deg = 0 if sd == 1 else degree - k
    phi = polynomial_set.ONPolynomialSet(facet_el, phi_deg, shape=shape[1:])
    Q = create_quadrature(facet_el, 2 * phi_deg)
    qpts, qwts = Q.get_points(), Q.get_weights()
    phi_at_qpts = phi.tabulate(qpts)[(0,) * (sd-1)]
    weights = numpy.multiply(phi_at_qpts, qwts)
    ax = tuple(range(1, weights.ndim))

    rows = []
    for facet in ref_el.get_interior_facets(sd-1):
        normal = ref_el.compute_scaled_normal(facet)
        ncoeffs = numpy.tensordot(coeffs, normal, axes=(len(shape), 0))
        jumps = expansion_set.tabulate_normal_jumps(degree, qpts, facet, order=order)
        for r in range(k, order+1):
            njump = numpy.dot(ncoeffs, jumps[r])
            rows.append(numpy.tensordot(weights, njump, axes=(ax, ax)))

    if len(rows) > 0:
        dual_mat = numpy.vstack(rows)
        nsp = polynomial_set.spanning_basis(dual_mat, nullspace=True)
        coeffs = numpy.tensordot(nsp, coeffs, axes=(1, 0))
    return coeffs


class HDivPolynomialSet(polynomial_set.PolynomialSet):
    """Constructs a vector-valued PolynomialSet with continuous
       normal components on a simplicial complex.

    :arg ref_el: The simplicial complex.
    :arg degree: The polynomial degree.
    :kwarg order: The order of continuity across subcells.
    :kwarg variant: The variant for the underlying ExpansionSet.
    :kwarg scale: The scale for the underlying ExpansionSet.
    """
    def __init__(self, ref_el, degree, order=0, **kwargs):
        sd = ref_el.get_spatial_dimension()
        U = polynomial_set.ONPolynomialSet(ref_el, degree, shape=(sd,), **kwargs)
        coeffs = hdiv_conforming_coefficients(U, order=order)
        super().__init__(ref_el, degree, degree, U.expansion_set, coeffs)


class HDivSymPolynomialSet(polynomial_set.PolynomialSet):
    """Constructs a symmetric tensor-valued PolynomialSet with continuous
       normal components on a simplicial complex.

    :arg ref_el: The simplicial complex.
    :arg degree: The polynomial degree.
    :kwarg order: The order of continuity across subcells.
    :kwarg variant: The variant for the underlying ExpansionSet.
    :kwarg scale: The scale for the underlying ExpansionSet.
    """
    def __init__(self, ref_el, degree, order=0, **kwargs):
        U = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree, **kwargs)
        coeffs = hdiv_conforming_coefficients(U, order=order)
        super().__init__(ref_el, degree, degree, U.expansion_set, coeffs)


def pullback(phi, mapping, J=None, Jinv=None, Jdet=None):
    """Transforms a reference tabulation into physical space.

    :arg phi: The tabulation on reference space.
    :arg mapping: A string indicating the pullback type.
    :arg J: The Jacobian of the transformation from reference to physical space.
    :arg Jinv: The inverse of the Jacobian.
    :arg Jdet: The determinant of the Jacobian.

    :returns: The tabulation on physical space.
    """
    try:
        formdegree = {
            "affine": (0,),
            "covariant piola": (1,),
            "contravariant piola": (2,),
            "double covariant piola": (1, 1),
            "double contravariant piola": (2, 2),
            "covariant contravariant piola": (1, 2),
            "contravariant covariant piola": (2, 1), }[mapping]
    except KeyError:
        raise ValueError(f"Unrecognized mapping {mapping}")

    if J is None:
        J = numpy.linalg.pinv(Jinv)
    if Jinv is None:
        Jinv = numpy.linalg.pinv(J)
    if Jdet is None:
        Jdet = numpy.linalg.det(J)
    F1 = Jinv.T
    F2 = J / Jdet

    for i, k in enumerate(formdegree):
        if k == 0:
            continue
        F = F1 if k == 1 else F2

        perm = list(range(phi.ndim))
        perm[i+1], perm[-1] = perm[-1], perm[i+1]

        phi = phi.transpose(perm)
        phi = phi.dot(F.T)
        phi = phi.transpose(perm)

    return phi


class MacroPolynomialSet(polynomial_set.PolynomialSet):
    """Constructs a PolynomialSet by tiling a CiarletElement on a
       SimplicialComplex.

    :arg ref_el: The SimplicialComplex.
    :arg element: The CiarletElement.
    """
    def __init__(self, ref_el, element):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()

        mapping, = set(element.mapping())
        base_ref_el = element.get_reference_element()
        base_entity_ids = element.entity_dofs()
        n = element.degree()

        base_expansion_set = element.get_nodal_basis().get_expansion_set()
        expansion_set = base_expansion_set.reconstruct(ref_el=ref_el)

        shp = element.value_shape()
        num_bfs = expansions.polynomial_dimension(ref_el, n, base_entity_ids)
        num_members = expansion_set.get_num_members(n)
        coeffs = numpy.zeros((num_bfs, *shp, num_members))
        base_coeffs = element.get_coeffs()

        rmap = expansions.polynomial_cell_node_map(ref_el, n, base_entity_ids)
        cmap = expansion_set.get_cell_node_map(n)
        for cell in sorted(top[sd]):
            cell_verts = ref_el.get_vertices_of_subcomplex(top[sd][cell])
            A, b = reference_element.make_affine_mapping(base_ref_el.vertices, cell_verts)

            indices = numpy.ix_(rmap[cell], *map(range, shp), cmap[cell])
            coeffs[indices] = pullback(base_coeffs, mapping, J=A)

        super().__init__(ref_el, element.degree(), element.degree(), expansion_set, coeffs)
