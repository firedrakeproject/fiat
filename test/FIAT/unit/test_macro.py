import math
import numpy
import pytest
from FIAT import (Lagrange, DiscontinuousLagrange, IntegratedLegendre, Legendre, P0,
                  Nedelec, NedelecSecondKind, RaviartThomas, BrezziDouglasMarini,
                  Regge, HellanHerrmannJohnson, GopalakrishnanLedererSchoberlSecondKind,
                  CrouzeixRaviart)
from FIAT.macro import AlfeldSplit, IsoSplit, PowellSabinSplit, CkPolynomialSet, MacroPolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import ufc_simplex, UFCSimplex
from FIAT.expansions import polynomial_entity_ids, polynomial_cell_node_map
from FIAT.polynomial_set import make_bubbles, PolynomialSet, ONPolynomialSet
from FIAT.barycentric_interpolation import get_lagrange_points


@pytest.fixture(params=(1, 2, 3), ids=("I", "T", "S"))
def cell(request):
    dim = request.param
    return ufc_simplex(dim)


def test_split_cache(cell):
    A = AlfeldSplit(cell)
    B = AlfeldSplit(cell)
    assert B is A
    fe = Lagrange(cell, 1, variant="alfeld")
    C = fe.ref_complex
    assert C is A


@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_split_entity_transform(split, cell):
    split_cell = split(cell)
    top = split_cell.get_topology()
    sdim = cell.get_spatial_dimension()
    for dim in range(1, sdim+1):
        ref_el = split_cell.construct_subelement(dim)
        b = numpy.average(ref_el.get_vertices(), axis=0)
        for entity in top[dim]:
            mapped_bary = split_cell.get_entity_transform(dim, entity)(b)
            computed_bary = numpy.average(split_cell.get_vertices_of_subcomplex(top[dim][entity]), axis=0)
            assert numpy.allclose(mapped_bary, computed_bary)


@pytest.mark.parametrize("degree", (4,))
@pytest.mark.parametrize("variant", ("gll", "equispaced"))
@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_split_make_points(split, cell, degree, variant):
    split_cell = split(cell)
    top = split_cell.get_topology()
    sdim = cell.get_spatial_dimension()
    for i in range(1, sdim+1):
        ref_el = split_cell.construct_subelement(i)
        pts_ref = ref_el.make_points(i, 0, degree, variant=variant)
        for entity in top[i]:
            pts_entity = split_cell.make_points(i, entity, degree, variant=variant)
            mapping = split_cell.get_entity_transform(i, entity)
            mapped_pts = mapping(pts_ref)
            assert numpy.allclose(mapped_pts, pts_entity)


def test_split_child_to_parent(cell):
    split_cell = IsoSplit(cell)

    dim = cell.get_spatial_dimension()
    degree = 2 if dim == 3 else 4
    parent_degree = 2*degree

    top = cell.get_topology()
    parent_pts = {dim: {} for dim in top}
    for dim in top:
        for entity in top[dim]:
            parent_pts[dim][entity] = cell.make_points(dim, entity, parent_degree)

    top = split_cell.get_topology()
    child_pts = {dim: {} for dim in top}
    for dim in top:
        for entity in top[dim]:
            child_pts[dim][entity] = split_cell.make_points(dim, entity, degree)

    child_to_parent = split_cell.get_child_to_parent()
    for dim in top:
        for entity in top[dim]:
            parent_dim, parent_entity = child_to_parent[dim][entity]
            assert set(child_pts[dim][entity]) <= set(parent_pts[parent_dim][parent_entity])


@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_macro_quadrature(split, cell):
    ref_el = split(cell)
    sd = ref_el.get_spatial_dimension()

    degree = 3
    Q = create_quadrature(ref_el, 2*degree)
    pts, wts = Q.get_points(), Q.get_weights()

    # Test that the mass matrix for an orthogonal basis is diagonal
    fe = Legendre(ref_el, degree)
    phis = fe.tabulate(0, pts)[(0,)*sd]
    M = numpy.dot(numpy.multiply(phis, wts), phis.T)
    M = M - numpy.diag(M.diagonal())
    assert numpy.allclose(M, 0)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("variant", ("equispaced", "gll"))
@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_macro_lagrange(variant, degree, split, cell):
    ref_el = split(cell)

    fe = Lagrange(ref_el, degree, variant=variant)
    poly_set = fe.get_nodal_basis()

    # Test that the polynomial set is defined on the split and not on the parent cell
    assert poly_set.get_reference_element() is ref_el

    # Test that the finite element is defined on the parent cell and not on the split
    assert fe.get_reference_element() is cell

    # Test that parent entities are the ones exposed
    entity_ids = fe.entity_dofs()
    parent_top = ref_el.get_parent().get_topology()
    for dim in parent_top:
        assert len(entity_ids[dim]) == len(parent_top[dim])

    # Test that tabulation onto lattice points gives the identity
    sd = ref_el.get_spatial_dimension()
    parent_to_children = ref_el.get_parent_to_children()
    pts = []
    for dim in sorted(parent_to_children):
        for entity in sorted(parent_to_children[dim]):
            for cdim, centity in parent_to_children[dim][entity]:
                pts.extend(ref_el.make_points(cdim, centity, degree, variant=variant))

    phis = fe.tabulate(2, pts)
    assert numpy.allclose(phis[(0,)*sd], numpy.eye(fe.space_dimension()))

    # Test that we can reproduce the Vandermonde matrix by tabulating the expansion set
    U = poly_set.get_expansion_set()
    V = U.tabulate(degree, pts).T
    assert numpy.allclose(fe.V, V)


def test_powell_sabin(cell):
    dim = cell.get_spatial_dimension()
    A = AlfeldSplit(cell)
    assert A > cell

    PS = PowellSabinSplit(cell, dim)
    assert PS == A

    for split_dim in range(1, dim):
        PS = PowellSabinSplit(cell, split_dim)
        assert PS > A
        assert PS > cell
        assert len(PS.get_topology()[dim]) == math.factorial(dim+1) // math.factorial(split_dim)


def make_mass_matrix(fe, order=0):
    sd = fe.ref_el.get_spatial_dimension()
    Q = create_quadrature(fe.ref_complex, 2*fe.degree())
    qpts, qwts = Q.get_points(), Q.get_weights()
    phi = fe.tabulate(order, qpts)[(0,) * sd]
    M = numpy.dot(numpy.multiply(phi, qwts), phi.T)
    return M


@pytest.mark.parametrize("degree", (1, 2, 4))
@pytest.mark.parametrize("variant", ("equispaced", "gll"))
def test_lagrange_alfeld_duals(cell, degree, variant):
    Pk = Lagrange(cell, degree, variant=variant)
    alfeld = Lagrange(AlfeldSplit(cell), degree, variant=variant)

    Pk_pts = numpy.asarray(get_lagrange_points(Pk.dual_basis()))
    alfeld_pts = numpy.asarray(get_lagrange_points(alfeld.dual_basis()))
    ids = alfeld.entity_dofs()

    sd = cell.get_spatial_dimension()
    facet_dim = sum(len(ids[dim][entity]) for dim in range(sd) for entity in ids[dim])
    assert numpy.allclose(alfeld_pts[:facet_dim], Pk_pts[:facet_dim])

    phi = Pk.tabulate(0, alfeld_pts)[(0,) * sd]
    M_Pk = make_mass_matrix(Pk)
    M_alfeld = make_mass_matrix(alfeld)
    M_galerkin = numpy.dot(numpy.dot(phi, M_alfeld), phi.T)
    assert numpy.allclose(M_Pk, M_galerkin)


@pytest.mark.parametrize("degree", (1, 2, 4))
def test_lagrange_iso_duals(cell, degree):
    Pk = Lagrange(cell, 2*degree, variant="equispaced")
    Piso = Lagrange(IsoSplit(cell), degree, variant="equispaced")

    Pk_pts = numpy.asarray(get_lagrange_points(Pk.dual_basis()))
    Piso_pts = numpy.asarray(get_lagrange_points(Piso.dual_basis()))
    ids = Piso.entity_dofs()

    reorder = []
    for dim in ids:
        for entity in ids[dim]:
            reorder.extend(ids[dim][entity])
    assert numpy.allclose(Piso_pts[reorder], Pk_pts)

    poly_set = Piso.get_nodal_basis().take(reorder)
    assert numpy.allclose(numpy.eye(Piso.space_dimension()),
                          numpy.dot(Pk.get_dual_set().to_riesz(poly_set),
                                    poly_set.get_coeffs().T))


@pytest.mark.parametrize("variant", ("gll", "Alfeld,equispaced", "gll,iso"))
def test_is_macro_lagrange(variant):
    is_macro = "alfeld" in variant.lower() or "iso" in variant.lower()

    fe = Lagrange(ufc_simplex(2), 2, variant)
    assert not fe.get_reference_element().is_macrocell()
    assert fe.is_macroelement() == is_macro
    assert fe.get_reference_complex().is_macrocell() == is_macro
    assert fe.get_nodal_basis().get_reference_element().is_macrocell() == is_macro


@pytest.mark.parametrize("variant", ("gl", "Alfeld,equispaced_interior", "chebyshev,iso"))
@pytest.mark.parametrize("degree", (0, 2))
def test_is_macro_discontinuous_lagrange(degree, variant):
    is_macro = "alfeld" in variant.lower() or "iso" in variant.lower()

    fe = DiscontinuousLagrange(ufc_simplex(2), degree, variant)
    if degree == 0 and not is_macro:
        assert isinstance(fe, P0)
    assert not fe.get_reference_element().is_macrocell()
    assert fe.is_macroelement() == is_macro
    assert fe.get_reference_complex().is_macrocell() == is_macro
    assert fe.get_nodal_basis().get_reference_element().is_macrocell() == is_macro


@pytest.mark.parametrize('split', [None, AlfeldSplit])
@pytest.mark.parametrize('codim', range(3))
def test_make_bubbles(cell, split, codim):
    sd = cell.get_spatial_dimension()
    if codim > sd:
        return
    degree = 5
    if split is not None:
        cell = split(cell)
    B = make_bubbles(cell, degree, codim=codim)

    # basic tests
    assert isinstance(B, PolynomialSet)
    assert B.degree == degree
    num_members = B.get_num_members()
    top = cell.get_topology()
    assert num_members == math.comb(degree-1, sd-codim) * len(top[sd - codim])

    # tabulate onto a lattice
    points = []
    for dim in range(sd+1-codim):
        for entity in sorted(top[dim]):
            points.extend(cell.make_points(dim, entity, degree))
    values = B.tabulate(points)[(0,) * sd]

    # test that bubbles vanish on the boundary
    num_pts_on_facet = len(points) - num_members
    facet_values = values[:, :num_pts_on_facet]
    assert numpy.allclose(facet_values, 0, atol=1E-12)

    # test linear independence
    interior_values = values[:, num_pts_on_facet:]
    assert numpy.linalg.matrix_rank(interior_values.T, tol=1E-12) == num_members

    # test block diagonal tabulation
    bubbles_per_entity = num_members // len(top[sd-codim])
    for entity in top[sd-codim]:
        i0 = entity * bubbles_per_entity
        i1 = (entity+1) * bubbles_per_entity
        assert numpy.allclose(interior_values[i0:i1, :i0], 0, atol=1E-12)
        assert numpy.allclose(interior_values[i0:i1, i1:], 0, atol=1E-12)

    # test trace similarity
    dim = sd - codim
    nfacets = len(top[dim])
    if nfacets > 1 and dim > 0:
        ref_facet = cell.construct_subelement(dim)
        ref_bubbles = make_bubbles(ref_facet, degree)
        ref_points = ref_facet.make_points(dim, 0, degree)
        ref_values = ref_bubbles.tabulate(ref_points)[(0,) * dim]

        scale = None
        bubbles_per_entity = ref_bubbles.get_num_members()
        cur = 0
        for entity in sorted(top[dim]):
            indices = list(range(cur, cur + bubbles_per_entity))
            cur_values = interior_values[numpy.ix_(indices, indices)]
            if scale is None:
                scale = numpy.max(abs(cur_values)) / numpy.max(abs(ref_values))
            assert numpy.allclose(ref_values * scale, cur_values)
            cur += bubbles_per_entity

    # test that bubbles do not have components in span(P_{degree+2} \ P_{degree})
    Pkdim = math.comb(degree + sd, sd)
    entity_ids = polynomial_entity_ids(cell, degree + 2)
    indices = []
    for entity in top[sd]:
        indices.extend(entity_ids[sd][entity][Pkdim:])
    P = ONPolynomialSet(cell, degree + 2)
    P = P.take(indices)

    Q = create_quadrature(cell, P.degree + B.degree)
    qpts, qwts = Q.get_points(), Q.get_weights()
    P_at_qpts = P.tabulate(qpts)[(0,) * sd]
    B_at_qpts = B.tabulate(qpts)[(0,) * sd]
    assert numpy.allclose(numpy.dot(numpy.multiply(P_at_qpts, qwts), B_at_qpts.T), 0.0)


@pytest.mark.parametrize("degree", (4,))
@pytest.mark.parametrize("variant", (None, "bubble"))
@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_macro_expansion(cell, split, variant, degree):
    ref_complex = split(cell)
    top = ref_complex.get_topology()
    sd = ref_complex.get_spatial_dimension()
    P = ONPolynomialSet(ref_complex, degree, variant=variant, scale=1)

    npoints = degree + sd + 1
    cell_point_map = []
    pts = []
    for cell in top[sd]:
        cur = len(pts)
        pts.extend(ref_complex.make_points(sd, cell, npoints))
        cell_point_map.append(list(range(cur, len(pts))))

    order = 2
    values = P.tabulate(pts, order)
    cell_node_map = polynomial_cell_node_map(ref_complex, degree, continuity=P.expansion_set.continuity)
    for cell in top[sd]:
        sub_el = ref_complex.construct_subelement(sd)
        sub_el.vertices = ref_complex.get_vertices_of_subcomplex(top[sd][cell])
        Pcell = ONPolynomialSet(sub_el, degree, variant=variant, scale=1)

        cell_pts = sub_el.make_points(sd, 0, npoints)
        cell_values = Pcell.tabulate(cell_pts, order)

        ibfs = cell_node_map[cell]
        ipts = cell_point_map[cell]
        indices = numpy.ix_(ibfs, ipts)
        for alpha in values:
            assert numpy.allclose(cell_values[alpha], values[alpha][indices])


@pytest.mark.parametrize("order", (0, 1))
@pytest.mark.parametrize("variant", (None, "bubble"))
@pytest.mark.parametrize("degree", (1, 4))
def test_Ck_basis(cell, order, degree, variant):
    # Test that we can correctly tabulate on points on facets.
    # This breaks if we were binning points into more than one cell without a partition of unity.
    # It suffices to tabulate on the vertices of the simplicial complex.
    A = AlfeldSplit(cell)
    Ck = CkPolynomialSet(A, degree, order=order, variant=variant)
    U = Ck.get_expansion_set()
    cell_node_map = U.get_cell_node_map(degree)

    sd = A.get_spatial_dimension()
    top = A.get_topology()
    coeffs = Ck.get_coeffs()
    phis = Ck.tabulate(A.get_vertices())[(0,)*sd]

    for cell in top[sd]:
        ipts = list(top[sd][cell])
        verts = A.get_vertices_of_subcomplex(top[sd][cell])
        Uvals = U._tabulate_on_cell(degree, verts, 0, cell=cell)[(0,)*sd]
        local_phis = numpy.dot(coeffs[:, cell_node_map[cell]], Uvals)
        assert numpy.allclose(local_phis, phis[:, ipts])


def test_C2_double_alfeld():
    # Construct the quintic C2 spline on the double Alfeld split
    # See Section 7.5 of Lai & Schumacher
    K = ufc_simplex(2)
    DCT = AlfeldSplit(AlfeldSplit(K))

    degree = 5

    # C3 on major split facets, C2 elsewhere
    order = {}
    order[1] = dict.fromkeys(DCT.get_interior_facets(1), 2)
    order[1].update(dict.fromkeys(range(3, 6), 3))

    # C4 at minor split barycenters, C3 at major split barycenter
    order[0] = dict.fromkeys(DCT.get_interior_facets(0), 4)
    order[0][3] = 3

    P = CkPolynomialSet(DCT, degree, order=order, variant="bubble")
    assert P.get_num_members() == 27


def test_distance_to_point_l1(cell):
    A = AlfeldSplit(cell)
    dim = A.get_spatial_dimension()
    top = A.get_topology()
    p0, = cell.make_points(dim, 0, dim+1)

    # construct one point in front of each facet
    pts = []
    expected = []
    parent_top = cell.get_topology()
    for i in parent_top[dim-1]:
        Fi, = numpy.asarray(cell.make_points(dim-1, i, dim))
        n = cell.compute_normal(i)
        n *= numpy.dot(n, Fi - p0)
        n /= numpy.linalg.norm(n)
        d = 0.222 + i/10
        pts.append(Fi + d * n)
        expected.append(d)

    # the computed L1 distance agrees with the L2 distance for points in front of facets
    parent_distance = cell.distance_to_point_l1(pts, rescale=True)
    assert numpy.allclose(parent_distance, expected)

    # assert that the subcell measures the same distance as the parent
    for i in top[dim]:
        subcell_distance = A.distance_to_point_l1(pts, entity=(dim, i), rescale=True)
        assert numpy.isclose(subcell_distance[i], expected[i])
        assert all(subcell_distance[:i] > expected[:i])
        assert all(subcell_distance[i+1:] > expected[i+1:])


@pytest.mark.parametrize("element", (DiscontinuousLagrange, Lagrange))
def test_macro_sympy(cell, element):
    import sympy
    variant = "spectral,alfeld"
    K = IsoSplit(cell)
    ebig = element(K, 3, variant=variant)
    pts = get_lagrange_points(ebig.dual_basis())

    dim = cell.get_spatial_dimension()
    X = tuple(sympy.Symbol(f"X[{i}]") for i in range(dim))
    degrees = range(1, 3) if element is Lagrange else range(3)
    for degree in degrees:
        fe = element(cell, degree, variant=variant)
        tab_sympy = fe.tabulate(0, X)[(0,) * dim]

        phis = sympy.lambdify(X, tab_sympy)
        results = phis(*numpy.transpose(pts))
        tab_numpy = fe.tabulate(0, pts)[(0,) * dim]
        assert numpy.allclose(results, tab_numpy)


@pytest.mark.parametrize("element,degree", [
    (Lagrange, 1), (Nedelec, 1), (RaviartThomas, 1), (DiscontinuousLagrange, 0),
    (Regge, 0), (HellanHerrmannJohnson, 0),
    (GopalakrishnanLedererSchoberlSecondKind, 0)])
@pytest.mark.parametrize("dim", (2, 3))
def test_macro_polynomial_set(dim, element, degree):
    K = ufc_simplex(dim)
    A = IsoSplit(K)

    fe = element(K, degree)
    mapping = fe.mapping()[0]
    fdim = fe.formdegree
    if isinstance(fdim, tuple):
        fdim = max(fdim)
    comps = []
    pts = []
    for entity in A.topology[fdim]:
        pts_cur = A.make_points(fdim, entity, 1+fdim)
        pts.extend(pts_cur)

        if mapping == "affine":
            comp = numpy.ones(())
        elif mapping == "covariant contravariant piola":
            ts = A.compute_tangents(fdim, entity)
            n = A.compute_scaled_normal(entity)
            comp = ts[..., None] * n[None, None, :]

        elif mapping.endswith("covariant piola"):
            comp = A.compute_edge_tangent(entity)
        elif mapping.endswith("contravariant piola"):
            comp = A.compute_scaled_normal(entity)
        if mapping.startswith("double"):
            comp = numpy.outer(comp, comp)

        comps.extend(comp for pt in pts_cur)

    P = MacroPolynomialSet(A, fe)
    phis = P.tabulate(pts)[(0,)*dim]
    shape = phis.shape
    shp = shape[1:-1]
    ncomp = comp.shape[:-len(shp)]

    result = numpy.zeros((shape[0], *ncomp, shape[-1]))
    ax = (tuple(range(-len(shp), 0)), )*2
    for i, comp in enumerate(comps):
        result[..., i] = numpy.tensordot(comp, phis[..., i], ax).T
    result[abs(result) < 1E-14] = 0
    result = result[:len(pts)*numpy.prod(ncomp, dtype=int)]

    if ncomp:
        result = result.transpose((*range(1, len(ncomp)+1), 0, -1))
        result = result.reshape((shape[0], -1))
    assert numpy.allclose(numpy.diag(numpy.diag(result)), result)


def compare_macro_variant(element, K, degree, variant):
    """Test CiarletElement by comparing macro variant to their non-macro
       counterparts."""
    # Test unisolvence
    fe_macro = element(K, degree, variant=variant)

    # Ensure that we have a macroelement
    ref_complex = fe_macro.get_reference_complex()
    assert ref_complex.is_macrocell()
    top = ref_complex.topology
    dim = ref_complex.get_spatial_dimension()
    assert len(top[dim]) > 1

    # Construct the non-macro element on a subcell
    subcell = max(top[dim])
    vids = list(top[dim][subcell])
    subtop = {d: dict(enumerate(tuple(map(vids.index, top[d][e]))
                                for e in top[d]
                                if set(top[d][e]) <= set(vids)))
              for d in sorted(top)}
    vertices = ref_complex.get_vertices_of_subcomplex(vids)
    Ksub = UFCSimplex(K.shape, vertices, subtop)
    fe_ref = element(Ksub, degree)

    # Compute Vandermonde matrix on the subcell
    P = fe_macro.get_nodal_basis()
    B = P.get_coeffs()
    A = fe_ref.dual.to_riesz(P)
    V = numpy.tensordot(A, B, axes=(range(1, A.ndim), range(1, B.ndim)))

    # Assert that V = permutation matrix
    V[abs(V) < 1E-12] = 0
    rows, cols = numpy.nonzero(V)
    assert len(rows) == fe_ref.space_dimension()
    assert numpy.allclose(V[rows, cols], 1)

    # Test tabulation on interior points
    pts = Ksub.make_points(dim, 0, degree+dim, interior=1)
    tab_macro = fe_macro.tabulate(1, pts)
    tab_ref = fe_ref.tabulate(1, pts)
    for alpha in tab_ref:
        expected = tab_ref[alpha][rows]
        result = tab_macro[alpha][cols]
        assert numpy.allclose(result, expected)


@pytest.mark.parametrize("element,degree", [
    (IntegratedLegendre, 4), (Legendre, 1),
    (Nedelec, 3), (NedelecSecondKind, 3),
    (RaviartThomas, 2), (BrezziDouglasMarini, 2),
    (Regge, 3), (HellanHerrmannJohnson, 1),
    (GopalakrishnanLedererSchoberlSecondKind, 1),
])
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("variant", ("alfeld", "iso"))
def test_macro_variants(dim, element, degree, variant):
    K = ufc_simplex(dim)
    compare_macro_variant(element, K, degree, variant)


@pytest.mark.parametrize("element,degree", [
    (CrouzeixRaviart, 3),
])
@pytest.mark.parametrize("variant", ("alfeld", "iso"))
def test_macro_variants_triangle(element, degree, variant):
    K = ufc_simplex(2)
    compare_macro_variant(element, K, degree, variant)
