import pytest
import numpy
from math import factorial

from FIAT import (BrezziDouglasMarini as BDM,
                  HsiehCloughTocher as HCT,
                  AlfeldSorokina as AS,
                  ArnoldQin as AQ,
                  Lagrange as CG,
                  DiscontinuousLagrange as DG)
from FIAT.reference_element import ufc_simplex, symmetric_simplex

from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import CkPolynomialSet
from FIAT.alfeld_sorokina import AlfeldSorokinaSpace
from FIAT.arnold_qin import ArnoldQinSpace
from FIAT.bernardi_raugel import BernardiRaugel
from FIAT.christiansen_hu import ChristiansenHuSpace
from FIAT.guzman_neilan import (
    GuzmanNeilanFirstKindH1,
    GuzmanNeilanSecondKindH1,
    GuzmanNeilanH1div,
    GuzmanNeilanSpace)
from FIAT.restricted import RestrictedElement
from FIAT.quadrature import FacetQuadratureRule

T = ufc_simplex(2)
S = ufc_simplex(3)


def inner(u, v, qwts):
    return numpy.tensordot(numpy.multiply(u, qwts), v, axes=(tuple(range(1, u.ndim)), )*2)


def rHCT(cell):
    return RestrictedElement(HCT(cell, reduced=True), restriction_domain="vertex")


def rAQ(cell):
    return RestrictedElement(AQ(cell, reduced=True), indices=list(range(9)))


def span_greater_equal(A, B):
    # span(A) >= span(B)
    A = A.reshape(A.shape[0], -1).T
    B = B.reshape(B.shape[0], -1).T
    x, *_ = numpy.linalg.lstsq(A, B)
    return numpy.allclose(A @ x, B)


def span_equal(A, B):
    # span(A) == span(B)
    return span_greater_equal(A, B) and span_greater_equal(B, A)


def div(U):
    """Return divergence from dict of tabulations """
    return sum(U[k][:, k.index(1), :] for k in U if sum(k) == 1)


def rot(U):
    """Return rot from dict of tabulations """
    return numpy.stack([U[(0, 1)], -U[(1, 0)]], axis=1)


def make_points(K, degree):
    top = K.get_topology()
    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(K.make_points(dim, entity, degree))
    return pts


def check_h1div_space(V, degree, reduced=False, bubble=False):
    # Test that the divergence of the polynomial space V is spanned by a C0 basis
    A = V.get_reference_element()
    sd = A.get_spatial_dimension()
    z = (0,)*sd

    pts = make_points(A, degree+2)
    V_tab = V.tabulate(pts, 1)
    V_div = div(V_tab)

    C0 = CkPolynomialSet(A, degree-1, order=0, variant="bubble")
    C0_tab = C0.tabulate(pts)[z]
    assert span_equal(V_div, C0_tab)

    if bubble:
        # Test that div(Bubbles) = C0 int H^1_0
        assert span_equal(V_div[-(sd+1):], C0_tab[-1:])

    k = degree - 1 if reduced else degree
    # Test that V includes Pk
    cell = A.get_parent() or A
    Pk = ONPolynomialSet(cell, k, shape=(sd,))
    Pk_tab = Pk.tabulate(pts)[z]
    assert span_greater_equal(V_tab[z], Pk_tab)


@pytest.mark.parametrize("cell", (T, S))
@pytest.mark.parametrize("degree", (2, 3, 4))
def test_h1div_alfeld_sorokina(cell, degree):
    # Test that the divergence of the Alfeld-Sorokina space is spanned by a C0 basis
    V = AlfeldSorokinaSpace(cell, degree)
    check_h1div_space(V, degree)


@pytest.mark.parametrize("cell", (S,))
@pytest.mark.parametrize("reduced", (False, True), ids=("full", "reduced"))
def test_h1div_guzman_neilan(cell, reduced):
    # Test that the divergence of AS + GN Bubble is spanned by a C0 basis
    sd = cell.get_spatial_dimension()
    degree = 2
    fe = GuzmanNeilanH1div(cell, degree, reduced=reduced)
    reduced_dim = fe.space_dimension() - (sd-1)*(sd+1)
    V = fe.get_nodal_basis().take(list(range(reduced_dim)))
    check_h1div_space(V, degree, reduced=reduced, bubble=True)


def check_stokes_complex(spaces, degree):
    # Test that we have a discrete Stokes complex, verifying that the range of
    # the exterior derivative of each space is contained by the next space in
    # the sequence
    A = spaces[0].get_reference_complex()
    sd = A.get_spatial_dimension()
    z = (0,) * sd

    pts = make_points(A, degree+2)
    tab = [V.tabulate(1, pts) for V in spaces]
    if len(tab) > 2:
        # check rot(V0) in V1
        assert span_greater_equal(tab[1][z], rot(tab[0]))

    # check div(V1) = V2
    assert span_equal(tab[-1][z], div(tab[-2]))

    # Test that V1 includes Pk
    cell = A.get_parent() or A
    Pk = ONPolynomialSet(cell, degree, shape=(sd,))
    Pk_tab = Pk.tabulate(pts)[z]
    assert span_greater_equal(tab[-2][z], Pk_tab)


@pytest.mark.parametrize("reduced", (False, True), ids=("full", "reduced"))
@pytest.mark.parametrize("sobolev", ("H1", "H1div"))
@pytest.mark.parametrize("cell", (T,))
def test_hct_stokes_complex(cell, sobolev, reduced):
    if sobolev == "H1":
        if reduced:
            spaces = [rHCT(cell), rAQ(cell), DG(cell, 0)]
            degree = 1
        else:
            spaces = [HCT(cell), AQ(cell), DG(cell, 0)]
            degree = 1
    elif sobolev == "H1div":
        if reduced:
            spaces = [rHCT(cell),
                      GuzmanNeilanH1div(cell, reduced=True),
                      CG(cell, 1, variant="alfeld")]
            degree = 1
        else:
            spaces = [HCT(cell), AS(cell), CG(cell, 1, variant="alfeld")]
            degree = 2
    else:
        raise ValueError(f"Unexpected sobolev space {sobolev}")
    check_stokes_complex(spaces, degree)


@pytest.mark.parametrize("cell", (T, S))
@pytest.mark.parametrize("kind", (1, 2, "H1div", "H1div-red"))
def test_gn_stokes_pairs(cell, kind):
    order = cell.get_spatial_dimension() - 1
    if kind == 1:
        spaces = [GuzmanNeilanFirstKindH1(cell, order), DG(cell, order-1)]
        degree = order
    elif kind == 2:
        spaces = [GuzmanNeilanSecondKindH1(cell, order), DG(cell, order-1, variant="alfeld")]
        degree = order
    elif kind == "H1div":
        spaces = [GuzmanNeilanH1div(cell), CG(cell, 1, variant="alfeld")]
        degree = 2
    elif kind == "H1div-red":
        spaces = [GuzmanNeilanH1div(cell, reduced=True), CG(cell, 1, variant="alfeld")]
        degree = 1
    else:
        raise ValueError(f"Unexpected kind {kind}")
    check_stokes_complex(spaces, degree)


@pytest.mark.parametrize("element", (GuzmanNeilanFirstKindH1, BernardiRaugel))
@pytest.mark.parametrize("quad_scheme", (None, "KMV(2),powell-sabin"))
@pytest.mark.parametrize("sd", (2, 3))
def test_gn_dofs(element, sd, quad_scheme):
    cell = symmetric_simplex(sd)

    fe = element(cell, 1, quad_scheme=quad_scheme)
    degree = fe.degree()
    assert degree == sd
    ref_complex = fe.get_reference_complex()
    Q = create_quadrature(ref_complex, degree-1)
    wts = Q.get_weights()
    tab = fe.tabulate(1, Q.get_points())

    div_moments = numpy.dot(div(tab), wts)
    expected = numpy.zeros(div_moments.shape)

    entity_ids = fe.entity_dofs()
    for f in entity_ids[sd-1]:
        fdof = entity_ids[sd-1][f][0]
        expected[fdof] = (-1.0)**f / factorial(sd-1)

    assert numpy.allclose(div_moments, expected)


@pytest.mark.parametrize("element", (GuzmanNeilanFirstKindH1, BernardiRaugel))
@pytest.mark.parametrize("cell", (S, symmetric_simplex(3)))
def test_interior_bubbles(element, cell):
    # Test that the interior bubbles control the gradient of DG_{k-1}
    sd = cell.get_spatial_dimension()
    order = 2
    fe = element(cell, order)
    reduced_dim = fe.space_dimension() - (sd-1)*(sd+1)
    assert reduced_dim == 37

    # Test unisolvence using the public dual and polynomial basis APIs.
    basis = fe.get_nodal_basis()
    dual = fe.get_dual_set().to_riesz(basis)
    vandermonde = numpy.tensordot(dual, basis.get_coeffs(), axes=((1, 2), (1, 2)))
    assert numpy.allclose(vandermonde, numpy.eye(fe.space_dimension()))

    ref_complex = fe.get_reference_complex()
    Q = create_quadrature(ref_complex, 2*fe.degree())
    qpts, qwts = Q.get_points(), Q.get_weights()
    tab = fe.tabulate(1, qpts)

    # The divergence of the interior bubbles against the mean-free DG_{k-1} is nonsingular
    P = ONPolynomialSet(cell, order-1).tabulate(qpts)[(0,)*sd]
    P = P[1:] - numpy.dot(P[1:], qwts)[:, None] / sum(qwts)
    ids = fe.entity_dofs()[sd][0]
    assert len(ids) == len(P)
    B = inner(div(tab)[ids], P, qwts)
    assert numpy.linalg.matrix_rank(B) == len(P)


def modified_bubbles(cell, pts):
    """Tabulate M_d(K^r) = hat^d P0^d + sum_k hat^(d-k) BDM_k(facet) from (Guzman and Neilan, 2019)."""
    sd = cell.get_spatial_dimension()
    z = (0,)*sd
    CG1 = CG(cell, 1, variant="alfeld")
    barycenter = CG1.entity_dofs()[sd][0][0]
    hat = CG1.tabulate(0, pts)[z][barycenter]
    bubbles = [numpy.eye(sd)[:, :, None] * hat ** sd]
    for k in range(1, sd):
        fe = BDM(cell, k)
        facet_ids = [i for f in fe.entity_dofs()[sd-1].values() for i in f]
        bubbles.append(fe.tabulate(0, pts)[z][facet_ids] * hat ** (sd-k))
    return numpy.concatenate(bubbles)


@pytest.mark.parametrize("cell", (S, symmetric_simplex(3)))
def test_gn_interior_bubbles(cell):
    # Test that the interior bubbles vanish on the boundary, lie in the modified
    # bubble space, and invert the divergence from P1 / P0
    sd = cell.get_spatial_dimension()
    order = 2
    fe = GuzmanNeilanFirstKindH1(cell, order)
    ids = fe.entity_dofs()[sd][0]
    z = (0,)*sd

    for f in cell.get_topology()[sd-1]:
        pts = cell.make_points(sd-1, f, fe.degree()+sd)
        assert numpy.allclose(fe.tabulate(0, pts)[z][ids], 0)

    Q = create_quadrature(fe.get_reference_complex(), 2*fe.degree())
    qpts, qwts = Q.get_points(), Q.get_weights()
    tab = fe.tabulate(1, qpts)
    M = modified_bubbles(cell, qpts)
    assert numpy.linalg.matrix_rank(M.reshape(len(M), -1)) == len(M) == 39
    assert span_greater_equal(M, tab[z][ids])

    P = ONPolynomialSet(cell, order-1).tabulate(qpts)[z]
    P = P[1:] - numpy.dot(P[1:], qwts)[:, None] / sum(qwts)
    assert span_equal(P, div(tab)[ids])


@pytest.mark.parametrize("reduced", (False, True), ids=("full", "reduced"))
def test_gn_facet_bubbles(reduced):
    # Test that the facet bubbles, which are numbered last, have constant divergence
    sd = S.get_spatial_dimension()
    V = GuzmanNeilanSpace(S, 2, kind=2, reduced=reduced)
    num_facets = len(S.get_topology()[sd-1])
    num_bubbles = num_facets if reduced else sd * num_facets

    pts = make_points(V.get_reference_element(), 4)
    divV = div(V.tabulate(pts, 1))[-num_bubbles:]
    assert numpy.allclose(divV, divV[:, :1])


def test_gn1_order2_facet_restriction():
    gn = GuzmanNeilanFirstKindH1(S, order=2)
    # Keep normal facet moments, excluding tangential constraints.
    facet_dofs = [dofs[0] for dofs in gn.entity_dofs()[S.get_spatial_dimension() - 1].values()]
    V = RestrictedElement(gn, indices=facet_dofs)
    pts = make_points(V.get_reference_complex(), 4)
    pressure = DG(S, 0).tabulate(0, pts)[(0,)*S.get_spatial_dimension()]
    assert span_equal(div(V.tabulate(1, pts)), pressure)


@pytest.mark.parametrize("reduced", (False, True), ids=("full", "reduced"))
@pytest.mark.parametrize("kind", (1, 2))
def test_gn_nested(kind, reduced):
    # Test that the lowest-order space, with constant divergence, is strictly contained in the next order
    V1 = GuzmanNeilanSpace(S, 1, kind=kind, reduced=reduced)
    V2 = GuzmanNeilanSpace(S, 2, kind=kind, reduced=reduced)
    Q = create_quadrature(V2.get_reference_element(), 2*V2.degree)
    z = (0,)*S.get_spatial_dimension()
    V1_tab = V1.tabulate(Q.get_points())[z]
    V2_tab = V2.tabulate(Q.get_points())[z]
    assert numpy.linalg.matrix_rank(V1_tab.reshape(len(V1_tab), -1)) == len(V1_tab)
    assert numpy.linalg.matrix_rank(V2_tab.reshape(len(V2_tab), -1)) == len(V2_tab)
    assert len(V1_tab) < len(V2_tab)
    assert span_greater_equal(V2_tab, V1_tab)
    assert not span_greater_equal(V1_tab, V2_tab)


@pytest.mark.parametrize("sd, degree", ((2, 1), (3, 1), (3, 2)))
def test_gn_trace(sd, degree):
    cell = symmetric_simplex(sd)

    gn = GuzmanNeilanFirstKindH1(cell, degree)
    br = BernardiRaugel(cell, degree)

    sd = cell.get_spatial_dimension()
    ref_face = cell.construct_subelement(sd-1)
    Q_face = create_quadrature(ref_face, max(gn.degree(), br.degree()) + degree)

    Phis = ONPolynomialSet(ref_face, degree)
    phis_at_qpts = Phis.tabulate(Q_face.get_points())[(0,)*(sd-1)]

    for f in cell.topology[sd-1]:
        Q = FacetQuadratureRule(cell, sd-1, f, Q_face)
        vals = gn.tabulate(0, Q.get_points())[(0,)*sd]
        vals -= br.tabulate(0, Q.get_points())[(0,)*sd]

        # Assert that the normal component is the same
        n = cell.compute_normal(f)
        normal_trace = numpy.tensordot(vals, n, axes=(1, 0))
        normal_moments = inner(normal_trace, phis_at_qpts, Q.get_weights())
        assert numpy.allclose(normal_moments, 0)

        # Assert that the tangential components are the same
        for t in cell.compute_tangents(sd-1, f):
            tangential_trace = numpy.tensordot(vals, t, axes=(1, 0))
            tangential_moments = inner(tangential_trace, phis_at_qpts, Q.get_weights())
            assert numpy.allclose(tangential_moments, 0)


def test_rotated_br_trace():
    # The normal trace of the rotated element is linear on each edge
    br = BernardiRaugel(T, rotated=True)
    vertices = numpy.asarray(T.get_vertices())
    for e, (v0, v1) in T.get_topology()[1].items():
        pts = [vertices[v0], (vertices[v0] + vertices[v1]) / 2, vertices[v1]]
        vals = br.tabulate(0, pts)[(0, 0)][:9]
        normal_trace = numpy.tensordot(vals, T.compute_normal(e), axes=(1, 0))
        assert numpy.allclose(normal_trace[:, 1], (normal_trace[:, 0] + normal_trace[:, 2]) / 2)


@pytest.mark.parametrize("cell, family, degree",
                         [(cell, family, 1) for cell in (T, S) for family in ("AQ", "CH", "GN", "GN2")]
                         + [(S, "GN", 2), (S, "GN2", 2)])
def test_minimal_stokes_space(cell, family, degree):
    # Test that the C0 Stokes space is spanned by a C0 basis
    # Also test that its divergence is P_{k-1}
    sd = cell.get_spatial_dimension()
    if family == "GN":
        space = GuzmanNeilanSpace
    elif family == "GN2":
        space = lambda *args, **kwargs: GuzmanNeilanSpace(*args, kind=2, **kwargs)
    elif family == "CH":
        space = ChristiansenHuSpace
    elif family == "AQ":
        if sd != 2:
            return
        degree = 2
        space = ArnoldQinSpace

    W = space(cell, degree)
    V = space(cell, degree, reduced=True)
    Wdim = W.get_num_members()
    Vdim = V.get_num_members()
    K = W.get_reference_element()
    sd = K.get_spatial_dimension()

    pts = make_points(K, degree+2)
    C0 = CkPolynomialSet(K, sd, order=0, variant="bubble")
    C0_tab = C0.tabulate(pts)
    Wtab = W.tabulate(pts, 1)
    Vtab = V.tabulate(pts, 1)
    z = (0,) * sd
    for Xtab in (Vtab, Wtab):
        # Test that the space is full rank
        _, sig, _ = numpy.linalg.svd(Xtab[z].reshape(-1, sd*len(pts)).T, full_matrices=True)
        assert all(sig > 1E-10)

        # Test that the space is C0
        for k in range(sd):
            assert span_greater_equal(C0_tab[z], Xtab[z][:, k, :])

        # Test the divergence
        divX = div(Xtab)[:Vdim]
        if family in {"GN", "GN2"}:
            # Test that div(GN) = P_{k-1} and div(GN2) = P_{k-1}(Alfeld)
            ref_el = K if family == "GN2" else K.get_parent()
            pressure = ONPolynomialSet(ref_el, degree-1)
            pressure_tab = pressure.tabulate(pts)[z]
            assert span_equal(divX, pressure_tab)
        else:
            assert numpy.allclose(divX, divX[:, 0][:, None])

    # Test that the full space includes the reduced space
    assert Wdim > Vdim
    assert span_greater_equal(Wtab[z], Vtab[z])
