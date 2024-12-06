import pytest
import numpy

from FIAT import (HsiehCloughTocher as HCT,
                  AlfeldSorokina as AS,
                  ArnoldQin as AQ,
                  Lagrange as CG,
                  DiscontinuousLagrange as DG)
from FIAT.reference_element import ufc_simplex

from FIAT.polynomial_set import ONPolynomialSet
from FIAT.macro import CkPolynomialSet
from FIAT.alfeld_sorokina import AlfeldSorokinaSpace
from FIAT.arnold_qin import ArnoldQinSpace
from FIAT.christiansen_hu import ChristiansenHuSpace
from FIAT.guzman_neilan import (
    GuzmanNeilanFirstKindH1,
    GuzmanNeilanSecondKindH1,
    GuzmanNeilanH1div,
    GuzmanNeilanSpace)
from FIAT.restricted import RestrictedElement


T = ufc_simplex(2)
S = ufc_simplex(3)


def rHCT(cell):
    return RestrictedElement(HCT(cell, reduced=True), restriction_domain="vertex")


def rAQ(cell):
    return RestrictedElement(AQ(cell, reduced=True), indices=list(range(9)))


def span_greater_equal(A, B):
    # span(A) >= span(B)
    _, residual, *_ = numpy.linalg.lstsq(A.reshape(A.shape[0], -1).T,
                                         B.reshape(B.shape[0], -1).T)
    return numpy.allclose(residual, 0)


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


@pytest.mark.parametrize("cell", (T, S))
@pytest.mark.parametrize("family", ("AQ", "CH", "GN", "GN2"))
def test_minimal_stokes_space(cell, family):
    # Test that the C0 Stokes space is spanned by a C0 basis
    # Also test that its divergence is constant
    sd = cell.get_spatial_dimension()
    if family == "GN":
        degree = 1
        space = GuzmanNeilanSpace
    elif family == "GN2":
        degree = 1
        space = lambda *args, **kwargs: GuzmanNeilanSpace(*args, kind=2, **kwargs)
    elif family == "CH":
        degree = 1
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
            _, residual, *_ = numpy.linalg.lstsq(C0_tab[z].T, Xtab[z][:, k, :].T)
            assert numpy.allclose(residual, 0)

        # Test that divergence is in P0
        divX = div(Xtab)[:Vdim]
        if family in {"GN", "GN2"}:
            # Test that div(GN2) is in P0(Alfeld)
            ref_el = K if family == "GN2" else K.get_parent()
            P0 = ONPolynomialSet(ref_el, degree-1)
            P0_tab = P0.tabulate(pts)[z]
            assert span_equal(divX, P0_tab)
        else:
            assert numpy.allclose(divX, divX[:, 0][:, None])

    # Test that the full space includes the reduced space
    assert Wdim > Vdim
    assert span_greater_equal(Wtab[z], Vtab[z])
