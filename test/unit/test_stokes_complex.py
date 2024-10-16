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
from FIAT.guzman_neilan import GuzmanNeilanSpace, GuzmanNeilanH1div
from FIAT.restricted import RestrictedElement


T = ufc_simplex(2)
S = ufc_simplex(3)


def span_greater_equal(A, B):
    # span(A) >= span(B)
    dimA = A.shape[0]
    dimB = B.shape[0]
    _, residual, *_ = numpy.linalg.lstsq(A.reshape(dimA, -1).T, B.reshape(dimB, -1).T)
    return numpy.allclose(residual, 0)


def span_equal(A, B):
    # span(A) == span(B)
    return span_greater_equal(A, B) and span_greater_equal(B, A)


def check_h1div_space(V, degree, reduced=False, bubble=False):
    # Test that the divergence of the polynomial space V is spanned by a C0 basis
    A = V.get_reference_element()
    top = A.get_topology()
    sd = A.get_spatial_dimension()
    z = (0,)*sd

    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(A.make_points(dim, entity, degree+2))

    V_tab = V.tabulate(pts, 1)
    V_div = sum(V_tab[alpha][:, alpha.index(1), :]
                for alpha in V_tab if sum(alpha) == 1)

    C0 = CkPolynomialSet(A, degree-1, order=0, variant="bubble")
    C0_tab = C0.tabulate(pts)[z]
    assert span_equal(V_div, C0_tab)

    if bubble:
        # Test that div(Bubbles) = C0 int H^1_0
        assert span_equal(V_div[-(sd+1):], C0_tab[-1:])

    if not reduced:
        # Test that V includes Pk
        cell = A.get_parent() or A
        Pk = ONPolynomialSet(cell, degree, shape=(sd,))
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


@pytest.mark.parametrize("cell", (T,))
@pytest.mark.parametrize("reduced", (False, True))
def test_hct_stokes_complex(cell, reduced):
    # Test that we have the lowest-order discrete Stokes complex, verifying
    # that the range of the exterior derivative of each space is contained by
    # the next space in the sequence
    H2 = HCT(T, reduced=reduced)
    A = H2.get_reference_complex()
    if reduced:
        # HCT-red(3) -curl-> AQ-red(2) -div-> DG(0)
        H2 = RestrictedElement(H2, restriction_domain="vertex")
        H1 = RestrictedElement(AQ(T, reduced=reduced), indices=list(range(9)))
        L2 = DG(cell, 0)
    else:
        # HCT(3) -curl-> AS(2) -div-> CG(1, Alfeld)
        H1 = AS(cell)
        L2 = CG(A, 1)

    pts = []
    top = A.get_topology()
    for dim in top:
        for entity in top[dim]:
            pts.extend(A.make_points(dim, entity, 4))

    H2tab = H2.tabulate(1, pts)
    H1tab = H1.tabulate(1, pts)
    L2tab = L2.tabulate(0, pts)

    L2val = L2tab[(0, 0)]
    H1val = H1tab[(0, 0)]
    H1div = sum(H1tab[alpha][:, alpha.index(1), :] for alpha in H1tab if sum(alpha) == 1)
    H2curl = numpy.stack([H2tab[(0, 1)], -H2tab[(1, 0)]], axis=1)

    assert span_greater_equal(H1val, H2curl)
    assert span_greater_equal(L2val, H1div)


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
    top = K.get_topology()

    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(K.make_points(dim, entity, degree))

    C0 = CkPolynomialSet(K, sd, order=0, variant="bubble")
    C0_tab = C0.tabulate(pts)
    Wtab = W.tabulate(pts, 1)
    Vtab = V.tabulate(pts, 1)
    z = (0,)*sd
    for tab in (Vtab, Wtab):
        # Test that the space is full rank
        _, sig, _ = numpy.linalg.svd(tab[z].reshape(-1, sd*len(pts)).T, full_matrices=True)
        assert all(sig > 1E-10)

        # Test that the space is C0
        for k in range(sd):
            _, residual, *_ = numpy.linalg.lstsq(C0_tab[z].T, tab[z][:, k, :].T)
            assert numpy.allclose(residual, 0)

        # Test that divergence is in P0
        div = sum(tab[alpha][:, alpha.index(1), :]
                  for alpha in tab if sum(alpha) == 1)[:Vdim]
        if family == "GN2":
            # Test that div(GN2) is in P0(Alfeld)
            P0 = ONPolynomialSet(K, 0)
            P0_tab = P0.tabulate(pts)[(0,)*sd]
            assert span_equal(div, P0_tab)
        else:
            assert numpy.allclose(div, div[:, 0][:, None])

    # Test that the full space includes the reduced space
    assert Wdim > Vdim
    assert span_greater_equal(Wtab[z], Vtab[z])
