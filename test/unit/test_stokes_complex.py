import pytest
import numpy

from FIAT import HsiehCloughTocher as HCT
from FIAT import AlfeldSorokina as AS
from FIAT import ArnoldQin as AQ
from FIAT import Lagrange as CG
from FIAT import DiscontinuousLagrange as DG
from FIAT.restricted import RestrictedElement
from FIAT.reference_element import ufc_simplex

from FIAT.macro import CkPolynomialSet
from FIAT.alfeld_sorokina import AlfeldSorokinaSpace
from FIAT.arnold_qin import ArnoldQinSpace
from FIAT.christiansen_hu import ChristiansenHuSpace
from FIAT.guzman_neilan import ExtendedGuzmanNeilanSpace


T = ufc_simplex(2)
S = ufc_simplex(3)


@pytest.mark.parametrize("cell", (T, S))
@pytest.mark.parametrize("degree", (2, 3, 4))
def test_AlfeldSorokinaSpace(cell, degree):
    # Test that the divergence of the Alfeld-Sorokina space is spanned by a C0 basis
    V = AlfeldSorokinaSpace(cell, degree)
    A = V.get_reference_element()
    top = A.get_topology()
    sd = A.get_spatial_dimension()

    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(A.make_points(dim, entity, degree))

    V_tab = V.tabulate(pts, 1)
    V_div = sum(V_tab[alpha][:, alpha.index(1), :]
                for alpha in V_tab if sum(alpha) == 1)

    C0 = CkPolynomialSet(A, degree-1, order=0, variant="bubble")
    C0_tab = C0.tabulate(pts)[(0,)*sd]
    _, residual, *_ = numpy.linalg.lstsq(C0_tab.T, V_div.T)
    assert numpy.allclose(residual, 0)


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

    H2dim = H2.space_dimension()
    H1dim = H1.space_dimension()
    _, residual, *_ = numpy.linalg.lstsq(H1val.reshape(H1dim, -1).T, H2curl.reshape(H2dim, -1).T)
    assert numpy.allclose(residual, 0)

    _, residual, *_ = numpy.linalg.lstsq(L2val.T, H1div.T)
    assert numpy.allclose(residual, 0)


@pytest.mark.parametrize("cell", (T, S))
@pytest.mark.parametrize("family", ("AQ", "CH", "GN"))
def test_minimal_stokes_space(cell, family):
    # Test that the C0 Stokes space is spanned by a C0 basis
    # Also test that its divergence is constant
    sd = cell.get_spatial_dimension()
    if family == "GN":
        degree = 1
        space = ExtendedGuzmanNeilanSpace
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
        assert numpy.allclose(div, div[:, 0][:, None])

    # Test that the full space includes the reduced space
    assert Wdim > Vdim
    _, residual, *_ = numpy.linalg.lstsq(Wtab[z].reshape(Wdim, -1).T,
                                         Vtab[z].reshape(Vdim, -1).T)
    assert numpy.allclose(residual, 0)
