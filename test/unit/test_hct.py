import pytest
import numpy

from FIAT import RestrictedElement, HsiehCloughTocher as HCT
from FIAT.reference_element import ufc_simplex
from FIAT.functional import PointEvaluation
from FIAT.macro import CkPolynomialSet


@pytest.fixture
def cell():
    K = ufc_simplex(2)
    K.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    return K


def span_greater_equal(A, B):
    # span(A) >= span(B)
    _, residual, *_ = numpy.linalg.lstsq(A.reshape(A.shape[0], -1).T,
                                         B.reshape(B.shape[0], -1).T)
    return numpy.allclose(residual, 0)


def make_points(K, degree):
    top = K.get_topology()
    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(K.make_points(dim, entity, degree))
    return pts


@pytest.mark.parametrize("reduced", (False, True))
def test_hct_constant(cell, reduced):
    # Test that bfs associated with point evaluation sum up to 1
    fe = HCT(cell, reduced=reduced)

    pts = make_points(cell, 4)
    tab = fe.tabulate(2, pts)

    coefs = numpy.zeros((fe.space_dimension(),))
    nodes = fe.dual_basis()
    entity_dofs = fe.entity_dofs()
    for v in entity_dofs[0]:
        for k in entity_dofs[0][v]:
            if isinstance(nodes[k], PointEvaluation):
                coefs[k] = 1.0

    for alpha in tab:
        expected = 1 if sum(alpha) == 0 else 0
        vals = numpy.dot(coefs, tab[alpha])
        assert numpy.allclose(vals, expected)


@pytest.mark.parametrize("reduced", (False, True))
def test_full_polynomials(cell, reduced):
    # Test that HCT/HCT-red contains all cubics/quadratics
    fe = HCT(cell, reduced=reduced)
    if reduced:
        fe = RestrictedElement(fe, restriction_domain="vertex")

    ref_complex = fe.get_reference_complex()
    pts = make_points(ref_complex, 4)
    tab = fe.tabulate(0, pts)[(0, 0)]

    degree = fe.degree()
    if reduced:
        degree -= 1

    P = CkPolynomialSet(cell, degree, variant="bubble")
    P_tab = P.tabulate(pts)[(0, 0)]
    assert span_greater_equal(tab, P_tab)

    C1 = CkPolynomialSet(ref_complex, degree, order=1, variant="bubble")
    C1_tab = C1.tabulate(pts)[(0, 0)]
    assert span_greater_equal(tab, C1_tab)
