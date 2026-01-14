import pytest
import numpy

from FIAT import Walkington
from FIAT.expansions import polynomial_dimension
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import ufc_simplex


@pytest.fixture(params=[0, 1], ids=["K1", "K2"])
def cell(request):
    K = ufc_simplex(3)
    if request.param == 1:
        K.vertices = ((0, 0, 0),
                      (1., 0.1, -0.37),
                      (0.01, 0.987, -.23),
                      (-0.1, -0.2, 1.38))
    return K


def directional_derivative(direction, tab):
    return sum(direction[alpha.index(1)] * tab[alpha]
               for alpha in tab if sum(alpha) == 1)


def inner(u, v, wts):
    return numpy.dot(numpy.multiply(u, wts), v.T)


def test_walkington_basis_functions(cell):
    degree = 5
    fe = Walkington(cell, degree)
    space_dim = 45

    ref_el = fe.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    entity_ids = fe.entity_dofs()

    degree = fe.degree()

    ref_facet = ufc_simplex(sd-1)
    Qref = create_quadrature(ref_facet, 2*degree-1)

    P = ONPolynomialSet(ref_facet, degree-1)
    Ptab = P.tabulate(Qref.get_points())
    P_at_qpts = Ptab[(0,)*(sd-1)]
    P3dim = polynomial_dimension(ref_facet, degree-2)
    P4_at_qpts = P_at_qpts[P3dim:]

    for f in top[sd-1]:
        n = ref_el.compute_normal(f)
        Q = FacetQuadratureRule(ref_el, sd-1, f, Qref)
        qpts, qwts = Q.get_points(), Q.get_weights()

        ids = entity_ids[sd-1][f]

        tab = fe.tabulate(1, qpts)
        phi = tab[(0,) * sd]
        phi_n = directional_derivative(n, tab)

        # Test that normal derivative moment bfs have vanishing trace
        assert numpy.allclose(phi[ids[:1]], 0)

        # Test that trace moment bfs have vanishing normal derivative
        P4_moments = inner(phi_n[:space_dim], P4_at_qpts, qwts)
        assert numpy.allclose(P4_moments, 0)


def span_greater_equal(A, B):
    # span(A) >= span(B)
    _, residual, *_ = numpy.linalg.lstsq(A.reshape(A.shape[0], -1).T,
                                         B.reshape(B.shape[0], -1).T)
    return numpy.allclose(residual, 0)


def test_walkington_space(cell):
    degree = 5
    fe = Walkington(cell, degree)
    space_dim = 45

    V = fe.get_nodal_basis()
    ref_complex = V.ref_el
    pts = []
    top = ref_complex.topology
    for dim in top:
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree))
    V_tab = V.tabulate(pts)

    P = ONPolynomialSet(cell, degree)
    P_tab = P.tabulate(pts)

    # Test that the augmented space includes all quintics
    sd = cell.get_spatial_dimension()
    assert span_greater_equal(V_tab[(0,)*sd], P_tab[(0,)*sd])

    # Test that the reduced space includes all quartics
    dimP4 = polynomial_dimension(cell, degree-1)
    assert span_greater_equal(V_tab[(0,)*sd][:space_dim], P_tab[(0,)*sd][:dimP4])
