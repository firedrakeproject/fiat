import pytest
import numpy

from FIAT import Argyris, HsiehCloughTocher
from FIAT.jacobi import eval_jacobi_batch, eval_jacobi_deriv_batch
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import ufc_simplex


@pytest.fixture
def cell():
    return ufc_simplex(2)


def directional_derivative(direction, tab):
    return sum(direction[alpha.index(1)] * tab[alpha]
               for alpha in tab if sum(alpha) == 1)


def inner(u, v, wts):
    return numpy.dot(numpy.multiply(u, wts), v.T)


@pytest.mark.parametrize("family, degree", [
    *((Argyris, k) for k in range(6, 10)),
    *((HsiehCloughTocher, k) for k in range(4, 8))])
def test_argyris_basis_functions(cell, family, degree):
    fe = family(cell, degree)

    ref_el = fe.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    entity_ids = fe.entity_dofs()

    degree = fe.degree()
    lowest_p = 5 if isinstance(fe, Argyris) else 3
    a = (lowest_p - 1) // 2
    q = degree - lowest_p

    rline = ufc_simplex(1)
    Qref = create_quadrature(rline, 2*degree)
    xref = 2.0 * Qref.get_points() - 1

    ell_at_qpts = eval_jacobi_batch(0, 0, degree, xref)
    P_at_qpts = eval_jacobi_batch(a, a, degree, xref)
    DP_at_qpts = eval_jacobi_deriv_batch(a, a, degree, xref)

    for e in top[1]:
        n = ref_el.compute_normal(e)
        t = ref_el.compute_edge_tangent(e)
        Q = FacetQuadratureRule(ref_el, 1, e, Qref)
        qpts, qwts = Q.get_points(), Q.get_weights()

        ids = entity_ids[1][e]
        ids1 = ids[1:-q]
        ids0 = ids[-q:]

        tab = fe.tabulate(1, qpts)
        phi = tab[(0,) * sd]
        phi_n = directional_derivative(n, tab)
        phi_t = directional_derivative(t, tab)

        # Test that normal derivative moment bfs have vanishing trace
        assert numpy.allclose(phi[ids[:-q]], 0)

        # Test that trace moment bfs have vanishing normal derivative
        assert numpy.allclose(phi_n[ids0], 0)

        # Test that facet bfs are hierarchical
        coeffs = inner(ell_at_qpts[1+lowest_p:], phi[ids0], qwts)
        assert numpy.allclose(coeffs, numpy.triu(coeffs))
        coeffs = inner(ell_at_qpts[1+lowest_p:], phi_n[ids1], qwts)
        assert numpy.allclose(coeffs, numpy.triu(coeffs))

        # Test duality of normal derivative moments
        coeffs = inner(P_at_qpts[1:], phi_n[ids1], qwts)
        assert numpy.allclose(coeffs[q:], 0)
        assert numpy.allclose(coeffs[:q], numpy.diag(numpy.diag(coeffs[:q])))

        # Test duality of trace moments
        coeffs = inner(DP_at_qpts[1:], phi[ids0], qwts)
        assert numpy.allclose(coeffs[q:], 0)
        assert numpy.allclose(coeffs[:q], numpy.diag(numpy.diag(coeffs[:q])))

        # Test the integration by parts property arising from the choice
        # of normal derivative and trace moments DOFs.
        # The normal derivative of the normal derviative bfs must be equal
        # to minus the tangential derivative of the trace moment bfs
        assert numpy.allclose(numpy.dot((phi_n[ids1] + phi_t[ids0])**2, qwts), 0)
