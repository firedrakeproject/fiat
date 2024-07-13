import pytest
import numpy

from FIAT import Argyris
from FIAT.jacobi import eval_jacobi_batch, eval_jacobi_deriv_batch
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import ufc_simplex


@pytest.fixture
def cell():
    return ufc_simplex(2)


@pytest.mark.parametrize("degree", range(6, 10))
def test_argyris_basis_functions(cell, degree):
    fe = Argyris(cell, degree)

    entity_ids = fe.entity_dofs()
    sd = cell.get_spatial_dimension()
    top = cell.get_topology()
    rline = ufc_simplex(1)
    Qref = create_quadrature(rline, 2*degree)

    q = degree - 5
    xref = 2.0*Qref.get_points()-1
    ell_at_qpts = eval_jacobi_batch(0, 0, degree, xref)
    P_at_qpts = eval_jacobi_batch(2, 2, degree, xref)
    DP_at_qpts = eval_jacobi_deriv_batch(2, 2, degree, xref)

    for e in top[1]:
        n = cell.compute_normal(e)
        t = cell.compute_edge_tangent(e)
        Q = FacetQuadratureRule(cell, 1, e, Qref)
        qpts, qwts = Q.get_points(), Q.get_weights()

        ids = entity_ids[1][e]
        ids1 = ids[1:-q]
        ids0 = ids[-q:]

        tab = fe.tabulate(1, qpts)
        # Test that normal derivative moment bfs have vanishing trace
        assert numpy.allclose(tab[(0,) * sd][ids[:-q]], 0)

        phi = tab[(0,) * sd][ids0]

        phi_n = sum(n[alpha.index(1)] * tab[alpha][ids1]
                    for alpha in tab if sum(alpha) == 1)

        phi_t = sum(t[alpha.index(1)] * tab[alpha][ids0]
                    for alpha in tab if sum(alpha) == 1)

        # Test that facet bfs are hierarchical
        coeffs = numpy.dot(numpy.multiply(phi, qwts), ell_at_qpts[6:].T)
        assert numpy.allclose(numpy.triu(coeffs, k=1), 0)

        # Test duality of normal derivarive moments
        coeffs = numpy.dot(numpy.multiply(phi_n, qwts), P_at_qpts[1:].T)
        assert numpy.allclose(coeffs[:, q:], 0)
        assert numpy.allclose(coeffs[:, :q], numpy.diag(numpy.diag(coeffs[:, :q])))

        # Test duality of trace moments
        coeffs = numpy.dot(numpy.multiply(phi, qwts), DP_at_qpts[1:].T)
        assert numpy.allclose(coeffs[:, q:], 0)
        assert numpy.allclose(coeffs[:, :q], numpy.diag(numpy.diag(coeffs[:, :q])))

        # Test the integration by parts property arising from the choice
        # of normal derivative and trace moments DOFs.
        # The normal derivative of the normal derviative bfs must be equal
        # to minus the tangential derivative of the trace moment bfs
        assert numpy.allclose(numpy.dot((phi_n + phi_t)**2, qwts), 0)
