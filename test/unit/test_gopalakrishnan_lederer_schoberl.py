import pytest
import numpy

from FIAT import GopalakrishnanLedererSchoberl as GLS
from FIAT.reference_element import ufc_simplex
from FIAT.expansions import polynomial_dimension
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


@pytest.fixture(params=("T", "S"))
def cell(request):
    dim = {"I": 1, "T": 2, "S": 3}[request.param]
    return ufc_simplex(dim)


@pytest.mark.parametrize("degree", (1, 2, 3))
def test_gls_bubbles(cell, degree):
    fe = GLS(cell, degree)
    sd = cell.get_spatial_dimension()
    facet_el = cell.construct_subelement(sd-1)
    poly_set = fe.get_nodal_basis()

    # test dimension of constrained space
    dimPkm1 = polynomial_dimension(facet_el, degree-1)
    dimPk = polynomial_dimension(facet_el, degree)
    expected = (sd**2-1)*(polynomial_dimension(cell, degree) - (dimPk - dimPkm1))
    assert poly_set.get_num_members() == expected

    # test dimension of the bubbles
    entity_dofs = fe.entity_dofs()
    bubbles = poly_set.take(entity_dofs[sd][0])
    expected = (sd**2-1)*polynomial_dimension(cell, degree-1)
    assert bubbles.get_num_members() == expected

    top = cell.get_topology()
    Qref = create_quadrature(facet_el, 2*degree)
    Pk = ONPolynomialSet(facet_el, degree)
    PkH = Pk.take(list(range(dimPkm1, dimPk)))
    PkH_at_qpts = PkH.tabulate(Qref.get_points())[(0,)*(sd-1)]
    weights = numpy.transpose(numpy.multiply(PkH_at_qpts, Qref.get_weights()))
    for facet in top[sd-1]:
        n = cell.compute_scaled_normal(facet)
        rts = cell.compute_tangents(sd-1, facet)
        Q = FacetQuadratureRule(cell, sd-1, facet, Qref)
        qpts, qwts = Q.get_points(), Q.get_weights()

        # test the degree of normal-tangential components
        phi_at_pts = fe.tabulate(0, qpts)[(0,) * sd]
        for t in rts:
            nt = numpy.outer(t, n)
            phi_nt = numpy.tensordot(nt, phi_at_pts, axes=((0, 1), (1, 2)))
            assert numpy.allclose(numpy.dot(phi_nt, weights), 0)

        # test the support of the normal-tangential bubble
        phi_at_pts = bubbles.tabulate(qpts)[(0,) * sd]
        for t in rts:
            nt = numpy.outer(t, n)
            phi_nt = numpy.tensordot(nt, phi_at_pts, axes=((0, 1), (1, 2)))
            norms = numpy.dot(phi_nt**2, qwts)
            assert numpy.allclose(norms, 0)
