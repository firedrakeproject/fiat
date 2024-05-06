import pytest
import numpy

from FIAT import GopalakrishnanLedererSchoberl as GLS
from FIAT.reference_element import ufc_simplex
from FIAT.expansions import polynomial_dimension
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

    entity_dofs = fe.entity_dofs()
    poly_set = fe.get_nodal_basis()
    bubbles = poly_set.take(entity_dofs[sd][0])
    expansion_set = poly_set.get_expansion_set()

    # test dimension of constrained space
    dimPkm1 = polynomial_dimension(facet_el, degree-1)
    dimPk = polynomial_dimension(facet_el, degree)
    expected = (sd**2-1)*(expansion_set.get_num_members(degree) - (dimPk - dimPkm1))
    assert poly_set.get_num_members() == expected

    # test normal-tangential bubble support
    Qref = create_quadrature(facet_el, 2*degree)
    norms = numpy.zeros((bubbles.get_num_members(),))
    top = cell.get_topology()
    for facet in top[sd-1]:
        Q = FacetQuadratureRule(cell, sd-1, facet, Qref)
        qpts, qwts = Q.get_points(), Q.get_weights()
        phi_at_pts = bubbles.tabulate(qpts)[(0,) * sd]
        n = cell.compute_normal(facet)
        rts = cell.compute_normalized_tangents(sd-1, facet)
        for t in rts:
            nt = numpy.outer(t, n)
            phi_nt = numpy.tensordot(nt, phi_at_pts, axes=((0, 1), (1, 2)))
            norms += numpy.dot(phi_nt**2, qwts)

    assert numpy.allclose(norms, 0)
    expected = (sd**2-1)*expansion_set.get_num_members(degree-1)
    assert bubbles.get_num_members() == expected
