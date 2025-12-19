import pytest

from FIAT import ufc_cell
from finat.quadrature import make_quadrature
from finat.quadrature_element import make_quadrature_element


@pytest.fixture(params=["interval", "triangle", "interval * interval", "triangle * interval"])
def cell(request):
    return ufc_cell(request.param)


def test_create_from_quadrature(cell):
    degree = 4
    scheme = "default"
    fe1 = make_quadrature_element(cell, degree, scheme=scheme)

    quadrature = make_quadrature(cell, degree, scheme=scheme)
    fe2 = make_quadrature_element(cell, degree, scheme=quadrature)

    Q1, ps1 = fe1.dual_basis
    Q2, ps2 = fe2.dual_basis
    assert ps1.almost_equal(ps2)
