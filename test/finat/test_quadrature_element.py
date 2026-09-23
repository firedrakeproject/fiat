import numpy
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


@pytest.mark.parametrize("cellname", ["quadrilateral", "hexahedron"])
@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_facet_factorisation_ordering(cellname, degree):
    """The products dual evaluation splits into number the same points."""
    element = make_quadrature_element(ufc_cell(cellname), degree, codim=1)

    blocks = []
    for k, summand in enumerate(element._facet_factorisation):
        points = summand.dual_basis[1].points.reshape(summand.index_shape + (-1,))
        # Factor k numbers the facets of its direction; dual_evaluation brings
        # that index outermost, where this element carries the facet number.
        points = numpy.moveaxis(points, k, 0)
        blocks.append(points.reshape(-1, points.shape[-1]))

    assert numpy.allclose(numpy.concatenate(blocks), element._point_set.points)
