import numpy
import pytest

import gem
from FIAT import ufc_cell
from FIAT.quadrature_schemes import create_quadrature as fiat_scheme
from finat.quadrature import make_quadrature


@pytest.mark.parametrize(
    "cell_name",
    ["interval", "triangle", "interval * interval", "triangle * interval"]
)
def test_quadrature_rules_are_hashable(cell_name):
    ref_cell = ufc_cell(cell_name)
    quadrature1 = make_quadrature(ref_cell, 3)
    quadrature2 = make_quadrature(ref_cell, 3)

    assert quadrature1 is not quadrature2
    assert hash(quadrature1) == hash(quadrature2)
    assert repr(quadrature1) == repr(quadrature2)
    assert quadrature1 == quadrature2


@pytest.mark.parametrize("cell_name", ["interval", "triangle", "tetrahedron"])
@pytest.mark.parametrize("degree", [3, 8])
def test_collapsed_quadrature(cell_name, degree):
    ref_cell = ufc_cell(cell_name)
    dim = ref_cell.get_spatial_dimension()
    rule = make_quadrature(ref_cell, degree, scheme="collapsed")
    ps = rule.point_set
    result, = gem.interpreter.evaluate([rule.weight_expression])
    weights = result.broadcast(ps.indices).ravel()

    reference = fiat_scheme(ref_cell, degree, "canonical")
    ref_points = reference.get_points()
    ref_weights = reference.get_weights()
    for alpha in numpy.ndindex((degree + 1,) * dim):
        if sum(alpha) > degree:
            continue
        monomial = lambda pts: numpy.prod(pts ** numpy.asarray(alpha), axis=-1)
        exact = numpy.dot(ref_weights, monomial(ref_points))
        assert numpy.allclose(numpy.dot(weights, monomial(ps.points)), exact)
