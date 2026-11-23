import numpy
import pytest

import gem
from FIAT import ufc_cell
from FIAT.reference_element import Hypercube
from finat import FlattenedDimensions, Lagrange, TensorProductElement
from finat.point_set import PointSet
from finat.quadrature import make_quadrature
from finat.quadrature_element import make_quadrature_element
from gem.interpreter import evaluate


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


@pytest.mark.parametrize("cellname", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
@pytest.mark.parametrize("degree", range(1, 6))
def test_facet_dual_evaluation(cellname, degree):
    """Check tabulation structure and reproduction of coordinate functions."""
    cell = ufc_cell(cellname)
    target = make_quadrature_element(cell, degree, codim=1)
    if isinstance(cell, Hypercube):
        source = FlattenedDimensions(TensorProductElement(
            Lagrange(factor, 1) for factor in cell.product.cells))
    else:
        source = Lagrange(cell, 1)
    source_indices = source.get_indices()

    def tabulate(point_set):
        table, = source.basis_evaluation(0, point_set).values()
        return table[source_indices]

    expr, point_indices, basis_indices = target.dual_evaluation(tabulate)
    assert tuple(index.extent for index in basis_indices) == target.index_shape
    tables = gem.extract_type([expr], gem.Literal)
    if isinstance(cell, Hypercube):
        assert not point_indices
        concatenated, = gem.extract_type([expr], gem.Concatenate)
        assert len(concatenated.children) == cell.get_spatial_dimension()
        assert all(block.shape == (2,) for block in concatenated.children)
        assert max(len(table.shape) for table in tables) == 2
    else:
        assert not gem.extract_type([expr], gem.Concatenate)
        table, = tables
        assert table.shape == source.index_shape + target.index_shape

    result, = evaluate([gem.ComponentTensor(gem.IndexSum(expr, point_indices),
                                            basis_indices + source_indices)])
    _, target_points = target.dual_basis
    _, source_points = source.dual_basis
    tabulation = result.arr.reshape(target.space_dimension(), source.space_dimension())
    numpy.testing.assert_allclose(tabulation @ source_points.points, target_points.points, rtol=0, atol=1e-12)


def test_point_rule_weight():
    rule = make_quadrature(ufc_cell("interval").construct_subelement(0), 0)
    assert rule.point_set.indices == ()
    assert rule.weight_expression.shape == ()
    weight, = evaluate([rule.weight_expression])
    assert weight.arr == 1


@pytest.mark.parametrize("dtype, equal", [(numpy.float32, True), (numpy.float64, False)])
def test_point_comparison_tolerance(dtype, equal):
    points = PointSet(numpy.array([[0.0]], dtype=dtype))
    nearby = PointSet(numpy.array([[1e-7]], dtype=dtype))
    assert points.almost_equal(nearby) == equal
    assert not points.almost_equal(nearby, tolerance=1e-12)
