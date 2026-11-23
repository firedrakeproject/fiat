import numpy
import pytest

import gem
from FIAT import ufc_cell
from FIAT.reference_element import Hypercube
from finat import FlattenedDimensions, Lagrange, TensorProductElement
from finat.point_set import PointSet, UnionPointSet
from finat.quadrature import QuadratureRule, make_quadrature
from finat.quadrature_element import make_quadrature_element
from gem.interpreter import evaluate
from gem.node import traversal
from gem.unconcatenate import split_contraction


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
    nfacets, *point_shape = target.index_shape
    assert tuple(index.extent for index in point_indices) == tuple(point_shape)
    # The tabulations and the points are stacked over the facets.
    concatenated = gem.extract_type([expr], gem.Concatenate)
    assert concatenated
    assert all(len(c.children) == nfacets for c in concatenated)
    tables = gem.extract_type([expr], gem.Literal)
    if isinstance(cell, Hypercube):
        # Each factor is tabulated on each facet at a vertex or at the
        # points of a tangential factor.
        nq = point_shape[-1]
        assert len(concatenated) == cell.get_spatial_dimension()
        assert {table.shape for table in tables} == {(2,), (2, nq)}
    else:
        assert len(concatenated) == 1
        assert {table.shape for table in tables} == {source.index_shape + tuple(point_shape)}

    result, = evaluate([gem.ComponentTensor(gem.IndexSum(expr, point_indices),
                                            basis_indices + source_indices)])
    _, target_points = target.dual_basis
    _, source_points = source.dual_basis
    tabulation = result.arr.reshape(target.space_dimension(), source.space_dimension())
    numpy.testing.assert_allclose(tabulation @ source_points.points, target_points.points, rtol=0, atol=1e-12)


def test_facet_dual_evaluation_on_union_of_points() -> None:
    cell = ufc_cell("quadrilateral")
    points = UnionPointSet([PointSet([[0.2]]), PointSet([[0.4], [0.7]])])
    rule = QuadratureRule(points, numpy.ones(3), ref_el=cell.construct_subelement(1))
    element = make_quadrature_element(cell, None, scheme=rule, codim=1)

    expr, point_indices, basis_indices = element.dual_evaluation(
        lambda ps: ps.expression[0] + 2 * ps.expression[1])
    result, = evaluate([gem.ComponentTensor(gem.IndexSum(expr, point_indices), basis_indices)])
    _, facet_points = element.dual_basis
    x, y = facet_points.points.T
    numpy.testing.assert_allclose(result.arr, (x + 2 * y).reshape(element.index_shape))


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


def test_facet_dual_evaluation_on_runtime_facet():
    """Evaluate an interpolant into facet quadrature on a facet known only at runtime."""
    cell = ufc_cell("quadrilateral")
    degree = 2
    element = make_quadrature_element(cell, degree, codim=1)
    rule = make_quadrature(cell.construct_subelement(1), degree)
    facet = gem.Variable("facet", (), dtype=gem.uint_type)
    tables = [element.basis_evaluation(0, rule.point_set, entity=(1, f))[(0, 0)]
              for f in cell.get_topology()[1]]
    table = gem.select_expression(tables, gem.VariableIndex(facet))

    expr, point_indices, beta = element.dual_evaluation(lambda ps: ps.expression[0] + 2 * ps.expression[1])
    dofs = gem.ComponentTensor(gem.IndexSum(expr, point_indices), beta)
    terms = split_contraction(gem.Product(table[beta], dofs[beta]), beta)
    value = gem.optimise.make_sum([gem.optimise.contraction(gem.IndexSum(term, indices))
                                   for term, indices in terms])
    assert not any(isinstance(node, (gem.Concatenate, gem.Delta)) for node in traversal([value]))

    _, facet_points = element.dual_basis
    x, y = facet_points.points.reshape(*element.index_shape, -1).transpose(2, 0, 1)
    value = gem.ComponentTensor(value, rule.point_set.indices)
    for f in range(len(tables)):
        result, = evaluate([value], {facet: numpy.array(f, dtype=gem.uint_type)})
        numpy.testing.assert_allclose(result.arr, x[f] + 2 * y[f], rtol=0, atol=1e-12)
