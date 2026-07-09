import gem
import pytest
import ufl

import finat
import finat.ufl
from finat.element_factory import create_element
from finat.point_set import GaussLobattoLegendrePointSet, TensorPointSet


def _point_set_leaves(point_set):
    if isinstance(point_set, TensorPointSet):
        for factor in point_set.factors:
            yield from _point_set_leaves(factor)
    else:
        yield point_set


def _assert_only_delta_terminals(expr):
    found_delta = False
    nodes = [expr]
    while nodes:
        node = nodes.pop()
        if isinstance(node, gem.Delta):
            found_delta = True
        elif isinstance(node, gem.gem.Terminal):
            pytest.fail(f"Expected only gem.Delta terminals, found {type(node).__name__}")
        else:
            nodes.extend(node.children)
    assert found_delta


def _quadrilateral_x_interval_gll_element(degree):
    cell = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
    quadrilateral = finat.ufl.FiniteElement("Q", ufl.quadrilateral, degree, variant="spectral")
    interval = finat.ufl.FiniteElement("CG", ufl.interval, degree, variant="spectral")
    return create_element(finat.ufl.TensorProductElement(quadrilateral, interval, cell=cell))


def _hexahedron_gll_element(degree):
    element = finat.ufl.FiniteElement("Q", ufl.hexahedron, degree, variant="spectral")
    return create_element(element)


@pytest.mark.parametrize(
    "element_factory",
    [_quadrilateral_x_interval_gll_element, _hexahedron_gll_element],
    ids=["quadrilateral_x_interval", "hexahedron"],
)
def test_spectral_gll_element_tabulates_to_delta_on_gll_point_set(element_factory):
    degree = 3
    element = element_factory(degree)
    point_set = finat.quadrature.make_quadrature(element.cell, element.degree, "KMV").point_set

    assert len(point_set.points) == element.space_dimension()
    assert all(isinstance(ps, GaussLobattoLegendrePointSet) for ps in _point_set_leaves(point_set))

    derivative = (0,) * element.cell.get_spatial_dimension()
    tabulation = element.basis_evaluation(0, point_set)[derivative]

    _assert_only_delta_terminals(tabulation)
