import pytest

from FIAT import ufc_cell
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
