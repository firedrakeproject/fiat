import numpy
import pytest
from FIAT import QuadraticPowellSabin6 as PS6
from FIAT import QuadraticPowellSabin12 as PS12
from FIAT.functional import PointEvaluation
from FIAT.reference_element import make_lattice, ufc_simplex


@pytest.fixture
def cell():
    return ufc_simplex(2)


@pytest.mark.parametrize("el", (PS6, PS12))
def test_powell_sabin_constant(cell, el):
    # Test that bfs associated with point evaluation sum up to 1
    fe = el(cell)

    pts = make_lattice(cell.get_vertices(), 3)
    tab = fe.tabulate(2, pts)

    coeffs = numpy.zeros((fe.space_dimension(),))
    nodes = fe.dual_basis()
    entity_dofs = fe.entity_dofs()
    for v in entity_dofs[0]:
        for k in entity_dofs[0][v]:
            if isinstance(nodes[k], PointEvaluation):
                coeffs[k] = 1.0

    for alpha in tab:
        expected = 1 if sum(alpha) == 0 else 0
        assert numpy.allclose(coeffs @ tab[alpha], expected)
