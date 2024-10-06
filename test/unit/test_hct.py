import pytest
import numpy

from FIAT import HsiehCloughTocher as HCT
from FIAT.reference_element import ufc_simplex, make_lattice
from FIAT.functional import PointEvaluation


@pytest.fixture
def cell():
    K = ufc_simplex(2)
    K.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    return K


@pytest.mark.parametrize("reduced", (False, True))
def test_hct_constant(cell, reduced):
    # Test that bfs associated with point evaluation sum up to 1
    fe = HCT(cell, reduced=reduced)

    pts = make_lattice(cell.get_vertices(), 3)
    tab = fe.tabulate(2, pts)

    coefs = numpy.zeros((fe.space_dimension(),))
    nodes = fe.dual_basis()
    entity_dofs = fe.entity_dofs()
    for v in entity_dofs[0]:
        for k in entity_dofs[0][v]:
            if isinstance(nodes[k], PointEvaluation):
                coefs[k] = 1.0

    for alpha in tab:
        expected = 1 if sum(alpha) == 0 else 0
        vals = numpy.dot(coefs, tab[alpha])
        assert numpy.allclose(vals, expected)
