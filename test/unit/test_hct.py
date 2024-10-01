import pytest
import numpy

from FIAT import HsiehCloughTocher as HCT
from FIAT import AlfeldSorokina as AS
from FIAT import ArnoldQin as AQ
from FIAT import Lagrange as CG
from FIAT import DiscontinuousLagrange as DG
from FIAT.restricted import RestrictedElement
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


@pytest.mark.parametrize("reduced", (False, True))
def test_stokes_complex(cell, reduced):
    # Test that we have the lowest-order discrete Stokes complex, verifying
    # that the range of the exterior derivative of each space is contained by
    # the next space in the sequence
    H2 = HCT(cell, reduced=reduced)
    ref_complex = H2.get_reference_complex()
    if reduced:
        # HCT-red(3) -curl-> AQ-red(2) -div-> DG(0)
        H2 = RestrictedElement(H2, restriction_domain="vertex")
        H1 = RestrictedElement(AQ(cell), indices=list(range(9)))
        L2 = DG(cell, 0)
    else:
        # HCT(3) -curl-> AS(2) -div-> CG(1, Alfeld)
        H1 = AS(cell)
        L2 = CG(ref_complex, 1)

    pts = []
    top = ref_complex.get_topology()
    for dim in top:
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, 4))

    H2tab = H2.tabulate(1, pts)
    H1tab = H1.tabulate(1, pts)
    L2tab = L2.tabulate(0, pts)

    L2val = L2tab[(0, 0)]
    H1val = H1tab[(0, 0)]
    H1div = sum(H1tab[alpha][:, alpha.index(1), :] for alpha in H1tab if sum(alpha) == 1)
    H2curl = numpy.stack([H2tab[(0, 1)], -H2tab[(1, 0)]], axis=1)

    H2dim = H2.space_dimension()
    H1dim = H1.space_dimension()
    _, residual, *_ = numpy.linalg.lstsq(H1val.reshape(H1dim, -1).T, H2curl.reshape(H2dim, -1).T)
    assert numpy.allclose(residual, 0)

    _, residual, *_ = numpy.linalg.lstsq(L2val.T, H1div.T)
    assert numpy.allclose(residual, 0)
