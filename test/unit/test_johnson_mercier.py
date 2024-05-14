import pytest
import numpy

from FIAT import JohnsonMercier, Nedelec
from FIAT.reference_element import ufc_simplex
from FIAT.quadrature_schemes import create_quadrature


@pytest.fixture(params=("T", "S"))
def cell(request):
    dim = {"T": 2, "S": 3}[request.param]
    return ufc_simplex(dim)


def test_johnson_mercier_divergence_rigid_body_motions(cell):
    # test that the divergence of interior JM basis functions is orthogonal to
    # the rigid-body motions
    sd = cell.get_spatial_dimension()
    JM = JohnsonMercier(cell, 1, variant="divergence")

    ref_complex = JM.get_reference_complex()
    Q = create_quadrature(ref_complex, 2)
    qpts, qwts = Q.get_points(), Q.get_weights()

    tab = JM.tabulate(1, qpts)
    div = sum(tab[alpha][:, alpha.index(1), :, :] for alpha in tab if sum(alpha) == 1)

    # construct rigid body motions
    N1 = Nedelec(cell, 1)
    rbms = N1.tabulate(0, qpts)
    ells = rbms[(0,)*sd] * qwts[None, None, :]

    edofs = JM.entity_dofs()
    idofs = edofs[sd][0]
    L = numpy.tensordot(ells, div[idofs, ...], axes=((1, 2), (1, 2)))
    assert numpy.allclose(L, 0)
