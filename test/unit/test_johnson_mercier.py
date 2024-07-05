import pytest
import numpy

from FIAT import JohnsonMercier, Nedelec
from FIAT.reference_element import ufc_simplex
from FIAT.quadrature_schemes import create_quadrature


@pytest.fixture(params=("T-ref", "T-phys", "S-ref", "S-phys"))
def cell(request):
    cell, deform = request.param.split("-")
    dim = {"T": 2, "S": 3}[cell]
    K = ufc_simplex(dim)
    if deform == "phys":
        if dim == 2:
            K.vertices = ((0.0, 0.0), (2.0, 0.1), (0.0, 1.0))
        else:
            K.vertices = ((0, 0, 0), (1., 0.1, -0.37),
                          (0.01, 0.987, -.23), (-0.1, -0.2, 1.38))
    return K


def test_johnson_mercier_divergence_rigid_body_motions(cell):
    # test that the divergence of interior JM basis functions is orthogonal to
    # the rigid-body motions
    degree = 1
    variant = None
    sd = cell.get_spatial_dimension()
    JM = JohnsonMercier(cell, degree, variant=variant)

    ref_complex = JM.get_reference_complex()
    Q = create_quadrature(ref_complex, 2*(degree)-1)
    qpts, qwts = Q.get_points(), Q.get_weights()

    tab = JM.tabulate(1, qpts)
    div = sum(tab[alpha][:, alpha.index(1), :, :] for alpha in tab if sum(alpha) == 1)

    # construct rigid body motions
    N1 = Nedelec(cell, 1)
    N1_at_qpts = N1.tabulate(0, qpts)
    rbms = N1_at_qpts[(0,)*sd]
    ells = rbms * qwts[None, None, :]

    edofs = JM.entity_dofs()
    idofs = edofs[sd][0]
    L = numpy.tensordot(div, ells, axes=((1, 2), (1, 2)))
    assert numpy.allclose(L[idofs], 0)

    if variant == "divergence":
        edofs = JM.entity_dofs()
        cdofs = []
        for entity in edofs[sd-1]:
            cdofs.extend(edofs[sd-1][entity][:sd])
        D = L[cdofs]
        M = numpy.tensordot(rbms, ells, axes=((1, 2), (1, 2)))
        X = numpy.linalg.solve(M, D.T)
        assert numpy.allclose(numpy.tensordot(X, rbms, axes=(0, 0)), div[cdofs])
