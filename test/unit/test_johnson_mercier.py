import pytest
import numpy

from FIAT import JohnsonMercier, Nedelec, polynomial_set
from FIAT.reference_element import ufc_simplex
from FIAT.quadrature_schemes import create_quadrature


@pytest.fixture(params=("T-ref", "T-phys", "S-ref", "S-phis"))
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


@pytest.mark.parametrize("variant", (None, "divergence"))
def test_johnson_mercier_divergence_rigid_body_motions(cell, variant):
    # test that the divergence of interior JM basis functions is orthogonal to
    # the rigid-body motions
    degree = 1
    sd = cell.get_spatial_dimension()
    JM = JohnsonMercier(cell, degree, variant=variant)

    ref_complex = JM.get_reference_complex()
    Q = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = Q.get_points(), Q.get_weights()

    tab = JM.tabulate(1, qpts)
    div = sum(tab[alpha][:, alpha.index(1), :, :] for alpha in tab if sum(alpha) == 1)

    # construct rigid body motions
    N1 = Nedelec(cell, 1)
    rbms = N1.tabulate(0, qpts)[(0,)*sd]
    ells = rbms * qwts[None, None, :]

    edofs = JM.entity_dofs()
    idofs = edofs[sd][0]
    fdofs = numpy.setdiff1d(numpy.arange(JM.space_dimension()), idofs)
    L = numpy.tensordot(div[idofs, ...], ells, axes=((1, 2), (1, 2)))
    assert numpy.allclose(L, 0)

    if variant == "divergence":
        # construct the L2-orthogonal complement of the rigid-body motions
        P = polynomial_set.ONPolynomialSet(ref_complex, degree-1, shape=(sd,),
                                           scale="orthonormal")
        phis = P.tabulate(qpts)[(0,) * sd]
        dual_mat = numpy.tensordot(ells, phis, axes=((1, 2), (1, 2)))

        u, sig, vt = numpy.linalg.svd(dual_mat, full_matrices=True)
        num_sv = len([s for s in sig if abs(s) > 1.e-10])

        cphis = numpy.tensordot(vt[num_sv:], phis, axes=(1, 0))
        perp_ells = cphis * qwts[None, None, :]
        # test L2-orthogonality of the complement of rigid-body motions
        assert numpy.allclose(numpy.tensordot(rbms, perp_ells, axes=((1, 2), (1, 2))), 0)

        # Test Kronecker property for the internal divergence DOFs
        Lperp = numpy.tensordot(div, perp_ells, axes=((1, 2), (1, 2)))
        assert numpy.allclose(Lperp[fdofs], 0)
        assert numpy.allclose(Lperp[idofs], numpy.eye(len(ells)))
