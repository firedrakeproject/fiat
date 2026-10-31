import pytest
import numpy

from FIAT import JohnsonMercier, ReducedJohnsonMercier, Nedelec, polynomial_set
from FIAT.reference_element import ufc_simplex
from FIAT.quadrature import FacetQuadratureRule
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


@pytest.mark.parametrize("reduced", (False, True))
def test_johnson_mercier_divergence_rigid_body_motions(cell, reduced):
    # test that the divergence of interior JM basis functions is orthogonal to
    # the rigid-body motions
    degree = 1
    sd = cell.get_spatial_dimension()
    if reduced:
        JM = ReducedJohnsonMercier(cell, degree)
    else:
        JM = JohnsonMercier(cell, degree)

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
    L = numpy.tensordot(div, ells, axes=((1, 2), (1, 2)))

    edofs = JM.entity_dofs()
    idofs = list(edofs[sd][0])
    L = L[idofs].reshape(len(idofs), -1)
    assert numpy.allclose(L, 0)


def reduced_space_dimension(cell):
    # Normal-normal moments against P1(F) and normal-tangential moments
    # against RM(F) on each facet F
    sd = cell.get_spatial_dimension()
    return (sd + 1) * (sd + (sd * (sd - 1)) // 2)


def test_reduced_johnson_mercier_divergence(cell):
    # test that div(Sigma^R) = P_T RM(T), where P_T is the L2 projection
    # onto piecewise constants on the Alfeld split
    sd = cell.get_spatial_dimension()
    tol = 1E-10
    JM = ReducedJohnsonMercier(cell)
    space_dim = reduced_space_dimension(cell)

    ref_complex = JM.get_reference_complex()
    Q = create_quadrature(ref_complex, 1)
    qpts, qwts = Q.get_points(), Q.get_weights()

    tab = JM.tabulate(1, qpts)
    div = sum(tab[alpha][:space_dim, alpha.index(1)] for alpha in tab if sum(alpha) == 1)

    rbms = Nedelec(cell, 1).tabulate(0, qpts)[(0,) * sd]
    P0 = polynomial_set.ONPolynomialSet(ref_complex, 0, shape=(sd,))
    P0_at_qpts = P0.tabulate(qpts)[(0,) * sd]
    ells = P0_at_qpts * qwts[None, None, :]

    # Coefficients of div(Sigma^R) and P_T RM in the basis for P0(Alfeld)
    M = numpy.tensordot(ells, P0_at_qpts, axes=((1, 2), (1, 2)))
    div_coeffs = numpy.linalg.solve(M, numpy.tensordot(ells, div, axes=((1, 2), (1, 2))))
    rbm_coeffs = numpy.linalg.solve(M, numpy.tensordot(ells, rbms, axes=((1, 2), (1, 2))))

    num_rbms = len(rbms)
    assert numpy.linalg.matrix_rank(rbm_coeffs, tol=tol) == num_rbms
    assert numpy.linalg.matrix_rank(div_coeffs, tol=tol) == num_rbms
    assert numpy.linalg.matrix_rank(numpy.hstack((rbm_coeffs, div_coeffs)), tol=tol) == num_rbms


def test_reduced_johnson_mercier_traction(cell):
    # test that the traction of Sigma^R lies in n P1(F) + RM(F) on each facet F,
    # and that Sigma^R is determined by its traction
    sd = cell.get_spatial_dimension()
    tol = 1E-10
    JM = ReducedJohnsonMercier(cell)
    space_dim = reduced_space_dimension(cell)

    ref_facet = cell.construct_subelement(sd-1)
    Qref = create_quadrature(ref_facet, 2)
    P1 = polynomial_set.ONPolynomialSet(ref_facet, 1)
    P1_at_qpts = P1.tabulate(Qref.get_points())[(0,) * (sd-1)]
    N1 = Nedelec(cell, 1)

    tractions = []
    for f in sorted(cell.get_topology()[sd-1]):
        Qf = FacetQuadratureRule(cell, sd-1, f, Qref)
        qpts = Qf.get_points()
        n = cell.compute_scaled_normal(f)

        sig = JM.tabulate(0, qpts)[(0,) * sd][:space_dim]
        sig_n = numpy.tensordot(sig, n, axes=(2, 0))
        tractions.append(sig_n)

        # The traces of RM(T) span n P1(F) + RM(F)
        rbms = N1.tabulate(0, qpts)[(0,) * sd]
        n_P1 = n[None, :, None] * P1_at_qpts[:, None, :]
        trace_space = numpy.concatenate((n_P1, rbms)).reshape(-1, sd * len(qpts))
        trace_dim = numpy.linalg.matrix_rank(trace_space, tol=tol)
        assert trace_dim == sd + (sd * (sd - 1)) // 2

        sig_n = sig_n.reshape(space_dim, -1)
        assert numpy.linalg.matrix_rank(numpy.vstack((trace_space, sig_n)), tol=tol) == trace_dim

    tractions = numpy.concatenate(tractions, axis=-1).reshape(space_dim, -1)
    assert numpy.linalg.matrix_rank(tractions, tol=tol) == space_dim
