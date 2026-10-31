import numpy as np
import pytest
from FIAT import ufc_simplex, MardalTaiWinther, Nedelec, expansions, polynomial_set
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


def test_dofs():
    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.asarray([(0.0, 0.0), (1.0, 0.25), (-0.75, 1.1)])
    MTW = MardalTaiWinther(T)
    assert MTW.degree() == T.get_spatial_dimension()+1

    Qline = create_quadrature(line, 2*MTW.degree())

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        wts = np.asarray(Qline.wts)

        vals = MTW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nvals = np.dot(np.transpose(vals, (0, 2, 1)), n)
        normal_moments = np.zeros((9, 2))
        for bf in range(9):
            for k in range(len(Qline.wts)):
                for m in (0, 1):
                    normal_moments[bf, m] += wts[k] * nvals[bf, k] * linevals[m, k]
        right = np.zeros((9, 2))
        right[3*ed, 0] = 1.0
        right[3*ed+1, 1] = 1.0
        assert np.allclose(normal_moments, right)
    for ed in range(3):
        t = T.compute_edge_tangent(ed)
        wts = np.asarray(Qline.wts)

        vals = MTW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        tvals = np.dot(np.transpose(vals, (0, 2, 1)), t)
        tangent_moments = np.zeros(9)
        for bf in range(9):
            for k in range(len(Qline.wts)):
                tangent_moments[bf] += wts[k] * tvals[bf, k] * linevals[0, k]
        right = np.zeros(9)
        right[3*ed+2] = 1.0
        assert np.allclose(tangent_moments, right)


@pytest.mark.parametrize("deform", (False, True))
def test_tai_winther_facet_dofs(deform):
    # The Tai-Winther facet functionals are the moments of v.n against P1(f)
    # and of the tangential component v_t against the rigid motions RM(f).
    # They must only see the dofs of the facet f.
    K = ufc_simplex(3)
    if deform:
        K.vertices = ((0, 0, 0), (1., 0.1, -0.37), (0.01, 0.987, -.23), (-0.1, -0.2, 1.38))
    sd = K.get_spatial_dimension()
    tol = 1E-10
    MTW = MardalTaiWinther(K, 1)

    ref_facet = K.construct_subelement(sd-1)
    Qref = create_quadrature(ref_facet, MTW.degree() + 1)
    P1 = polynomial_set.ONPolynomialSet(ref_facet, 1)
    P1_at_qpts = P1.tabulate(Qref.get_points())[(0,) * (sd-1)]
    N1 = Nedelec(K, 1)

    for f, dofs in MTW.entity_dofs()[sd-1].items():
        Qf = FacetQuadratureRule(K, sd-1, f, Qref)
        qpts, qwts = Qf.get_points(), Qf.get_weights()
        n = K.compute_scaled_normal(f)

        # The traces of RM(T) span n P1(f) + RM(f)
        rbms = N1.tabulate(0, qpts)[(0,) * sd]
        n_P1 = n[None, :, None] * P1_at_qpts[:, None, :]
        test_functions = np.concatenate((n_P1, rbms))

        vals = MTW.tabulate(0, qpts)[(0,) * sd]
        moments = np.tensordot(vals * qwts, test_functions, axes=((1, 2), (1, 2)))

        others = [i for i in range(MTW.space_dimension()) if i not in dofs]
        assert np.allclose(moments[others], 0, atol=tol)
        assert np.linalg.matrix_rank(moments[dofs], tol=tol) == len(dofs)
