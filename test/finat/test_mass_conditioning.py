import finat
import numpy as np
import pytest
from gem.interpreter import evaluate


@pytest.mark.parametrize("sd,element,degree,variant", [
    (2, finat.Hermite, 3, None),
    (2, finat.ArnoldWintherNC, 2, None),
    (2, finat.ArnoldWinther, 3, None),
    (2, finat.HuZhang, 3, "integral"),
    (2, finat.HuZhang, 3, "point"),
    (2, finat.QuadraticPowellSabin6, 2, None),
    (2, finat.QuadraticPowellSabin12, 2, None),
    (2, finat.ReducedHsiehCloughTocher, 3, None),
    (2, finat.HsiehCloughTocher, 3, None),
    (2, finat.HsiehCloughTocher, 4, None),
    (2, finat.Bell, 5, None),
    (2, finat.Argyris, 5, "point"),
    (2, finat.Argyris, 5, None),
    (2, finat.Argyris, 6, None),
    (2, finat.WuXuH3NC, 4, None),
    (2, finat.WuXuRobustH3NC, 7, None),
    (2, finat.BrambleZlamalC2, 9, None),
    (2, finat.AlfeldC2, 5, None),
    (3, finat.Walkington, 5, None),
])
def test_mass_scaling(scaled_ref_to_phys, sd, element, degree, variant):
    ref_cell = scaled_ref_to_phys[sd][0].ref_cell
    if variant is not None:
        ref_element = element(ref_cell, degree, variant=variant)
    else:
        ref_element = element(ref_cell, degree)

    Q = finat.quadrature.make_quadrature(ref_element.complex, 2*degree)
    qpts = Q.point_set
    qwts = Q.weights

    kappa = []
    for mapping in scaled_ref_to_phys[sd]:
        J_gem = mapping.jacobian_at(ref_cell.make_points(sd, 0, sd+1)[0])
        J = evaluate([J_gem])[0].arr

        z = (0,) * ref_element.cell.get_spatial_dimension()
        finat_vals_gem = ref_element.basis_evaluation(0, qpts, coordinate_mapping=mapping)[z]
        value_size = np.prod(ref_element.value_shape, dtype=int)
        phis = evaluate([finat_vals_gem])[0].arr.reshape(len(qwts), -1, value_size)

        weighted_phis = phis * (qwts * abs(np.linalg.det(J)))[:, None, None]
        M = np.tensordot(weighted_phis, phis, axes=((0, 2), (0, 2)))
        kappa.append(np.linalg.cond(M))

    kappa = np.array(kappa)
    ratio = kappa[1:] / kappa[:-1]
    assert np.allclose(ratio, 1, atol=0.1)
