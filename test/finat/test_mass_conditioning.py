import finat
import numpy as np
import pytest
from gem.interpreter import evaluate


@pytest.mark.parametrize("element, degree, variant", [
    (finat.Hermite, 3, None),
    (finat.QuadraticPowellSabin6, 2, None),
    (finat.QuadraticPowellSabin12, 2, None),
    (finat.ReducedHsiehCloughTocher, 3, None),
    (finat.HsiehCloughTocher, 3, None),
    (finat.HsiehCloughTocher, 4, None),
    (finat.Bell, 5, None),
    (finat.Argyris, 5, "point"),
    (finat.Argyris, 5, None),
    (finat.Argyris, 6, None),
])
def test_mass_scaling(scaled_ref_to_phys, element, degree, variant):
    sd = 2
    ref_cell = scaled_ref_to_phys[sd][0].ref_cell
    if variant is not None:
        ref_element = element(ref_cell, degree, variant=variant)
    else:
        ref_element = element(ref_cell, degree)

    Q = finat.quadrature.make_quadrature(ref_cell, 2*degree)
    qpts = Q.point_set
    qwts = Q.weights

    kappa = []
    for mapping in scaled_ref_to_phys[sd]:
        J_gem = mapping.jacobian_at(ref_cell.make_points(sd, 0, sd+1)[0])
        J = evaluate([J_gem])[0].arr

        z = (0,) * ref_element.cell.get_spatial_dimension()
        finat_vals_gem = ref_element.basis_evaluation(0, qpts, coordinate_mapping=mapping)[z]
        phis = evaluate([finat_vals_gem])[0].arr.T

        M = np.dot(np.multiply(phis, qwts * abs(np.linalg.det(J))), phis.T)
        kappa.append(np.linalg.cond(M))

    kappa = np.array(kappa)
    ratio = kappa[1:] / kappa[:-1]
    assert np.allclose(ratio, 1, atol=0.1)
