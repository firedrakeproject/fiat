import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate


def make_unisolvent_points(element, interior=False):
    degree = element.degree()
    ref_complex = element.get_reference_complex()
    top = ref_complex.get_topology()
    pts = []
    if interior:
        dim = ref_complex.get_spatial_dimension()
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree+dim+1, variant="gll"))
    else:
        for dim in top:
            for entity in top[dim]:
                pts.extend(ref_complex.make_points(dim, entity, degree, variant="gll"))
    return pts


def check_zany_mapping(element, ref_to_phys, *args, **kwargs):
    phys_cell = ref_to_phys.phys_cell
    ref_cell = ref_to_phys.ref_cell
    phys_element = element(phys_cell, *args, **kwargs).fiat_equivalent
    finat_element = element(ref_cell, *args, **kwargs)

    ref_element = finat_element._element
    ref_cell = ref_element.get_reference_element()
    phys_cell = phys_element.get_reference_element()
    sd = ref_cell.get_spatial_dimension()

    shape = ref_element.value_shape()
    ref_pts = make_unisolvent_points(ref_element, interior=True)
    ref_vals = ref_element.tabulate(0, ref_pts)[(0,)*sd]

    phys_pts = make_unisolvent_points(phys_element, interior=True)
    phys_vals = phys_element.tabulate(0, phys_pts)[(0,)*sd]

    mapping = ref_element.mapping()[0]
    if mapping == "affine":
        ref_vals_piola = ref_vals
    else:
        # Piola map the reference elements
        J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                          phys_cell.vertices)
        K = []
        if "covariant" in mapping:
            K.append(np.linalg.inv(J).T)
        if "contravariant" in mapping:
            K.append(J / np.linalg.det(J))

        if len(shape) == 2:
            piola_map = lambda x: K[0] @ x @ K[-1].T
        else:
            piola_map = lambda x: K[0] @ x

        ref_vals_piola = np.zeros(ref_vals.shape)
        for i in range(ref_vals.shape[0]):
            for k in range(ref_vals.shape[-1]):
                ref_vals_piola[i, ..., k] = piola_map(ref_vals[i, ..., k])

    # Zany map the results
    num_bfs = phys_element.space_dimension()
    num_dofs = finat_element.space_dimension()
    try:
        Mgem = finat_element.basis_transformation(ref_to_phys)
        M = evaluate([Mgem])[0].arr
        ref_vals_zany = np.tensordot(M, ref_vals_piola, (-1, 0))
    except AttributeError:
        M = np.eye(num_dofs, num_bfs)
        ref_vals_zany = ref_vals_piola

    # Solve for the basis transformation and compare results
    Phi = ref_vals_piola.reshape(num_bfs, -1)
    phi = phys_vals.reshape(num_bfs, -1)
    Vh, residual, *_ = np.linalg.lstsq(Phi.T, phi.T)
    Mh = Vh.T
    Mh = Mh[:num_dofs]
    Mh[abs(Mh) < 1E-10] = 0
    M[abs(M) < 1E-10] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        error = M.T / Mh.T - 1
    error[error != error] = 0
    error[abs(error) < 1E-10] = 0
    error = error[np.ix_(*map(np.unique, np.nonzero(error)))]
    error[error != 0] += 1

    assert np.allclose(residual, 0), str(error)
    assert np.allclose(ref_vals_zany, phys_vals[:num_dofs]), str(error)


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.Hermite,
                         finat.Bell,
                         ])
def test_C1_triangle(ref_to_phys, element):
    check_zany_mapping(element, ref_to_phys[2])


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         ])
def test_C1_tetrahedron(ref_to_phys, element):
    check_zany_mapping(element, ref_to_phys[3])


@pytest.mark.parametrize("element", [
                         finat.QuadraticPowellSabin6,
                         finat.QuadraticPowellSabin12,
                         finat.ReducedHsiehCloughTocher,
                         ])
def test_C1_macroelements(ref_to_phys, element):
    kwargs = {}
    if element == finat.QuadraticPowellSabin12:
        kwargs = dict(avg=True)
    check_zany_mapping(element, ref_to_phys[2], **kwargs)


@pytest.mark.parametrize("element, degree", [
    *((finat.Argyris, k) for k in range(5, 8)),
    *((finat.HsiehCloughTocher, k) for k in range(3, 6))
])
def test_high_order_C1_elements(ref_to_phys, element, degree):
    check_zany_mapping(element, ref_to_phys[2], degree, avg=True)


def test_argyris_point(ref_to_phys):
    check_zany_mapping(finat.Argyris, ref_to_phys[2], variant="point")


zany_piola_elements = {
    2: [
        finat.MardalTaiWinther,
        finat.ReducedArnoldQin,
        finat.ArnoldWinther,
        finat.ArnoldWintherNC,
    ],
    3: [
        finat.BernardiRaugel,
        finat.BernardiRaugelBubble,
        finat.AlfeldSorokina,
        finat.ChristiansenHu,
        finat.JohnsonMercier,
        finat.GuzmanNeilanFirstKindH1,
        finat.GuzmanNeilanSecondKindH1,
        finat.GuzmanNeilanBubble,
        finat.GuzmanNeilanH1div,
    ],
}


@pytest.mark.parametrize("dimension, element", [
                         *((2, e) for e in zany_piola_elements[2]),
                         *((2, e) for e in zany_piola_elements[3]),
                         *((3, e) for e in zany_piola_elements[3]),
                         ])
def test_piola(ref_to_phys, element, dimension):
    check_zany_mapping(element, ref_to_phys[dimension])


@pytest.mark.parametrize("element, degree, variant", [
    *((finat.HuZhang, k, v) for v in ("integral", "point") for k in range(3, 6)),
])
def test_piola_triangle_high_order(ref_to_phys, element, degree, variant):
    check_zany_mapping(element, ref_to_phys[2], degree, variant)


@pytest.mark.parametrize("element, degree", [
                         *((finat.Regge, k) for k in range(3)),
                         *((finat.HellanHerrmannJohnson, k) for k in range(3)),
                         *((finat.GopalakrishnanLedererSchoberlFirstKind, k) for k in range(1, 4)),
                         *((finat.GopalakrishnanLedererSchoberlSecondKind, k) for k in range(0, 3)),
                         ])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("variant", [None, "alfeld"])
def test_affine(ref_to_phys, element, degree, variant, dimension):
    check_zany_mapping(element, ref_to_phys[dimension], degree, variant=variant)


@pytest.mark.parametrize("element", [finat.BrezziDouglasMarini, finat.NedelecSecondKind])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("variant", [None, "iso"])
def test_macro_piola(ref_to_phys, element, degree, variant, dimension):
    check_zany_mapping(element, ref_to_phys[dimension], degree, variant=variant)
