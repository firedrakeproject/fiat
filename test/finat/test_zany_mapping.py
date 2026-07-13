import finat
import pytest


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.Hermite,
                         finat.Bell,
                         finat.WuXuH3NC,
                         finat.WuXuRobustH3NC,
                         ])
def test_C1_triangle(check_zany_mapping, ref_to_phys, element):
    check_zany_mapping(element, ref_to_phys[2])


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.Walkington,
                         ])
def test_C1_tetrahedron(check_zany_mapping, ref_to_phys, element):
    check_zany_mapping(element, ref_to_phys[3])


@pytest.mark.parametrize("element", [
                         finat.QuadraticPowellSabin6,
                         finat.QuadraticPowellSabin12,
                         finat.ReducedHsiehCloughTocher,
                         ])
def test_C1_macroelements(check_zany_mapping, ref_to_phys, element):
    kwargs = {}
    if element == finat.QuadraticPowellSabin12:
        kwargs = dict(avg=True)
    check_zany_mapping(element, ref_to_phys[2], **kwargs)


@pytest.mark.parametrize("element, degree", [
    *((finat.Argyris, k) for k in range(5, 8)),
    *((finat.HsiehCloughTocher, k) for k in range(3, 6)),
    *((finat.AlfeldC2, k) for k in range(5, 7)),
    *((finat.BrambleZlamalC2, k) for k in range(9, 11)),
])
def test_high_order_Ck_elements(check_zany_mapping, ref_to_phys, element, degree):
    check_zany_mapping(element, ref_to_phys[2], degree, avg=True)


def test_argyris_point(check_zany_mapping, ref_to_phys):
    check_zany_mapping(finat.Argyris, ref_to_phys[2], variant="point")


zany_piola_elements = {
    2: [
        finat.ReducedArnoldQin,
        finat.ArnoldWinther,
        finat.ArnoldWintherNC,
    ],
    3: [
        finat.MardalTaiWinther,
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
def test_piola(check_zany_mapping, ref_to_phys, element, dimension):
    check_zany_mapping(element, ref_to_phys[dimension])


@pytest.mark.parametrize("dimension, element, degree", [
    (3, finat.MardalTaiWinther, 2),
    (3, finat.GuzmanNeilanFirstKindH1, 2),
])
def test_high_order_stokes_elements(check_zany_mapping, ref_to_phys, element, dimension, degree):
    check_zany_mapping(element, ref_to_phys[dimension], degree)


@pytest.mark.parametrize("element, degree, variant", [
    *((finat.HuZhang, k, v) for v in ("integral", "point") for k in range(3, 6)),
])
def test_piola_triangle_high_order(check_zany_mapping, ref_to_phys, element, degree, variant):
    check_zany_mapping(element, ref_to_phys[2], degree, variant)


@pytest.mark.parametrize("element, degree", [
                         *((finat.Regge, k) for k in range(3)),
                         *((finat.HellanHerrmannJohnson, k) for k in range(3)),
                         *((finat.GopalakrishnanLedererSchoberlFirstKind, k) for k in range(1, 4)),
                         *((finat.GopalakrishnanLedererSchoberlSecondKind, k) for k in range(0, 3)),
                         ])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("variant", [None, "alfeld"])
def test_affine(check_zany_mapping, ref_to_phys, element, degree, variant, dimension):
    check_zany_mapping(element, ref_to_phys[dimension], degree, variant=variant)


@pytest.mark.parametrize("element", [finat.BrezziDouglasMarini, finat.NedelecSecondKind])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("variant", [None, "iso"])
def test_macro_piola(check_zany_mapping, ref_to_phys, element, degree, variant, dimension):
    check_zany_mapping(element, ref_to_phys[dimension], degree, variant=variant)
