import numpy
from FIAT import (NodalEnrichedElement, RestrictedElement,
                  BernardiRaugel, GuzmanNeilanFirstKindH1,
                  ufc_simplex)


def test_nodal_enriched_mismatching_expansion_set():
    # Add two elements with mismatching expansion sets
    sd = 2
    ref_el = ufc_simplex(sd)

    BR = BernardiRaugel(ref_el, 1, hierarchical=True)
    P1 = RestrictedElement(BR, restriction_domain="vertex", take_closure=False)

    GN = GuzmanNeilanFirstKindH1(ref_el, 1)
    GNB = RestrictedElement(GN, restriction_domain="facet", take_closure=False)

    fe = NodalEnrichedElement(P1, GNB)
    coeffs = fe.poly_set.get_coeffs()
    e = numpy.tensordot(GN.dual.to_riesz(fe.poly_set),
                        coeffs, axes=(range(1, coeffs.ndim),)*2)
    e[abs(e) < 1E-12] = 0
    print(e)

    degree = GN.degree()
    ref_complex = GN.get_reference_complex()
    top = ref_complex.topology
    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree))

    result = fe.tabulate(0, pts)[(0,)*sd]
    expected = GN.tabulate(0, pts)[(0,)*sd]

    result -= expected
    expected = 0
    result[abs(result) < 1E-12] = 0
    result = result.reshape((result.shape[0], -1))
    assert numpy.allclose(result, expected)
