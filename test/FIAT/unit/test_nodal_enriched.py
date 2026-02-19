import numpy
from FIAT import (NodalEnrichedElement, RestrictedElement,
                  BernardiRaugel, GuzmanNeilanFirstKindH1,
                  ufc_simplex)


def test_nodal_enriched_mismatching_expansion_set():
    sd = 2
    ref_el = ufc_simplex(sd)

    # Extract (non-macro) vector P1 from BR
    BR = BernardiRaugel(ref_el, 1, hierarchical=True)
    P1 = RestrictedElement(BR, restriction_domain="vertex", take_closure=False)

    # Extract the macro face bubbles from GN
    GN = GuzmanNeilanFirstKindH1(ref_el, 1)
    MFB = RestrictedElement(GN, restriction_domain="facet", take_closure=False)

    # Add two elements with mismatching expansion sets
    # The resulting element should be identical to GN
    fe = NodalEnrichedElement(P1, MFB)

    # Test nodality
    coeffs = fe.poly_set.get_coeffs()
    e = numpy.tensordot(GN.dual.to_riesz(fe.poly_set),
                        coeffs, axes=(range(1, coeffs.ndim),)*2)
    assert numpy.allclose(e, numpy.eye(*e.shape))

    # Test that the spaces are equal
    degree = GN.degree()
    ref_complex = GN.get_reference_complex()
    top = ref_complex.topology
    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree))

    result = fe.tabulate(0, pts)[(0,)*sd]
    expected = GN.tabulate(0, pts)[(0,)*sd]
    assert numpy.allclose(result, expected)
