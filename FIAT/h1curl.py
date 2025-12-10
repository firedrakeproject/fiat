# Copyright (C) 2024 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import ComponentPointEvaluation, PointDivergence
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import CkPolynomialSet, AlfeldSplit

import numpy


def H1CurlSpace(ref_el, degree):
    """Return a vector-valued C0 PolynomialSet on an Alfeld split with C0
    curl. This works on any simplex and for all polynomial degrees."""
    ref_complex = AlfeldSplit(ref_el)
    sd = ref_complex.get_spatial_dimension()
    C0 = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), variant="bubble")
    expansion_set = C0.get_expansion_set()
    num_members = C0.get_num_members()
    coeffs = C0.get_coeffs()

    facet_el = ref_complex.construct_subelement(sd-1)
    phi = polynomial_set.ONPolynomialSet(facet_el, 0 if sd == 1 else degree-1)
    Q = create_quadrature(facet_el, 2 * phi.degree)
    qpts, qwts = Q.get_points(), Q.get_weights()
    phi_at_qpts = phi.tabulate(qpts)[(0,) * (sd-1)]
    weights = numpy.multiply(phi_at_qpts, qwts)

    rows = []
    for facet in ref_complex.get_interior_facets(sd-1):
        ts = ref_complex.compute_tangents(sd-1, facet)
        jumps = expansion_set.tabulate_normal_jumps(degree, qpts, facet, order=1)
        for t in ts:
            tjump = t[:, None, None] * jumps[1][None, ...]
            r = numpy.tensordot(tjump, weights, axes=(-1, -1))
            rows.append(r.reshape(num_members, -1).T)

    if len(rows) > 0:
        dual_mat = numpy.vstack(rows)
        nsp = polynomial_set.spanning_basis(dual_mat, nullspace=True)
        coeffs = numpy.tensordot(nsp, coeffs, axes=(-1, 0))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, coeffs)


if __name__ == "__main__":
    from FIAT import ufc_simplex
    for dim in range(2, 4):
        for degree in range(2, 5):
            ref_el = ufc_simplex(dim)
            V = H1CurlSpace(ref_el, degree)
            print(dim, degree, len(V))
