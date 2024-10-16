# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, polynomial_set
from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TRIANGLE
from FIAT.macro import CkPolynomialSet
from FIAT.hct import HsiehCloughTocher

import numpy


def ArnoldQinSpace(ref_el, degree, reduced=False):
    """Return a basis for the Arnold-Qin space
    curl(HCT-red) + P0 x if reduced = True, and
    curl(HCT) + P0 x if reduced = False."""
    if ref_el.get_shape() != TRIANGLE:
        raise ValueError("Arnold-Qin only defined on triangles")
    if degree != 2:
        raise ValueError("Arnold-Qin only defined for degree = 2")
    sd = ref_el.get_spatial_dimension()
    HCT = HsiehCloughTocher(ref_el, degree+1, reduced=True)
    ref_complex = HCT.get_reference_complex()
    Q = create_quadrature(ref_complex, 2 * degree)
    qpts, qwts = Q.get_points(), Q.get_weights()

    x = qpts.T
    bary = numpy.asarray(ref_el.make_points(sd, 0, sd+1))
    P0x_at_qpts = x[None, :, :] - bary[:, :, None]

    tab = HCT.tabulate(1, qpts)
    curl_at_qpts = numpy.stack([tab[(0, 1)], -tab[(1, 0)]], axis=1)
    if reduced:
        curl_at_qpts = curl_at_qpts[:9]

    C0 = CkPolynomialSet(ref_complex, degree, order=0, scale=1, variant="bubble")
    C0_at_qpts = C0.tabulate(qpts)[(0,) * sd]
    duals = numpy.multiply(C0_at_qpts, qwts)
    M = numpy.dot(duals, C0_at_qpts.T)
    duals = numpy.linalg.solve(M, duals)

    # Remove the constant nullspace
    ids = [0, 3, 6]
    A = numpy.asarray([[1, 1, 1], [1, -1, 0], [0, -1, 1]])
    phis = curl_at_qpts
    phis[ids] = numpy.tensordot(A, phis[ids], axes=(-1, 0))
    # Replace the constant nullspace with P_0 x
    phis[0] = P0x_at_qpts
    coeffs = numpy.tensordot(phis, duals, axes=(-1, -1))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                        C0.get_expansion_set(), coeffs)


class ArnoldQin(finite_element.CiarletElement):
    """The Arnold-Qin C0(Alfeld) quadratic macroelement with divergence in P0.
    This element belongs to a Stokes complex, and is paired with unsplit DG0."""
    def __init__(self, ref_el, degree=2, reduced=False):
        poly_set = ArnoldQinSpace(ref_el, degree)
        if reduced:
            order = 1
            mapping = "contravariant piola"
        else:
            order = degree
            mapping = "affine"
        dual = BernardiRaugelDualSet(ref_el, order, degree=degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
