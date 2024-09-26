# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, polynomial_set
from FIAT.hct import HsiehCloughTocher
from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import PowellSabinSplit, CkPolynomialSet

import numpy


def ArnoldQinSpace2D(ref_el, degree):
    """Return a basis for the Arnold-Qin space.
    curl(HCT) + P_0 x"""
    sd = ref_el.get_spatial_dimension()

    HCT = HsiehCloughTocher(ref_el, degree+1, reduced=True)
    ref_complex = HCT.get_reference_complex()
    Q = create_quadrature(ref_complex, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    x = Qpts.T
    bary = numpy.asarray(ref_el.make_points(sd, 0, sd+1))
    P0x_at_Qpts = x[None, :, :] - bary[:, :, None]

    tab = HCT.tabulate(1, Qpts)
    curl_at_qpts = numpy.stack([tab[(0, 1)], -tab[(1, 0)]], axis=1)

    Pk = polynomial_set.ONPolynomialSet(ref_complex, degree, scale=1, variant="bubble")
    Pk_at_Qpts = Pk.tabulate(Qpts)[(0,) * sd]
    duals = numpy.multiply(Pk_at_Qpts, Qwts)
    M = numpy.dot(duals, Pk_at_Qpts.T)
    duals = numpy.linalg.solve(M, duals)

    # Remove the constant nullspace
    ids = [0, 3, 6]
    A = numpy.asarray([[1, 1, 1], [1, -1, 0], [0, -1, 1]])
    phis = curl_at_qpts
    phis[ids] = numpy.tensordot(A, phis[ids], axes=(-1, 0))
    # Replace the constant nullspace with P_0 x
    phis[0] = P0x_at_Qpts
    coeffs = numpy.tensordot(phis, duals, axes=(-1, -1))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree, Pk.get_expansion_set(), coeffs)


def ArnoldQinSpace(ref_el, degree):
    """Return a basis for the Arnold-Qin space.
    v in C0 P1(WF)^d : div(v) in P_0"""
    sd = ref_el.get_spatial_dimension()
    if sd == 2:
        return ArnoldQinSpace2D(ref_el, degree)

    shp = (sd,)
    ref_complex = PowellSabinSplit(ref_el, codim=2)
    Q = create_quadrature(ref_complex, degree)
    qpts, qwts = Q.get_points(), Q.get_weights()

    C0 = CkPolynomialSet(ref_complex, degree, order=0, shape=shp, scale=1, variant="bubble")
    tab = C0.tabulate(qpts, 1)
    divC0 = sum(tab[alpha][:, alpha.index(1), :] for alpha in tab if sum(alpha) == 1)

    duals = numpy.multiply(qpts.T, qwts)
    F = numpy.dot(duals, divC0.T)
    _, sig, vt = numpy.linalg.svd(F, full_matrices=True)
    tol = sig[0] * 1E-10
    num_sv = len([s for s in sig if abs(s) > tol])
    coeffs = numpy.tensordot(vt[num_sv:], C0.get_coeffs(), axes=(-1, 0))

    return polynomial_set.PolynomialSet(ref_complex, degree, degree, C0.get_expansion_set(), coeffs)


class ArnoldQin(finite_element.CiarletElement):
    """The Arnold-Qin macroelement."""
    def __init__(self, ref_el, degree=None):
        sd = ref_el.get_spatial_dimension()
        if degree is None:
            degree = 4 - sd
        poly_set = ArnoldQinSpace(ref_el, degree)
        ref_complex = poly_set.get_reference_element()
        dual = BernardiRaugelDualSet(ref_complex, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super(ArnoldQin, self).__init__(poly_set, dual, degree, formdegree,
                                        mapping="contravariant piola")
