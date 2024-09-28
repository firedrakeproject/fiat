# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, polynomial_set
from FIAT.hct import HsiehCloughTocher
from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import WorseyFarinSplit, CkPolynomialSet

import numpy


def ArnoldQinSpace2D(ref_el, degree, reduced=False):
    """Return a basis for the Arnold-Qin space.
    curl(HCT) + P_0 x"""
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

    C0 = polynomial_set.ONPolynomialSet(ref_complex, degree, scale=1, variant="bubble")
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


def ArnoldQinSpace(ref_el, degree, reduced=False):
    """Return a basis for the Arnold-Qin space.
    v in C0 P1(WF)^d : div(v) in P_0"""
    sd = ref_el.get_spatial_dimension()
    if sd == 2:
        return ArnoldQinSpace2D(ref_el, degree, reduced=reduced)

    ref_complex = WorseyFarinSplit(ref_el)
    C0 = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), scale=1, variant="bubble")
    Q = create_quadrature(ref_complex, degree-1)
    tab = C0.tabulate(Q.get_points(), 1)
    divC0 = sum(tab[alpha][:, alpha.index(1), :] for alpha in tab if sum(alpha) == 1)

    _, sig, vt = numpy.linalg.svd(divC0.T, full_matrices=True)
    tol = sig[0] * 1E-10
    num_sv = len([s for s in sig if abs(s) > tol])
    coeffs = numpy.tensordot(vt[num_sv:], C0.get_coeffs(), axes=(-1, 0))

    bary = numpy.mean(ref_el.get_vertices(), axis=0, keepdims=True)
    P0x_coeffs = numpy.transpose(ref_complex.get_vertices())
    P0x_coeffs -= bary.T
    coeffs = numpy.concatenate((coeffs, P0x_coeffs[None, ...]), axis=0)

    if not reduced:
        dual = BernardiRaugelDualSet(ref_complex, degree, reduced=True)
        dualmat = dual.to_riesz(C0)
        V = numpy.tensordot(dualmat, coeffs, axes=((1, 2), (1, 2)))
        coeffs = numpy.tensordot(numpy.linalg.inv(V.T), coeffs, axes=(-1, 0))
        facet_bubbles = coeffs[-(sd+1):]
        top = ref_el.get_topology()
        ext = []
        for f in top[sd-1]:
            thats = ref_el.compute_tangents(sd-1, f)
            nhat = numpy.cross(*thats)
            for that in thats:
                tn = numpy.outer(that, nhat)
                ext.append(numpy.dot(tn, facet_bubbles[f]))
        ext_coeffs = numpy.array(ext)
        coeffs = numpy.concatenate((coeffs, ext_coeffs), axis=0)

    return polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                        C0.get_expansion_set(), coeffs)


class ArnoldQin(finite_element.CiarletElement):
    """The Arnold-Qin macroelement."""
    def __init__(self, ref_el, degree=None, reduced=False):
        sd = ref_el.get_spatial_dimension()
        if degree is None:
            degree = 4 - sd
        poly_set = ArnoldQinSpace(ref_el, degree, reduced=reduced)
        ref_complex = poly_set.get_reference_element()
        dual = BernardiRaugelDualSet(ref_complex, degree, reduced=reduced)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super(ArnoldQin, self).__init__(poly_set, dual, degree, formdegree,
                                        mapping="contravariant piola")
