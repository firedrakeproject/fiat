# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import (ComponentPointEvaluation,
                             IntegralMomentOfScaledNormalEvaluation,
                             IntegralMomentOfEdgeTangentEvaluation)
from FIAT.hct import HsiehCloughTocher
from FIAT.reference_element import TRIANGLE
from FIAT.quadrature_schemes import create_quadrature
from FIAT.jacobi import eval_jacobi

import numpy


def ChristiansenHuSpace(ref_el, degree):
    """Return a basis for the Christiansen-Hu space.
    curl(HCT-red) + P_0 x"""
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
    M = numpy.dot(Pk_at_Qpts, duals.T)
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


class ChristiansenHuDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree, reduced=False):
        if degree != 2:
            raise ValueError("Christiansen-Hu only defined for degree = 2")
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Christiansen-Hu only defined on triangles")
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        nodes = []
        dim = 0
        for entity in sorted(top[dim]):
            cur = len(nodes)
            pts = ref_el.make_points(dim, entity, degree)
            nodes.extend(ComponentPointEvaluation(ref_el, k, (sd,), pt)
                         for pt in pts for k in range(sd))
            entity_ids[dim][entity].extend(range(cur, len(nodes)))

        dim = 1
        k = 2
        facet = ref_el.construct_subelement(dim)
        scale = 1 / facet.volume()
        Q = create_quadrature(facet, degree+k)
        qpts = Q.get_points()
        xref = scale * sum(qpts - v[0] for v in facet.get_vertices())
        f_at_qpts = eval_jacobi(0, 0, k, xref[:, 0])

        for ell in (IntegralMomentOfScaledNormalEvaluation,
                    IntegralMomentOfEdgeTangentEvaluation):
            for entity in sorted(top[dim]):
                cur = len(nodes)
                nodes.append(ell(ref_el, Q, f_at_qpts, entity))
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super(ChristiansenHuDualSet, self).__init__(nodes, ref_el, entity_ids)


class ChristiansenHu(finite_element.CiarletElement):
    """The Christiansen-Hu macroelement."""
    def __init__(self, ref_el, degree=2):
        dual = ChristiansenHuDualSet(ref_el, degree)
        poly_set = ChristiansenHuSpace(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super(ChristiansenHu, self).__init__(poly_set, dual, degree, formdegree,
                                             mapping="contravariant piola")
