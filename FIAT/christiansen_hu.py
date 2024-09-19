# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import ComponentPointEvaluation, FrobeniusIntegralMoment
from FIAT.hct import HsiehCloughTocher
from FIAT.restricted import RestrictedElement
from FIAT.reference_element import TRIANGLE
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule

import numpy


def ChristiansenHuSpace(ref_el, degree):
    """Return a basis for the Christiansen-Hu space.
    curl(HCT-red) + P_0 x"""
    sd = ref_el.get_spatial_dimension()

    HCT = HsiehCloughTocher(ref_el, degree+1, reduced=True)
    RHCT = RestrictedElement(HCT, restriction_domain="vertex")

    ref_complex = RHCT.get_reference_complex()
    Q = create_quadrature(ref_complex, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    tab = RHCT.tabulate(1, Qpts)
    curl_RHCT_at_Qpts = numpy.stack([tab[(0, 1)], -tab[(1, 0)]], axis=1)

    Pk = polynomial_set.ONPolynomialSet(ref_complex, degree)
    Pk_at_Qpts = Pk.tabulate(Qpts)[(0,) * sd]

    x = Qpts.T
    P0x_at_Qpts = x[None, :, :]

    expansion_set = Pk.get_expansion_set()
    duals = numpy.transpose(numpy.multiply(Pk_at_Qpts, Qwts))
    pieces = [polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, numpy.dot(T, duals))
              for T in (curl_RHCT_at_Qpts, P0x_at_Qpts)]
    return polynomial_set.polynomial_set_union_normalized(*pieces)


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
        q = degree - 2
        facet = ref_el.construct_subelement(dim)
        Q_ref = create_quadrature(facet, degree + q)
        Pq = polynomial_set.ONPolynomialSet(facet, q)
        Pq_at_qpts = Pq.tabulate(Q_ref.get_points())[(0,)*dim]
        for entity in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
            Jdet = Q.jacobian_determinant()
            n = ref_el.compute_scaled_normal(entity) / Jdet
            phis = n[None, :, None] * Pq_at_qpts[:, None, :]
            nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
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
