# -*- coding: utf-8 -*-
"""Implementation of the Mardal-Tai-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT import dual_set, expansions, finite_element, polynomial_set
from FIAT.functional import (IntegralMomentOfNormalEvaluation,
                             IntegralMomentOfTangentialEvaluation,
                             IntegralLegendreNormalMoment,
                             IntegralMomentOfDivergence)

from FIAT.quadrature_schemes import create_quadrature


def DivergenceDubinerMoments(ref_el, start_deg, stop_deg, comp_deg):
    sd = ref_el.get_spatial_dimension()
    P = polynomial_set.ONPolynomialSet(ref_el, stop_deg)
    Q = create_quadrature(ref_el, comp_deg + stop_deg)

    dim0 = expansions.polynomial_dimension(ref_el, start_deg-1)
    dim1 = expansions.polynomial_dimension(ref_el, stop_deg)
    indices = list(range(dim0, dim1))
    phis = P.take(indices).tabulate(Q.get_points())[(0,)*sd]
    for phi in phis:
        yield IntegralMomentOfDivergence(ref_el, Q, phi)


class MardalTaiWintherDual(dual_set.DualSet):
    """Degrees of freedom for Mardal-Tai-Winther elements."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()

        if sd != 2:
            raise ValueError("Mardal-Tai-Winther elements are only defined in dimension 2.")

        if degree != 3:
            raise ValueError("Mardal-Tai-Winther elements are only defined for degree 3.")

        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # no vertex dofs

        # On each facet, let n be its normal.  We need to integrate
        # u.n and u.t against the first Legendre polynomial (constant)
        # and u.n against the second (linear).
        facet = ref_el.get_facet_element()
        # Facet nodes are \int_F v.n p ds where p \in P_{q-1}
        # degree is q - 1
        Q = create_quadrature(facet, degree+1)
        Pq = polynomial_set.ONPolynomialSet(facet, 1)
        phis = Pq.tabulate(Q.get_points())[(0,)*(sd - 1)]
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            nodes.append(IntegralMomentOfNormalEvaluation(ref_el, Q, phis[0], f))
            nodes.append(IntegralMomentOfTangentialEvaluation(ref_el, Q, phis[0], f))
            nodes.append(IntegralMomentOfNormalEvaluation(ref_el, Q, phis[1], f))
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Generate constraint nodes on the cell and facets
        # * div(v) must be constant on the cell.  Since v is a cubic and
        #   div(v) is quadratic, we need the integral of div(v) against the
        #   linear and quadratic Dubiner polynomials to vanish.
        #   There are two linear and three quadratics, so these are five
        #   constraints
        # * v.n must be linear on each facet.  Since v.n is cubic, we need
        #   the integral of v.n against the cubic and quadratic Legendre
        #   polynomial to vanish on each facet.

        # So we introduce functionals whose kernel describes this property,
        # as described in the FIAT paper.
        start_order = 2
        stop_order = 3
        qdegree = degree + stop_order
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            nodes.extend(IntegralLegendreNormalMoment(ref_el, f, order, qdegree)
                         for order in range(start_order, stop_order+1))
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        cur = len(nodes)
        nodes.extend(DivergenceDubinerMoments(ref_el, start_order-1, stop_order-1, degree))
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class MardalTaiWinther(finite_element.CiarletElement):
    """The definition of the Mardal-Tai-Winther element.
    """
    def __init__(self, ref_el, degree=3):
        sd = ref_el.get_spatial_dimension()
        assert degree == 3, "Only defined for degree 3"
        assert sd == 2, "Only defined for dimension 2"
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, (sd,))
        dual = MardalTaiWintherDual(ref_el, degree)
        formdegree = sd-1
        mapping = "contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
