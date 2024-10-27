# -*- coding: utf-8 -*-
"""Implementation of the Hu-Zhang finite elements."""

# Copyright (C) 2024 by Francis Aznaran (University of Notre Dame)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT import finite_element, polynomial_set, dual_set
from FIAT.check_format_variant import check_format_variant
from FIAT.reference_element import TRIANGLE
from FIAT.quadrature_schemes import create_quadrature
from FIAT.functional import (ComponentPointEvaluation,
                             PointwiseInnerProductEvaluation,
                             TensorBidirectionalIntegralMoment,
                             IntegralLegendreNormalNormalMoment,
                             IntegralLegendreNormalTangentialMoment)


class HuZhangDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, qdegree):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        shp = (sd, sd)
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # vertex dofs
        for v in sorted(top[0]):
            cur = len(nodes)
            pt, = ref_el.make_points(0, v, degree)
            nodes.extend(ComponentPointEvaluation(ref_el, (i, j), shp, pt)
                         for i in range(sd) for j in range(i, sd))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # edge dofs
        for entity in sorted(top[1]):
            cur = len(nodes)
            if variant == "point":
                # nn and nt components evaluated at edge points
                n = ref_el.compute_scaled_normal(entity)
                t = ref_el.compute_edge_tangent(entity)
                pts = ref_el.make_points(1, entity, degree)
                nodes.extend(PointwiseInnerProductEvaluation(ref_el, n, s, pt)
                             for pt in pts for s in (n, t))

            elif variant == "integral":
                # bidirectional nn and nt moments against P_{k-2}
                moments = (IntegralLegendreNormalNormalMoment, IntegralLegendreNormalTangentialMoment)
                nodes.extend(mu(ref_el, entity, order, qdegree + degree-2)
                             for order in range(degree-1) for mu in moments)
            entity_ids[1][entity].extend(range(cur, len(nodes)))

        # interior dofs
        cur = len(nodes)
        if variant == "point":
            # unique components evaluated at interior points
            pts = ref_el.make_points(sd, 0, degree+1)
            nodes.extend(ComponentPointEvaluation(ref_el, (i, j), shp, pt)
                         for pt in pts for i in range(sd) for j in range(i, sd))

        elif variant == "integral":
            # Moments of unique components against a basis for P_{k-2}
            n = list(map(ref_el.compute_scaled_normal, sorted(top[sd-1])))
            Q = create_quadrature(ref_el, 2*degree-2)
            P = polynomial_set.ONPolynomialSet(ref_el, degree-2, scale="L2 piola")
            phis = P.tabulate(Q.get_points())[(0,)*sd]
            nodes.extend(TensorBidirectionalIntegralMoment(ref_el, n[i+1], n[j+1], Q, phi)
                         for phi in phis for i in range(sd) for j in range(i, sd))

        entity_ids[2][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class HuZhang(finite_element.CiarletElement):
    """The definition of the Hu-Zhang element."""
    def __init__(self, ref_el, degree=3, variant=None):
        if degree < 3:
            raise ValueError(f"{type(self).__name__} only defined for degree >= 3")
        if ref_el.shape != TRIANGLE:
            raise ValueError(f"{type(self).__name__} only defined on triangles")
        variant, qdegree = check_format_variant(variant, degree)
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HuZhangDual(ref_el, degree, variant, qdegree)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
