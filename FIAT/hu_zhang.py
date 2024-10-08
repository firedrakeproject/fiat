# -*- coding: utf-8 -*-
"""Implementation of the Hu-Zhang finite elements."""

# Copyright (C) 2024 by Francis Aznaran (University of Notre Dame)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT import finite_element, polynomial_set, dual_set
from FIAT.reference_element import TRIANGLE
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.functional import (PointwiseInnerProductEvaluation as InnerProduct,
                             FrobeniusIntegralMoment as FIM,
                             IntegralMomentOfTensorDivergence,
                             IntegralLegendreNormalNormalMoment,
                             IntegralLegendreNormalTangentialMoment,
                             IntegralLegendreTangentialTangentialMoment,
                             )

import numpy


class HuZhangDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # vertex dofs
        e1 = numpy.array([1.0, 0.0])
        e2 = numpy.array([0.0, 1.0])
        basis = [(e1, e1), (e1, e2), (e2, e2)]
        for entity_id in sorted(top[0]):
            cur = len(nodes)
            pt, = ref_el.make_points(0, entity_id, degree)
            nodes.extend(InnerProduct(ref_el, v1, v2, pt) for (v1, v2) in basis)
            entity_ids[0][entity_id].extend(range(cur, len(nodes)))

        # edge dofs: moments of normal component of sigma against degree p - 2.
        for entity_id in sorted(top[1]):
            cur = len(nodes)
            for k in range(degree-1):
                nodes.append(IntegralLegendreNormalNormalMoment(ref_el, entity_id, k, k + degree))
                nodes.append(IntegralLegendreNormalTangentialMoment(ref_el, entity_id, k, k + degree))
            # NB, mom_deg should actually be k + degree <= 2 degree, but in AW have 6 = 2 degree
            entity_ids[1][entity_id].extend(range(cur, len(nodes)))

        # internal dofs
        cur = len(nodes)
        if variant == "integral":
            Q = create_quadrature(ref_el, 2*degree-sd-1)
            # Moments against P_{degree-3} for each component
            phi = polynomial_set.ONPolynomialSet(ref_el, degree-sd-1)
            phi_at_qpts = phi.tabulate(Q.get_points())[(0,) * sd]
            for (v1, v2) in basis:
                Phi_at_qpts = numpy.outer(v1, v2)[None, :, :, None] * phi_at_qpts[:, None, None, :]
                nodes.extend(FIM(ref_el, Q, Phi) for Phi in Phi_at_qpts)

            # More internal dofs: tangential-tangential moments against bubbles for each edge
            # Note these are evaluated on the edge, but not shared between cells (hence internal).
            facet = ref_el.get_facet_element()
            Q = create_quadrature(facet, 2*degree-sd)
            phi = polynomial_set.ONPolynomialSet(facet, degree-sd)
            phi_at_qpts = phi.tabulate(Q.get_points())[(0,) * (sd-1)]
            for entity_id in sorted(top[1]):
                Q_mapped = FacetQuadratureRule(ref_el, sd-1, entity_id, Q)
                t = ref_el.compute_edge_tangent(entity_id)
                Phi_at_qpts = numpy.outer(t, t)[None, :, :, None] * phi_at_qpts[:, None, None, :]
                nodes.extend(FIM(ref_el, Q_mapped, Phi) for Phi in Phi_at_qpts)

        elif variant == "point":
            # Evaluation at interior points for each component
            interior_points = ref_el.make_points(sd, 0, degree)
            nodes.extend(InnerProduct(ref_el, v1, v2, pt)
                         for pt in interior_points for (v1, v2) in basis)

            # More internal dofs: tangential-tangential point evaluations.
            # Note these are evaluated on the edge, but not shared between cells (hence internal).
            for entity_id in sorted(top[1]):
                t = ref_el.compute_edge_tangent(entity_id)
                pts = ref_el.make_points(sd-1, entity_id, degree)
                nodes.extend(InnerProduct(ref_el, t, t, pt) for pt in pts)
        else:
            raise ValueError(f"Unsupported variant {variant}")

        entity_ids[2][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class HuZhang(finite_element.CiarletElement):
    """The definition of the Hu-Zhang element.
    """
    def __init__(self, ref_el, degree=3, variant="integral"):
        if degree < 3:
            raise ValueError("Hu-Zhang only defined for degree >= 3")
        if ref_el.shape != TRIANGLE:
            raise ValueError("Hu-Zhang only defined on triangles")
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HuZhangDual(ref_el, degree, variant)
        formdegree = ref_el.get_spatial_dimension() - 1
        super().__init__(poly_set, dual, degree, formdegree=formdegree, mapping="double contravariant piola")
