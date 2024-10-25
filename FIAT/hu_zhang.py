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
from FIAT.functional import (PointwiseInnerProductEvaluation as InnerProduct,
                             FrobeniusIntegralMoment as FIM,
                             IntegralLegendreNormalNormalMoment,
                             IntegralLegendreNormalTangentialMoment,
                             )

import numpy


class HuZhangDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, qdegree):
        if qdegree is None:
            qdegree = degree
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
                nodes.append(IntegralLegendreNormalNormalMoment(ref_el, entity_id, k, qdegree))
                nodes.append(IntegralLegendreNormalTangentialMoment(ref_el, entity_id, k, qdegree))
            entity_ids[1][entity_id].extend(range(cur, len(nodes)))

        # internal dofs
        cur = len(nodes)
        if variant == "point":
            # Evaluation at interior points for each component
            interior_points = ref_el.make_points(sd, 0, degree+1)
            nodes.extend(InnerProduct(ref_el, v1, v2, pt)
                         for pt in interior_points for (v1, v2) in basis)

        elif variant == "integral":
            Q = create_quadrature(ref_el, degree + qdegree)
            qpts = Q.get_points()
            P = polynomial_set.ONPolynomialSet(ref_el, degree-2)
            Phis = P.tabulate(qpts)[(0,)*sd]
            v = numpy.array(ref_el.vertices)
            x = numpy.transpose(ref_el.compute_barycentric_coordinates(qpts))
            for k in sorted(top[1]):
                i = (k+1) % (sd+1)
                j = (k+2) % (sd+1)
                t = v[i] - v[j]
                phis = numpy.outer(t, t)[None, :, :, None] * Phis[:, None, None, :]
                phis = numpy.multiply(phis, x[i] * x[j], out=phis)
                nodes.extend(FIM(ref_el, Q, phi) for phi in phis)

        else:
            raise ValueError(f"Unsupported variant {variant}")

        entity_ids[2][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class HuZhang(finite_element.CiarletElement):
    """The definition of the Hu-Zhang element.
    """
    def __init__(self, ref_el, degree=3, variant="point"):
        if degree < 3:
            raise ValueError("Hu-Zhang only defined for degree >= 3")
        if ref_el.shape != TRIANGLE:
            raise ValueError("Hu-Zhang only defined on triangles")
        variant, qdegree = check_format_variant(variant, degree)
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HuZhangDual(ref_el, degree, variant, qdegree)
        formdegree = ref_el.get_spatial_dimension() - 1
        super().__init__(poly_set, dual, degree, formdegree=formdegree, mapping="double contravariant piola")
