# -*- coding: utf-8 -*-
"""Implementation of the Arnold-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
# Modified by Francis Aznaran (Oxford University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT import finite_element, dual_set, polynomial_set
from FIAT.reference_element import TRIANGLE
from FIAT.quadrature_schemes import create_quadrature
from FIAT.functional import (ComponentPointEvaluation,
                             TensorBidirectionalIntegralMoment,
                             IntegralMomentOfTensorDivergence,
                             IntegralLegendreNormalNormalMoment,
                             IntegralLegendreNormalTangentialMoment)

import numpy


class ArnoldWintherNCDual(dual_set.DualSet):
    def __init__(self, ref_el, degree=2):
        if degree != 2:
            raise ValueError("Nonconforming Arnold-Winther elements are only defined for degree 2.")
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # no vertex dofs
        # proper edge dofs now (not the contraints)
        # edge dofs: bidirectional nn and nt moments against P1.
        qdegree = degree + 2
        for entity in sorted(top[1]):
            cur = len(nodes)
            for order in range(2):
                nodes.append(IntegralLegendreNormalNormalMoment(ref_el, entity, order, qdegree))
                nodes.append(IntegralLegendreNormalTangentialMoment(ref_el, entity, order, qdegree))
            entity_ids[1][entity].extend(range(cur, len(nodes)))

        # internal dofs: constant moments of three unique components
        cur = len(nodes)
        n = list(map(ref_el.compute_scaled_normal, sorted(top[sd-1])))
        Q = create_quadrature(ref_el, degree)
        phi = numpy.full(Q.get_weights().shape, 1/ref_el.volume())
        nodes.extend(TensorBidirectionalIntegralMoment(ref_el, n[i+1], n[j+1], Q, phi)
                     for i in range(sd) for j in range(i, sd))
        entity_ids[2][0].extend(range(cur, len(nodes)))

        # put the constraint dofs last.
        for entity in sorted(top[1]):
            cur = len(nodes)
            nodes.append(IntegralLegendreNormalNormalMoment(ref_el, entity, 2, qdegree))
            entity_ids[1][entity].append(cur)

        super().__init__(nodes, ref_el, entity_ids)


class ArnoldWintherNC(finite_element.CiarletElement):
    """The definition of the nonconforming Arnold-Winther element.
    """
    def __init__(self, ref_el, degree=2):
        if ref_el.shape != TRIANGLE:
            raise ValueError(f"{type(self).__name__} only defined on triangles")
        Ps = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        Ls = ArnoldWintherNCDual(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(Ps, Ls, degree, formdegree, mapping=mapping)


class ArnoldWintherDual(dual_set.DualSet):
    def __init__(self, ref_el, degree=3):
        if degree != 3:
            raise ValueError("Arnold-Winther elements are only defined for degree 3.")
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

        # edge dofs: bidirectional nn and nt moments against P_{k-2}
        max_order = degree - 2
        qdegree = degree + max_order
        for entity in sorted(top[1]):
            cur = len(nodes)
            for order in range(max_order+1):
                nodes.append(IntegralLegendreNormalNormalMoment(ref_el, entity, order, qdegree))
                nodes.append(IntegralLegendreNormalTangentialMoment(ref_el, entity, order, qdegree))
            entity_ids[1][entity].extend(range(cur, len(nodes)))

        # internal dofs: moments of unique components against P_{k-3}
        n = list(map(ref_el.compute_scaled_normal, sorted(top[sd-1])))
        Q = create_quadrature(ref_el, 2*(degree-1))
        P = polynomial_set.ONPolynomialSet(ref_el, degree-3, scale="L2 piola")
        phis = P.tabulate(Q.get_points())[(0,)*sd]
        nodes.extend(TensorBidirectionalIntegralMoment(ref_el, n[i+1], n[j+1], Q, phi)
                     for phi in phis for i in range(sd) for j in range(i, sd))

        # constraint dofs: moments of divergence against P_{k-1} \ P_{k-2}
        P = polynomial_set.ONPolynomialSet(ref_el, degree-1, shape=(sd,))
        dimPkm1 = P.expansion_set.get_num_members(degree-1)
        dimPkm2 = P.expansion_set.get_num_members(degree-2)
        PH = P.take([i + j * dimPkm1 for j in range(sd) for i in range(dimPkm2, dimPkm1)])
        phis = PH.tabulate(Q.get_points())[(0,)*sd]
        nodes.extend(IntegralMomentOfTensorDivergence(ref_el, Q, phi) for phi in phis)

        entity_ids[2][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class ArnoldWinther(finite_element.CiarletElement):
    """The definition of the conforming Arnold-Winther element.
    """
    def __init__(self, ref_el, degree=3):
        if ref_el.shape != TRIANGLE:
            raise ValueError(f"{type(self).__name__} only defined on triangles")
        Ps = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        Ls = ArnoldWintherDual(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(Ps, Ls, degree, formdegree, mapping=mapping)
