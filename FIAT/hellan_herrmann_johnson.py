# -*- coding: utf-8 -*-
"""Implementation of the Hellan-Herrmann-Johnson finite elements."""

# Copyright (C) 2016-2018 Lizao Li <lzlarryli@gmail.com>
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2024
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import dual_set, finite_element, polynomial_set
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import (PointwiseInnerProductEvaluation,
                             ComponentPointEvaluation,
                             TensorBidirectionalIntegralMoment as BidirectionalMoment)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


class HellanHerrmannJohnsonDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, qdegree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        n = list(map(ref_el.compute_scaled_normal, sorted(top[sd-1])))
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # Face dofs
        if variant == "point":
            # n^T u n evaluated on a Pk lattice
            for f in sorted(top[sd-1]):
                cur = len(nodes)
                pts = ref_el.make_points(sd-1, f, degree + sd)
                nodes.extend(PointwiseInnerProductEvaluation(ref_el, n[f], n[f], pt)
                             for pt in pts)
                entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        elif variant == "integral":
            # n^T u n integrated against a basis for Pk
            facet = ref_el.construct_subelement(sd-1)
            Q = create_quadrature(facet, qdegree + degree)
            P = polynomial_set.ONPolynomialSet(facet, degree)
            phis = P.tabulate(Q.get_points())[(0,)*(sd-1)]
            for f in sorted(top[sd-1]):
                cur = len(nodes)
                Q_mapped = FacetQuadratureRule(ref_el, sd-1, f, Q)
                detJ = Q_mapped.jacobian_determinant()
                nodes.extend(BidirectionalMoment(ref_el, n[f], n[f]/detJ, Q_mapped, phi) for phi in phis)
                entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Interior dofs
        cur = len(nodes)
        if sd == 2 and variant == "point":
            # FIXME Keeping Cartesian dofs in 2D just to make regression test pass
            pts = ref_el.make_points(sd, 0, degree + sd)
            nodes.extend(ComponentPointEvaluation(ref_el, (i, j), (sd, sd), pt)
                         for i in range(sd) for j in range(i, sd) for pt in pts)
        elif variant == "point":
            # n[f]^T u n[f] evaluated on a P_{k-1} lattice
            pts = ref_el.make_points(sd, 0, degree + sd)
            nodes.extend(PointwiseInnerProductEvaluation(ref_el, n[f], n[f], pt)
                         for pt in pts for f in sorted(top[sd-1]))

            # n[i+1]^T u n[i+2] evaluated on a Pk lattice
            pts = ref_el.make_points(sd, 0, degree + sd + 1)
            nodes.extend(PointwiseInnerProductEvaluation(ref_el, n[i+1], n[i+2], pt)
                         for pt in pts for i in range((sd-1)*(sd-2)))
        else:
            Q = create_quadrature(ref_el, qdegree + degree)
            P = polynomial_set.ONPolynomialSet(ref_el, degree)
            phis = P.tabulate(Q.get_points())[(0,)*sd]
            phis /= ref_el.volume()
            dimPkm1 = P.expansion_set.get_num_members(degree-1)
            # n[f]^T u n[f] integrated against a basis for P_{k-1}
            nodes.extend(BidirectionalMoment(ref_el, n[f], n[f], Q, phi)
                         for phi in phis[:dimPkm1] for f in sorted(top[sd-1]))

            # n[i+1]^T u n[i+2] integrated against a basis for Pk
            nodes.extend(BidirectionalMoment(ref_el, n[i+1], n[i+2], Q, phi)
                         for phi in phis for i in range((sd-1)*(sd-2)))

        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HellanHerrmannJohnson(finite_element.CiarletElement):
    """The definition of Hellan-Herrmann-Johnson element.
       HHJ(k) is the space of symmetric-matrix-valued polynomials of degree k
       or less with normal-normal continuity.
    """
    def __init__(self, ref_el, degree=0, variant=None):
        if degree < 0:
            raise ValueError(f"{type(self).__name__} only defined for degree >= 0")

        variant, qdegree = check_format_variant(variant, degree)
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HellanHerrmannJohnsonDual(ref_el, degree, variant, qdegree)
        sd = ref_el.get_spatial_dimension()
        formdegree = (sd-1, sd-1)
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
