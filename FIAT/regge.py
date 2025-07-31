# -*- coding: utf-8 -*-
"""Implementation of the generalized Regge finite elements."""

# Copyright (C) 2015-2018 Lizao Li
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2024
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import dual_set, finite_element, polynomial_set
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import (PointwiseInnerProductEvaluation,
                             TensorBidirectionalIntegralMoment as BidirectionalMoment)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


class ReggeDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, qdegree):
        top = ref_el.get_topology()
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []
        if variant == "point":
            # On a dim-facet, for all the edge tangents of the facet,
            # t^T u t is evaluated on a Pk lattice, where k = degree - dim + 1.
            for dim in sorted(top):
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    tangents = ref_el.compute_face_edge_tangents(dim, entity)
                    pts = ref_el.make_points(dim, entity, degree + 2)
                    nodes.extend(PointwiseInnerProductEvaluation(ref_el, t, t, pt)
                                 for pt in pts for t in tangents)
                    entity_ids[dim][entity].extend(range(cur, len(nodes)))

        elif variant == "integral":
            # On a dim-facet, for all the edge tangents of the facet,
            # t^T u t is integrated against a basis for Pk, where k = degree - dim + 1.
            for dim in sorted(top):
                k = degree - dim + 1
                if dim == 0 or k < 0:
                    continue
                facet = ref_el.construct_subelement(dim)
                Q = create_quadrature(facet, qdegree + k)
                P = polynomial_set.ONPolynomialSet(facet, k)
                phis = P.tabulate(Q.get_points())[(0,)*dim]
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    tangents = ref_el.compute_face_edge_tangents(dim, entity)
                    Q_mapped = FacetQuadratureRule(ref_el, dim, entity, Q)
                    detJ = Q_mapped.jacobian_determinant()
                    nodes.extend(BidirectionalMoment(ref_el, t, t/detJ, Q_mapped, phi)
                                 for phi in phis for t in tangents)
                    entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Regge(finite_element.CiarletElement):
    """The generalized Regge elements for symmetric-matrix-valued functions.
       REG(k) is the space of symmetric-matrix-valued polynomials of degree k
       or less with tangential-tangential continuity.
    """
    def __init__(self, ref_el, degree=0, variant=None):
        if degree < 0:
            raise ValueError(f"{type(self).__name__} only defined for degree >= 0")

        splitting, variant, qdegree = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)

        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = ReggeDual(ref_el, degree, variant, qdegree)
        formdegree = (1, 1)
        mapping = "double covariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
