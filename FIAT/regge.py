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
from FIAT.functional import PointwiseInnerProductEvaluation, FrobeniusIntegralMoment
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


class ReggeDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant):
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
                Q = create_quadrature(facet, degree + k)
                P = polynomial_set.ONPolynomialSet(facet, k)
                phis = P.tabulate(Q.get_points())[(0,)*dim]
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    tangents = ref_el.compute_face_edge_tangents(dim, entity)
                    basis = [t[:, None] * t[None, :] for t in tangents]
                    Q_mapped = FacetQuadratureRule(ref_el, dim, entity, Q)
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q_mapped,
                                 comp[:, :, None] * phi[None, None, :])
                                 for phi in phis for comp in basis)
                    entity_ids[dim][entity].extend(range(cur, len(nodes)))
        else:
            raise ValueError(f"Invalid variant {variant}")

        super().__init__(nodes, ref_el, entity_ids)


class Regge(finite_element.CiarletElement):
    """The generalized Regge elements for symmetric-matrix-valued functions.
       REG(r) is the space of symmetric-matrix-valued polynomials of degree r
       or less with tangential-tangential continuity.
    """
    def __init__(self, ref_el, degree, variant="integral"):
        assert degree >= 0, "Regge start at degree 0!"
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = ReggeDual(ref_el, degree, variant)
        formdegree = (1, 1)
        mapping = "double covariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
