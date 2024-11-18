# -*- coding: utf-8 -*-
"""Implementation of the generalized Regge finite elements."""

# Copyright (C) 2015-2018 Lizao Li
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import dual_set, finite_element, polynomial_set
from FIAT.functional import PointwiseInnerProductEvaluation as InnerProduct


class ReggeDual(dual_set.DualSet):
    """Degrees of freedom for generalized Regge finite elements.

    On a k-face for degree r, the dofs are given by the value of
       t^T u t
    evaluated at enough points to control P(r-k+1) for all the edge
    tangents of the face.
    `ref_el.make_points(dim, entity, degree + 2)` happens to
    generate exactly those points needed.
    """
    def __init__(self, ref_el, degree):
        top = ref_el.get_topology()
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []
        for dim in sorted(top):
            if dim == 0:
                # no vertex dofs
                continue
            for entity in sorted(top[dim]):
                cur = len(nodes)
                tangents = ref_el.compute_face_edge_tangents(dim, entity)
                pts = ref_el.make_points(dim, entity, degree + 2)
                nodes.extend(InnerProduct(ref_el, t, t, pt)
                             for pt in pts
                             for t in tangents)
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Regge(finite_element.CiarletElement):
    """The generalized Regge elements for symmetric-matrix-valued functions.
       REG(r) in dimension n is the space of polynomial symmetric-matrix-valued
       functions of degree r or less with tangential-tangential continuity.
    """
    def __init__(self, ref_el, degree):
        assert degree >= 0, "Regge start at degree 0!"
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = ReggeDual(ref_el, degree)
        mapping = "double covariant piola"
        super().__init__(poly_set, dual, degree, mapping=mapping)
