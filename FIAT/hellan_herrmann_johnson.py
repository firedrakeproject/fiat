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
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        cell_to_faces = ref_el.get_connectivity()[(sd, sd-1)]
        n = list(map(ref_el.compute_scaled_normal, sorted(top[sd-1])))
        if variant == "point":
            for f in sorted(top[sd-1]):
                cur = len(nodes)
                # n^T u n evaluated on a Pk lattice
                pts = ref_el.make_points(sd-1, f, degree + sd)
                nodes.extend(PointwiseInnerProductEvaluation(ref_el, n[f], n[f], pt)
                             for pt in pts)
                entity_ids[sd-1][f].extend(range(cur, len(nodes)))

            if sd == 2:
                # FIXME Keeping Cartesian dofs in 2D just to make regression test pass
                for entity in sorted(top[sd]):
                    faces = cell_to_faces[entity]
                    cur = len(nodes)
                    pts = ref_el.make_points(sd, entity, degree + sd)
                    nodes.extend(ComponentPointEvaluation(ref_el, (i, j), (sd, sd), pt)
                                 for i in range(sd) for j in range(i, sd) for pt in pts)
                    entity_ids[sd][entity].extend(range(cur, len(nodes)))
            else:
                for entity in sorted(top[sd]):
                    faces = cell_to_faces[entity]
                    cur = len(nodes)
                    # n[f]^T u n[f] evaluated on a P_{k-1} lattice
                    pts = ref_el.make_points(sd, entity, degree + sd)
                    nodes.extend(PointwiseInnerProductEvaluation(ref_el, n[f], n[f], pt)
                                 for pt in pts for f in faces)

                    # n[i+1]^T u n[i+2] evaluated on a Pk lattice
                    pts = ref_el.make_points(sd, entity, degree + sd + 1)
                    nodes.extend(PointwiseInnerProductEvaluation(ref_el, n[faces[i+1]], n[faces[i+2]], pt)
                                 for pt in pts for i in range((sd-1)*(sd-2)))
                    entity_ids[sd][entity].extend(range(cur, len(nodes)))

        elif variant == "integral":
            # Face dofs
            ref_facet = ref_el.construct_subelement(sd-1)
            Q_ref = create_quadrature(ref_facet, qdegree + degree)
            P = polynomial_set.ONPolynomialSet(ref_facet, degree)
            Phis = P.tabulate(Q_ref.get_points())[(0,)*(sd-1)]

            for f in sorted(top[sd-1]):
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, sd-1, f, Q_ref)
                detJ = Q.jacobian_determinant()
                # n[f]^T u n[f] integrated against a basis for Pk
                nodes.extend(BidirectionalMoment(ref_el, n[f], n[f]/detJ, Q, phi) for phi in Phis)
                entity_ids[sd-1][f].extend(range(cur, len(nodes)))

            ref_facet = ref_el.construct_subelement(sd)
            Q_ref = create_quadrature(ref_facet, qdegree + degree)
            P = polynomial_set.ONPolynomialSet(ref_facet, degree)
            Phis = P.tabulate(Q_ref.get_points())[(0,) * sd]
            dimPkm1 = P.expansion_set.get_num_members(degree-1)

            # Interior dofs
            for entity in sorted(top[sd]):
                cur = len(nodes)
                faces = cell_to_faces[entity]
                Q = FacetQuadratureRule(ref_el, sd, entity, Q_ref)
                detJ = Q.jacobian_determinant()
                # n[f]^T u n[f] integrated against a basis for P_{k-1}
                nodes.extend(BidirectionalMoment(ref_el, n[f], n[f]/detJ, Q, phi)
                             for phi in Phis[:dimPkm1] for f in faces)
                # n[i+1]^T u n[i+2] integrated against a basis for Pk
                nodes.extend(BidirectionalMoment(ref_el, n[faces[i+1]], n[faces[i+2]]/detJ, Q, phi)
                             for phi in Phis for i in range((sd-1)*(sd-2)))
                entity_ids[sd][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HellanHerrmannJohnson(finite_element.CiarletElement):
    """The definition of Hellan-Herrmann-Johnson element.
       HHJ(k) is the space of symmetric-matrix-valued polynomials of degree k
       or less with normal-normal continuity.
    """
    def __init__(self, ref_el, degree=0, variant=None):
        if degree < 0:
            raise ValueError(f"{type(self).__name__} only defined for degree >= 0")

        splitting, variant, qdegree = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)

        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HellanHerrmannJohnsonDual(ref_el, degree, variant, qdegree)
        sd = ref_el.get_spatial_dimension()
        formdegree = (sd-1, sd-1)
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
