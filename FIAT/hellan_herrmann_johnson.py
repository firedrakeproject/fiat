# -*- coding: utf-8 -*-
"""Implementation of the Hellan-Herrmann-Johnson finite elements."""

# Copyright (C) 2016-2018 Lizao Li <lzlarryli@gmail.com>
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2024
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import dual_set, finite_element, polynomial_set, expansions
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import (PointwiseInnerProductEvaluation,
                             ComponentPointEvaluation,
                             FrobeniusIntegralMoment)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
import numpy


class HellanHerrmannJohnsonDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, qdegree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # Face dofs
        if variant == "point":
            # n^T u n evaluated on a Pk lattice
            for entity in sorted(top[sd-1]):
                cur = len(nodes)
                normal = ref_el.compute_scaled_normal(entity)
                pts = ref_el.make_points(sd-1, entity, degree + sd)
                nodes.extend(PointwiseInnerProductEvaluation(ref_el, normal, normal, pt)
                             for pt in pts)
                entity_ids[sd-1][entity].extend(range(cur, len(nodes)))

        elif variant == "integral":
            # n^T u n integrated against a basis for Pk
            facet = ref_el.construct_subelement(sd-1)
            Q = create_quadrature(facet, qdegree + degree)
            P = polynomial_set.ONPolynomialSet(facet, degree)
            Phis = P.tabulate(Q.get_points())[(0,)*(sd-1)]
            for entity in sorted(top[sd-1]):
                cur = len(nodes)
                Q_mapped = FacetQuadratureRule(ref_el, sd-1, entity, Q)
                detJ = Q_mapped.jacobian_determinant()
                n = ref_el.compute_scaled_normal(entity)
                comp = (n[:, None] * n[None, :]) / detJ
                phis = comp[None, :, :, None] * Phis[:, None, None, :]
                nodes.extend(FrobeniusIntegralMoment(ref_el, Q_mapped, phi) for phi in phis)
                entity_ids[sd-1][entity].extend(range(cur, len(nodes)))

        # Interior dofs
        cur = len(nodes)
        if variant == "point":
            if sd != 2:
                raise NotImplementedError("Pointwise dofs only implemented in 2D")
            # independent components evaluated at a P_{k-1} lattice
            shp = (sd, sd)
            basis = [(i, j) for i in range(sd) for j in range(i, sd)]
            pts = ref_el.make_points(sd, 0, degree + sd)
            nodes.extend(ComponentPointEvaluation(ref_el, comp, shp, pt)
                         for comp in basis for pt in pts)
        else:
            # u integrated against a P_k basis of nn bubbles
            Q = create_quadrature(ref_el, qdegree + degree)
            P = polynomial_set.ONPolynomialSet(ref_el, degree)
            Phis = P.tabulate(Q.get_points())[(0,)*sd]

            v = numpy.array(ref_el.get_vertices())
            sym_outer = lambda u, v: 0.5*(u[:, None]*v[None, :] + v[:, None]*u[None, :])
            tt = lambda i, j, k, l: sym_outer(v[i % (sd+1)] - v[j % (sd+1)],
                                              v[k % (sd+1)] - v[l % (sd+1)])

            basis = [tt(i, i+1, i+2, i+3) for i in range((sd-2)*(sd-1))]
            for comp in basis:
                phis = comp[None, :, :, None] * Phis[:, None, None, :]
                nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)

            dimPkm1 = expansions.polynomial_dimension(ref_el, degree-1)
            if dimPkm1 > 0:
                x = numpy.transpose(ref_el.compute_barycentric_coordinates(Q.get_points()))
                for i in sorted(top[0]):
                    comp = tt(i, i+1, i+2, i)
                    phis = comp[None, :, :, None] * Phis[:dimPkm1, None, None, :]
                    phis = numpy.multiply(phis, x[i], out=phis)
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)

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
