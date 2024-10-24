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
from FIAT.functional import (PointwiseInnerProductEvaluation,
                             ComponentPointEvaluation,
                             FrobeniusIntegralMoment)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
import numpy


class HellanHerrmannJohnsonDual(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []
        if variant == "point":
            # On each codim=1 facet, n^T u n is evaluated on a Pk lattice, where k = degree.
            for entity in sorted(top[sd-1]):
                cur = len(nodes)
                normal = ref_el.compute_scaled_normal(entity)
                pts = ref_el.make_points(sd-1, entity, degree + sd)
                nodes.extend(PointwiseInnerProductEvaluation(ref_el, normal, normal, pt)
                             for pt in pts)
                entity_ids[sd-1][entity].extend(range(cur, len(nodes)))

        elif variant == "integral":
            # On each codim=1 facet, n^T u n is integrated against a basis for Pk, where k = degree.
            facet = ref_el.construct_subelement(sd-1)
            Q = create_quadrature(facet, 2*degree)
            P = polynomial_set.ONPolynomialSet(facet, degree)
            phis = P.tabulate(Q.get_points())[(0,)*(sd-1)]
            for entity in sorted(top[sd-1]):
                cur = len(nodes)
                Q_mapped = FacetQuadratureRule(ref_el, sd-1, entity, Q)
                detJ = Q_mapped.jacobian_determinant()
                n = ref_el.compute_scaled_normal(entity)
                comp = (n[:, None] * n[None, :]) / detJ
                nodes.extend(FrobeniusIntegralMoment(ref_el, Q_mapped,
                             comp[:, :, None] * phi[None, None, :])
                             for phi in phis)
                entity_ids[sd-1][entity].extend(range(cur, len(nodes)))
        else:
            raise ValueError(f"Invalid variant {variant}")

        cur = len(nodes)
        if variant == "point":
            if sd != 2:
                raise NotImplementedError("Pointwise dofs only implemented in 2D")
            # On the interior, the independent components uij, i <= j,
            # are evaluated at a Pk lattice, where k = degree - 1.
            shp = (sd, sd)
            basis = [(i, j) for i in range(sd) for j in range(i, sd)]
            pts = ref_el.make_points(sd, 0, degree + sd)
            nodes.extend(ComponentPointEvaluation(ref_el, comp, shp, pt)
                         for comp in basis for pt in pts)
        else:
            # On the interior, u is integrated against a basis of nn bubbles
            Q = create_quadrature(ref_el, 2*degree)
            P = polynomial_set.ONPolynomialSet(ref_el, degree)
            Phis = P.tabulate(Q.get_points())[(0,)*sd]
            sym = lambda A: 0.5*(A + A.T)
            v = numpy.array(ref_el.get_vertices())

            if sd == 3:
                basis = [sym(numpy.outer(v[i] - v[j], v[m] - v[n]))
                         for i, j, m, n in [(0, 1, 2, 3), (0, 2, 1, 3)]]
                for comp in basis:
                    phis = comp[None, :, :, None] * Phis[:, None, None, :]
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
            elif sd > 3:
                raise NotImplementedError(f"HHJ is not implemented in {sd} dimensions")

            if degree > 0:
                dimPkm1 = expansions.polynomial_dimension(ref_el, degree-1)
                x = ref_el.compute_barycentric_coordinates(Q.get_points())
                for i in sorted(top[0]):
                    comp = sym(numpy.outer(v[(i+1) % (sd+1)] - v[i], v[(i-1) % (sd+1)] - v[i]))
                    phis = comp[None, :, :, None] * x[None, None, None, :, i] * Phis[:dimPkm1, None, None, :]
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)

        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HellanHerrmannJohnson(finite_element.CiarletElement):
    """The definition of Hellan-Herrmann-Johnson element.
       HHJ(r) is the space of symmetric-matrix-valued polynomials of degree r
       or less with normal-normal continuity.
    """
    def __init__(self, ref_el, degree, variant=None):
        assert degree >= 0, "Hellan-Herrmann-Johnson starts at degree 0!"
        if variant is None:
            variant = "integral"
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HellanHerrmannJohnsonDual(ref_el, degree, variant)
        sd = ref_el.get_spatial_dimension()
        formdegree = (sd-1, sd-1)
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
