# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import (finite_element, polynomial_set, dual_set, functional, quadrature,
                  jacobi, barycentric_interpolation)
from FIAT.reference_element import LINE
from FIAT.lagrange import make_entity_permutations


class LegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree):
        verts = ref_el.get_vertices()
        x0 = verts[0][0]
        x1 = verts[1][0]

        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        scale = 2.0 / (x1 - x0)
        xhat = scale * (numpy.array(rule.pts) - x0) - 1.0

        basis = jacobi.eval_jacobi_batch(0, 0, degree, xhat)
        nodes = [functional.IntegralMoment(ref_el, rule, f) for f in basis]

        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}
        entity_permutations = {}
        entity_permutations[0] = {0: {0: []}, 1: {0: []}}
        entity_permutations[1] = {0: make_entity_permutations(1, degree + 1)}
        super(LegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class Legendre(finite_element.CiarletElement):
    """1D discontinuous element with Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = LegendreDual(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()
        super(Legendre, self).__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree):
        verts = ref_el.get_vertices()
        x0 = verts[0][0]
        x1 = verts[1][0]

        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        xhat = 2.0 * (numpy.array(rule.pts) - x0) / (x1 - x0) - 1.0
        P = jacobi.eval_jacobi_batch(0, 0, degree-1, xhat)
        D, _ = barycentric_interpolation.make_dmat(numpy.array(rule.pts).flatten())
        W = rule.get_weights()
        duals = numpy.dot(numpy.multiply(P, W), numpy.multiply(D.T, 1.0/W))

        nodes = [functional.PointEvaluation(ref_el, x) for x in verts]
        nodes.extend([functional.IntegralMoment(ref_el, rule, f) for f in duals[1:]])

        entity_ids = {0: {0: [0], 1: [1]},
                      1: {0: list(range(2, degree+1))}}
        entity_permutations = {}
        entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
        entity_permutations[1] = {0: make_entity_permutations(1, degree-1)}
        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class IntegratedLegendre(finite_element.CiarletElement):
    """1D continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = IntegratedLegendreDual(ref_el, degree)
        formdegree = 0
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)
