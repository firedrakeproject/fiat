# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import (finite_element, reference_element,
                  dual_set, functional, quadrature,
                  jacobi, barycentric_interpolation)
from FIAT.lagrange import make_entity_permutations
from FIAT.barycentric_interpolation import LagrangePolynomialSet


class LegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree, rule):
        v1 = ref_el.get_vertices()
        A, b = reference_element.make_affine_mapping(v1, [(-1.0,), (1.0,)])
        mapping = lambda x: numpy.dot(A, x) + b
        xhat = numpy.array([mapping(pt) for pt in rule.pts])

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
        if ref_el.shape != reference_element.LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        poly_set = LagrangePolynomialSet(ref_el, rule.get_points())
        dual = LegendreDual(ref_el, degree, rule)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(Legendre, self).__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements."""
    def __init__(self, ref_el, degree, rule):
        v1 = ref_el.get_vertices()
        A, b = reference_element.make_affine_mapping(v1, [(-1.0,), (1.0,)])
        mapping = lambda x: numpy.dot(A, x) + b
        xhat = numpy.array([mapping(pt) for pt in rule.pts])

        W = rule.get_weights()
        D, _ = barycentric_interpolation.make_dmat(numpy.array(rule.pts).flatten())
        P = jacobi.eval_jacobi_batch(0, 0, degree-1, xhat)
        basis = numpy.dot(numpy.multiply(P, W), numpy.multiply(D.T, 1.0/W))

        nodes = [functional.PointEvaluation(ref_el, x) for x in v1]
        nodes += [functional.IntegralMoment(ref_el, rule, f) for f in basis[2::2]]
        nodes += [functional.IntegralMoment(ref_el, rule, f) for f in basis[1::2]]

        entity_ids = {0: {0: [0], 1: [1]},
                      1: {0: list(range(2, degree+1))}}
        entity_permutations = {}
        entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
        entity_permutations[1] = {0: make_entity_permutations(1, degree - 1)}
        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class IntegratedLegendre(finite_element.CiarletElement):
    """1D continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape != reference_element.LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        poly_set = LagrangePolynomialSet(ref_el, rule.get_points())
        dual = IntegratedLegendreDual(ref_el, degree, rule)
        formdegree = 0  # 0-form
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)
