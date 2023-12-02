# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import (finite_element, dual_set, functional, reference_element,
                  jacobi, polynomial_set)
from FIAT.reference_element import POINT, LINE, TRIANGLE, TETRAHEDRON, make_affine_mapping
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet, make_dmat
from FIAT.quadrature_schemes import create_quadrature


class LegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree, poly_set):
        entity_ids = {}
        entity_permutations = {}
        top = ref_el.get_topology()
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = make_entity_permutations_simplex(dim, degree + 1 if dim == len(top)-1 else -1)
            for entity in sorted(top[dim]):
                entity_ids[dim][entity] = []
                entity_permutations[dim][entity] = perms

        dim = ref_el.get_spatial_dimension()
        Q = create_quadrature(ref_el, 2 * degree)
        phis = poly_set.tabulate(Q.get_points())[(0,) * dim]
        nodes = [functional.IntegralMoment(ref_el, Q, phi) for phi in phis]

        super(LegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class Legendre(finite_element.CiarletElement):
    """Simplicial discontinuous element with Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape not in {POINT, LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("%s is only defined on simplices." % type(self))
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = LegendreDual(ref_el, degree, poly_set)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(Legendre, self).__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements."""
    def __init__(self, ref_el, degree, rule):
        v1 = ref_el.get_vertices()
        A, b = make_affine_mapping(v1, [(-1.0,), (1.0,)])
        mapping = lambda x: numpy.dot(A, x) + b
        xhat = numpy.array([mapping(pt) for pt in rule.pts])

        W = rule.get_weights()
        D, _ = make_dmat(numpy.array(rule.pts).flatten())
        P = jacobi.eval_jacobi_batch(0, 0, degree-1, xhat)
        basis = numpy.dot(numpy.multiply(P, W), numpy.multiply(D.T, 1.0/W))

        nodes = [functional.PointEvaluation(ref_el, x) for x in v1]
        nodes += [functional.IntegralMoment(ref_el, rule, f) for f in basis[2::2]]
        nodes += [functional.IntegralMoment(ref_el, rule, f) for f in basis[1::2]]

        entity_ids = {0: {0: [0], 1: [1]},
                      1: {0: list(range(2, degree+1))}}
        entity_permutations = {}
        entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
        entity_permutations[1] = {0: make_entity_permutations_simplex(1, degree - 1)}
        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class IntegratedLegendre(finite_element.CiarletElement):
    """1D continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape != reference_element.LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        rule = create_quadrature(ref_el, 2 * degree)
        poly_set = LagrangePolynomialSet(ref_el, rule.get_points())
        dual = IntegratedLegendreDual(ref_el, degree, rule)
        formdegree = 0  # 0-form
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)
