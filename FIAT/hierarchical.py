# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy
import scipy

from FIAT import finite_element, dual_set, functional
from FIAT.reference_element import (POINT, LINE, TRIANGLE, TETRAHEDRON,
                                    symmetric_simplex)
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.polynomial_set import ONPolynomialSet, make_bubbles


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
        entity_ids[dim][0] = list(range(len(nodes)))

        super(LegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class Legendre(finite_element.CiarletElement):
    """Simplicial discontinuous element with Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape not in {POINT, LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("%s is only defined on simplices." % type(self))
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = LegendreDual(ref_el, degree, poly_set)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(Legendre, self).__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements."""
    def __init__(self, ref_el, degree):
        duals = self._beuchler_integral_duals

        nodes = []
        entity_ids = {}
        entity_permutations = {}

        top = ref_el.get_topology()
        for dim in sorted(top):
            perms = make_entity_permutations_simplex(dim, degree - dim)
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            if dim == 0 or degree <= dim:
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(functional.PointEvaluation(ref_el, pt) for pt in pts)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                    entity_permutations[dim][entity] = perms
                continue

            ref_facet = symmetric_simplex(dim)
            Q_ref, phis = duals(ref_facet, degree)
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref)

                # phis must transform like a d-form to undo the measure transformation
                scale = 1 / Q_facet.jacobian_determinant()
                Jphis = scale * phis

                nodes.extend(functional.IntegralMoment(ref_el, Q_facet, phi) for phi in Jphis)
                entity_ids[dim][entity] = list(range(cur, len(nodes)))
                entity_permutations[dim][entity] = perms

        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)

    def _beuchler_integral_duals(self, ref_el, degree):
        Q = create_quadrature(ref_el, 2 * degree)
        qpts, qwts = Q.get_points(), Q.get_weights()
        inner = lambda v, u: numpy.dot(numpy.multiply(v, qwts), u.T)
        dim = ref_el.get_spatial_dimension()

        B = make_bubbles(ref_el, degree)
        B_table = B.expansion_set.tabulate(degree, qpts)

        P = ONPolynomialSet(ref_el, degree)
        P_table = P.tabulate(qpts, 0)[(0,) * dim]

        # TODO sparse LU
        V = inner(P_table, B_table)
        PLU = scipy.linalg.lu_factor(V)
        phis = scipy.linalg.lu_solve(PLU, P_table)
        phis = numpy.dot(B.get_coeffs(), phis)
        return Q, phis


class IntegratedLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree, variant=None):
        if ref_el.shape not in {POINT, LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("%s is only defined on simplices." % type(self))
        if degree < 1:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= 1")

        poly_set = ONPolynomialSet(ref_el, degree, variant="integral")
        dual = IntegratedLegendreDual(ref_el, degree)
        formdegree = 0  # 0-form
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)
