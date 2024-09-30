# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy
import scipy

from FIAT import finite_element, dual_set, functional, P0
from FIAT.reference_element import symmetric_simplex
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.polynomial_set import ONPolynomialSet, make_bubbles
from FIAT.check_format_variant import parse_lagrange_variant


class LegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree, codim=0):
        nodes = []
        entity_ids = {}
        entity_permutations = {}

        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        for dim in sorted(top):
            npoints = degree + 1 if dim == sd - codim else 0
            perms = make_entity_permutations_simplex(dim, npoints)
            entity_permutations[dim] = {}
            entity_ids[dim] = {}
            if npoints == 0:
                for entity in sorted(top[dim]):
                    entity_ids[dim][entity] = []
                    entity_permutations[dim][entity] = perms
                continue

            ref_facet = ref_el.construct_subelement(dim)
            poly_set = ONPolynomialSet(ref_facet, degree)
            Q_ref = create_quadrature(ref_facet, 2 * degree)
            phis = poly_set.tabulate(Q_ref.get_points())[(0,) * dim]
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                nodes.extend(functional.IntegralMoment(ref_el, Q_facet, phi) for phi in phis)
                entity_ids[dim][entity] = list(range(cur, len(nodes)))
                entity_permutations[dim][entity] = perms

        super().__init__(nodes, ref_el, entity_ids, entity_permutations)


class Legendre(finite_element.CiarletElement):
    """Simplicial discontinuous element with Legendre polynomials."""
    def __new__(cls, ref_el, degree, variant=None):
        if degree == 0:
            splitting, _ = parse_lagrange_variant(variant, integral=True)
            if splitting is None:
                # FIXME P0 on the split requires implementing SplitSimplicialComplex.symmetry_group_size()
                return P0.P0(ref_el)
        return super().__new__(cls)

    def __init__(self, ref_el, degree, variant=None):
        splitting, _ = parse_lagrange_variant(variant, integral=True)
        if splitting is not None:
            ref_el = splitting(ref_el)
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = LegendreDual(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements."""
    def __init__(self, ref_el, degree):
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
            Q_ref, phis = self.make_reference_duals(ref_facet, degree)
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref)

                # phis must transform like a d-form to undo the measure transformation
                scale = 1 / Q_facet.jacobian_determinant()
                Jphis = scale * phis

                nodes.extend(functional.IntegralMoment(ref_el, Q_facet, phi) for phi in Jphis)
                entity_ids[dim][entity] = list(range(cur, len(nodes)))
                entity_permutations[dim][entity] = perms

        super().__init__(nodes, ref_el, entity_ids, entity_permutations)

    def make_reference_duals(self, ref_el, degree):
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
        splitting, _ = parse_lagrange_variant(variant, integral=True)
        if splitting is not None:
            ref_el = splitting(ref_el)
        if degree < 1:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= 1")
        poly_set = ONPolynomialSet(ref_el, degree, variant="bubble")
        dual = IntegratedLegendreDual(ref_el, degree)
        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree)
