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
                                    make_lattice, symmetric_simplex)
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import make_dmat
from FIAT.quadrature import QuadratureRule, FacetQuadratureRule
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
    def __init__(self, ref_el, degree, variant=None):
        if variant is None:
            variant = "beuchler"

        duals = {
            "beuchler": self._beuchler_integral_duals,
            "beuchler_point": self._beuchler_point_duals,
            "demkowicz": self._demkowicz_duals,
            "orthonormal": self._orthonormal_duals,
        }[variant]

        nodes = []
        entity_ids = {}
        entity_permutations = {}

        top = ref_el.get_topology()
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            if dim == 0:
                perms = {0: [0]}
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pt, = ref_el.make_points(0, entity, degree)
                    nodes.append(functional.PointEvaluation(ref_el, pt))
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                    entity_permutations[dim][entity] = perms
                continue

            perms = make_entity_permutations_simplex(dim, degree - dim)
            ref_facet = symmetric_simplex(dim)
            Q_ref, phis = duals(ref_facet, degree)
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                J = Q.jacobian()

                # phis must transform like a d-form to undo the measure transformation
                scale = 1/numpy.sqrt(abs(numpy.linalg.det(numpy.dot(J.T, J))))
                Jphis = scale * phis

                nodes.extend(functional.IntegralMoment(ref_el, Q, phi) for phi in reversed(Jphis))
                entity_ids[dim][entity] = list(range(cur, len(nodes)))
                entity_permutations[dim][entity] = perms

        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)

    def _beuchler_point_duals(self, ref_el, degree, variant="gll"):
        points = make_lattice(ref_el.vertices, degree, variant=variant)
        weights = (1.0,) * len(points)
        Q = QuadratureRule(ref_el, points, weights)
        B = make_bubbles(ref_el, degree)
        V = numpy.transpose(B.expansion_set.tabulate(degree, points))

        PLU = scipy.linalg.lu_factor(V)
        phis = scipy.linalg.lu_solve(PLU, B.get_coeffs().T, trans=1).T
        return Q, phis

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

    def _demkowicz_duals(self, ref_el, degree):
        Q = create_quadrature(ref_el, 2 * degree)
        qpts, qwts = Q.get_points(), Q.get_weights()
        moments = lambda v: numpy.dot(numpy.multiply(v, qwts), v.T)

        dim = ref_el.get_spatial_dimension()
        if dim == 1:
            # Assemble a stiffness matrix in the Lagrange basis
            dmat, _ = make_dmat(qpts.flatten())
            K = moments(dmat)
        else:
            # Get ON basis
            P = ONPolynomialSet(ref_el, degree)
            P_table = P.tabulate(qpts, 1)
            # Assemble a stiffness matrix in the ON basis
            K = sum(moments(P_table[alpha]) for alpha in P_table if sum(alpha) == 1)
            # Change of basis to Lagrange polynomials at the quadrature nodes
            v = numpy.multiply(P_table[(0, ) * dim], qwts)
            K = numpy.dot(numpy.dot(v.T, K), v)

        B = make_bubbles(ref_el, degree)
        phis = B.tabulate(qpts)[(0,) * dim]
        phis = numpy.multiply(numpy.dot(phis, K), 1/qwts)
        return Q, phis

    def _orthonormal_duals(self, ref_el, degree, solver=None):
        dim = ref_el.get_spatial_dimension()

        Q = create_quadrature(ref_el, 2 * degree)
        qpts, qwts = Q.get_points(), Q.get_weights()
        inner = lambda v, u: numpy.dot(numpy.multiply(v, qwts), u.T)
        galerkin = lambda order, v, u: sum(inner(v[k], u[k]) for k in v if sum(k) == order)

        B = make_bubbles(ref_el, degree)
        B_table = B.tabulate(qpts, 1)

        P = ONPolynomialSet(ref_el, degree)
        P_table = P.tabulate(qpts, 1)

        KBP = galerkin(1, B_table, P_table)
        phis = P_table[(0,) * dim]
        phis = numpy.dot(KBP, phis)

        if len(phis) > 0:
            KBB = galerkin(1, B_table, B_table)
            V = numpy.linalg.cholesky(KBB)
            phis = numpy.linalg.solve(V, phis)
        return Q, phis


class IntegratedLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree, variant=None):
        if ref_el.shape not in {POINT, LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("%s is only defined on simplices." % type(self))
        poly_set = ONPolynomialSet(ref_el, degree, variant="integral")
        dual = IntegratedLegendreDual(ref_el, degree, variant=variant)
        formdegree = 0  # 0-form
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)
