# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy
from itertools import chain

from FIAT import finite_element, dual_set, functional
from FIAT.reference_element import POINT, LINE, TRIANGLE, TETRAHEDRON
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
        entity_ids = {}
        entity_permutations = {}

        # vertex dofs
        vertices = ref_el.get_vertices()
        nodes = [functional.PointEvaluation(ref_el, pt) for pt in vertices]
        entity_ids[0] = {k: [k] for k in range(len(vertices))}
        entity_permutations[0] = {k: {0: [0]} for k in range(len(vertices))}

        top = ref_el.get_topology()
        for dim in range(1, len(top)):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = make_entity_permutations_simplex(dim, degree - dim)

            ref_facet = ref_el.construct_subelement(dim)
            Q_ref = create_quadrature(ref_facet, 2 * degree)
            if dim == 1 and False:
                phis = self._tabulate_H1_duals(ref_facet, degree, Q_ref)
            else:
                phis = self._tabulate_L2_duals(ref_facet, degree, Q_ref)

            for entity in range(len(top[dim])):
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                nodes.extend(functional.IntegralMoment(ref_el, Q, phi) for phi in reversed(phis))
                entity_ids[dim][entity] = list(range(cur, cur + len(phis)))
                entity_permutations[dim][entity] = perms

        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)

    def _tabulate_H1_duals(self, ref_el, degree, Q):
        qpts = Q.get_points()
        qwts = Q.get_weights()
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
        return phis

    def _tabulate_L2_duals(self, ref_el, degree, Q):
        qpts = Q.get_points()
        qwts = Q.get_weights()
        moments = lambda v, u: numpy.dot(numpy.multiply(v, qwts), u.T)
        dim = ref_el.get_spatial_dimension()

        B = make_bubbles(ref_el, degree)
        B_table = B.tabulate(qpts, 1)

        phis = B_table[(0,) * dim]
        if len(phis) > 0:
            phis[:, abs(phis[0]) <= 1E-12] = 1.0
            phis = phis / abs(phis[0])

        P = ONPolynomialSet(ref_el, degree)
        P_table = P.tabulate(qpts, 1)
        phis = P_table[(0,) * dim]

        k = dim == 1
        K00 = sum(moments(B_table[alpha], B_table[alpha]) for alpha in B_table if sum(alpha) == k)
        K01 = sum(moments(B_table[alpha], P_table[alpha]) for alpha in B_table if sum(alpha) == k)
        phis = numpy.linalg.solve(K00, numpy.dot(K01, phis))
        return phis


class IntegratedLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape not in {POINT, LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("%s is only defined on simplices." % type(self))
        poly_set = ONPolynomialSet(ref_el, degree, variant="integral")
        dual = IntegratedLegendreDual(ref_el, degree)
        formdegree = 0  # 0-form
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)


def super_quadrature(cell, degree):
    sd = cell.get_spatial_dimension()
    top = cell.get_topology()
    Qs = []
    for dim in sorted(top, reverse=True):
        if dim == 0:
            Qs.append(QuadratureRule(cell, cell.vertices, (1.,)*len(cell.vertices)))
        elif dim == sd:
            Qs.append(create_quadrature(cell, degree))
        else:
            facet = cell.construct_subelement(dim)
            Qfacet = create_quadrature(facet, degree + sd - dim)
            Qs.extend(FacetQuadratureRule(cell, dim, entity, Qfacet) for entity in top[dim])

    qpts = tuple(chain.from_iterable(Q.pts for Q in Qs))
    qwts = tuple(chain.from_iterable(Q.wts for Q in Qs))
    return QuadratureRule(cell, qpts, qwts)
