# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import finite_element, dual_set, functional
from FIAT.reference_element import POINT, LINE, TRIANGLE, TETRAHEDRON, ufc_simplex
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import make_dmat
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.polynomial_set import ONPolynomialSet


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
    def __init__(self, ref_el, degree, poly_set):
        entity_ids = {}
        entity_permutations = {}

        # vertex dofs
        top = ref_el.get_topology()
        nodes = [functional.PointEvaluation(ref_el, pt) for pt in ref_el.vertices]
        nvertices = len(top[0])
        entity_ids[0] = {k: [k] for k in range(nvertices)}
        entity_permutations[0] = {k: {0: [0]} for k in range(nvertices)}

        for dim in range(1, len(top)):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = make_entity_permutations_simplex(dim, degree - dim)

            quad_degree = 2 * degree
            ref_facet = ufc_simplex(dim)
            Q_ref = create_quadrature(ref_facet, quad_degree)
            phis = self._tabulate_bubbles(ref_facet, degree, Q_ref)

            for entity in range(len(top[dim])):
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                nodes.extend(functional.IntegralMoment(ref_el, Q, phi) for phi in phis)
                entity_ids[dim][entity] = list(range(cur, cur + len(phis)))
                entity_permutations[dim][entity] = perms

        super(IntegratedLegendreDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)

    def _tabulate_bubbles(self, ref_el, degree, Q):
        qpts = Q.get_points()
        dim = ref_el.get_spatial_dimension()
        k = degree - dim - 1
        if k < 0:
            return numpy.zeros((0, len(qpts)))
        P = ONPolynomialSet(ref_el, k, alpha=1, beta=1)
        P_at_qpts = P.tabulate(qpts)[(0,) * dim]

        x = qpts.T
        bubble = (1.0 - numpy.sum(x, axis=0)) * numpy.prod(x, axis=0)
        bubbles = P_at_qpts * bubble

        W = Q.get_weights()
        if dim == 1:
            dmat, _ = make_dmat(qpts.flatten())
            K = numpy.dot(numpy.multiply(dmat, W), dmat.T)
        else:
            # Get an ON basis
            P = ONPolynomialSet(ref_el, degree)
            tab = P.tabulate(qpts, 1)
            # Assemble a stiffness matrix in the ON basis
            moments = lambda dv: numpy.dot(numpy.multiply(dv, W), dv.T)
            K = sum(moments(tab[alpha]) for alpha in tab if sum(alpha) == 1)
            # Change of basis to Lagrange polynomials in the quadrauture nodes
            v = numpy.multiply(tab[(0, ) * dim], W)
            K = numpy.dot(numpy.dot(v.T, K), v)

        phis = numpy.multiply(numpy.dot(bubbles, K), 1/W)
        phis = numpy.concatenate([phis[1::2], phis[0::2]])
        return phis


class IntegratedLegendre(finite_element.CiarletElement):
    """1D continuous element with integrated Legendre polynomials."""

    def __init__(self, ref_el, degree):
        if ref_el.shape not in {POINT, LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("%s is only defined on simplices." % type(self))
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = IntegratedLegendreDual(ref_el, degree, poly_set)
        formdegree = 0  # 0-form
        super(IntegratedLegendre, self).__init__(poly_set, dual, degree, formdegree)
