# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy
from FIAT import finite_element, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre


class HistopolationDualSet(dual_set.DualSet):
    """The dual basis for 1D histopolation elements"""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}

        embedded = GaussLobattoLegendre(ref_el, degree+1)
        points = []
        for node in embedded.dual_basis():
            # Assert singleton point for each node.
            pt, = node.get_point_dict().keys()
            points.append(pt[0])
        h = numpy.diff(numpy.array(points))
        B = numpy.diag(1.0 / h[:-1], k=-1)
        numpy.fill_diagonal(B, -1.0 / h)

        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        self.rule = rule

        phi = embedded.tabulate(1, rule.get_points())
        wts = rule.get_weights()
        D = phi[(1, )][:-1]
        A = numpy.dot(numpy.multiply(D, wts), D.T)

        C = numpy.linalg.solve(A, B)
        F = numpy.dot(C.T, D)
        nodes = [functional.IntegralMoment(ref_el, rule, f) for f in F]

        entity_permutations = {}
        entity_permutations[0] = {0: {0: []}, 1: {0: []}}
        entity_permutations[1] = {0: make_entity_permutations_simplex(1, degree + 1)}

        super(HistopolationDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class Histopolation(finite_element.CiarletElement):
    """1D discontinuous element with average between consecutive Gauss-Lobatto-Legendre points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Histopolation elements are only defined in one dimension.")

        dual = HistopolationDualSet(ref_el, degree)
        poly_set = LagrangePolynomialSet(ref_el, dual.rule.pts)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(Histopolation, self).__init__(poly_set, dual, degree, formdegree)
