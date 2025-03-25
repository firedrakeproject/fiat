# Copyright (C) 2025 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2025

import numpy
from FIAT import finite_element, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet, get_lagrange_points
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre


class HistopolationDualSet(dual_set.DualSet):
    r"""The dual basis for 1D histopolation elements.

    We define window functions w_j that satisfy

    \int_K w_j v dx = \ell_j(v)   for all v in P_{k}

    where

    \ell_j(v) = 1/h_j \int_{[x_j, x_{j+1}]} v dx

    is the usual histopolation dual basis.

    The DOFs are defined as integral moments against w_j.
    """
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}

        fe = GaussLobattoLegendre(ref_el, degree+1)
        points = get_lagrange_points(fe.dual_basis())
        h = numpy.diff(numpy.reshape(points, (-1,)))
        B = numpy.diag(1.0 / h[:-1], k=-1)
        numpy.fill_diagonal(B, -1.0 / h)

        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        self.rule = rule

        phi = fe.tabulate(1, rule.get_points())
        wts = rule.get_weights()
        D = phi[(1, )][:-1]
        A = numpy.dot(numpy.multiply(D, wts), D.T)

        C = numpy.linalg.solve(A, B)
        F = numpy.dot(C.T, D)
        nodes = [functional.IntegralMoment(ref_el, rule, f) for f in F]

        entity_permutations = {}
        entity_permutations[0] = {0: {0: []}, 1: {0: []}}
        entity_permutations[1] = {0: make_entity_permutations_simplex(1, degree + 1)}

        super().__init__(nodes, ref_el, entity_ids, entity_permutations)


class Histopolation(finite_element.CiarletElement):
    """1D discontinuous element with integral DOFs on GLL subgrid."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Histopolation elements are only defined in one dimension.")

        dual = HistopolationDualSet(ref_el, degree)
        poly_set = LagrangePolynomialSet(ref_el, dual.rule.pts)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)
