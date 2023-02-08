# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

from FIAT import finite_element, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.lagrange import make_entity_permutations
from FIAT.barycentric_interpolation import LagrangePolynomialSet


class GaussLobattoLegendreDualSet(dual_set.DualSet):
    """The dual basis for 1D continuous elements with nodes at the
    Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [0], 1: [degree]},
                      1: {0: list(range(1, degree))}}
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]
        entity_permutations = {}
        entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
        entity_permutations[1] = {0: make_entity_permutations(1, degree - 1)}

        super(GaussLobattoLegendreDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class GaussLobattoLegendre(finite_element.CiarletElement):
    """1D continuous element with nodes at the Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Gauss-Lobatto-Legendre elements are only defined in one dimension.")
        dual = GaussLobattoLegendreDualSet(ref_el, degree)
        points = []
        for node in dual.nodes:
            # Assert singleton point for each node.
            pt, = node.get_point_dict().keys()
            points.append(pt)
        poly_set = LagrangePolynomialSet(ref_el, points)
        formdegree = 0  # 0-form
        super(GaussLobattoLegendre, self).__init__(poly_set, dual, degree, formdegree)
