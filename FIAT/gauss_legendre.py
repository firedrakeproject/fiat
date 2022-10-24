# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

from FIAT import finite_element, polynomial_set, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.lagrange import make_entity_permutations
from FIAT.barycentric_interpolation import LagrangePolynomialSet


class GaussLegendreDualSet(dual_set.DualSet):
    """The dual basis for 1D discontinuous elements with nodes at the
    Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}
        lr = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]
        entity_permutations = {}
        entity_permutations[0] = {0: {0: []}, 1: {0: []}}
        entity_permutations[1] = {0: make_entity_permutations(1, degree + 1)}

        super(GaussLegendreDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class GaussLegendre(finite_element.CiarletElement):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Gauss-Legendre elements are only defined in one dimension.")
        # poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = GaussLegendreDualSet(ref_el, degree)
        points = []
        for node in dual.nodes:
            # Assert singleton point for each node.
            pt, = node.get_point_dict().keys()
            points.append(pt)
        poly_set = LagrangePolynomialSet(ref_el, points)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(GaussLegendre, self).__init__(poly_set, dual, degree, formdegree)
