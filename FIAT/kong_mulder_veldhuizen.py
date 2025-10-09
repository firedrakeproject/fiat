# Copyright (C) 2020 Robert C. Kirby (Baylor University)
#
# contributions by Keith Roberts (University of São Paulo)
# and Alexandre Olender (University of São Paulo)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import finite_element, functional, dual_set
from FIAT.check_format_variant import parse_lagrange_variant
from FIAT.expansions import polynomial_entity_ids
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import (LINE, TRIANGLE, TETRAHEDRON,
                                    point_entity_ids)
import math


def bump(T, deg):
    """Return a tuple with the degree raise for each codimension"""
    sd = T.get_spatial_dimension()
    if deg == 1 or sd == 1:
        return ()
    elif sd == 2:
        if deg < 5:
            return (1,)
        elif deg == 5 or deg == 6:
            return (2,)
        else:
            raise ValueError("Degree not supported")
    elif sd == 3:
        if deg < 4:
            return (2, 1)
        else:
            raise ValueError("Degree not supported")
    else:
        raise ValueError("Dimension of element is not supported")


def KongMulderVeldhuizenSpace(ref_el, deg):
    sd = ref_el.get_spatial_dimension()
    degree = [deg] * (sd+1)
    for codim, degree_raise in enumerate(bump(ref_el, deg)):
        degree[sd-codim] += degree_raise

    k = max(degree)
    P = ONPolynomialSet(ref_el, k, variant="bubble")
    U = P.get_expansion_set()
    entity_ids = polynomial_entity_ids(ref_el, k, continuity=U.continuity)

    ids = []
    for dim in entity_ids:
        num_bubbles = math.comb(degree[dim] - 1, dim)
        for entity in entity_ids[dim]:
            ids.extend(entity_ids[dim][entity][:num_bubbles])
    return P.take(ids)


class KongMulderVeldhuizenDualSet(dual_set.DualSet):
    """The dual basis for KMV simplical elements."""

    def __init__(self, ref_el, degree):
        Q = create_quadrature(ref_el, degree, scheme="KMV")
        points = Q.get_points()
        entity_ids = point_entity_ids(ref_el, points)
        nodes = [functional.PointEvaluation(ref_el, x) for x in points]
        super().__init__(nodes, ref_el, entity_ids)


class KongMulderVeldhuizen(finite_element.CiarletElement):
    """The "lumped" simplical finite element (NB: requires custom quad. "KMV" points to achieve a diagonal mass matrix).

    References
    ----------

    Higher-order triangular and tetrahedral finite elements with mass
    lumping for solving the wave equation
    M. J. S. CHIN-JOE-KONG, W. A. MULDER and M. VAN VELDHUIZEN

    HIGHER-ORDER MASS-LUMPED FINITE ELEMENTS FOR THE WAVE EQUATION
    W.A. MULDER

    NEW HIGHER-ORDER MASS-LUMPED TETRAHEDRAL ELEMENTS
    S. GEEVERS, W.A. MULDER, AND J.J.W. VAN DER VEGT

    More Continuous Mass-Lumped Triangular Finite Elements
    W. A. MULDER

    """

    def __init__(self, ref_el, degree, variant=None):
        splitting, variant = parse_lagrange_variant(variant)
        if splitting:
            ref_el = splitting(ref_el)

        if ref_el.shape not in {LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("KMV is only valid for simplices of dimension <= 3.")
        if degree > 6 and ref_el.shape == TRIANGLE:
            raise NotImplementedError("Only P < 7 for triangles are implemented.")
        if degree > 3 and ref_el.shape == TETRAHEDRON:
            raise NotImplementedError("Only P < 4 for tetrahedrals are implemented.")
        S = KongMulderVeldhuizenSpace(ref_el, degree)

        dual = KongMulderVeldhuizenDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super().__init__(S, dual, S.degree, formdegree)
