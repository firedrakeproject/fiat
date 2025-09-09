# Copyright (C) 2020 Robert C. Kirby (Baylor University)
#
# contributions by Keith Roberts (University of São Paulo)
# and Alexandre Olender (University of São Paulo)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numpy
from FIAT import (
    finite_element,
    dual_set,
    functional,
    Bubble,
    FacetBubble,
    IntegratedLegendre,
    Lagrange,
    NodalEnrichedElement,
    RestrictedElement,
)
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TRIANGLE, TETRAHEDRON


def _get_entity_ids(ref_el, points, tol=1e-12):
    """The topological association in a dictionary"""
    top = ref_el.topology
    invtop = {top[d][e]: (d, e) for d in top for e in top[d]}
    bary = ref_el.compute_barycentric_coordinates(points)

    entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
    for i in numpy.lexsort(bary.T):
        verts = tuple(numpy.flatnonzero(abs(bary[i]) > tol))
        dim, entity = invtop[verts]
        entity_ids[dim][entity].append(i)
    return entity_ids


def bump(T, deg):
    """Increase degree of polynomial along face/edges"""
    sd = T.get_spatial_dimension()
    if deg == 1:
        return (0, 0)
    else:
        if sd == 2:
            if deg < 5:
                return (1, 1)
            elif deg == 5 or deg == 6:
                return (2, 2)
            else:
                raise ValueError("Degree not supported")
        elif sd == 3:
            if deg < 4:
                return (1, 2)
            else:
                raise ValueError("Degree not supported")
        else:
            raise ValueError("Dimension of element is not supported")


def KongMulderVeldhuizenSpace(T, deg):
    sd = T.get_spatial_dimension()
    if deg == 1:
        return Lagrange(T, 1).poly_set
    else:
        # NOTE The Lagrange bubbles may lead to an ill-conditioned
        # Vandermonde system (depending on the points and expansion set).
        variant = "integral"
        L = IntegratedLegendre(T, deg, variant=variant)
        # Toss the bubble from Lagrange since it's dependent
        # on the higher-dimensional bubbles
        RL = RestrictedElement(L, restriction_domain="edge")

        # interior cell bubble
        facet_bump, interior_bump = bump(T, deg)
        B = Bubble(T, deg + interior_bump, variant=variant)

        elems = [RL, B]
        if sd == 3:
            # bubble on the facet
            elems.append(FacetBubble(T, deg + facet_bump, variant=variant))

        return NodalEnrichedElement(*elems).get_nodal_basis()


class KongMulderVeldhuizenDualSet(dual_set.DualSet):
    """The dual basis for KMV simplical elements."""

    def __init__(self, ref_el, degree):
        lr = create_quadrature(ref_el, degree, scheme="KMV")
        entity_ids = _get_entity_ids(ref_el, lr.get_points())
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]
        super(KongMulderVeldhuizenDualSet, self).__init__(nodes, ref_el, entity_ids)


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

    def __init__(self, ref_el, degree):
        if ref_el.shape not in {TRIANGLE, TETRAHEDRON}:
            raise ValueError("KMV is only valid for triangles and tetrahedrals")
        if degree > 6 and ref_el.shape == TRIANGLE:
            raise NotImplementedError("Only P < 7 for triangles are implemented.")
        if degree > 3 and ref_el.shape == TETRAHEDRON:
            raise NotImplementedError("Only P < 4 for tetrahedrals are implemented.")
        S = KongMulderVeldhuizenSpace(ref_el, degree)

        dual = KongMulderVeldhuizenDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(KongMulderVeldhuizen, self).__init__(
            S, dual, degree + max(bump(ref_el, degree)), formdegree
        )
