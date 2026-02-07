# -*- coding: utf-8 -*-
"""Implementation of the Mardal-Tai-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import numpy
from FIAT import dual_set, expansions, finite_element, polynomial_set
from FIAT.functional import FrobeniusIntegralMoment, IntegralMomentOfDivergence
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.nedelec import Nedelec


def MardalTaiWintherSpace(ref_el):
    # Generate constraint nodes on the cell and facets
    # * div(v) must be constant on the cell.  Since v is a cubic and
    #   div(v) is quadratic, we need the integral of div(v) against the
    #   linear and quadratic Dubiner polynomials to vanish.
    #   There are two linear and three quadratics, so these are five
    #   constraints
    # * v.n must be linear on each facet.  Since v.n is cubic, we need
    #   the integral of v.n against the cubic and quadratic Legendre
    #   polynomial to vanish on each facet.

    # So we introduce functionals whose kernel describes this property,
    # as described in the FIAT paper.

    top = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()
    # Polynomials of degree sd+1
    degree = sd + 1
    poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, shape=(sd,))

    constraints = []
    # Normal component in P1
    ref_facet = ref_el.construct_subelement(sd-1)
    P = polynomial_set.ONPolynomialSet(ref_facet, degree)
    start = expansions.polynomial_dimension(ref_facet, 1)
    stop = expansions.polynomial_dimension(ref_facet, P.degree)
    Q = create_quadrature(ref_facet, degree + P.degree)
    Phis = P.take(range(start, stop)).tabulate(Q.get_points())[(0,)*(sd-1)]
    for f in sorted(top[sd-1]):
        n = ref_el.compute_normal(f)
        Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
        phis = n[None, :, None] * Phis[:, None, :]
        constraints.extend(FrobeniusIntegralMoment(ref_el, Qf, phi) for phi in phis)

    # Divergence in P0
    P = polynomial_set.ONPolynomialSet(ref_el, degree-1)
    start = expansions.polynomial_dimension(ref_el, 0)
    stop = expansions.polynomial_dimension(ref_el, P.degree)
    Q = create_quadrature(ref_el, degree-1 + P.degree)
    phis = P.take(range(start, stop)).tabulate(Q.get_points())[(0,)*sd]
    constraints.extend(IntegralMomentOfDivergence(ref_el, Q, phi) for phi in phis)

    return polynomial_set.ConstrainedPolynomialSet(constraints, poly_set)


class MardalTaiWintherDual(dual_set.DualSet):
    """Degrees of freedom for Mardal-Tai-Winther elements."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()

        if sd not in (2, 3):
            raise ValueError("Mardal-Tai-Winther elements are only defined in dimension 2.")

        if degree != sd+1:
            raise ValueError("Mardal-Tai-Winther elements are only defined for degree = dim+1.")

        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # no vertex dofs

        # On each facet, let n be its normal.  We need to integrate
        # u.n and u.t against the first Legendre polynomial (constant)
        # and u.n against the second (linear).
        ref_facet = ref_el.get_facet_element()
        # Facet nodes are \int_F v.n p ds where p \in P_{q-1}
        # degree is q - 1
        Q = create_quadrature(ref_facet, degree+1)
        P1 = polynomial_set.ONPolynomialSet(ref_facet, 1)
        P1_at_qpts = P1.tabulate(Q.get_points())[(0,)*(sd - 1)]
        if sd == 2:
            Phis = P1_at_qpts[:1, None, :]
        else:
            Ned1 = Nedelec(ref_facet, 1)
            Phis = Ned1.tabulate(0, Q.get_points())[(0,)*(sd - 1)]
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
            Jf = Qf.jacobian()
            n = ref_el.compute_scaled_normal(f)
            phis = numpy.tensordot(Jf, Phis.transpose(1, 0, 2), axes=(-1, 0)).transpose(1, 0, 2)

            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, phi) for phi in phis)
            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, numpy.outer(n, phi)) for phi in P1_at_qpts)
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class MardalTaiWinther(finite_element.CiarletElement):
    """The definition of the Mardal-Tai-Winther element.
    """
    def __init__(self, ref_el, degree=3):
        dual = MardalTaiWintherDual(ref_el, degree)
        poly_set = MardalTaiWintherSpace(ref_el)
        formdegree = ref_el.get_spatial_dimension() - 1
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
