# -*- coding: utf-8 -*-
"""Implementation of the Mardal-Tai-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import numpy
from FIAT import dual_set, expansions, finite_element, polynomial_set
from FIAT.functional import FrobeniusIntegralMoment
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


def curl(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    d = len(grad_u)
    if d == 2:
        curl_u = [grad_u[1], -grad_u[0]]
        return numpy.concatenate(curl_u, axis=1)
    else:
        indices = ((i, j) for i in reversed(range(d)) for j in reversed(range(i+1, d)))
        curl_u = [((-1)**k) * (grad_u[j][:, i, :] - grad_u[i][:, j, :]) for k, (i, j) in enumerate(indices)]
        return numpy.transpose(curl_u, (1, 0, 2))


def MardalTaiWintherSpace(ref_el):
    """Construct the MTW space [P1]^d + curl(B [P1]^d)."""
    sd = ref_el.get_spatial_dimension()
    # Polynomials of degree sd+1
    degree = sd + 1
    Pk = polynomial_set.ONPolynomialSet(ref_el, degree, shape=(sd,), scale="orthonormal")

    # Grab [P1]^d from [Pk]^d
    dimP1 = expansions.polynomial_dimension(ref_el, 1)
    dimPk = expansions.polynomial_dimension(ref_el, degree)
    ids = [i+dimPk*j for i in range(dimP1) for j in range(sd)]
    P1 = Pk.take(ids)
    # Project curl(B [P1]^d) into [Pk]^d
    BP1 = polynomial_set.make_bubbles(ref_el, degree+1, shape=((sd*(sd-1))//2,))

    Q = create_quadrature(ref_el, degree*2)
    qpts = Q.get_points()
    qwts = Q.get_weights()
    Pk_at_qpts = Pk.tabulate(qpts)
    BP1_at_qpts = BP1.tabulate(qpts, 1)

    inner = lambda u, v, qwts: numpy.tensordot(u, numpy.multiply(v, qwts), axes=(range(1, u.ndim),)*2)
    C = inner(curl(BP1_at_qpts), Pk_at_qpts[(0,)*sd], qwts)
    coeffs = numpy.tensordot(C, Pk.get_coeffs(), axes=(1, 0))
    curlBP1 = polynomial_set.PolynomialSet(ref_el, degree, degree, Pk.get_expansion_set(), coeffs)

    return polynomial_set.polynomial_set_union_normalized(P1, curlBP1)


class MardalTaiWintherDual(dual_set.DualSet):
    """Degrees of freedom for Mardal-Tai-Winther elements."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()

        if sd not in (2, 3):
            raise ValueError("Mardal-Tai-Winther elements are only defined in dimension 2 and 3.")

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
            Phis = numpy.zeros((3, sd-1, P1_at_qpts.shape[-1]))
            Phis[0, 0, :] = P1_at_qpts[0, None, :]
            Phis[1, 1, :] = P1_at_qpts[0, None, :]
            Phis[2, 0, :] = P1_at_qpts[1, None, :]
            Phis[2, 1, :] = P1_at_qpts[2, None, :]

        for f in sorted(top[sd-1]):
            cur = len(nodes)
            n = ref_el.compute_scaled_normal(f)
            Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
            Jf = ref_el.compute_tangents(sd-1, f)
            phis = numpy.tensordot(Jf.T, Phis.transpose((1, 0, 2)), (1, 0)).transpose((1, 0, 2))

            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, numpy.cross(n, phi, axis=0) if i == 2 else phi) for i, phi in enumerate(phis))
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
