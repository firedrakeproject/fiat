# -*- coding: utf-8 -*-
"""Implementation of the Mardal-Tai-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import numpy
from FIAT import dual_set, expansions, finite_element, polynomial_set
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.functional import FrobeniusIntegralMoment
from FIAT.nedelec import Nedelec
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


def curl(tabulation):
    """
    Compute the curl of a vector (the skew-symmetric part of the gradient) or the rot of scalar in 2D.

    :arg tabulation: a dictionary with at least the first order tabulation.
    :returns: a numpy.ndarray with the curl.
    """
    grad_u = {alpha.index(1): tabulation[alpha] for alpha in tabulation if sum(alpha) == 1}
    shp = grad_u[0].shape[1:-1]
    if shp == ():
        curl_u = [grad_u[1], -grad_u[0]]
    else:
        d = len(grad_u)
        indices = ((i, j) for i in reversed(range(d)) for j in reversed(range(i+1, d)))
        curl_u = [((-1)**k) * (grad_u[j][:, i, :] - grad_u[i][:, j, :]) for k, (i, j) in enumerate(indices)]
    return numpy.transpose(curl_u, (1, 0, 2))


def MardalTaiWintherSpace(ref_el, order=1):
    """Construct the MTW space BDM(order) + curl(B [P1]^d)."""
    sd = ref_el.get_spatial_dimension()
    k = sd + 1
    assert order < k
    # [Pk]^d = vector polynomials of degree k = sd+1
    Pk = polynomial_set.ONPolynomialSet(ref_el, k, shape=(sd,), scale="orthonormal")

    # Grab BDM(order) = [P_order]^d from [Pk]^d
    dimP1 = expansions.polynomial_dimension(ref_el, order)
    dimPk = expansions.polynomial_dimension(ref_el, k)
    ids = [i+dimPk*j for i in range(dimP1) for j in range(sd)]
    BDM = Pk.take(ids)

    # Project curl(B [P1]^d) into [Pk]^d
    shape = () if sd == 2 else ((sd*(sd-1))//2,)
    BP1 = polynomial_set.make_bubbles(ref_el, k+1, shape=shape)

    Q = create_quadrature(ref_el, 2*k)
    qpts = Q.get_points()
    qwts = Q.get_weights()
    Pk_at_qpts = Pk.tabulate(qpts)
    BP1_at_qpts = BP1.tabulate(qpts, 1)

    inner = lambda u, v, qwts: numpy.tensordot(u, numpy.multiply(v, qwts), axes=(range(1, u.ndim),)*2)
    C = inner(curl(BP1_at_qpts), Pk_at_qpts[(0,)*sd], qwts)
    coeffs = numpy.tensordot(C, Pk.get_coeffs(), axes=(1, 0))
    curlBP1 = polynomial_set.PolynomialSet(ref_el, k, k, Pk.get_expansion_set(), coeffs)

    return polynomial_set.polynomial_set_union_normalized(BDM, curlBP1)


class MardalTaiWintherDual(dual_set.DualSet):
    """Degrees of freedom for Mardal-Tai-Winther elements."""
    def __init__(self, ref_el, order, quad_scheme):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []
        degree = sd + 1

        # On each facet, let n be its normal.  We need to integrate
        # u.n against a Dubiner basis for P1
        # and u x n against a basis for lowest-order RT.
        ref_facet = ref_el.get_facet_element()
        Q = parse_quadrature_scheme(ref_facet, degree+order, quad_scheme)

        P1 = polynomial_set.ONPolynomialSet(ref_facet, order)
        P1_at_qpts = P1.tabulate(Q.get_points())[(0,)*(sd - 1)]
        if sd == 2:
            # For 2D just take the constant
            RT_at_qpts = P1_at_qpts[:1, None, :]
        else:
            # Basis for lowest-order RT [(1, 0), (0, 1), (x, y)]
            RT_at_qpts = numpy.zeros((3, sd-1, P1_at_qpts.shape[-1]))
            RT_at_qpts[0, 0, :] = P1_at_qpts[0, None, :]
            RT_at_qpts[1, 1, :] = P1_at_qpts[0, None, :]
            RT_at_qpts[2, 0, :] = P1_at_qpts[1, None, :]
            RT_at_qpts[2, 1, :] = P1_at_qpts[2, None, :]

        for f in sorted(top[sd-1]):
            cur = len(nodes)
            n = ref_el.compute_scaled_normal(f)
            Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
            # Normal moments against P_{order}
            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, numpy.outer(n, phi)) for phi in P1_at_qpts)
            # Map the RT basis into the facet
            Jf = Qf.jacobian()
            phis = numpy.tensordot(Jf, RT_at_qpts.transpose(1, 0, 2), (1, 0)).transpose(1, 0, 2)
            if sd == 3:
                # Moments against cross(n, RT)
                phis = numpy.cross(n[None, :, None], phis, axis=1)
            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, phi) for phi in phis)
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Interior nodes: moments against Nedelec(order-1)
        if order > 1:
            Q = parse_quadrature_scheme(ref_el, degree+order-1, quad_scheme)
            Ned = Nedelec(ref_el, order-1)
            phis = Ned.tabulate(0, Q.get_points())[(0,) * sd]
            cur = len(nodes)
            nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
            entity_ids[sd][0] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class MardalTaiWinther(finite_element.CiarletElement):
    """The definition of the Mardal-Tai-Winther element.
    """
    def __init__(self, ref_el, order=1, quad_scheme=None):
        sd = ref_el.get_spatial_dimension()
        if sd not in (2, 3):
            raise ValueError(f"{type(self).__name__} only defined in dimension 2 and 3.")
        if not ref_el.is_simplex():
            raise ValueError(f"{type(self).__name__} only defined on simplices.")
        if order >= sd:
            raise ValueError(f"{type(self).__name__} only defined for 1 <= order < dim. "
                             "The order is defined as the embedded sub-degree, with 1 for lowest-order case.")

        dual = MardalTaiWintherDual(ref_el, order, quad_scheme)
        poly_set = MardalTaiWintherSpace(ref_el, order)
        formdegree = sd - 1
        super().__init__(poly_set, dual, order, formdegree, mapping="contravariant piola")
