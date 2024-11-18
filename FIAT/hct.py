# Copyright (C) 2024 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT.functional import (PointEvaluation, PointDerivative,
                             IntegralMoment,
                             IntegralMomentOfNormalDerivative)
from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.reference_element import TRIANGLE, ufc_simplex
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.jacobi import eval_jacobi, eval_jacobi_batch, eval_jacobi_deriv_batch


class HCTDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, reduced=False):
        if reduced and degree != 3:
            raise ValueError("Reduced HCT only defined for degree = 3")
        if degree < 3:
            raise ValueError("HCT only defined for degree >= 3")
        ref_el = ref_complex.get_parent()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("HCT only defined on triangles")
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # get first order jet at each vertex
        alphas = polynomial_set.mis(sd, 1)
        nodes = []
        for v in sorted(top[0]):
            pt = verts[v]
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, pt))
            nodes.extend(PointDerivative(ref_el, pt, alpha) for alpha in alphas)
            entity_ids[0][v].extend(range(cur, len(nodes)))

        k = 2 if reduced else degree - 3
        facet = ufc_simplex(1)
        Q = create_quadrature(facet, degree-1+k)
        qpts = Q.get_points()
        xref = 2.0 * qpts - 1.0
        if reduced:
            f_at_qpts = eval_jacobi(0, 0, k, xref[:, 0])
            for e in sorted(top[1]):
                cur = len(nodes)
                nodes.append(IntegralMomentOfNormalDerivative(ref_el, e, Q, f_at_qpts))
                entity_ids[1][e].extend(range(cur, len(nodes)))
        else:
            phis = eval_jacobi_batch(1, 1, k, xref)
            dphis = eval_jacobi_deriv_batch(1, 1, k, xref)
            for e in sorted(top[1]):
                Q_mapped = FacetQuadratureRule(ref_el, 1, e, Q)
                scale = 2 / Q_mapped.jacobian_determinant()
                cur = len(nodes)
                nodes.extend(IntegralMomentOfNormalDerivative(ref_el, e, Q, phi) for phi in phis)
                nodes.extend(IntegralMoment(ref_el, Q_mapped, dphi * scale) for dphi in dphis[1:])
                entity_ids[1][e].extend(range(cur, len(nodes)))

            q = degree - 4
            if q >= 0:
                Q = create_quadrature(ref_complex, degree + q)
                Pq = polynomial_set.ONPolynomialSet(ref_el, q, scale=1)
                phis = Pq.tabulate(Q.get_points())[(0,) * sd]
                scale = 1 / ref_el.volume()
                cur = len(nodes)
                nodes.extend(IntegralMoment(ref_el, Q, phi * scale) for phi in phis)
                entity_ids[sd][0] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HsiehCloughTocher(finite_element.CiarletElement):
    """The HCT macroelement. For degree higher than 3, we implement the
    super-smooth C^1 space from Groselj and Knez (2022) on a barycentric split,
    although there the basis functions are positive on an incenter split.
    """
    def __init__(self, ref_el, degree=3, reduced=False):
        ref_complex = macro.AlfeldSplit(ref_el)
        dual = HCTDualSet(ref_complex, degree, reduced=reduced)
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1, vorder=degree-1, variant="bubble")
        formdegree = 0
        super().__init__(poly_set, dual, degree, formdegree=formdegree)
