# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import (PointEvaluation, PointDerivative, PointNormalDerivative,
                             IntegralMoment,
                             IntegralMomentOfNormalDerivative)
from FIAT.jacobi import eval_jacobi_batch, eval_jacobi_deriv_batch
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TRIANGLE, ufc_simplex


class ArgyrisDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, interpolant_deg):
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Argyris only defined on triangles")

        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # get second order jet at each vertex
        verts = ref_el.get_vertices()
        alphas = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, verts[v]))
            nodes.extend(PointDerivative(ref_el, verts[v], alpha) for alpha in alphas)
            entity_ids[0][v] = list(range(cur, len(nodes)))

        if variant == "integral":
            # edge dofs
            k = degree - 5
            rline = ufc_simplex(1)
            Q = create_quadrature(rline, interpolant_deg+k-1)
            x = 2.0 * Q.get_points() - 1.0
            phis = eval_jacobi_batch(2, 2, k, x)
            dphis = eval_jacobi_deriv_batch(2, 2, k, x)
            for e in sorted(top[1]):
                Q_mapped = FacetQuadratureRule(ref_el, 1, e, Q)
                scale = 2 / Q_mapped.jacobian_determinant()
                cur = len(nodes)
                nodes.extend(IntegralMomentOfNormalDerivative(ref_el, e, Q, phi) for phi in phis)
                nodes.extend(IntegralMoment(ref_el, Q_mapped, dphi * scale) for dphi in dphis[1:])
                entity_ids[1][e].extend(range(cur, len(nodes)))

            # interior dofs
            q = degree - 6
            if q >= 0:
                cell = ref_el.construct_subelement(sd)
                Q_ref = create_quadrature(cell, interpolant_deg + q)
                Pq = polynomial_set.ONPolynomialSet(cell, q, scale=1)
                Phis = Pq.tabulate(Q_ref.get_points())[(0,) * sd]
                for entity in sorted(top[sd]):
                    Q = FacetQuadratureRule(ref_el, sd, entity, Q_ref)
                    scale = 1 / Q.jacobian_determinant()
                    phis = scale * Phis
                    cur = len(nodes)
                    nodes.extend(IntegralMoment(ref_el, Q, phi) for phi in phis)
                    entity_ids[sd][entity] = list(range(cur, len(nodes)))

        elif variant == "point":
            # edge dofs
            for e in sorted(top[1]):
                cur = len(nodes)
                # normal derivatives at degree - 4 points on each edge
                ndpts = ref_el.make_points(1, e, degree - 3)
                nodes.extend(PointNormalDerivative(ref_el, e, pt) for pt in ndpts)

                # point value at degree - 5 points on each edge
                ptvalpts = ref_el.make_points(1, e, degree - 4)
                nodes.extend(PointEvaluation(ref_el, pt) for pt in ptvalpts)
                entity_ids[1][e] = list(range(cur, len(nodes)))

            # interior dofs
            if degree > 5:
                cur = len(nodes)
                for entity in sorted(top[sd]):
                    internalpts = ref_el.make_points(sd, entity, degree - 3)
                    nodes.extend(PointEvaluation(ref_el, pt) for pt in internalpts)
                    entity_ids[sd][entity] = list(range(cur, len(nodes)))
        else:
            raise ValueError("Invalid variant for Argyris")
        super().__init__(nodes, ref_el, entity_ids)


class Argyris(finite_element.CiarletElement):
    """
    The Argyris finite element.

    :arg ref_el: The reference element.
    :arg degree: The degree.
    :arg variant: optional variant specifying the types of nodes.

    variant can be chosen from ["point", "integral", "integral(q)"]
    "point" -> dofs are evaluated by point evaluation.
    "integral" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence.
    "integral(q)" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence plus q.
    """

    def __init__(self, ref_el, degree=5, variant=None):

        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            raise NotImplementedError(f"{type(self).__name__} is not implemented as a macroelement.")

        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, variant="bubble")
        dual = ArgyrisDualSet(ref_el, degree, variant, interpolant_deg)
        super().__init__(poly_set, dual, degree)
