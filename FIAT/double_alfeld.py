# Copyright (C) 2026 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2026

from FIAT.functional import (PointEvaluation, PointDerivative,
                             IntegralMoment, IntegralMomentOfDerivative)
from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.polynomial_set import mis
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.reference_element import TRIANGLE, ufc_simplex
from FIAT.quadrature import FacetQuadratureRule
from FIAT.jacobi import eval_jacobi_batch, eval_jacobi_deriv_batch


class C2DualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, vorder=None, reduced=False, quad_scheme=None):
        if vorder is None:
            vorder = 2 if ref_complex.is_macrocell() else 4

        if degree < 2*vorder+1:
            raise ValueError(f"{type(self).__name__} only defined for degree >= {2*vorder+1}")

        ref_el = ref_complex.get_parent() or ref_complex
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError(f"{type(self).__name__} only defined on triangles")

        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # vorder jet at vertices
        nodes = []
        for v in sorted(top[0]):
            pt = verts[v]
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, pt))
            nodes.extend(PointDerivative(ref_el, pt, alpha) for i in range(1, vorder+1) for alpha in mis(sd, i))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        k = degree - 2*vorder
        facet = ufc_simplex(1)
        Q_ref = parse_quadrature_scheme(facet, degree-2+k, quad_scheme)
        x = facet.compute_barycentric_coordinates(Q_ref.get_points())
        xref = x[:, [1]] - x[:, [0]]

        if reduced:
            raise NotImplementedError
        else:
            # Integral moments of normal derivatives against Jacobi polynomials along edges
            phis = eval_jacobi_batch(vorder, vorder, k, xref)
            dphis = 2*eval_jacobi_deriv_batch(vorder, vorder, k, xref, order=1)
            ddphis = 4*eval_jacobi_deriv_batch(vorder, vorder, k, xref, order=2)
            for e in sorted(top[1]):
                Q = FacetQuadratureRule(ref_el, 1, e, Q_ref, avg=True)
                n = ref_el.compute_normal(e)
                cur = len(nodes)
                nodes.extend(IntegralMoment(ref_el, Q, ddphi) for ddphi in ddphis[2:])
                nodes.extend(IntegralMomentOfDerivative(ref_el, Q, dphi, n) for dphi in dphis[1:])
                nodes.extend(IntegralMomentOfDerivative(ref_el, Q, phi, n, n) for phi in phis)
                entity_ids[1][e].extend(range(cur, len(nodes)))

            # Interior moments against a basis for Pq
            q = degree - 3 * (vorder // 2 + 1)
            if q >= 0:
                Q = parse_quadrature_scheme(ref_complex, degree + q, quad_scheme)
                Pq = polynomial_set.ONPolynomialSet(ref_el, q, scale=1)
                phis = Pq.tabulate(Q.get_points())[(0,) * sd]
                phis *= 1/ref_el.volume()
                cur = len(nodes)
                nodes.extend(IntegralMoment(ref_el, Q, phi) for phi in phis)
                entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class DoubleAlfeld(finite_element.CiarletElement):
    """The double Alfeld C^2 macroelement on a double barycentric split.
    See Section 7.5 of Lai & Schumacher for the quintic C^2 spline.
    """
    def __init__(self, ref_el, degree=5, reduced=False, quad_scheme=None):
        # Construct the quintic C2 spline on the double Alfeld split
        ref_complex = macro.AlfeldSplit(macro.AlfeldSplit(ref_el))
        # C3 on major split facets, C2 elsewhere
        order = {}
        order[1] = dict.fromkeys(ref_complex.get_interior_facets(1), 2)
        order[1].update(dict.fromkeys(range(3, 6), degree-2))
        # C4 at minor split barycenters, C3 at major split barycenter
        order[0] = dict.fromkeys(ref_complex.get_interior_facets(0), degree-1)
        order[0][3] = degree-2
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=order, variant="bubble")

        dual = C2DualSet(ref_complex, degree, reduced=reduced, quad_scheme=quad_scheme)
        super().__init__(poly_set, dual, degree, formdegree=0)
