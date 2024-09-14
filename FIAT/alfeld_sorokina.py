# Copyright (C) 2024 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT.functional import ComponentPointEvaluation, PointDivergence, FrobeniusIntegralMoment
from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.reference_element import TRIANGLE
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule

import numpy


def C0DivPolynomialSet(ref_complex, degree):
    """Return a vector-valued C^0 PolynomialSet whose divergence is also C^0.
    """
    sd = ref_complex.get_spatial_dimension()
    shp = (sd,)
    P = polynomial_set.ONPolynomialSet(ref_complex, degree, shape=shp, variant="bubble")
    expansion_set = P.get_expansion_set()
    num_members = P.get_num_members()
    coeffs = P.get_coeffs()

    facet_el = ref_complex.construct_subelement(sd-1)
    phi = polynomial_set.ONPolynomialSet(facet_el, 0 if sd == 1 else degree-1)
    Q = create_quadrature(facet_el, 2 * phi.degree)
    qpts, qwts = Q.get_points(), Q.get_weights()
    phi_at_qpts = phi.tabulate(qpts)[(0,) * (sd-1)]
    weights = numpy.multiply(phi_at_qpts, qwts)

    rows = []
    for facet in ref_complex.get_interior_facets(sd-1):
        n = ref_complex.compute_normal(facet)
        jumps = expansion_set.tabulate_normal_jumps(degree, qpts, facet, order=1)
        div_jump = n[:, None, None] * jumps[1][None, ...]
        r = numpy.tensordot(div_jump, weights, axes=(-1, -1))
        rows.append(r.reshape(num_members, -1).T)

    if len(rows) > 0:
        dual_mat = numpy.vstack(rows)
        _, sig, vt = numpy.linalg.svd(dual_mat, full_matrices=True)
        tol = sig[0] * 1E-10
        num_sv = len([s for s in sig if abs(s) > tol])
        coeffs = numpy.tensordot(vt[num_sv:], coeffs, axes=(-1, 0))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, coeffs)


class AlfeldSorokinaDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, reduced=False):
        if degree != 2:
            raise ValueError("Alfeld-Sorokina only defined for degree = 2")
        ref_el = ref_complex.get_parent()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Alfeld-Sorokina only defined on triangles")
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        nodes = []
        dims = (0,) if reduced else (0, 1)
        for dim in dims:
            for entity in sorted(top[dim]):
                cur = len(nodes)
                pts = ref_el.make_points(dim, entity, degree)
                if dim == 0:
                    pt, = pts
                    nodes.append(PointDivergence(ref_el, pt))
                nodes.extend(ComponentPointEvaluation(ref_el, k, (sd,), pt)
                             for pt in pts for k in range(sd))
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        if reduced:
            dim = 1
            q = degree - 2
            facet = ref_el.construct_subelement(dim)
            Q_ref = create_quadrature(facet, degree + q)
            Pq = polynomial_set.ONPolynomialSet(facet, q)
            Pq_at_qpts = Pq.tabulate(Q_ref.get_points())[(0,)*(sd - 1)]
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                Jdet = Q.jacobian_determinant()
                n = ref_el.compute_normal(entity)
                ts = ref_el.compute_tangents(dim, entity)
                comps = (n, *ts)
                nodes.extend(FrobeniusIntegralMoment(ref_el, Q, comp[:, None]*phi[None, :]/Jdet)
                             for phi in Pq_at_qpts for comp in comps)
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super(AlfeldSorokinaDualSet, self).__init__(nodes, ref_el, entity_ids)


class AlfeldSorokina(finite_element.CiarletElement):
    """The Alfeld-Sorokina macroelement."""
    def __init__(self, ref_el, degree=2):
        ref_complex = macro.AlfeldSplit(ref_el)
        dual = AlfeldSorokinaDualSet(ref_complex, degree)
        poly_set = C0DivPolynomialSet(ref_complex, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super(AlfeldSorokina, self).__init__(poly_set, dual, degree, formdegree,
                                             mapping="contravariant piola")
