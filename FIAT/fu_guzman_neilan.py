# Copyright (C) 2024 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import ComponentPointEvaluation, PointDerivative, FrobeniusIntegralMoment
from FIAT.macro import CkPolynomialSet, AlfeldSplit
from FIAT.reference_element import TETRAHEDRON
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.check_format_variant import parse_quadrature_scheme

import numpy


def FuGuzmanNeilanH1curlSpace(ref_complex, degree):
    """Return a vector-valued C0 PolynomialSet with C0 curl.
    This works on any simplex and for all polynomial degrees."""

    sd = ref_complex.get_spatial_dimension()
    C0 = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), variant="bubble")
    expansion_set = C0.get_expansion_set()
    num_members = C0.get_num_members()
    coeffs = C0.get_coeffs()

    facet_el = ref_complex.construct_subelement(sd-1)
    phi = polynomial_set.ONPolynomialSet(facet_el, 0 if sd == 1 else degree-1)
    Q = create_quadrature(facet_el, 2 * phi.degree)
    qpts, qwts = Q.get_points(), Q.get_weights()
    phi_at_qpts = phi.tabulate(qpts)[(0,) * (sd-1)]
    weights = numpy.multiply(phi_at_qpts, qwts)

    rows = []
    for facet in ref_complex.get_interior_facets(sd-1):
        ts = ref_complex.compute_tangents(sd-1, facet)
        jumps = expansion_set.tabulate_normal_jumps(degree, qpts, facet, order=1)
        for t in ts:
            tjump = t[:, None, None] * jumps[1][None, ...]
            r = numpy.tensordot(tjump, weights, axes=(-1, -1))
            rows.append(r.reshape(num_members, -1).T)

    if len(rows) > 0:
        dual_mat = numpy.vstack(rows)
        nsp = polynomial_set.spanning_basis(dual_mat, nullspace=True)
        coeffs = numpy.tensordot(nsp, coeffs, axes=(-1, 0))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, coeffs)


class FuGuzmanNeilanH1curlDualSet(dual_set.DualSet):

    def __init__(self, ref_el, degree, quad_scheme=None):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        shape = (sd,)
        # Vertex dofs: first order jet
        for v in sorted(top[0]):
            cur = len(nodes)
            x, = ref_el.make_points(0, v, degree)
            nodes.append(ComponentPointEvaluation(ref_el, x, shape, cmp) for cmp in numpy.ndindex(shape))
            nodes.extend(PointDerivative(ref_el, x, alpha, shape, cmp)
                         for alpha in polynomial_set.mis(sd, 1)
                         for cmp in numpy.ndindex(shape))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # Edge dofs:
        phi_deg = degree - 5
        dim = 1
        facet = ref_el.construct_subelement(dim)
        Q_ref = parse_quadrature_scheme(facet, degree + phi_deg, quad_scheme)
        Pqmd = polynomial_set.ONPolynomialSet(facet, phi_deg, (dim,))
        Phis = Pqmd.tabulate(Q_ref.get_points())[(0,) * dim]
        Phis = numpy.transpose(Phis, (0, 2, 1))

        for entity in top[dim]:
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref, avg=True)
            R = numpy.array(ref_el.compute_tangents(dim, entity))
            phis = numpy.dot(Phis, R)
            phis = numpy.transpose(phis, (0, 2, 1))
            nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)

            # nodes.extend(IntegralMomentOfCurl(ref_el, Q, phi) for phi in phis)
            entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class FuGuzmanNeilanH1curl(finite_element.CiarletElement):

    def __init__(self, ref_el, degree=4, quad_scheme=None):
        if ref_el.get_shape() != TETRAHEDRON:
            raise ValueError(f"{type(self).__name__} only defined on tetrahedron")
        if degree < 4:
            raise ValueError(f"{type(self).__name__} only defined for degree>=4.")

        dual = FuGuzmanNeilanH1curlDualSet(ref_el, degree, quad_scheme=quad_scheme)
        ref_complex = AlfeldSplit(ref_el)
        poly_set = FuGuzmanNeilanH1curlSpace(ref_complex, degree)
        super().__init__(poly_set, dual, degree, formdegree=1, mapping="covariant piola")


if __name__ == "__main__":
    from FIAT import ufc_simplex
    for dim in (2, 3):
        for degree in range(2, 5):
            ref_el = ufc_simplex(dim)
            ref_complex = AlfeldSplit(ref_el)
            V = FuGuzmanNeilanH1curlSpace(ref_complex, degree)
            print(dim, degree, len(V))
