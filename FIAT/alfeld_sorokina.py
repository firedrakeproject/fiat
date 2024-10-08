# Copyright (C) 2024 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import ComponentPointEvaluation, PointDivergence, IntegralMoment
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.macro import CkPolynomialSet, AlfeldSplit

import numpy


def AlfeldSorokinaSpace(ref_el, degree):
    """Return a vector-valued C^0 PolynomialSet on an Alfeld split whose
    divergence is also C^0.  This works on any simplex and for all
    polynomial degrees."""
    ref_complex = AlfeldSplit(ref_el)
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
    def __init__(self, ref_el, degree, variant=None):
        if degree != 2:
            raise NotImplementedError("Alfeld-Sorokina only defined for degree = 2")
        if variant is None:
            variant = "integral"
        if variant not in {"integral", "point"}:
            raise ValueError(f"Invalid variant {variant}")

        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        nodes = []
        dims = (0, 1) if variant == "point" else (0,)
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

        if variant == "integral":
            # Edge moments against quadratic Legendre (mean-free bubble)
            dim = 1
            facet = ref_el.construct_subelement(dim)
            Q = create_quadrature(facet, degree+dim+1)
            f_at_qpts = facet.compute_bubble(Q.get_points())
            f_at_qpts -= numpy.dot(f_at_qpts, Q.get_weights()) / facet.volume()
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q_mapped = FacetQuadratureRule(ref_el, dim, entity, Q)
                detJ = Q_mapped.jacobian_determinant()
                phi = f_at_qpts / detJ
                nodes.extend(IntegralMoment(ref_el, Q_mapped, phi, comp=k, shp=(sd,))
                             for k in range(sd))
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class AlfeldSorokina(finite_element.CiarletElement):
    """The Alfeld-Sorokina C^0 quadratic macroelement with C^0 divergence. This element
    belongs to a Stokes complex, and is paired with Lagrange(ref_el, 1, variant="alfeld")."""
    def __init__(self, ref_el, degree=2):
        dual = AlfeldSorokinaDualSet(ref_el, degree)
        poly_set = AlfeldSorokinaSpace(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree,
                         mapping="contravariant piola")
