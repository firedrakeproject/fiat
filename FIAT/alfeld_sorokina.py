# Copyright (C) 2024 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

import copy

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import ComponentPointEvaluation, PointDivergence
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import CkPolynomialSet, AlfeldSplit
from FIAT.reference_element import cast_vertices

import numpy


def AlfeldSorokinaSpace(ref_el, degree):
    """Return a vector-valued C0 PolynomialSet on an Alfeld split with C0
    divergence. This works on any simplex and for all polynomial degrees."""
    ref_complex = AlfeldSplit(ref_el)
    sd = ref_complex.get_spatial_dimension()
    C0 = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), variant="bubble")
    expansion_set = C0.get_expansion_set()
    num_members = C0.get_num_members()
    coeffs = C0.get_coeffs()

    interior_facets = ref_complex.get_interior_facets(sd-1)
    if len(interior_facets) > 0:
        # Redo this in double precision, on a copy of the actual geometry.
        ref_el_fp64 = copy.copy(ref_el)
        ref_el_fp64.vertices = cast_vertices(ref_el.vertices, float)
        ref_el_fp64._split_cache = {}
        ref_complex_fp64 = AlfeldSplit(ref_el_fp64)
        C0_fp64 = CkPolynomialSet(ref_complex_fp64, degree, order=0, shape=(sd,), variant="bubble")
        expansion_set_fp64 = C0_fp64.get_expansion_set()

        facet_el_fp64 = ref_complex_fp64.construct_subelement(sd-1)
        phi_fp64 = polynomial_set.ONPolynomialSet(facet_el_fp64, 0 if sd == 1 else degree-1)
        Q_fp64 = create_quadrature(facet_el_fp64, 2 * phi_fp64.degree)
        qpts_fp64, qwts_fp64 = Q_fp64.get_points(), Q_fp64.get_weights()
        phi_at_qpts_fp64 = phi_fp64.tabulate(qpts_fp64)[(0,) * (sd-1)]
        weights_fp64 = numpy.multiply(phi_at_qpts_fp64, qwts_fp64)

        rows_fp64 = []
        for facet in ref_complex_fp64.get_interior_facets(sd-1):
            n_fp64 = ref_complex_fp64.compute_normal(facet)
            jumps_fp64 = expansion_set_fp64.tabulate_normal_jumps(degree, qpts_fp64, facet, order=1)
            div_jump_fp64 = n_fp64[:, None, None] * jumps_fp64[1][None, ...]
            r_fp64 = numpy.tensordot(div_jump_fp64, weights_fp64, axes=(-1, -1))
            rows_fp64.append(r_fp64.reshape(num_members, -1).T)

        dual_mat = numpy.vstack(rows_fp64)
        nsp = polynomial_set.spanning_basis(dual_mat, nullspace=True)
        coeffs = numpy.tensordot(nsp.astype(coeffs.dtype), coeffs, axes=(-1, 0))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, coeffs)


class AlfeldSorokinaDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        if degree != 2:
            raise NotImplementedError(f"{type(self).__name__} only defined for degree = 2")

        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        nodes = []
        for dim in sorted(top):
            for entity in sorted(top[dim]):
                cur = len(nodes)

                dpts = ref_el.make_points(dim, entity, degree-1)
                nodes.extend(PointDivergence(ref_el, pt) for pt in dpts)

                pts = ref_el.make_points(dim, entity, degree)
                nodes.extend(ComponentPointEvaluation(ref_el, k, (sd,), pt)
                             for pt in pts for k in range(sd))
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class AlfeldSorokina(finite_element.CiarletElement):
    """The Alfeld-Sorokina C0 quadratic macroelement with C0 divergence.

    This element belongs to a Stokes complex, and is paired with CG1(Alfeld).
    """
    def __init__(self, ref_el, degree=2):
        dual = AlfeldSorokinaDualSet(ref_el, degree)
        poly_set = AlfeldSorokinaSpace(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree,
                         mapping="contravariant piola")
