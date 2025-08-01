# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import finite_element, dual_set, functional
from FIAT.reference_element import symmetric_simplex
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.polynomial_set import ONPolynomialSet, make_bubbles
from FIAT.check_format_variant import check_format_variant
from FIAT.P0 import P0


def make_dual_bubbles(ref_el, degree, codim=0, interpolant_deg=None):
    """Tabulate the L2-duals of the hierarchical C0 basis."""
    if ref_el.get_spatial_dimension() == 0:
        degree = 0
    if interpolant_deg is None:
        interpolant_deg = degree
    Q = create_quadrature(ref_el, degree + interpolant_deg)
    B = make_bubbles(ref_el, degree, codim=codim, scale="orthonormal")
    P_at_qpts = B.expansion_set.tabulate(degree, Q.get_points())
    M = numpy.dot(numpy.multiply(P_at_qpts, Q.get_weights()), P_at_qpts.T)
    phis = numpy.linalg.solve(M, P_at_qpts)
    phis = numpy.dot(B.get_coeffs(), phis)
    return Q, phis


class LegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree, codim=0, interpolant_deg=None):
        if interpolant_deg is None:
            interpolant_deg = degree
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        dim = sd - codim
        ref_facet = ref_el.construct_subelement(dim)
        poly_set = ONPolynomialSet(ref_facet, degree, scale="L2 piola")
        Q_ref = create_quadrature(ref_facet, degree + interpolant_deg)
        Phis = poly_set.tabulate(Q_ref.get_points())[(0,) * dim]
        for entity in sorted(top[dim]):
            cur = len(nodes)
            Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
            # phis must transform like a d-form to undo the measure transformation
            scale = 1 / Q_facet.jacobian_determinant()
            phis = scale * Phis
            nodes.extend(functional.IntegralMoment(ref_el, Q_facet, phi) for phi in phis)
            entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Legendre(finite_element.CiarletElement):
    """Simplicial discontinuous element with Legendre polynomials."""
    def __new__(cls, ref_el, degree, variant=None):
        if degree == 0:
            splitting, variant, interpolant_deg = check_format_variant(variant, degree)
            if splitting is None and interpolant_deg == 0:
                # FIXME P0 on the split requires implementing SplitSimplicialComplex.symmetry_group_size()
                return P0(ref_el)
        return super().__new__(cls)

    def __init__(self, ref_el, degree, variant=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = LegendreDual(ref_el, degree, interpolant_deg=interpolant_deg)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements."""
    def __init__(self, ref_el, degree, interpolant_deg=None):
        if interpolant_deg is None:
            interpolant_deg = degree
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        for dim in sorted(top):
            if degree <= dim:
                continue
            ref_facet = symmetric_simplex(dim)
            Q_ref, Phis = make_dual_bubbles(ref_facet, degree, interpolant_deg=interpolant_deg)
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                # phis must transform like a d-form to undo the measure transformation
                scale = 1 / Q_facet.jacobian_determinant()
                phis = scale * Phis
                nodes.extend(functional.IntegralMoment(ref_el, Q_facet, phi) for phi in phis)
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class IntegratedLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""
    def __init__(self, ref_el, degree, variant=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)
        if degree < 1:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= 1")
        poly_set = ONPolynomialSet(ref_el, degree, variant="bubble")
        dual = IntegratedLegendreDual(ref_el, degree, interpolant_deg=interpolant_deg)
        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree)
