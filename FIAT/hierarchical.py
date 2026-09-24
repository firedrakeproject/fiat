# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import finite_element, dual_set, functional
from FIAT.expansions import polynomial_dimension
from FIAT.reference_element import symmetric_simplex, lattice_iter
from FIAT.quadrature import FacetQuadratureRule
from FIAT.polynomial_set import ONPolynomialSet, make_bubbles
from FIAT.check_format_variant import check_format_variant, parse_quadrature_scheme
from FIAT.P0 import P0


def make_projected_bubble_moments(ref_el, degree, interpolant_deg=None, quad_scheme=None):
    """Tabulate moments against projected bubbles of the given degree."""
    if interpolant_deg is None:
        interpolant_deg = degree
    sd = ref_el.get_spatial_dimension()
    num_cells = len(ref_el.get_topology()[sd])
    deg = numpy.array([sum(alpha) for alpha in lattice_iter(0, degree - sd, sd)])
    lex2hier = numpy.argsort(deg, kind="stable")

    Q = parse_quadrature_scheme(ref_el, degree + interpolant_deg, quad_scheme)
    P = ONPolynomialSet(ref_el, degree, scale="orthonormal")
    phis = P.tabulate(Q.get_points())[(0,) * sd]
    duals = numpy.multiply(phis, Q.get_weights())

    B = make_bubbles(ref_el, degree)
    bubbles = B.tabulate(Q.get_points())[(0,) * sd]
    bubbles = bubbles.reshape(num_cells, len(B)//num_cells, -1)
    bubbles = bubbles[:, lex2hier, :]
    bubbles = bubbles.reshape(len(B), -1)
    coeffs = numpy.dot(duals, bubbles.T)
    coeffs = coeffs.reshape(num_cells, len(duals)//num_cells, num_cells, -1)

    for k in range(degree-sd):
        dimPk0 = polynomial_dimension(ref_el, k) // num_cells
        dimPk1 = polynomial_dimension(ref_el, k-1) // num_cells
        dimPkd = polynomial_dimension(ref_el, k+sd) // num_cells
        coeffs[:, :dimPkd, :, dimPk1:dimPk0] = 0

    coeffs = coeffs.reshape(len(duals), -1)
    coeffs /= numpy.linalg.norm(coeffs, axis=0)
    return Q, numpy.dot(coeffs.T, phis)


class LegendreDual(dual_set.DualSet):
    """The dual basis for Legendre elements."""
    def __init__(self, ref_el, degree, codim=0, interpolant_deg=None, quad_scheme=None):
        if interpolant_deg is None:
            interpolant_deg = degree
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        dim = sd - codim
        ref_facet = ref_el.construct_subelement(dim)
        poly_set = ONPolynomialSet(ref_facet, degree, scale="L2 piola")
        Q_ref = parse_quadrature_scheme(ref_facet, degree + interpolant_deg, quad_scheme)
        phis = poly_set.tabulate(Q_ref.get_points())[(0,) * dim]
        for entity in sorted(top[dim]):
            cur = len(nodes)
            Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref, avg=True)
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

    def __init__(self, ref_el, degree, variant=None, quad_scheme=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = LegendreDual(ref_el, degree, interpolant_deg=interpolant_deg, quad_scheme=quad_scheme)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements."""
    def __init__(self, ref_el, degree, interpolant_deg=None, quad_scheme=None):
        if interpolant_deg is None:
            interpolant_deg = degree
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        for dim in sorted(top):
            if degree <= dim:
                continue
            test_deg = degree - dim - 1 if dim > 0 else 0
            ref_facet = symmetric_simplex(dim)
            Q_ref = parse_quadrature_scheme(ref_facet, test_deg + interpolant_deg, quad_scheme)
            poly_set = ONPolynomialSet(ref_facet, test_deg, scale="L2 piola", variant="dual")
            phis = poly_set.tabulate(Q_ref.get_points())[(0,) * dim]
            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref, avg=True)
                nodes.extend(functional.IntegralMoment(ref_el, Q_facet, phi) for phi in phis)
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class IntegratedLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""
    def __init__(self, ref_el, degree, variant=None, quad_scheme=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)
        if degree < 1:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= 1")
        poly_set = ONPolynomialSet(ref_el, degree, scale=1, variant="bubble")
        dual = IntegratedLegendreDual(ref_el, degree, interpolant_deg=interpolant_deg, quad_scheme=quad_scheme)
        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree, recombine_dual=True)
