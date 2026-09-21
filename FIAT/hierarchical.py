# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2022

import numpy

from FIAT import expansions, finite_element, dual_set, functional
from FIAT.expansions import ExpansionSet
from FIAT.reference_element import symmetric_simplex
from FIAT.quadrature import CompositeQuadratureRule, FacetQuadratureRule
from FIAT.polynomial_set import ONPolynomialSet, make_bubbles
from FIAT.check_format_variant import check_format_variant, parse_quadrature_scheme
from FIAT.P0 import P0


def make_dual_bubbles(ref_el, degree, codim=0, interpolant_deg=None, quad_scheme=None, scale="orthonormal"):
    """Tabulate the L2-duals of the hierarchical C0 basis."""
    dim = ref_el.get_spatial_dimension()
    if dim == 0:
        quad_scheme = None
        degree = 0
    if interpolant_deg is None:
        interpolant_deg = degree
    Q = parse_quadrature_scheme(ref_el, degree + interpolant_deg, quad_scheme)
    B = make_bubbles(ref_el, degree, codim=codim, scale=scale)
    P_at_qpts = B.expansion_set.tabulate(degree, Q.get_points())
    M = numpy.dot(numpy.multiply(P_at_qpts, Q.get_weights()), P_at_qpts.T)
    phis = numpy.linalg.solve(M, P_at_qpts)
    phis = numpy.dot(B.get_coeffs(), phis)
    return Q, phis


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


def tabulate_bubble_duals(ref_el, degree, interpolant_deg=None, quad_scheme=None):
    """Tabulate the L2-duals of the interior bubbles on the reference simplex.

    The interior bubbles of the hierarchical basis are the cell bubble times
    the members of the ``"dual"`` expansion set, which are orthogonal with
    respect to the bubble weight.  The duals are therefore members of that
    expansion set, read off at the shifted multi-index, and no projection
    problem has to be solved.  They are dual to the interior bubbles alone,
    not to the whole C0 basis, which is what the closure correction in
    :class:`IntegratedLegendreDual` repairs.

    :arg ref_el: The reference simplex.
    :arg degree: The polynomial degree.
    :kwarg interpolant_deg: The degree of the interpolated functions.
    :kwarg quad_scheme: The quadrature scheme, see ``parse_quadrature_scheme``.

    :returns: A tuple with the quadrature rule and the duals tabulated on its points.
    """
    dim = ref_el.get_spatial_dimension()
    if interpolant_deg is None:
        interpolant_deg = degree
    if dim == 0:
        Q = parse_quadrature_scheme(ref_el, 0, None)
        return Q, numpy.ones((1, len(Q.get_weights())))
    Q = parse_quadrature_scheme(ref_el, degree + interpolant_deg, quad_scheme)
    duals = ExpansionSet(ref_el, variant="dual")
    phis = duals.tabulate(degree - dim - 1, Q.get_points())
    return Q, phis[expansions.dual_bubble_indices(dim, degree)]


class IntegratedLegendreDual(dual_set.DualSet):
    """The dual basis for integrated Legendre elements.

    The functional owned by an entity is a moment against the L2-dual of an
    interior bubble of that entity, corrected by the functionals of the entities
    in its closure.  Those corrections are needed because the dual of an interior
    bubble annihilates the other interior bubbles but not the basis functions
    that carry a trace on the entity.  Since the correction only involves the
    closure, each functional integrates over a :class:`CompositeQuadratureRule`
    that all the functionals of the same entity share.
    """
    def __init__(self, ref_el, degree, interpolant_deg=None, quad_scheme=None):
        top = ref_el.get_topology()
        entity_ids = expansions.C0_entity_ids(ref_el, degree)
        expansion_set = ExpansionSet(ref_el, variant="bubble")
        nodes = [None for _ in range(expansion_set.get_num_members(degree))]

        # Quadrature rule and duals of each entity, keyed by (dim, entity)
        rules = {}
        duals = {}
        for dim in sorted(top):
            if degree <= dim:
                continue
            Q_ref, phis = tabulate_bubble_duals(symmetric_simplex(dim), degree,
                                                interpolant_deg=interpolant_deg,
                                                quad_scheme=quad_scheme)
            for entity in sorted(top[dim]):
                rules[(dim, entity)] = FacetQuadratureRule(ref_el, dim, entity, Q_ref, avg=True)
                duals[(dim, entity)] = phis

        # Tabulate the hierarchical basis on every entity at once
        keys = sorted(rules)
        offsets = numpy.cumsum([0] + [len(rules[key].get_weights()) for key in keys])
        tabulation = expansion_set.tabulate(degree, numpy.concatenate(
            [rules[key].get_points() for key in keys]))
        slices = {key: slice(offsets[i], offsets[i+1]) for i, key in enumerate(keys)}

        weights = {}
        for dim in sorted(top):
            if degree <= dim:
                continue
            for entity in sorted(top[dim]):
                dofs = entity_ids[dim][entity]
                Q = rules[(dim, entity)]
                phis = duals[(dim, entity)]

                # Moments of the hierarchical basis against the duals
                moments = numpy.dot(numpy.multiply(phis, Q.get_weights()),
                                    tabulation[:, slices[(dim, entity)]].T)
                # The duals are biorthogonal to the interior bubbles of this
                # entity, so this block of moments is diagonal
                scale = 1.0 / numpy.diagonal(moments[:, dofs])
                moments *= scale[:, None]

                # Subtract the functionals of the entities in the closure
                closure = sorted(key for key in ref_el.sub_entities[dim][entity] if key in rules)
                cur = {key: numpy.zeros((len(dofs), len(rules[key].get_weights())))
                       for key in closure}
                cur[(dim, entity)] += numpy.multiply(phis, scale[:, None])
                for d, sub_entity in closure:
                    if d == dim:
                        continue
                    sub_dofs = entity_ids[d][sub_entity]
                    for key in weights[sub_dofs[0]]:
                        phis_sub = numpy.array([weights[dof][key] for dof in sub_dofs])
                        cur[key] -= numpy.dot(moments[:, sub_dofs], phis_sub)

                Q_closure = (Q if len(closure) == 1 else
                             CompositeQuadratureRule(ref_el, [rules[key] for key in closure]))
                for k, dof in enumerate(dofs):
                    weights[dof] = {key: cur[key][k] for key in closure}
                    phi = numpy.concatenate([cur[key][k] for key in closure])
                    nodes[dof] = functional.IntegralMoment(ref_el, Q_closure, phi)

        super().__init__(nodes, ref_el, entity_ids)


class IntegratedLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""
    def __init__(self, ref_el, degree, variant=None, quad_scheme=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)
        if degree < 1:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= 1")
        poly_set = ONPolynomialSet(ref_el, degree, variant="bubble")
        dual = IntegratedLegendreDual(ref_el, degree, interpolant_deg=interpolant_deg, quad_scheme=quad_scheme)
        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree)
