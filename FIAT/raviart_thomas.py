# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (expansions, polynomial_set, dual_set,
                  finite_element, functional, macro)
import numpy
from itertools import chain
from FIAT.check_format_variant import check_format_variant
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


def RTSpace(ref_el, degree):
    """Constructs a basis for the Raviart-Thomas space
    (P_{degree-1})^d + P_{degree-1} x"""
    sd = ref_el.get_spatial_dimension()

    k = degree - 1
    vec_Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, k + 1)
    dimPk = expansions.polynomial_dimension(ref_el, k)
    dimPkm1 = expansions.polynomial_dimension(ref_el, k - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk_from_Pkp1 = vec_Pkp1.take(vec_Pk_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1)
    PkH = Pkp1.take(list(range(dimPkm1, dimPk)))

    Q = create_quadrature(ref_el, 2 * (k + 1))
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    # have to work on this through "tabulate" interface
    # first, tabulate PkH at quadrature points
    PkH_at_Qpts = PkH.tabulate(Qpts)[(0,) * sd]

    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[(0,) * sd]

    x = Qpts.T
    PkHx_at_Qpts = PkH_at_Qpts[:, None, :] * x[None, :, :]
    PkHx_coeffs = numpy.dot(numpy.multiply(PkHx_at_Qpts, Qwts), Pkp1_at_Qpts.T)
    PkHx = polynomial_set.PolynomialSet(ref_el,
                                        k,
                                        k + 1,
                                        vec_Pkp1.get_expansion_set(),
                                        PkHx_coeffs)
    return polynomial_set.polynomial_set_union_normalized(vec_Pk_from_Pkp1, PkHx)


class RTDualSet(dual_set.DualSet):
    """Dual basis for Raviart-Thomas elements consisting of point
    evaluation of normals on facets of codimension 1 and internal
    moments against polynomials"""

    def __init__(self, ref_el, degree, variant, interpolant_deg):
        nodes = []
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()

        entity_ids = {}
        # set to empty
        for dim in top:
            entity_ids[dim] = {}
            for entity in top[dim]:
                entity_ids[dim][entity] = []

        if variant == "integral":
            facet = ref_el.construct_subelement(sd-1)
            # Facet nodes are \int_F v\cdot n p ds where p \in P_q
            q = degree - 1
            Q_ref = create_quadrature(facet, interpolant_deg + q)
            Pq = polynomial_set.ONPolynomialSet(facet, q if sd > 1 else 0)
            Pq_at_qpts = Pq.tabulate(Q_ref.get_points())[(0,)*(sd - 1)]
            for f in top[sd - 1]:
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, sd-1, f, Q_ref)
                Jdet = Q.jacobian_determinant()
                n = ref_el.compute_scaled_normal(f) / Jdet
                phis = n[None, :, None] * Pq_at_qpts[:, None, :]
                nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q, phi)
                             for phi in phis)
                entity_ids[sd - 1][f] = list(range(cur, len(nodes)))

            # internal nodes. These are \int_T v \cdot p dx where p \in P_{q-1}^d
            if q > 0:
                cell = ref_el.construct_subelement(sd)
                Q_ref = create_quadrature(cell, interpolant_deg + q - 1)
                Pqm1 = polynomial_set.ONPolynomialSet(cell, q - 1)
                Pqm1_at_qpts = Pqm1.tabulate(Q_ref.get_points())[(0,) * sd]

                for entity in top[sd]:
                    Q = FacetQuadratureRule(ref_el, sd, entity, Q_ref)
                    cur = len(nodes)
                    nodes.extend(functional.IntegralMoment(ref_el, Q, phi, (d,), (sd,))
                                 for d in range(sd)
                                 for phi in Pqm1_at_qpts)
                    entity_ids[sd][entity] = list(range(cur, len(nodes)))

        elif variant == "point":
            # codimension 1 facets
            for i in top[sd - 1]:
                cur = len(nodes)
                pts_cur = ref_el.make_points(sd - 1, i, sd + degree - 1)
                nodes.extend(functional.PointScaledNormalEvaluation(ref_el, i, pt)
                             for pt in pts_cur)
                entity_ids[sd - 1][i] = list(range(cur, len(nodes)))

            # internal nodes.  Let's just use points at a lattice
            if degree > 1:
                cur = len(nodes)
                pts = ref_el.make_points(sd, 0, sd + degree - 1)
                nodes.extend(functional.ComponentPointEvaluation(ref_el, d, (sd,), pt)
                             for d in range(sd)
                             for pt in pts)
                entity_ids[sd][0] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class RaviartThomas(finite_element.CiarletElement):
    """
    The Raviart Thomas element

    :arg ref_el: The reference element.
    :arg degree: The degree.
    :arg variant: optional variant specifying the types of nodes.

    variant can be chosen from ["point", "integral", "integral(q)"]
    "point" -> dofs are evaluated by point evaluation. Note that this variant
    has suboptimal convergence order in the H(div)-norm
    "integral" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence.
    "integral(q)" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence plus q. You might want to choose a high
    quadrature degree to make sure that expressions will be interpolated
    exactly. This is important when you want to have (nearly) div-preserving
    interpolation.
    """

    def __init__(self, ref_el, degree, variant=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)

        if ref_el.is_macrocell():
            base_element = RaviartThomas(ref_el.get_parent(), degree)
            poly_set = macro.MacroPolynomialSet(ref_el, base_element)
        else:
            poly_set = RTSpace(ref_el, degree)
        dual = RTDualSet(ref_el, degree, variant, interpolant_deg)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
