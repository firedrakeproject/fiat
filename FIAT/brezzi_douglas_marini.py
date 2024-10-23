# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (finite_element, functional, dual_set,
                  polynomial_set, nedelec)
from FIAT.check_format_variant import check_format_variant
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


class BDMDualSet(dual_set.DualSet):
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
            facet = ref_el.get_facet_element()
            # Facet nodes are \int_F v\cdot n p ds where p \in P_{q}
            # degree is q
            Q_ref = create_quadrature(facet, interpolant_deg + degree)
            Pq = polynomial_set.ONPolynomialSet(facet, degree)
            Pq_at_qpts = Pq.tabulate(Q_ref.get_points())[(0,)*(sd - 1)]
            for f in top[sd - 1]:
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, sd - 1, f, Q_ref)
                Jdet = Q.jacobian_determinant()
                n = ref_el.compute_scaled_normal(f) / Jdet
                phis = n[None, :, None] * Pq_at_qpts[:, None, :]
                nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q, phi)
                             for phi in phis)
                entity_ids[sd - 1][f] = list(range(cur, len(nodes)))

        elif variant == "point":
            # Define each functional for the dual set
            # codimension 1 facets
            for f in top[sd - 1]:
                cur = len(nodes)
                pts_cur = ref_el.make_points(sd - 1, f, sd + degree)
                nodes.extend(functional.PointScaledNormalEvaluation(ref_el, f, pt)
                             for pt in pts_cur)
                entity_ids[sd - 1][f] = list(range(cur, len(nodes)))

        # internal nodes
        if degree > 1:
            if interpolant_deg is None:
                interpolant_deg = degree
            cur = len(nodes)
            Q = create_quadrature(ref_el, interpolant_deg + degree - 1)
            Nedel = nedelec.Nedelec(ref_el, degree - 1, variant)
            Nedfs = Nedel.get_nodal_basis()
            Ned_at_qpts = Nedfs.tabulate(Q.get_points())[(0,) * sd]
            nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q, phi)
                         for phi in Ned_at_qpts)
            entity_ids[sd][0] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class BrezziDouglasMarini(finite_element.CiarletElement):
    """
    The BDM element

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

        variant, interpolant_deg = check_format_variant(variant, degree)

        if degree < 1:
            raise Exception("BDM_k elements only valid for k >= 1")

        sd = ref_el.get_spatial_dimension()
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, (sd, ))
        dual = BDMDualSet(ref_el, degree, variant, interpolant_deg)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
