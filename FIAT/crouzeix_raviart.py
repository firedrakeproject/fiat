# Copyright (C) 2010 Marie E. Rognes
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Marie E. Rognes <meg@simula.no> based on original
# implementation by Robert C. Kirby.
#
# Last changed: 2010-01-28

import numpy
from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.check_format_variant import check_format_variant
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


class CrouzeixRaviartDualSet(dual_set.DualSet):

    def __init__(self, ref_el, degree, variant, interpolant_deg):
        # Get topology dictionary
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()

        if degree > 1 and sd != 2:
            raise NotImplementedError("High-order Crouzeix-Raviart is only implemented on triangles.")

        # Initialize empty nodes and entity_ids
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # Construct nodes and entity_ids
        if variant == "integral":
            for dim in sorted(top):
                if dim == 0 and dim != sd-1:
                    # Skip vertex dofs
                    continue
                facet = ref_el.construct_subelement(dim)
                if dim == 0:
                    Q_facet = create_quadrature(facet, degree + interpolant_deg-1)
                    Phis = numpy.ones((1, len(Q_facet.pts)))
                else:
                    k = degree - 1 if dim == sd-1 else degree - (1+dim)
                    if k < 0:
                        continue
                    Q_facet = create_quadrature(facet, k + interpolant_deg)
                    poly_set = polynomial_set.ONPolynomialSet(facet, k)
                    Phis = poly_set.tabulate(Q_facet.get_points())[(0,) * dim]

                for i in sorted(top[dim]):
                    cur = len(nodes)
                    Q = FacetQuadratureRule(ref_el, dim, i, Q_facet)
                    scale = 1 / Q.jacobian_determinant()
                    phis = scale * Phis
                    nodes.extend(functional.IntegralMoment(ref_el, Q, phi) for phi in phis)
                    entity_ids[dim][i].extend(range(cur, len(nodes)))
        else:
            for dim in sorted(top):
                if dim == 0 and dim != sd-1:
                    # Skip vertex dofs
                    continue
                for i in sorted(top[dim]):
                    cur = len(nodes)
                    if dim == sd-1 and dim != 0:
                        pts = ref_el.make_points(dim, i, degree-1, variant="gl", interior=0)
                    else:
                        pts = ref_el.make_points(dim, i, degree, variant="gll")
                    nodes.extend(functional.PointEvaluation(ref_el, x) for x in pts)
                    entity_ids[dim][i].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class CrouzeixRaviart(finite_element.CiarletElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_k
    Dual basis:        Evaluation at points or integral moments
    """

    def __init__(self, ref_el, degree, variant=None):
        if degree % 2 != 1:
            raise ValueError("Crouzeix-Raviart only defined for odd degree")

        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)

        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = CrouzeixRaviartDualSet(ref_el, degree, variant, interpolant_deg)
        super().__init__(poly_set, dual, degree)
