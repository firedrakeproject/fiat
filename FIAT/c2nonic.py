# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import (PointEvaluation, PointDerivative, PointNormalDerivative,
                             IntegralMoment,
                             IntegralMomentOfNormalDerivative)
from FIAT.jacobi import eval_jacobi_batch, eval_jacobi_deriv_batch
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TRIANGLE, ufc_simplex


class C2NonicDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, interpolant_deg):
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Argyris only defined on triangles")

        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # get fourth jet at each vertex
        verts = ref_el.get_vertices()

        # FIXME: go up to fourth order
        alphas = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, verts[v]))
            nodes.extend(PointDerivative(ref_el, verts[v], alpha) for alpha in alphas)
            entity_ids[0][v] = list(range(cur, len(nodes)))

        # And the other dofs!
            
        super().__init__(nodes, ref_el, entity_ids)


class C2Nonic(finite_element.CiarletElement):
    """
    The C2 Nonic element

    :arg ref_el: The reference element.
    :arg degree: The degree.
    :arg variant: optional variant specifying the types of nodes.

    """
    def __init__(self, ref_el, degree=9):
        assert degree == 9
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = C2NonicDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)
