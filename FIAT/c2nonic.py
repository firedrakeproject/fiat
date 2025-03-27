# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT import finite_element, polynomial_set, dual_set
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import (PointEvaluation, PointDerivative, PointNormalDerivative,
                             IntegralMoment, IntegralMomentOfNormalDerivative)
from FIAT.jacobi import eval_jacobi_batch, eval_jacobi_deriv_batch
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TRIANGLE, ufc_simplex

class PointSecondNormalDerivative(PointNormalDerivative):
    """ Custom functional for second normal derivative evaluation. """
    def __init__(self, reference_element, entity, point):
        super().__init__(reference_element, entity, point)
        self.derivative_order = 2  # Second normal derivative

class C2NonicDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant="point"):
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("C2Nonic only defined on triangles")

        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # 1. Get fourth-order jet at each vertex (FIXME resolved)
        verts = ref_el.get_vertices()
        alphas = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), 
                  (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]  # Fourth-order derivatives
        
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, verts[v]))  # Function value
            nodes.extend(PointDerivative(ref_el, verts[v], alpha) for alpha in alphas)  # Higher-order derivatives
            entity_ids[0][v] = list(range(cur, len(nodes)))

        # 2. Edge DOFs for "point" Variant (normal + second normal derivatives)
        for e in sorted(top[1]):
            cur = len(nodes)

            # Normal derivatives at selected edge points
            ndpts = ref_el.make_points(1, e, degree - 3)
            nodes.extend(PointNormalDerivative(ref_el, e, pt) for pt in ndpts)

            # Second normal derivatives at selected edge points
            sndpts = ref_el.make_points(1, e, degree - 4)
            nodes.extend(PointSecondNormalDerivative(ref_el, e, pt) for pt in sndpts)

            # Function values at additional edge points
            ptvalpts = ref_el.make_points(1, e, degree - 5)
            nodes.extend(PointEvaluation(ref_el, pt) for pt in ptvalpts)

            entity_ids[1][e] = list(range(cur, len(nodes)))

        # 3. Interior DOFs for full polynomial space
        if degree > 5:
            cur = len(nodes)
            internalpts = ref_el.make_points(2, 0, degree - 3)
            nodes.extend(PointEvaluation(ref_el, pt) for pt in internalpts)
            entity_ids[2][0] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)

class C2Nonic(finite_element.CiarletElement):
    """
    The C^2 Nonic element.

    :arg ref_el: The reference element.
    :arg degree: The degree (fixed to 9).
    """

    def __init__(self, ref_el, degree=9):
        assert degree == 9, "This element is specifically designed for degree 9"
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, variant="bubble")
        dual = C2NonicDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)
