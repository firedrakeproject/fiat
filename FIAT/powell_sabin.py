# Copyright (C) 2024 Robert C. Kirby
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Robert C. Kirby (robert.c.kirby@gmail.com), 2024

from FIAT import dual_set, finite_element, macro, polynomial_set
from FIAT.functional import (IntegralMomentOfNormalDerivative, PointDerivative,
                             PointEvaluation)
from FIAT.jacobi import eval_jacobi_batch
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TRIANGLE, ufc_simplex


class QuadraticPowellSabin6DualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree=2):
        if degree != 2:
            raise ValueError("PS6 only defined for degree = 2")
        ref_el = ref_complex.get_parent()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("PS6 only defined on triangles")
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # get first order jet at each vertex
        alphas = polynomial_set.mis(sd, 1)
        nodes = []

        for v in sorted(top[0]):
            pt = verts[v]
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, pt))
            nodes.extend(PointDerivative(ref_el, pt, alpha) for alpha in alphas)
            entity_ids[0][v].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class QuadraticPowellSabin6(finite_element.CiarletElement):
    """The PS6 macroelement is a C^1 quadratic macroelement defined
    on the 6-way Powell-Sabin split of a triangle.
    """
    def __init__(self, ref_el, degree=2):
        if degree != 2:
            raise ValueError("PS6 only defined for degree = 2")
        ref_complex = macro.PowellSabinSplit(ref_el)
        dual = QuadraticPowellSabin6DualSet(ref_complex, degree)
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1)

        super().__init__(poly_set, dual, degree)


class QuadraticPowellSabin12DualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree=2):
        if degree != 2:
            raise ValueError("PS12 only defined for degree = 2")
        ref_el = ref_complex.get_parent()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("PS12 only defined on triangles")
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # get first order jet at each vertex
        alphas = polynomial_set.mis(sd, 1)
        nodes = []

        for v in sorted(top[0]):
            pt = verts[v]
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, pt))
            nodes.extend(PointDerivative(ref_el, pt, alpha) for alpha in alphas)
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # integral moment of normal derivatives
        rline = macro.AlfeldSplit(ufc_simplex(1))
        Q = create_quadrature(rline, degree-1)
        qpts = Q.get_points()

        x = 2.0*qpts - 1
        phis = eval_jacobi_batch(1, 1, 0, x)
        for e in sorted(top[1]):
            cur = len(nodes)
            nodes.extend(IntegralMomentOfNormalDerivative(ref_el, e, Q, phi) for phi in phis)
            entity_ids[1][e].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class QuadraticPowellSabin12(finite_element.CiarletElement):
    """The PS12 macroelement is a C^1 quadratic macroelement defined
    on the 12-way Powell-Sabin split of a triangle.
    """
    def __init__(self, ref_el, degree=2):
        if degree != 2:
            raise ValueError("PS12 only defined for degree = 2")
        ref_complex = macro.PowellSabin12Split(ref_el)
        dual = QuadraticPowellSabin12DualSet(ref_complex, degree)
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1)

        super().__init__(poly_set, dual, degree)
