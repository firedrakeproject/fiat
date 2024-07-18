# Copyright (C) 2024 Robert C. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Robert C. Kirby (robert.c.kirby@gmail.com), 2024

from FIAT import dual_set, finite_element, macro, polynomial_set
from FIAT.functional import PointDerivative, PointEvaluation
from FIAT.reference_element import TRIANGLE


class QuadraticPowellSabin6DualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree=2):
        if degree != 2:
            raise ValueError("PS6 only defined for degree = 2")
        ref_el = ref_complex.get_parent()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("HCT only defined on triangles")
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

        super(QuadraticPowellSabin6DualSet, self).__init__(
            nodes, ref_el, entity_ids)


class QuadraticPowellSabin6(finite_element.CiarletElement):
    """The PS6 macroelement.
    """
    def __init__(self, ref_el, degree=2):
        if degree != 2:
            raise ValueError("PS6 only defined for degree = 2")
        ref_complex = macro.PowellSabinSplit(ref_el)
        dual = QuadraticPowellSabin6DualSet(ref_complex, degree)
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1)

        super(QuadraticPowellSabin6, self).__init__(poly_set, dual, degree)
