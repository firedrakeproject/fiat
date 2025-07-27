# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import TETRAHEDRON, TRIANGLE
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
import numpy
import math


class MorleyDualSet(dual_set.DualSet):
    """The dual basis for Morley elements.  This class works for
    simplices of any dimension.  Nodes are average on codim=2 entities
    and average of normal derivative on codim=1 entities."""

    def __init__(self, ref_el, degree):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        def duals(ref_el, dim, degree):
            facet = ref_el.construct_subelement(dim)
            Q_ref = create_quadrature(facet, degree)
            scale = numpy.ones(Q_ref.get_weights().shape)
            return Q_ref, scale

        # codim=2 dof -- integral average
        dim = sd - 2
        Q_ref, scale = duals(ref_el, dim, degree)
        for entity in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
            nodes.append(functional.IntegralMoment(ref_el, Q, scale / Q.jacobian_determinant()))
            entity_ids[dim][entity].extend(range(cur, len(nodes)))

        # codim=1 dof -- average of normal derivative at each facet
        dim = sd - 1
        Q_ref, scale = duals(ref_el, dim, degree-1)
        # normalized normals do not have unit norm!
        scale /= math.factorial(sd-1)
        for entity in sorted(top[dim]):
            cur = len(nodes)
            nodes.append(functional.IntegralMomentOfNormalDerivative(ref_el, entity, Q_ref, scale))
            entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Morley(finite_element.CiarletElement):
    """The Morley finite element."""

    def __init__(self, ref_el, degree=2):
        if ref_el.get_shape() not in {TRIANGLE, TETRAHEDRON}:
            raise ValueError("Morley only defined on simplices of dimension >= 2")
        if degree != 2:
            raise ValueError("{type(self).__name__} only defined for degree == 2")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = MorleyDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)
