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


class MorleyDualSet(dual_set.DualSet):
    """The dual basis for Morley elements.  This class works for
    simplices of any dimension.  Nodes are average on codim=2 entities
    and average of normal derivative on codim=1 entities."""

    def __init__(self, ref_el, degree):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []
        for dim in (sd-2, sd-1):
            facet = ref_el.construct_subelement(dim)
            Q_ref = create_quadrature(facet, degree-1)
            scale = numpy.ones(Q_ref.get_weights().shape)

            for entity in sorted(top[dim]):
                cur = len(nodes)
                if dim == sd-1:
                    # codim=1 dof -- average of normal derivative at each facet
                    ell = functional.IntegralMomentOfNormalDerivative(ref_el, entity, Q_ref, scale)
                elif dim == 0:
                    # codim=2 vertex dof -- point evaluation
                    pt, = ref_el.make_points(dim, entity, degree)
                    ell = functional.PointEvaluation(ref_el, pt)
                else:
                    # codim=2 edge dof -- integral average
                    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                    ell = functional.IntegralMoment(ref_el, Q, scale / Q.jacobian_determinant())

                nodes.append(ell)
                entity_ids[dim][entity].extend(list(range(cur, len(nodes))))

        super().__init__(nodes, ref_el, entity_ids)


class Morley(finite_element.CiarletElement):
    """The Morley finite element."""

    def __init__(self, ref_el, degree=2):
        if ref_el.get_shape() not in {TRIANGLE, TETRAHEDRON}:
            raise ValueError("Morley only defined on simplices")
        if degree != 2:
            raise ValueError("{type(self).__name__} only defined for degree=2")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = MorleyDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)
