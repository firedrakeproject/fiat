# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified 2017 by RCK
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional


class CubicHermiteDualSet(dual_set.DualSet):
    """The dual basis for Hermite elements.  This class works for
    simplices of any dimension.  Nodes are first order jet at
    vertices and point evaluation at barycenters of 2D entities."""

    def __init__(self, ref_el, degree, variant=None):
        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        sd = ref_el.get_topological_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # get first order jet at each vertex
        for v in sorted(top[0]):
            pt, = ref_el.make_points(0, v, degree, variant=variant)
            cur = len(nodes)
            nodes.append(functional.PointEvaluation(ref_el, pt))
            if sd == 1:
                # use normal derivative to support manifolds in 1D
                nodes.append(functional.PointNormalDerivative(ref_el, v, pt))
            else:
                nodes.extend(functional.PointDerivative(ref_el, pt, alpha)
                             for alpha in polynomial_set.mis(sd, 1))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        if sd == 1:
            # edge dofs: point evaluations to support higher order in 1D
            for e in sorted(top[1]):
                cur = len(nodes)
                pts = ref_el.make_points(1, e, degree-2, variant=variant)
                nodes.extend(functional.PointEvaluation(ref_el, pt) for pt in pts)
                entity_ids[1][e].extend(range(cur, len(nodes)))
        else:
            assert degree == 3
            # no edge dof
            # face dof: point evaluation at barycenter
            for f in sorted(top[2]):
                cur = len(nodes)
                pt, = ref_el.make_points(2, f, degree, variant=variant)
                nodes.append(functional.PointEvaluation(ref_el, pt))
                entity_ids[2][f].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class CubicHermite(finite_element.CiarletElement):
    """The cubic Hermite finite element.  It is what it is."""

    def __init__(self, ref_el, degree=3, variant=None):
        if variant is None:
            variant = "gll"
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = CubicHermiteDualSet(ref_el, degree, variant=variant)

        super().__init__(poly_set, dual, degree)
