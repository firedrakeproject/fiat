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

    def __init__(self, ref_el):
        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_topological_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # get first order jet at each vertex
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.append(functional.PointEvaluation(ref_el, verts[v]))
            if sd == 1:
                # in 1D use normal derivative to support manifolds
                nodes.append(functional.PointNormalDerivative(ref_el, v, verts[v]))
            else:
                nodes.extend(functional.PointDerivative(ref_el, verts[v], alpha)
                             for alpha in polynomial_set.mis(sd, 1))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # no edge dof
        # now only add dofs at the barycenter of the 2D entities
        if sd > 1:
            # face dof
            # point evaluation at barycenter
            for f in sorted(top[2]):
                cur = len(nodes)
                pt = ref_el.make_points(2, f, 3)[0]
                nodes.append(functional.PointEvaluation(ref_el, pt))
                entity_ids[2][f].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class CubicHermite(finite_element.CiarletElement):
    """The cubic Hermite finite element.  It is what it is."""

    def __init__(self, ref_el, deg=3):
        assert deg == 3
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 3)
        dual = CubicHermiteDualSet(ref_el)

        super().__init__(poly_set, dual, 3)
