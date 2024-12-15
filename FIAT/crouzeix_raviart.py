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

from FIAT import finite_element, polynomial_set, dual_set, functional


class CrouzeixRaviartDualSet(dual_set.DualSet):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)."""

    def __init__(self, cell, degree, variant=None):

        # Get topology dictionary
        sd = cell.get_spatial_dimension()
        top = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []
        # Construct nodes and entity_ids
        for i in top[sd - 1]:
            cur = len(nodes)
            pts = cell.make_points(sd-1, i, degree+sd-1, variant=variant)
            # Degree of freedom number i is evaluation at midpoint
            nodes.extend(functional.PointEvaluation(cell, x) for x in pts)
            entity_ids[sd - 1][i].extend(range(cur, len(nodes)))

        cur = len(nodes)
        pts = cell.make_points(sd, 0, degree, variant=variant)
        nodes.extend(functional.PointEvaluation(cell, x) for x in pts)
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        # Initialize super-class
        super().__init__(nodes, cell, entity_ids)


class CrouzeixRaviart(finite_element.CiarletElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_1 or P_3
    Dual basis:        Evaluation at facet midpoints
    """

    def __init__(self, cell, degree, variant=None):
        # Crouzeix Raviart is only defined for polynomial degree == 1 or 3
        if degree not in [1, 3]:
            raise Exception("Crouzeix-Raviart only defined for degree 1 or 3")
        # Construct polynomial spaces, dual basis and initialize FiniteElement
        space = polynomial_set.ONPolynomialSet(cell, degree)
        dual = CrouzeixRaviartDualSet(cell, degree, variant=variant)
        super().__init__(space, dual, degree)
