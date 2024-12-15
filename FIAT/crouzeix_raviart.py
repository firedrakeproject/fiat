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
import math
import numpy as np

def _initialize_entity_ids(topology):
    entity_ids = {}
    for (i, entity) in list(topology.items()):
        entity_ids[i] = {}
        for j in entity:
            entity_ids[i][j] = []
    return entity_ids

gauss_points =  {
    1: {
            0: [(1/2, 1/2)],
            1: [(0  , 1/2)],
            2: [(1/2  , 0)]
        },
    3: {
        0: [(1-(1/2)+(1/2)*math.sqrt((3/5)),(1/2)-(1/2)*math.sqrt((3/5))),
            ((1/2), (1/2)),
            (1-(1/2)-(1/2)*math.sqrt((3/5)),(1/2)+(1/2)*math.sqrt((3/5)))],
        1: [(0,(1/2)-(1/2)*math.sqrt((3/5))),
            (0, (1/2)),
            (0,(1/2)+(1/2)*math.sqrt((3/5)))],
        2: [((1/2)-(1/2)*math.sqrt((3/5)),0),
            ((1/2),0),
            ((1/2)+(1/2)*math.sqrt((3/5)),0)]
    }
}

class CrouzeixRaviartDualSet(dual_set.DualSet):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)."""

    def __init__(self, cell, degree):

        # Get topology dictionary
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = _initialize_entity_ids(topology)
        nodes = [None for _ in range(int(0.5*(degree+1)*(degree+2)))]

        # Construct nodes and entity_ids
        dof_counter = 0
        for i in sorted(topology[d - 1]):
            # Construct midpoint
            #import pdb; pdb.set_trace()
            #x = cell.make_points(d - 1, i, d)[0]
            #print(x)
            for pt_idx in range(len(gauss_points[degree][i])):
                print(gauss_points[degree][i][pt_idx])
                x = gauss_points[degree][i][pt_idx]
                # Degree of freedom number i is evaluation at midpoint
                nodes[dof_counter] = functional.PointEvaluation(cell, x)
                entity_ids[d - 1][i] += [dof_counter]
                dof_counter += 1
        if degree == 3:
            nodes[-1] = functional.PointEvaluation(cell, (1/3, 1/3))
            entity_ids[d][0] += [dof_counter]
        # Initialize super-class
        print(nodes, entity_ids)
        super(CrouzeixRaviartDualSet, self).__init__(nodes, cell, entity_ids)


class CrouzeixRaviart(finite_element.CiarletElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_1
    Dual basis:        Evaluation at facet midpoints
    """

    def __init__(self, cell, degree):

        # Crouzeix Raviart is only defined for polynomial degree == 1
        if not (degree in [1,3]):
            raise Exception("Crouzeix-Raviart only defined for degree 1")

        # Construct polynomial spaces, dual basis and initialize
        # FiniteElement
        space = polynomial_set.ONPolynomialSet(cell, degree)
        dual = CrouzeixRaviartDualSet(cell, degree)
        super(CrouzeixRaviart, self).__init__(space, dual, 1)
