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
from FIAT.check_format_variant import parse_lagrange_variant


def _initialize_entity_ids(topology):
    entity_ids = {}
    for (i, entity) in list(topology.items()):
        entity_ids[i] = {}
        for j in entity:
            entity_ids[i][j] = []
    return entity_ids


class CrouzeixRaviartDualSet(dual_set.DualSet):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)."""

    def __init__(self, cell, degree):

        # Get topology dictionary
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = _initialize_entity_ids(topology)
        nodes = [None for i in list(topology[d - 1].keys())]

        # Construct nodes and entity_ids
        for i in topology[d - 1]:

            # Construct midpoint
            x = cell.make_points(d - 1, i, d)[0]

            # Degree of freedom number i is evaluation at midpoint
            nodes[i] = functional.PointEvaluation(cell, x)
            entity_ids[d - 1][i] += [i]

        # Initialize super-class
        super(CrouzeixRaviartDualSet, self).__init__(nodes, cell, entity_ids)

class CrouzeixRaviartThreeDualSet(dual_set.DualSet):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)."""

    def __init__(self, cell, degree):

        # Get topology dictionary
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = _initialize_entity_ids(topology)
        nodes = [None for i in range(10)]
        # Construct nodes and entity_ids
        entity_permutations = {0: {0: {0: []}, 1: {0: []}, 2: {0: []}},
                               1: {0: {0: [0, 1, 2], 1: [2, 1, 0]}, 1: {0: [0, 1, 2], 1: [2, 1, 0]}, 2: {0: [0, 1, 2], 1: [2, 1, 0]}},
                               2: {0: {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0]}}
                            }
        dof_counter = 0
        for i in topology[d - 1]:
            x = cell.make_points(d-1,i,4, variant="equi")
            # Degree of freedom number i is evaluation at midpoint
            nodes[dof_counter] = functional.PointEvaluation(cell, x[1])
            nodes[dof_counter+1] = functional.PointEvaluation(cell, x[0])
            nodes[dof_counter+2] = functional.PointEvaluation(cell, x[2])
            entity_ids[d - 1][i] += [dof_counter]
            entity_ids[d - 1][i] += [dof_counter+1]
            entity_ids[d - 1][i] += [dof_counter+2]
            dof_counter += 3
        x = cell.make_points(d,0,3, variant="equi")
        nodes[-1] = functional.PointEvaluation(cell,x[0])
        entity_ids[2][0] += [9]
        # Initialize super-class
        super(CrouzeixRaviartThreeDualSet, self).__init__(nodes, cell, entity_ids, entity_permutations)



class CrouzeixRaviart(finite_element.CiarletElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_1
    Dual basis:        Evaluation at facet midpoints
    """

    def __init__(self, cell, degree,variant=None):

        # Crouzeix Raviart is only defined for polynomial degree == 1
        if not (degree in [1,3]):
            raise Exception("Crouzeix-Raviart only defined for degree 1")
        # Construct polynomial spaces, dual basis and initialize
        # FiniteElement
        if degree == 1:
            space = polynomial_set.ONPolynomialSet(cell, 1)
            dual = CrouzeixRaviartDualSet(cell, 1)
            super(CrouzeixRaviart, self).__init__(space, dual, 1)
        elif degree == 3:
            ref_el  = cell
            splitting, point_variant = parse_lagrange_variant(variant)
            if splitting is not None:
                ref_el = splitting(ref_el)
            poly_variant = "bubble" if ref_el.is_macrocell() else None
            space = polynomial_set.ONPolynomialSet(ref_el, degree, variant=poly_variant)
            formdegree = 0  # 0-form
            dual = CrouzeixRaviartThreeDualSet(cell, degree) 
            super(CrouzeixRaviart, self).__init__(space, dual, 3, formdegree)

