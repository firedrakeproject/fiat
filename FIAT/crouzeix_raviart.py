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

import numpy
from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.check_format_variant import check_format_variant
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


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

    def __init__(self, cell, degree, variant, interpolant_deg):

        # Get topology dictionary
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = _initialize_entity_ids(topology)
        nodes = []

        # Construct nodes and entity_ids
        if variant == "point":
            for i in sorted(topology[d - 1]):
                # Construct midpoint
                pt, = cell.make_points(d - 1, i, d)
                # Degree of freedom number i is evaluation at midpoint
                nodes.append(functional.PointEvaluation(cell, pt))
                entity_ids[d - 1][i].append(i)
        else:
            facet = cell.construct_subelement(d-1)
            Q_facet = create_quadrature(facet, degree-1 + interpolant_deg)
            for i in sorted(topology[d - 1]):
                # Map quadrature
                Q = FacetQuadratureRule(cell, d-1, i, Q_facet)
                f = 1 / Q.jacobian_determinant()
                f_at_qpts = numpy.full(Q.get_weights().shape, f)
                # Degree of freedom number i is integral moment on facet
                nodes.append(functional.IntegralMoment(cell, Q, f_at_qpts))
                entity_ids[d - 1][i].append(i)

        # Initialize super-class
        super().__init__(nodes, cell, entity_ids)


class CrouzeixRaviart(finite_element.CiarletElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_1
    Dual basis:        Evaluation at facet midpoints
    """

    def __init__(self, cell, degree, variant=None):

        variant, interpolant_deg = check_format_variant(variant, degree)

        # Crouzeix Raviart is only defined for polynomial degree == 1
        if not (degree == 1):
            raise Exception("Crouzeix-Raviart only defined for degree 1")

        # Construct polynomial spaces, dual basis and initialize
        # FiniteElement
        space = polynomial_set.ONPolynomialSet(cell, 1)
        dual = CrouzeixRaviartDualSet(cell, 1, variant, interpolant_deg)
        super().__init__(space, dual, 1)
