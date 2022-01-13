# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import itertools
import numpy as np
from FIAT import finite_element, polynomial_set, dual_set, functional, P0
from FIAT.polynomial_set import mis


def make_entity_permutations(dim, npoints):
    if npoints <= 0:
        return {o: [] for o in range(np.math.factorial(dim + 1))}
    # DG nodes are numbered, in order of significance,
    # - by g0: entity dim (vertices first, then edges, then ...)
    # - by g1: entity ids (DoFs on entities of smaller ids first)
    # - lexicographically as in CG
    #
    # Example:
    # dim = 2, degree = 3 (npoints = degree + 1)
    #
    #     facet ids     Lexicographic  DG (degree = 3)
    #    +              DoF numbering  DoF numbering
    #    | \            9              2
    #    |   \  0       7 8            6 4
    #  1 |     \        4 5 6          5 9 3
    #    |       \      0 1 2 3        0 7 8 1
    #    +--------+
    #        2
    # where DG degrees of freedom are numbered to geometrically
    # coincide with CG degrees of freedom.
    #
    # In the below we will show example outputs corresponding to
    # the above example.

    a = np.array(sorted(mis(dim + 1, npoints - 1)), dtype=int)
    # >>> a
    # [[0, 0, 3],
    #  [0, 1, 2],   # (3,0,0)
    #  [0, 2, 1],   #
    #  [0, 3, 0],   # (2,0,1)(2,1,0)
    #  [1, 0, 2],   #
    #  [1, 1, 1],   # (1,0,2)(1,1,1)(1,2,0)
    #  [1, 2, 0],   #
    #  [2, 0, 1],   # (0,0,3)(0,1,2)(0,2,1)(0,3,0)
    #  [2, 1, 0],   #
    #  [3, 0, 0]]   # Lattice points that a represents
    # a.shape[0] = number of DoFs
    # a.shape[1] = dim + 1
    # sum of each row = degree (bary centric lattice coordinates)

    # Flip along the axis 1 for convenience.
    # This would make: np.lexsort(a.transpose()) = [0, 1, 2, ..., 9].
    a = a[:, ::-1]

    index_perms = sorted(itertools.permutations(range(dim + 1)))
    # >>> index_perms
    # [[0, 1, 2],
    #  [0, 2, 1],
    #  [1, 0, 2],
    #  [1, 2, 0],
    #  [2, 0, 1],
    #  [2, 1, 0]]

    # First separate by entity dimension.
    g0 = dim - (a == 0).astype(int).sum(axis=1)
    # >>> g0
    # [ 0, 1, 1, 0, 1, 2, 1, 1, 1, 0]  # 0 for vertices
    #                                  # 1 for edges
    #                                  # 2 for cell

    # Then separate by entity number.
    g1 = np.zeros_like(g0)
    for d in range(dim + 1):
        on_facet_d = (a[:, d] == 0).astype(int)
        g1 += d * on_facet_d
    # The above logic is consistent with the FIAT entity numbering
    # convention ("entities that are not incident to vertex 0 are
    # numbered first, then ..."), but vertices are not numbered using
    # the same logic in FIAT, so we need to fix:
    g1[g0 == 0] = -g1[g0 == 0]
    # >>> g1
    # [-3, 2, 2,-2, 1, 0, 0, 1, 0,-1]

    # Compoute the map from the DG DoFs to the lattice points on the cell.
    # For each entity dimension, DoFs that have smaller numbers in g1
    # will be assigned smaller DG node numbers.
    # Order first by g0, then by g1, and finally by a (lexicographically as in CG)
    g0 = g0.reshape((a.shape[0], 1))
    g1 = g1.reshape((a.shape[0], 1))
    dg_to_lattice = np.lexsort(np.transpose(np.concatenate((a, g1, g0), axis=1)))
    # >>> dg_to_lattice
    # [ 0, 3, 9, 6, 8, 4, 7, 1, 2, 5]

    # Compute the inverse map.
    lattice_to_dg = np.empty_like(dg_to_lattice)
    for i, im in enumerate(dg_to_lattice):
        lattice_to_dg[im] = i
    # >>> lattice_to_dg
    # [ 0, 7, 8, 1, 5, 9, 3, 6, 4, 2]
    perms = {}
    for o, index_perm in enumerate(index_perms):
        # First compute permutation in lattice point numbers in lattice point order (as we do for CG cell DoFs)
        perm = np.lexsort(np.transpose(a[:, index_perm]))
        # Then convert to DG DoF numbers in DG DoF order:
        # lattice_to_dg[perm]                -> convert lattice point numbers to DG DoF numbers
        # lattice_to_dg[perm][dg_to_lattice] -> reorder for DG DoF order
        #
        # Example:
        # Under one CW rotation, a DG element on a physical cell
        # is mapped to the FIAT reference as:
        #
        # 0
        # 7 5
        # 8 9 6
        # 1 3 4 2
        #
        # Under the same transformation, the lattice points would
        # be mapped as:
        #
        # 0
        # 1 4
        # 2 5 7
        # 3 6 8 9
        #
        # In this case we will have:
        #
        # perm                               = [3, 6, 8, 9, 2, 5, 7, 1, 4, 0]
        # lattice_to_dg[perm]                = [1, 3, 4, 2, 8, 9, 6, 7, 5, 0]
        # lattice_to_dg[perm][dg_to_lattice] = [1, 2, 0, 6, 5, 8, 7, 3, 4, 9]
        #
        # Note:
        # Sane thing to do is to just number DG dofs on a lattice.
        perm = lattice_to_dg[perm][dg_to_lattice]
        perms[o] = perm.tolist()
    return perms


class DiscontinuousLagrangeDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []
        entity_permutations = {}

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()

        cur = 0
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = make_entity_permutations(dim, degree + 1 if dim == len(top) - 1 else -1)
            for entity in sorted(top[dim]):
                pts_cur = ref_el.make_points(dim, entity, degree)
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                entity_ids[dim][entity] = []
                cur += nnodes_cur
                entity_permutations[dim][entity] = perms
        entity_ids[dim][0] = list(range(len(nodes)))

        super(DiscontinuousLagrangeDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class HigherOrderDiscontinuousLagrange(finite_element.CiarletElement):
    """The discontinuous Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = DiscontinuousLagrangeDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(HigherOrderDiscontinuousLagrange, self).__init__(poly_set, dual, degree, formdegree)


def DiscontinuousLagrange(ref_el, degree):
    if degree == 0:
        return P0.P0(ref_el)
    else:
        return HigherOrderDiscontinuousLagrange(ref_el, degree)
