# Copyright (C) 2005 The University of Chicago
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Robert C. Kirby
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

from FIAT import dual_set, functional, polynomial_set, finite_element
import numpy


class P0Dual(dual_set.DualSet):
    def __init__(self, ref_el):
        entity_ids = {}
        entity_permutations = {}
        sd = ref_el.get_dimension()
        top = ref_el.get_topology()
        if sd == 0:
            pts = [tuple() for entity in sorted(top[sd])]
        else:
            pts = [tuple(numpy.average(ref_el.get_vertices_of_subcomplex(top[sd][entity]), 0))
                   for entity in sorted(top[sd])]
        nodes = [functional.PointEvaluation(ref_el, pt) for pt in pts]
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            sym_size = ref_el.symmetry_group_size(dim)
            num_points = 1 if dim == sd else 0
            if isinstance(dim, tuple):
                assert isinstance(sym_size, tuple)
                perms = {o: list(range(num_points)) for o in numpy.ndindex(sym_size)}
            else:
                perms = {o: list(range(num_points)) for o in range(sym_size)}
            for entity in sorted(top[dim]):
                entity_ids[dim][entity] = [entity] if dim == sd else []
                entity_permutations[dim][entity] = perms

        super().__init__(nodes, ref_el, entity_ids, entity_permutations)


class P0(finite_element.CiarletElement):
    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 0)
        dual = P0Dual(ref_el)
        degree = 0
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)
