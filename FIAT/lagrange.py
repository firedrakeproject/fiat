# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet, get_lagrange_points
from FIAT.reference_element import LINE
from FIAT.check_format_variant import parse_lagrange_variant


class LagrangeDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplicial complexes of any dimension.  Nodes are point evaluation at
    recursively-defined points.

    :arg ref_el: The simplicial complex.
    :arg degree: The polynomial degree.
    :arg point_variant: The point distribution variant passed on to recursivenodes.
    :arg sort_entities: A flag to sort entities by support vertex ids.
                        If false then entities are sorted first by dimension and then by
                        entity id. The DOFs are always sorted by the entity ordering
                        and then lexicographically by lattice multiindex.
    """
    def __init__(self, ref_el, degree, point_variant="equispaced", sort_entities=False):
        nodes = []
        entity_ids = {}
        entity_permutations = {}
        top = ref_el.get_topology()
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = {0: [0]} if dim == 0 else make_entity_permutations_simplex(dim, degree - dim)
            for entity in sorted(top[dim]):
                entity_permutations[dim][entity] = perms

        entities = [(dim, entity) for dim in sorted(top) for entity in sorted(top[dim])]
        if sort_entities:
            # sort the entities by support vertex ids
            support = [top[dim][entity] for dim, entity in entities]
            entities = [entity for verts, entity in sorted(zip(support, entities))]

        # make nodes by getting points
        # need to do this entity-by-entity
        for dim, entity in entities:
            cur = len(nodes)
            pts_cur = ref_el.make_points(dim, entity, degree, variant=point_variant)
            nodes.extend(functional.PointEvaluation(ref_el, x) for x in pts_cur)
            entity_ids[dim][entity] = list(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids, entity_permutations)


class Lagrange(finite_element.CiarletElement):
    """The Lagrange finite element.

    :arg ref_el:  The reference element, which could be a standard FIAT simplex or a split complex
    :arg degree:  The polynomial degree
    :arg variant: A comma-separated string that may specify the type of point distribution
                  and the splitting strategy if a macro element is desired.
                  Either option may be omitted.  The default point type is equispaced
                  and the default splitting strategy is None.
                  Example: variant='gll' gives a standard unsplit point distribution with
                              spectral points.
                           variant='equispaced,Iso(2)' with degree=1 gives the P2:P1 iso element.
                           variant='Alfeld' can be used to obtain a barycentrically refined
                              macroelement for Scott-Vogelius.
    :arg sort_entities: A flag to sort entities by support vertex ids.
                        If false then entities are sorted first by dimension and then by
                        entity id. The DOFs are always sorted by the entity ordering
                        and then lexicographically by lattice multiindex.
    """
    def __init__(self, ref_el, degree, variant="equispaced", sort_entities=False):
        splitting, point_variant = parse_lagrange_variant(variant)
        if splitting is not None:
            ref_el = splitting(ref_el)
        dual = LagrangeDualSet(ref_el, degree, point_variant=point_variant, sort_entities=sort_entities)
        if ref_el.shape == LINE:
            # In 1D we can use the primal basis as the expansion set,
            # avoiding any round-off coming from a basis transformation
            points = get_lagrange_points(dual)
            poly_set = LagrangePolynomialSet(ref_el, points)
        else:
            poly_variant = "bubble" if ref_el.is_macrocell() else None
            poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, variant=poly_variant)
        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree)
