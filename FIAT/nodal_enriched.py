# Copyright (C) 2013 Andrew T. T. McRae, 2015-2016 Jan Blechta
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import math

from FIAT.expansions import polynomial_entity_ids
from FIAT.polynomial_set import PolynomialSet
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
from FIAT.barycentric_interpolation import LagrangeLineExpansionSet

__all__ = ['NodalEnrichedElement']


class NodalEnrichedElement(CiarletElement):
    """NodalEnriched element is a direct sum of a sequence of
    finite elements. Primal basis is reorthogonalized to the
    dual basis for nodality.

    The following is equivalent:
        * the constructor is well-defined,
        * the resulting element is unisolvent and its basis is nodal,
        * the supplied elements are unisolvent with nodal basis and
          their primal bases are mutually linearly independent,
        * the supplied elements are unisolvent with nodal basis and
          their dual bases are mutually linearly independent.
    """

    def __init__(self, *elements):

        # Test elements are nodal
        if not all(e.is_nodal() for e in elements):
            raise ValueError("Not all elements given for construction "
                             "of NodalEnrichedElement are nodal")

        # Extract common data
        embedded_degrees = [e.get_nodal_basis().get_embedded_degree() for e in elements]
        embedded_degree = max(embedded_degrees)
        degree = max(e.degree() for e in elements)
        order = max(e.get_order() for e in elements)
        formdegree = None if any(e.get_formdegree() is None for e in elements) \
            else max(e.get_formdegree() for e in elements)
        # LagrangeExpansionSet has fixed degree, ensure we grab the highest one
        elem = elements[embedded_degrees.index(embedded_degree)]
        ref_el = elem.get_reference_complex()
        expansion_set = elem.get_nodal_basis().get_expansion_set()
        mapping = elem.mapping()[0]
        value_shape = elem.value_shape()

        # Sanity check
        assert all(e.get_reference_complex() == ref_el for e in elements)
        assert all(set(e.mapping()) == {mapping, } for e in elements)
        assert all(e.value_shape() == value_shape for e in elements)

        # Merge polynomial sets
        if isinstance(expansion_set, LagrangeLineExpansionSet):
            # Obtain coefficients via interpolation
            points = expansion_set.get_points()
            coeffs = np.vstack([e.tabulate(0, points)[(0,)] for e in elements])
        else:
            assert all(e.get_nodal_basis().get_expansion_set() == expansion_set
                       for e in elements)
            coeffs = [e.get_coeffs() for e in elements]
            coeffs = _merge_coeffs(coeffs, ref_el, embedded_degrees, expansion_set.continuity)
        poly_set = PolynomialSet(ref_el,
                                 degree,
                                 embedded_degree,
                                 expansion_set,
                                 coeffs)

        # Renumber dof numbers
        offsets = np.cumsum([0] + [e.space_dimension() for e in elements[:-1]])
        entity_ids = _merge_entity_ids((e.entity_dofs() for e in elements),
                                       offsets)

        # Merge dual bases
        nodes = [node for e in elements for node in e.dual_basis()]
        ref_el = ref_el.get_parent() or ref_el
        dual_set = DualSet(nodes, ref_el, entity_ids)

        # CiarletElement constructor adjusts poly_set coefficients s.t.
        # dual_set is really dual to poly_set
        super().__init__(poly_set, dual_set, order, formdegree=formdegree, mapping=mapping)


def _merge_coeffs(coeffss, ref_el, degrees, continuity):
    # Indices of the hierachical expansion set on each facet
    entity_ids = polynomial_entity_ids(ref_el, max(degrees), continuity)

    # Number of bases members
    total_dim = sum(c.shape[0] for c in coeffss)

    # Value shape
    value_shape = coeffss[0].shape[1:-1]
    assert all(c.shape[1:-1] == value_shape for c in coeffss)

    # Number of expansion polynomials
    max_expansion_dim = max(c.shape[-1] for c in coeffss)

    # Compose new coeffs
    shape = (total_dim, *value_shape, max_expansion_dim)
    new_coeffs = np.zeros(shape, dtype=coeffss[0].dtype)
    counter = 0
    for c, degree in zip(coeffss, degrees):
        ids = []
        if continuity == "C0":
            dims = sorted(entity_ids)
        else:
            dims = (ref_el.get_spatial_dimension(),)
        for dim in dims:
            if continuity == "C0":
                dimPk = math.comb(degree - 1, dim)
            else:
                dimPk = math.comb(degree + dim, dim)
            for entity in sorted(entity_ids[dim]):
                ids.extend(entity_ids[dim][entity][:dimPk])

        num_members = c.shape[0]
        new_coeffs[counter:counter+num_members, ..., ids] = c
        counter += num_members
    assert counter == total_dim
    return new_coeffs


def _merge_entity_ids(entity_ids, offsets):
    ret = {}
    for i, ids in enumerate(entity_ids):
        for dim in ids:
            if dim not in ret:
                ret[dim] = {}
            for entity in ids[dim]:
                if entity not in ret[dim]:
                    ret[dim][entity] = []
                ret[dim][entity].extend(np.array(ids[dim][entity]) + offsets[i])
    return ret
