# Copyright (C) 2013 Andrew T. T. McRae, 2015-2016 Jan Blechta
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from FIAT.polynomial_set import PolynomialSet
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
from FIAT.barycentric_interpolation import LagrangeLineExpansionSet

__all__ = ['NodalEnrichedElement']


class NodalEnrichedElement(CiarletElement):
    """NodalEnriched element is a direct sum of a sequence of
    finite elements. Dual basis is reorthogonalized to the
    primal basis for nodality.

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
        degree = max(e.get_nodal_basis().get_degree() for e in elements)
        embedded_degree = max(e.get_nodal_basis().get_embedded_degree()
                              for e in elements)
        order = max(e.get_order() for e in elements)
        formdegree = None if any(e.get_formdegree() is None for e in elements) \
            else max(e.get_formdegree() for e in elements)
        # LagrangeExpansionSet has fixed degree, ensure we grab the embedding one
        elem = next(e for e in elements
                    if e.get_nodal_basis().get_embedded_degree() == embedded_degree)
        ref_el = elem.get_reference_element()
        expansion_set = elem.get_nodal_basis().get_expansion_set()
        mapping = elem.mapping()[0]
        value_shape = elem.value_shape()

        # Sanity check
        assert all(e.get_nodal_basis().get_reference_element() ==
                   ref_el for e in elements)
        assert all(e_mapping == mapping for e in elements
                   for e_mapping in e.mapping())
        assert all(e.value_shape() == value_shape for e in elements)

        # Merge polynomial sets
        if isinstance(expansion_set, LagrangeLineExpansionSet):
            # Obtain coefficients via interpolation
            points = expansion_set.get_points()
            coeffs = [e.tabulate(0, points)[(0,)] for e in elements]
        else:
            assert all(type(e.get_nodal_basis().get_expansion_set()) ==
                       type(expansion_set) for e in elements)
            coeffs = [e.get_coeffs() for e in elements]

        coeffs = _merge_coeffs(coeffs)
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
        dual_set = DualSet(nodes, ref_el, entity_ids)

        # CiarletElement constructor adjusts poly_set coefficients s.t.
        # dual_set is really dual to poly_set
        super(NodalEnrichedElement, self).__init__(poly_set, dual_set, order,
                                                   formdegree=formdegree, mapping=mapping)


def _merge_coeffs(coeffss):
    # Number of bases members
    total_dim = sum(c.shape[0] for c in coeffss)

    # Value shape
    value_shape = coeffss[0].shape[1:-1]
    assert all(c.shape[1:-1] == value_shape for c in coeffss)

    # Number of expansion polynomials
    max_expansion_dim = max(c.shape[-1] for c in coeffss)

    # Compose new coeffs
    shape = (total_dim,) + value_shape + (max_expansion_dim,)
    new_coeffs = np.zeros(shape, dtype=coeffss[0].dtype)
    counter = 0
    for c in coeffss:
        dim = c.shape[0]
        expansion_dim = c.shape[-1]
        new_coeffs[counter:counter+dim, ..., :expansion_dim] = c
        counter += dim
    assert counter == total_dim
    return new_coeffs


def _merge_entity_ids(entity_ids, offsets):
    ret = {}
    for i, ids in enumerate(entity_ids):
        for dim in ids:
            if not ret.get(dim):
                ret[dim] = {}
            for entity in ids[dim]:
                if not ret[dim].get(entity):
                    ret[dim][entity] = []
                ret[dim][entity] += (np.array(ids[dim][entity]) + offsets[i]).tolist()
    return ret
