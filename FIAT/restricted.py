# Copyright (C) 2015-2016 Jan Blechta, Andrew T T McRae, and others
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement


class RestrictedDualSet(DualSet):
    """Restrict the given DualSet to the specified list of dofs."""

    def __init__(self, dual, indices):
        indices = list(sorted(indices))
        ref_el = dual.get_reference_element()
        nodes_old = dual.get_nodes()
        entity_ids = {}
        nodes = []
        for d, entities in dual.get_entity_ids().items():
            entity_ids[d] = {}
            for entity, dofs in entities.items():
                entity_ids[d][entity] = [indices.index(dof)
                                         for dof in dofs if dof in indices]
        nodes = [nodes_old[i] for i in indices]
        self._dual = dual

        super().__init__(nodes, ref_el, entity_ids)

    def get_indices(self, restriction_domain, take_closure=True):
        """Return the list of dofs with support on a given restriction domain.

        :arg restriction_domain: can be 'interior', 'vertex', 'edge', 'face' or 'facet'
        :kwarg take_closure: Are we taking the closure of the restriction domain?
        """
        # Call get_indices on the parent class to support multiple restriction domains
        return type(self._dual).get_indices(self, restriction_domain, take_closure=take_closure)


class RestrictedElement(CiarletElement):
    """Restrict the given element to the specified list of dofs."""

    def __init__(self, element, indices=None, restriction_domain=None, take_closure=True):
        '''For sake of argument, indices overrides restriction_domain'''

        if not (indices or restriction_domain):
            raise RuntimeError("Either indices or restriction_domain must be passed in")

        if not indices:
            indices = element.dual.get_indices(restriction_domain, take_closure=take_closure)

        if isinstance(indices, str):
            raise RuntimeError("variable 'indices' was a string; did you forget to use a keyword?")

        if len(indices) == 0:
            raise ValueError("No point in creating empty RestrictedElement.")

        self._element = element
        self._indices = indices

        # Restrict primal set
        poly_set = element.get_nodal_basis().take(indices)

        # Restrict dual set
        dual = RestrictedDualSet(element.get_dual_set(), indices)

        # Restrict mapping
        mapping_old = element.mapping()
        mapping_new = [mapping_old[dof] for dof in indices]
        assert all(e_mapping == mapping_new[0] for e_mapping in mapping_new)

        # Call constructor of CiarletElement
        super().__init__(poly_set, dual, element.degree(), element.get_formdegree(), mapping_new[0])
