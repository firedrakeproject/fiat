# Copyright (C) 2015-2016 Jan Blechta, Andrew T T McRae, and others
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement


class RestrictedElement(CiarletElement):
    """Restrict given element to specified list of dofs."""

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

        # Fetch reference element
        ref_el = element.get_reference_element()

        # Restrict primal set
        poly_set = element.get_nodal_basis().take(indices)

        # Restrict dual set
        dof_counter = 0
        entity_ids = {}
        nodes = []
        nodes_old = element.dual_basis()
        for d, entities in element.entity_dofs().items():
            entity_ids[d] = {}
            for entity, dofs in entities.items():
                entity_ids[d][entity] = []
                for dof in dofs:
                    if dof not in indices:
                        continue
                    entity_ids[d][entity].append(dof_counter)
                    dof_counter += 1
                    nodes.append(nodes_old[dof])
        assert dof_counter == len(indices)
        dual = DualSet(nodes, ref_el, entity_ids)

        # Restrict mapping
        mapping_old = element.mapping()
        mapping_new = [mapping_old[dof] for dof in indices]
        assert all(e_mapping == mapping_new[0] for e_mapping in mapping_new)

        # Call constructor of CiarletElement
        super(RestrictedElement, self).__init__(poly_set, dual, 0, element.get_formdegree(), mapping_new[0])
