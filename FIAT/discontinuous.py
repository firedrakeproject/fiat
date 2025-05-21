# Copyright (C) 2014 Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT.finite_element import CiarletElement, FiniteElement
from FIAT.dual_set import DualSet


class DiscontinuousElement(CiarletElement):
    """A copy of an existing element where all dofs are associated with the cell"""

    def __init__(self, element):
        self._element = element
        ref_el = element.get_reference_element()
        ref_complex = element.get_reference_complex()
        mapping, = set(element.mapping())

        new_entity_ids = {}
        topology = ref_el.get_topology()
        for dim in sorted(topology):
            new_entity_ids[dim] = {}
            for ent in sorted(topology[dim]):
                new_entity_ids[dim][ent] = []

        new_entity_ids[dim][0] = list(range(element.space_dimension()))
        # re-initialise the dual, so entity_closure_dofs is recalculated
        dual = DualSet(element.dual_basis(), element.get_reference_element(), new_entity_ids)
        order = element.get_order()

        # fully discontinuous
        formdegree = element.get_reference_element().get_spatial_dimension()

        FiniteElement.__init__(self, ref_el, dual, order, formdegree=formdegree,
                               mapping=mapping, ref_complex=ref_complex)

    def degree(self):
        "Return the degree of the (embedding) polynomial space."
        return self._element.degree()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        return self._element.get_nodal_basis()

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        return self._element.get_coeffs()

    def num_sub_elements(self):
        "Return the number of sub-elements."
        return self._element.num_sub_elements()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""
        return self._element.tabulate(order, points, entity)

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self._element.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        return self._element.dmats()

    def get_num_members(self, arg):
        "Return number of members of the expansion set."
        return self._element.get_num_members()
