# -*- coding: utf-8 -*-
"""Implementation of the Hellan-Herrmann-Johnson finite elements."""

# Copyright (C) 2016-2018 Lizao Li <lzlarryli@gmail.com>
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import dual_set, finite_element, polynomial_set
from FIAT.functional import ComponentPointEvaluation, PointwiseInnerProductEvaluation as InnerProduct


class HellanHerrmannJohnsonDual(dual_set.DualSet):
    """Degrees of freedom for Hellan-Herrmann-Johnson elements."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {i: [] for i in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # no vertex dofs
        edge_dofs, entity_ids[sd-1] = self._generate_facet_dofs(ref_el, degree, 0)
        nodes.extend(edge_dofs)
        cell_nodes, entity_ids[sd] = self._generate_interior_dofs(ref_el, degree, len(nodes))
        nodes.extend(cell_nodes)

        super().__init__(nodes, ref_el, entity_ids)

    @staticmethod
    def _generate_facet_dofs(ref_el, degree, offset):
        """Generate dofs on edges.
        On each edge, let n be its normal. For degree=r, the scalar function
              n^T u n
        is evaluated at enough points to control P(r).
        """
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        nodes = []
        entity_ids = {}
        for entity in sorted(top[sd-1]):
            pts = ref_el.make_points(sd-1, entity, degree + 2)
            normal = ref_el.compute_scaled_normal(entity)
            nodes.extend(InnerProduct(ref_el, normal, normal, pt) for pt in pts)
            num_new_nodes = len(pts)
            entity_ids[entity] = list(range(offset, offset + num_new_nodes))
            offset += num_new_nodes
        return nodes, entity_ids

    @staticmethod
    def _generate_interior_dofs(ref_el, degree, offset):
        """Generate dofs on the cell interior.
        On each interior, for degree=r, the independent components
              uij, i <= j
        are evaluated at enough points to control P(r-1).
        """
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        shp = (sd, sd)
        basis = [(i, j) for i in range(sd) for j in range(i, sd)]
        nodes = []
        entity_ids = {}
        for entity in sorted(top[sd]):
            pts = ref_el.make_points(sd, entity, degree + 2)
            nodes.extend(ComponentPointEvaluation(ref_el, comp, shp, pt)
                         for comp in basis for pt in pts)
            num_new_nodes = len(pts)
            entity_ids[entity] = list(range(offset, offset + len(nodes)))
            offset += num_new_nodes
        return nodes, entity_ids


class HellanHerrmannJohnson(finite_element.CiarletElement):
    """The definition of Hellan-Herrmann-Johnson element. It is defined only in
       dimension 2. It consists of piecewise polynomial symmetric-matrix-valued
       functions of degree r or less with normal-normal continuity.
    """
    def __init__(self, ref_el, degree):
        assert degree >= 0, "Hellan-Herrmann-Johnson starts at degree 0!"
        poly_set = polynomial_set.ONSymTensorPolynomialSet(ref_el, degree)
        dual = HellanHerrmannJohnsonDual(ref_el, degree)
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, mapping=mapping)
