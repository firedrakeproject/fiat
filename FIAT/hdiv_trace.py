# Copyright (C) 2016 Thomas H. Gibson
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import SimplicialComplex


class TraceSimplicialComplex(SimplicialComplex):

    def __init__(self, parent, codim=1):
        sd = parent.get_spatial_dimension() - codim
        parent_top = parent.topology
        topology = dict(parent_top)
        for dim in parent_top:
            if dim > sd:
                topology.pop(dim)

        facet = parent.construct_subelement(sd)
        self.base_ref_el = facet
        self._parent_complex = parent
        self._parent_simplex = parent

        # dict mapping child facets to their parent facet
        parent_to_children = {d: {e: [(d, e)]
                              for e in topology[d]}
                              for d in topology}
        # dict mapping parent facets to their children
        child_to_parent = {d: {e: (d, e)
                           for e in topology[d]}
                           for d in topology}
        self._child_to_parent = child_to_parent
        self._parent_to_children = parent_to_children

        # dict mapping cells to their boundary facets for each dimension,
        # while respecting the ordering on the parent simplex
        connectivity = {cell: {dim: [] for dim in topology} for cell in topology[sd]}
        for cell in topology[sd]:
            for dim in topology:
                for entity in topology[dim]:
                    if set(topology[dim][entity]) <= set(topology[sd][cell]):
                        connectivity[cell][dim].append(entity)
        self._cell_connectivity = connectivity

        # dict mapping subentity dimension to interior facets
        interior_facets = {dim: [entity for entity in child_to_parent[dim]
                                 if child_to_parent[dim][entity][0] == sd]
                           for dim in sorted(child_to_parent)}
        self._interior_facets = interior_facets
        super().__init__(facet.shape, parent.vertices, topology)

    def get_spatial_dimension(self):
        return self.base_ref_el.get_spatial_dimension()

    def get_dimension(self):
        return self.base_ref_el.get_dimension()

    def construct_subelement(self, dim):
        return self._parent_simplex.construct_subelement(dim)

    def get_parent(self):
        return self._parent_simplex

    def get_parent_complex(self):
        return self._parent_complex

    def get_parent_to_children(self):
        return self._parent_to_children

    def get_cell_connectivity(self):
        return self._cell_connectivity

    def is_trace(self):
        return True

    def is_macrocell(self):
        return True


class TraceError(Exception):
    """Exception caused by tabulating a trace element on the interior of a cell,
    or the gradient of a trace element."""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class HDivTrace(DiscontinuousLagrange):
    """Class implementing the trace of hdiv elements. This class
    is a stand-alone element family that produces a DG-facet field.
    This element is what's produced after performing the trace
    operation on an existing H(Div) element.

    This element is also known as the discontinuous trace field that
    arises in several DG formulations.
    """
    def __new__(cls, ref_el, degree, variant="equispaced_interior"):
        """Constructor for the HDivTrace element.

        :arg ref_el: A reference element, which may be a tensor product
                     cell.
        :arg degree: The degree of approximation. If on a tensor product
                     cell, then provide a tuple of degrees if you want
                     varying degrees.
        :arg variant: The point distribution variant passed on to recursivenodes.
        """
        facets = TraceSimplicialComplex(ref_el)
        return DiscontinuousLagrange(facets, degree, variant=variant)
