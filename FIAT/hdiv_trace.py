# Copyright (C) 2016 Thomas H. Gibson
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex


class TraceHDiv(object):
    """Class implementing the trace of hdiv elements (assumed on a simplex)
       -- Currently a work in progress."""

    def __init__(self, cell, polyDegree):

        spaceDim = cell.get_spatial_dimension()
        
        # Check to make sure spacial dim is sensible for trace
        if spaceDim == 1:
            raise ValueError("Hell")

        # Otherwise, initialize some neat stuff and proceed
        self.cell = cell   
        self.polyDegree = polyDegree
        
        # Constructing facet as a DC Lagrange element
        self.facet = ufc_simplex(spaceDim - 1)
        self.DCLagrange = DiscontinuousLagrange(self.facet, polyDegree)

        # Number of facets on simplex-type element
        self.num_facets = spaceDim + 1

        # Construct entity ids (assigning top. dim. and initializing as empty)
        self.entity_ids = {}
        self.empty_entity_ids = {}
        
        # Looping over dictionary of cell topology to construct the empty 
        # dictionary for entity ids of the trace element
        topology = cell.get_topology()
        
        for top_dim, entities in topology.items():
            self.entity_ids[top_dim] = {}
            
#            self.empty_entity_ids[top_dim] = {}
            
            for entity in entities:
                self.entity_ids[top_dim][entity] = {}
                
#                self.empty_entity_ids[top_dim][entity] = {}
    

        # For each facet, we have nf = dim(facet) number of dofs
        # In this case, the facet is a DCLagrange element
        nf = self.DCLagrange.space_dimension()
        
        # Filling in entity ids
        for f in range(self.num_facets):
            self.entity_ids[spaceDim-1][f] = range(f*nf, (f+1)*nf)
    

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polyDegree
        
#    def empty_entity_ids(self):
#        """Return the empty entity dictionary (to check it's properly set up)."""
#        return self.empty_entity_ids()
        
    def entity_ids(self):
        """Return the entity dictionary."""
        return self.entity_ids()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError

    def tabulate(self, order, points):
        raise NotImplementedError
