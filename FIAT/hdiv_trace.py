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
from . import finite_element


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
        
        # Looping over dictionary of cell topology to construct the empty 
        # dictionary for entity ids of the trace element
        topology = cell.get_topology()
        
        for top_dim, entities in topology.items():
            self.entity_ids[top_dim] = {}
            
            for entity in entities:
                self.entity_ids[top_dim][entity] = {}

        # For each facet, we have nf = dim(facet) number of dofs
        # In this case, the facet is a DCLagrange element
        nf = self.DCLagrange.space_dimension()
        
        # Filling in entity ids
        for f in range(self.num_facets):
            self.entity_ids[spaceDim-1][f] = range(f*nf, (f+1)*nf)
    

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polyDegree
    
    def space_dimension(self):
        "Return the dimension of the trace finite element space."
        return self.DCLagrange.space_dimension()*self.num_facets
        
    def entity_ids(self):
        """Return the entity dictionary."""
        return self.entity_ids()

    def tabulate(self, order, points):
        """Return tabulated values basis functions at given points."""
        
        # Derivatives on facets don't make sense, so we raise error:
        if (order > 0):
            raise ValueError("Only function evals, no derivatives!")
        
        # Initialize basis function values at nodes to be 0 since
        # all basis functions are 0 except for specific phi on a facet
        space_dim = self.space_dimension()
        phiVals = np.zeros((space_dim, len(points)))
        
        entity = self.entity_ids
        
        # Call modified tabulate and pass entity information along.
        # Return type is a dictionary!
        facet_dim = self.DCLagrange.space_dimension()
        nf = facet_dim
        
        for facet_id in range(self.num_facets):
            nonzeroVals = list(self.DCLagrange.tabulate(order, points, \
                               entity[facet_dim][facet_id]))[0]
            
            phiVals[:,nf*facet_id:nf*(facet_id+1),] = nonzeroVals
        
        return {phiVals}
