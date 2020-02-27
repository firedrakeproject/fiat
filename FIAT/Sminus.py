# Copyright (C) 2019 Cyrus Cheng (Imperial College London)
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
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2019

from sympy import symbols, legendre, Array, diff
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis
from FIAT.reference_element import compute_unflattening_map, flatten_reference_cube

x, y, z = symbols('x y z')
variables = (x, y, z)
leg = legendre


def triangular_number(n):
    return int((n+1)*n/2)


class TrimmedSerendipity(FiniteElement):
    def __init__(self, ref_el, degree, mapping):
        if degree < 1:
            raise Exception("Trimmed serendipity elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("Trimmed serendipity elements only valid for dimension 2")

        flat_topology = flat_el.get_topology()
        entity_ids = {}
        cur = 0

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []
        for j in sorted(flat_topology[1]):
            entity_ids[1][j] = list(range(cur, cur + degree)) #assign entity ids to everything in the first dimension.
            cur = cur + degree

        if(degree >= 2):
            entity_ids[2][0] = list(range(cur, cur + 2*triangular_number(degree - 2) + degree)) #Assigns an entity IDs to the face.  I need to figure out the pattern here for trimmed serendipity.
        #else:
        #    entity_ids[2][0] = list(range(cur, cur + 2*triangular_number(degree - 2) + degree)) #Assigns an entity IDs to the face.  I need to figure out the pattern here for trimmed serendipity.

        #print(entity_ids)
        cur += 2*triangular_number(degree - 2) + degree
        #print("cur is equal to")
        #print(cur)

        formdegree = 1

        entity_closure_ids = make_entity_closure_ids(flat_el, entity_ids)

        super(TrimmedSerendipity, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree,
                                                      mapping=mapping)

        topology = ref_el.get_topology()
        unflattening_map = compute_unflattening_map(topology)
        unflattened_entity_ids = {}
        unflattened_entity_closure_ids = {}

        for dim, entities in sorted(topology.items()):
            unflattened_entity_ids[dim] = {}
            unflattened_entity_closure_ids[dim] = {}
        for dim, entities in sorted(flat_topology.items()):
            for entity in entities:
                unflat_dim, unflat_entity = unflattening_map[(dim, entity)]
                unflattened_entity_ids[unflat_dim][unflat_entity] = entity_ids[dim][entity]
                unflattened_entity_closure_ids[unflat_dim][unflat_entity] = entity_closure_ids[dim][entity]
        self.entity_ids = unflattened_entity_ids
        self.entity_closure_ids = unflattened_entity_closure_ids
        self._degree = degree
        self.flat_el = flat_el

    def degree(self):
        return self._degree

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for trimmed serendipity")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for trimmed serendipity")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for trimmed serendipity")

    def tabulate(self, order, points, entity=None):

        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        phivals = {}

        for o in range(order+1):
            alphas = mis(2, o)
            for alpha in alphas:
                try:
                    polynomials = self.basis[alpha]
                except KeyError:
                    polynomials = diff(self.basis[(0, 0)], *zip(variables, alpha))
                    self.basis[alpha] = polynomials
                T = np.zeros((len(polynomials[:, 0]), 2, len(points)))
                for i in range(len(points)):
                    subs = {v: points[i][k] for k, v in enumerate(variables[:2])}
                    for j, f in enumerate(polynomials[:, 0]):
                        T[j, 0, i] = f.evalf(subs=subs)
                    for j, f in enumerate(polynomials[:, 1]):
                        T[j, 1, i] = f.evalf(subs=subs)
                phivals[alpha] = T

        return phivals

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.entity_ids

    def entity_closure_dofs(self):
        """Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element."""
        return self.entity_closure_ids

    def value_shape(self):
        return (2,)

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError

    def space_dimension(self):
        return int(len(self.basis[(0, 0)])/2)


#Splitting the E Lambda function into two seperate functions for E Lambda and E tilde Lambda.
#Correlating with Andrew's paper, leg(j, x_mid) should be a polynomial x^i, leg(j, y_mid) should be y^i,
#dy[0] should represent y-1, dy[1] should represent y+1 (and similar for the dx and x+/- 1).  
#Still not sure why we use for loops in only the EL tuple but not the ELTilde tuple.

def e_lambda_1_2d_part_one(deg, dx, dy, x_mid, y_mid):
    EL = tuple([(-leg(j, x_mid)*dy[0], 0) for j in range(deg)]+
               [(-leg(j, x_mid)*dy[1], 0) for j in range(deg)]+
               [(0, -leg(j, y_mid) * dx[0]) for j in range(deg)]+
               [(0, -leg(j, y_mid) * dx[1]) for j in range(deg)])
    return EL

def e_lambda_tilde_1_2d_part_two(deg, dx, dy, x_mid, y_mid):
    ELTilde = tuple([(-leg(deg, x_mid)*dy[0], -leg(deg-1, x_mid)*dx[0]*dx[1]/(deg+1))]+
               [(-leg(deg, x_mid)*dy[1], leg(deg-1, x_mid)*dx[0]*dx[1]/(deg+1))]+
               [(-leg(deg-1, y_mid)*dy[0]*dy[1]/(deg+1), -leg(deg, y_mid)*dx[0])]+
               [(leg(deg-1, y_mid)*dy[0]*dy[1]/(deg+1), -leg(deg, y_mid)*dx[1])])
    return ELTilde

def e_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    EL = e_lambda_1_2d_part_one(deg, dx, dy, x_mid, y_mid)
    ELTilde = e_lambda_tilde_1_2d_part_two(deg, dx, dy, x_mid, y_mid)
    
    result = EL + ELTilde
    return result

def determine_f_lambda_portions_2d(deg):
    if (deg < 2):
        DegsOfIteration = []
    else:
        DegsOfIteration = []
        for i in range(2, deg):
            DegsOfIteration += [i]
        
    return DegsOfIteration

def f_lambda_1_2d_pieces(current_deg, dx, dy, x_mid, y_mid):
    if (current_deg == 2):
       FLpiece = [(leg(0, x_mid) * leg(0, y_mid) * dy[0] * dy[1], 0)]
       FLpiece += [(0, leg(0, x_mid) * leg(0, y_mid) * dx[0] * dx[1])]
    else:
        target_power = current_deg - 2
        FLpiece = tuple([])
        for j in range(0, target_power + 1):
            k = target_power - j
            FLpiece += tuple([(leg(j, x_mid) * leg(k, y_mid) * dy[0] * dy[1], 0)])
            FLpiece += tuple([(0, leg(j, x_mid) * leg(k, y_mid) * dx[0] * dx[1])])
    return FLpiece

def f_lambda_1_2d_trim(deg, dx, dy, x_mid, y_mid):
    DegsOfIteration = determine_f_lambda_portions_2d(deg)
    FL = []
    for i in DegsOfIteration:
        FL += f_lambda_1_2d_pieces(i, dx, dy, x_mid, y_mid)
    return tuple(FL)



def f_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    FL = []
    for k in range(2, deg+1):
        for j in range(k-1):
            FL += [(0, leg(j, x_mid)*leg(k-2-j, y_mid)*dx[0]*dx[1])]
            FL += [(leg(k-2-j, x_mid)*leg(j, y_mid)*dy[0]*dy[1], 0)]
    return tuple(FL)

def f_lambda_1_2d_tilde(deg, dx, dy, x_mid, y_mid):
    FLTilde = tuple([])
    FLTilde += tuple([(leg(deg - 2, y_mid)*dy[0]*dy[1], 0)])
    FLTilde += tuple([(0, leg(deg - 2, x_mid)*dx[0]*dx[1])])
    for k in range (1, deg - 1):
        FLTilde += tuple([(leg(k, x_mid) * leg(deg - k - 2, y_mid) * dy[0] * dy[1], -leg(k - 1, x_mid) * leg(deg - k - 1, y_mid) * dx[0] * dx[1])])

    return tuple(FLTilde)

def trimmed_f_lambda(deg, dx, dy, x_mid, y_mid):
    FL = f_lambda_1_2d_trim(deg, dx, dy, x_mid, y_mid)
    FLT = f_lambda_1_2d_tilde(deg, dx, dy, x_mid, y_mid)
    result = FL + FLT

    return result



class TrimmedSerendipityEdge(TrimmedSerendipity):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("Trimmed Serendipity_k edge elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("Trimmed Serendipity_k edge elements only valid for dimension 2")

        verts = flat_el.get_vertices()

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])

        EL = e_lambda_1_2d_part_one(degree, dx, dy, x_mid, y_mid)
        if degree >= 2:
            FL = trimmed_f_lambda(degree, dx, dy, x_mid, y_mid)
        else:
            FL = ()
        
        bdmce_list = EL + FL
        self.basis = {(0, 0): Array(bdmce_list)}
        super(TrimmedSerendipityEdge, self).__init__(ref_el=ref_el, degree=degree, mapping="covariant piola")


class TrimmedSerendipityFace(TrimmedSerendipity):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("Trimmed serendipity face elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("Trimmed serendipity face elements only valid for dimension 2")

        verts = flat_el.get_vertices()

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])

        EL = e_lambda_1_2d_part_one(degree, dx, dy, x_mid, y_mid)
        if degree >= 2:
            FL = trimmed_f_lambda(degree, dx, dy, x_mid, y_mid)
        else:
            FL = ()
        bdmcf_list = EL + FL
        bdmcf_list = [[-a[1], a[0]] for a in bdmcf_list]
        self.basis = {(0, 0): Array(bdmcf_list)}
        super(TrimmedSerendipityFace, self).__init__(ref_el=ref_el, degree=degree, mapping="contravariant piola")

