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


class BrezziDouglasMariniCube(FiniteElement):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("BDMce_k elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("BDMce_k elements only valid for dimension 2")

        flat_topology = flat_el.get_topology()

        entity_ids = {}
        cur = 0

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        for j in sorted(flat_topology[1]):
            entity_ids[1][j] = list(range(cur, cur + degree + 1))
            cur = cur + degree + 1

        entity_ids[2][0] = list(range(cur, cur + 2*triangular_number(degree - 1)))
        cur += 2*triangular_number(degree - 1)

        formdegree = 1

        entity_closure_ids = make_entity_closure_ids(flat_el, entity_ids)

        super(BrezziDouglasMariniCube, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree,
                                                      mapping="contravariant piola")

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
        raise NotImplementedError("get_nodal_basis not implemented for bdmce")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for bdmce")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for bdmce")

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


def e_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    EL = tuple([(0, -leg(j, y_mid)*dx[0]) for j in range(deg)] +
               [(-leg(deg-1, y_mid)*dy[0]*dy[1]/deg, -leg(deg, y_mid)*dx[0])] +
               [(0, -leg(j, y_mid)*dx[1]) for j in range(deg)] +
               [(leg(deg-1, y_mid)*dy[0]*dy[1]/deg, -leg(deg, y_mid)*dx[1])] +
               [(-leg(j, x_mid)*dy[0], 0) for j in range(deg)] +
               [(-leg(deg, x_mid)*dy[0], -leg(deg-1, x_mid)*dx[0]*dx[1]/deg)] +
               [(-leg(j, x_mid)*dy[1], 0) for j in range(deg)] +
               [(-leg(deg, x_mid)*dy[1], leg(deg-1, x_mid)*dx[0]*dx[1]/deg)])

    return EL


def f_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    FL = []
    for k in range(2, deg+1):
        for j in range(k-1):
            FL += [(0, leg(j, x_mid)*leg(k-2-j, y_mid)*dx[0]*dx[1])]
            FL += [(leg(k-2-j, x_mid)*leg(j, y_mid)*dy[0]*dy[1], 0)]

    return tuple(FL)


class BrezziDouglasMariniCubeEdge(BrezziDouglasMariniCube):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("BDMcf_k elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("BDMcf_k elements only valid for dimension 2")

        verts = flat_el.get_vertices()

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])

        EL = e_lambda_1_2d(degree, dx, dy, x_mid, y_mid)
        if degree >= 2:
            FL = f_lambda_1_2d(degree, dx, dy, x_mid, y_mid)
        else:
            FL = ()
        bdmce_list = EL + FL
        self.basis = {(0, 0): Array(bdmce_list)}
        super(BrezziDouglasMariniCubeEdge, self).__init__(ref_el=ref_el, degree=degree)


class BrezziDouglasMariniCubeFace(BrezziDouglasMariniCube):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("BDMcf_k elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("BDMcf_k elements only valid for dimension 2")

        verts = flat_el.get_vertices()

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])

        EL = e_lambda_1_2d(degree, dx, dy, x_mid, y_mid)
        if degree >= 2:
            FL = f_lambda_1_2d(degree, dx, dy, x_mid, y_mid)
        else:
            FL = ()
        bdmcf_list = EL + FL
        bdmcf_list = [[-a[1], a[0]] for a in bdmcf_list]
        self.basis = {(0, 0): Array(bdmcf_list)}

        super(BrezziDouglasMariniCubeFace, self).__init__(ref_el=ref_el, degree=degree)
