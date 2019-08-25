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


class ArbogastCorreaEdge(FiniteElement):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("AA_k elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 3:
            raise Exception("AAe_k elements only valid for dimension 3")

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        dz = ((verts[-1][2] - z)/(verts[-1][2] - verts[0][2]), (z - verts[0][2])/(verts[-1][2] - verts[0][2]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])
        z_mid = 2*z-(verts[-1][2] + verts[0][2])

    super(ArbogastCorreaEdge, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree,
                                                  mapping="covariant piola")


def e_lambda_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    EL = tuple([(0, 0, leg(j, z_mid)*dx[0]*dy[0]) for j in range(deg)] +
               [(leg(deg-1, z_mid)*dy[0]*dz[0]*dz[1], leg(deg-1, z_mid)*dx[0]*dz[0]*dz[1], (deg+1)*leg(deg, z_mid)*dx[0]*dy[0])] +
               [(0, 0, leg(j, z_mid)*dx[0]*dy[1]) for j in range(deg)] +
               [(leg(deg-1, z_mid)*dy[1]*dz[0]*dz[1], leg(deg-1, z_mid)*dx[0]*dz[0]*dz[1], (deg+1)*leg(deg, z_mid)*dx[0]*dy[1])] +
               [(0, 0, leg(j, z_mid)*dx[1]*dy[0]) for j in range(deg)] +
               [(leg(deg-1, z_mid)*dy[0]*dz[0]*dz[1], leg(deg-1, z_mid)*dx[1]*dz[0]*dz[1], (deg+1)*leg(deg, z_mid)*dx[1]*dy[0])] +
               [(0, 0, leg(j, z_mid)*dx[1]*dy[1]) for j in range(deg)] +
               [(leg(deg-1, z_mid)*dy[1]*dz[0]*dz[1], leg(deg-1, z_mid)*dx[1]*dz[0]*dz[1], (deg+1)*leg(deg, z_mid)*dx[1]*dy[1])] +
               [(0, leg(j, y_mid)*dx[0]*dz[0], 0) for j in range(deg)] +
               [(leg(deg-1, y_mid)*dz[0]*dy[0]*dy[1], (deg+1)*leg(deg, y_mid)*dx[0]*dz[0], leg(deg-1, y_mid)*dx[0]*dy[0]*dy[1])] +
               [(0, leg(j, y_mid)*dx[0]*dz[1], 0) for j in range(deg)] +
               [(leg(deg-1, y_mid)*dz[1]*dy[0]*dy[1], (deg+1)*leg(deg, y_mid)*dx[0]*dz[1], leg(deg-1, y_mid)*dx[0]*dy[0]*dy[1])] +
               [(0, leg(j, y_mid)*dx[1]*dz[0], 0) for j in range(deg)] +
               [(leg(deg-1, y_mid)*dz[0]*dy[0]*dy[1], (deg+1)*leg(deg, y_mid)*dx[1]*dz[0], leg(deg-1, y_mid)*dx[1]*dy[0]*dy[1])] +
               [(0, leg(j, y_mid)*dx[1]*dz[1], 0) for j in range(deg)] +
               [(leg(deg-1, y_mid)*dz[1]*dy[0]*dy[1], (deg+1)*leg(deg, y_mid)*dx[1]*dz[1], leg(deg-1, y_mid)*dx[1]*dy[0]*dy[1])] +
               [(leg(j, x_mid)*dy[0]*dz[0], 0, 0) for j in range(deg)] +
               [((deg+1)*leg(deg, x_mid)*dy[0]*dz[0], leg(deg-1, x_mid)*dz[0]*dx[0]*dx[1], leg(deg-1, x_mid)*dy[0]*dx[0]*dx[1])] +
               [(leg(j, x_mid)*dy[0]*dz[1], 0, 0) for j in range(deg)] +
               [((deg+1)*leg(deg, x_mid)*dy[0]*dz[1], leg(deg-1, x_mid)*dz[1]*dx[0]*dx[1], leg(deg-1, x_mid)*dy[0]*dx[0]*dx[1])] +
               [(leg(j, x_mid)*dy[1]*dz[0], 0, 0) for j in range(deg)] +
               [((deg+1)*leg(deg, x_mid)*dy[1]*dz[0], leg(deg-1, x_mid)*dz[0]*dx[0]*dx[1], leg(deg-1, x_mid)*dy[1]*dx[0]*dx[1])] +
               [(leg(j, x_mid)*dy[1]*dz[1], 0, 0) for j in range(deg)] +
               [((deg+1)*leg(deg, x_mid)*dy[1]*dz[1], leg(deg-1, x_mid)*dz[1]*dx[0]*dx[1], leg(deg-1, x_mid)*dy[1]*dx[0]*dx[1])])

    return EL


def f_lambda_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    FL = tuple([(0, leg(j, y_mid)*leg(deg-2-j, z_mid)*dx[0]*dz[0]*dz[1], 0) for j in range(deg-1)] +
               [(0, 0, leg(j, z_mid)*leg(deg-2-j, y_mid)*dx[0]*dy[0]*dy[1]) for j in range(deg-1)] +
               [(leg(j-1, y_mid)*leg(deg-2-j, z_mid)*dy[0]*dy[1]*dz[0]*dz[1], (deg+1)*leg(j, y_mid)*leg(deg-2-j, z_mid)*dx[0]*dz[0]*dz[1], 0) for j in range(1, deg-1)] +
               [(leg(j-1, z_mid)*leg(deg-2-j, y_mid)*dy[0]*dy[1]*dz[0]*dz[1], 0, (deg+1)*leg(j, z_mid)*leg(deg-2-j, y_mid)*dx[0]*dy[0]*dy[1]) for j in range(1, deg-1)] +
               [(0, leg(j, y_mid)*leg(deg-2-j, z_mid)*dx[1]*dz[0]*dz[1], 0) for j in range(deg-1)] +
               [(0, 0, leg(j, z_mid)*leg(deg-2-j, y_mid)*dx[1]*dy[0]*dy[1]) for j in range(deg-1)] +
               [(leg(j-1, y_mid)*leg(deg-2-j, z_mid)*dy[0]*dy[1]*dz[0]*dz[1], (deg+1)*leg(j, y_mid)*leg(deg-2-j, z_mid)*dx[1]*dz[0]*dz[1], 0) for j in range(1, deg-1)] +
               [(leg(j-1, z_mid)*leg(deg-2-j, y_mid)*dy[0]*dy[1]*dz[0]*dz[1], 0, (deg+1)*leg(j, z_mid)*leg(deg-2-j, y_mid)*dx[1]*dy[0]*dy[1]) for j in range(1, deg-1)] +
               [(leg(j, x_mid)*leg(deg-2-j, z_mid)*dy[0]*dz[0]*dz[1], 0, 0) for j in range(deg-1)] +
               [(0, 0, leg(j, z_mid)*leg(deg-2-j, x_mid)*dy[0]*dx[0]*dx[1]) for j in range(deg-1)] +
               [((deg+1)*leg(j, x_mid)*leg(deg-2-j, z_mid)*dy[0]*dz[0]*dz[1], leg(j-1, x_mid)*leg(deg-2-j, z_mid)*dx[0]*dx[1]*dz[0]*dz[1], 0) for j in range(1, deg-1)] +
               [(0, leg(j-1, z_mid)*leg(deg-2-j, x_mid)*dx[0]*dx[1]*dz[0]*dz[1], (deg+1)*leg(j, z_mid)*leg(deg-2-j, x_mid)*dy[0]*dx[0]*dx[1]) for j in range(1, deg-1)] +
               [(leg(j, x_mid)*leg(deg-2-j, z_mid)*dy[1]*dz[0]*dz[1], 0, 0) for j in range(deg-1)] +
               [(0, 0, leg(j, z_mid)*leg(deg-2-j, x_mid)*dy[1]*dx[0]*dx[1]) for j in range(deg-1)] +
               [((deg+1)*leg(j, x_mid)*leg(deg-2-j, z_mid)*dy[1]*dz[0]*dz[1], leg(j-1, x_mid)*leg(deg-2-j, z_mid)*dx[0]*dx[1]*dz[0]*dz[1], 0) for j in range(1, deg-1)] +
               [(0, leg(j-1, z_mid)*leg(deg-2-j, x_mid)*dx[0]*dx[1]*dz[0]*dz[1], (deg+1)*leg(j, z_mid)*leg(deg-2-j, x_mid)*dy[1]*dx[0]*dx[1]) for j in range(1, deg-1)] +
               [(leg(j, x_mid)*leg(deg-2-j, y_mid)*dz[0]*dy[0]*dy[1], 0, 0) for j in range(deg-1)] +
               [(0, leg(j, y_mid)*leg(deg-2-j, x_mid)*dz[0]*dx[0]*dx[1], 0) for j in range(deg-1)] +
               [((deg+1)*leg(j, x_mid)*leg(deg-2-j, y_mid)*dz[0]*dy[0]*dy[1], 0, leg(j-1, x_mid)*leg(deg-2-j, y_mid)*dx[0]*dx[1]*dy[0]*dy[1]) for j in range(1, deg-1)] +
               [(0, (deg+1)*leg(j, y_mid)*leg(deg-2-j, x_mid)*dz[0]*dx[0]*dx[1], leg(j-1, y_mid)*leg(deg-2-j, x_mid)*dx[0]*dx[1]*dy[0]*dy[1]) for j in range(1, deg-1)] +
               [(leg(j, x_mid)*leg(deg-2-j, y_mid)*dz[1]*dy[0]*dy[1], 0, 0) for j in range(deg-1)] +
               [(0, leg(j, y_mid)*leg(deg-2-j, x_mid)*dz[1]*dx[0]*dx[1], 0) for j in range(deg-1)] +
               [((deg+1)*leg(j, x_mid)*leg(deg-2-j, y_mid)*dz[1]*dy[0]*dy[1], 0, leg(j-1, x_mid)*leg(deg-2-j, y_mid)*dx[0]*dx[1]*dy[0]*dy[1]) for j in range(1, deg-1)] +
               [(0, (deg+1)*leg(j, y_mid)*leg(deg-2-j, x_mid)*dz[1]*dx[0]*dx[1], leg(j-1, y_mid)*leg(deg-2-j, x_mid)*dx[0]*dx[1]*dy[0]*dy[1]) for j in range(1, deg-1)])

    return FL
