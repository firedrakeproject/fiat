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
from FIAT.lagrange import Lagrange
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis
from FIAT.serendipity import tr

x, y, z = symbols('x y z')
variables = (x, y, z)
leg = legendre


class BrezziDouglasMariniCubeEdge(FiniteElement):
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("BDMce_k elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("BDMce_k elements only valid for dimension 2")

        flat_topology = flat_el.get_topology()

        verts = flat_el.get_vertices()

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])

        EL = e_lambda_1_2d(degree, dx, dy, x_mid, y_mid)
        FL = f_lambda_1_2d(degree, dx, dy, x_mid, y_mid)
        bdmce_list = ET + FL

        entity_ids = {}
        cur = 0

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        for j in sorted(flat_topology[1]):
            entity_ids[1][j] = list(range(cur, cur + degree + 1))
            cur = cur + degree + 1

        entity_ids[2][0] = list(range(cur, cur + len(FL)))
        cur += len(FL)

        assert len(bdmce_list) == cur


def e_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    EL = tuple([(0, y_mid**j*dx[0]) for j in range(deg)] +
               [(y_mid**(r-1)*dy[0]*dy[1], (r+1)*y_mid**r*dx[0])] +
               [(0, y_mid**j*dx[1]) for j in range(deg)] +
               [(y_mid**(r-1)*dy[0]*dy[1], (r+1)*y_mid**r*dx[1])] +
               [(x_mid**j*dy[0], 0) for j in range(deg)] +
               [((r+1)*x_mid**r*dy[0], x_mid**(r-1)*dx[0]*dx[1])] +
               [(x_mid**j*dy[1], 0) for j in range(deg)] +
               [((r+1)*x_mid**r*dy[1], x_mid**(r-1)*dx[0]*dx[1])])

    return EL


def f_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    FL = tuple([(x_mid**j*y_mid**(deg-2-j)*dy[0]*dy[1], 0) for j in range(2, deg+1)] +
               [(0, x_mid**j*y_mid**(deg-2-j)*dx[0]*dx[1]) for j in range(2, deg+1)])

    return FL
