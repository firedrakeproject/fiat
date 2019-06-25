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
        ETL = e_tilda_lambda_1_2d(degree, dx, dy, x_mid, y_mid)


def e_lambda_1_2d(i, dx, dy, x_mid, y_mid):
    EL = tuple([(0, y_mid**j*a) for a in dx for j in range(i)] + [(x_mid**j*a, 0) for a in dy for j in range(i)])

    return EL

def e_tilda_lambda_1_2d(r, dx, dy, x_mid, y_mid):
    ETL = tuple([(y_mid**(r-1)*dy[0]*dy[1], (r+1)*y_mid**r*a) for a in dx] +
                [((r+1)*x_mid**r*a, x_mid**(r-1)*dx[0]*dx[1]) for a in dy])

    return ETL
