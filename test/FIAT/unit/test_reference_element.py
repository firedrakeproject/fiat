# Copyright (C) 2016 Miklos Homolya
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

import pytest
import numpy as np
import sys

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import Point, TensorProductCell, UFCQuadrilateral, UFCHexahedron
from FIAT.reference_element import is_ufc, is_hypercube, default_simplex, flatten_reference_cube, Hypercube

point = Point()
interval = UFCInterval()
triangle = UFCTriangle()
quadrilateral = UFCQuadrilateral()
hexahedron = UFCHexahedron()
tetrahedron = UFCTetrahedron()
interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)

default_interval = default_simplex(1)
default_triangle = default_simplex(2)
default_tetrahedron = default_simplex(3)
default_interval_x_interval = TensorProductCell(default_interval, default_interval)
default_hypercube = Hypercube(2, default_interval_x_interval)

ufc_tetrahedron_21connectivity = [(0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5)]
ufc_hexahedron_21connectivity = [(0, 1, 4, 5), (2, 3, 6, 7), (0, 2, 8, 9),
                                 (1, 3, 10, 11), (4, 6, 8, 10), (5, 7, 9, 11)]


@pytest.mark.parametrize(('cell', 'connectivity'),
                         [(tetrahedron, ufc_tetrahedron_21connectivity),
                          (hexahedron, ufc_hexahedron_21connectivity),
                          pytest.param(triangle_x_interval, [], marks=pytest.mark.xfail),
                          pytest.param(quadrilateral_x_interval, [], marks=pytest.mark.xfail)])
def test_ufc_connectivity_21(cell, connectivity):
    """Check face-edge connectivity builds what UFC expects.
    This is only non-trivial case ; the rest is x-0 and D-x,
    see below."""
    assert cell.get_connectivity()[(2, 1)] == connectivity


@pytest.mark.parametrize('cell',
                         [point, interval, triangle, tetrahedron,
                          quadrilateral, hexahedron,
                          pytest.param(interval_x_interval, marks=pytest.mark.xfail),
                          pytest.param(triangle_x_interval, marks=pytest.mark.xfail),
                          pytest.param(quadrilateral_x_interval, marks=pytest.mark.xfail)])
def test_ufc_connectivity_x0(cell):
    """Check x-0 connectivity is just what get_topology gives"""
    for dim0 in range(cell.get_spatial_dimension()+1):
        connectivity = cell.get_connectivity()[(dim0, 0)]
        topology = cell.get_topology()[dim0]
        assert len(connectivity) == len(topology)
        assert all(connectivity[i] == t for i, t in topology.items())


@pytest.mark.parametrize('cell',
                         [point, interval, triangle, tetrahedron,
                          quadrilateral, hexahedron,
                          pytest.param(interval_x_interval, marks=pytest.mark.xfail),
                          pytest.param(triangle_x_interval, marks=pytest.mark.xfail),
                          pytest.param(quadrilateral_x_interval, marks=pytest.mark.xfail)])
def test_ufc_connectivity_Dx(cell):
    """Check D-x connectivity is just [(0,1,2,...)]"""
    D = cell.get_spatial_dimension()
    for dim1 in range(D+1):
        connectivity = cell.get_connectivity()[(D, dim1)]
        assert len(connectivity) == 1
        assert connectivity[0] == tuple(range(len(connectivity[0])))


@pytest.mark.parametrize(('cell', 'volume'),
                         [pytest.param(point, 1, marks=pytest.mark.xfail(conditional=sys.version_info < (3, 6))),
                          (interval, 1),
                          (triangle, 1/2),
                          (quadrilateral, 1),
                          (tetrahedron, 1/6),
                          (interval_x_interval, 1),
                          (triangle_x_interval, 1/2),
                          (quadrilateral_x_interval, 1),
                          (hexahedron, 1)])
def test_volume(cell, volume):
    assert np.allclose(volume, cell.volume())


@pytest.mark.parametrize(('cell', 'normals'),
                         [(interval, [[-1],
                                      [1]]),
                          (triangle, [[1, 1],
                                      [-1, 0],
                                      [0, -1]]),
                          (quadrilateral, [[-1, 0],
                                           [1, 0],
                                           [0, -1],
                                           [0, 1]]),
                          (tetrahedron, [[1, 1, 1],
                                         [-1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, -1]]),
                          (hexahedron, [[-1, 0, 0],
                                        [1, 0, 0],
                                        [0, -1, 0],
                                        [0, 1, 0],
                                        [0, 0, -1],
                                        [0, 0, 1]])])
def test_reference_normal(cell, normals):
    facet_dim = cell.get_spatial_dimension() - 1
    for facet_number in range(len(cell.get_topology()[facet_dim])):
        assert np.allclose(normals[facet_number],
                           cell.compute_reference_normal(facet_dim, facet_number))


@pytest.mark.parametrize('cell',
                         [interval_x_interval,
                          triangle_x_interval,
                          quadrilateral_x_interval])
def test_reference_normal_horiz(cell):
    dim = cell.get_spatial_dimension()
    np.allclose((0,) * (dim - 1) + (-1,),
                cell.compute_reference_normal((dim - 1, 0), 0))  # bottom facet
    np.allclose((0,) * (dim - 1) + (1,),
                cell.compute_reference_normal((dim - 1, 0), 1))  # top facet


@pytest.mark.parametrize(('cell', 'normals'),
                         [(interval_x_interval, [[-1, 0],
                                                 [1, 0]]),
                          (triangle_x_interval, [[1, 1, 0],
                                                 [-1, 0, 0],
                                                 [0, -1, 0]]),
                          (quadrilateral_x_interval, [[-1, 0, 0],
                                                      [1, 0, 0],
                                                      [0, -1, 0],
                                                      [0, 1, 0]])])
def test_reference_normal_vert(cell, normals):
    dim = cell.get_spatial_dimension()
    vert_dim = (dim - 2, 1)
    for facet_number in range(len(cell.get_topology()[vert_dim])):
        assert np.allclose(normals[facet_number],
                           cell.compute_reference_normal(vert_dim, facet_number))


@pytest.mark.parametrize(('cell', 'point', 'epsilon', 'expected'),
                         [(interval, [0.5], 0.0, True),
                          (interval, [0.0], 1e-14, True),
                          (interval, [1.0], 1e-14, True),
                          (interval, [-1e-12], 1e-11, True),
                          (interval, [1+1e-12], 1e-11, True),
                          (interval, [-1e-12], 1e-13, False),
                          (interval, [1+1e-12], 1e-13, False),
                          (triangle, [0.25, 0.25], 0.0, True),
                          (triangle, [0.0, 0.0], 1e-14, True),
                          (triangle, [1.0, 0.0], 1e-14, True),
                          (triangle, [0.0, 1.0], 1e-14, True),
                          (triangle, [0.5, 0.5], 1e-14, True),
                          (triangle, [-1e-12, 0.0], 1e-11, True),
                          (triangle, [1+1e-12, 0.0], 1e-11, True),
                          (triangle, [0.0, -1e-12], 1e-11, True),
                          (triangle, [0.0, 1+1e-12], 1e-11, True),
                          (triangle, [-1e-12, 0.0], 1e-13, False),
                          (triangle, [1+1e-12, 0.0], 1e-13, False),
                          (triangle, [0.0, -1e-12], 1e-13, False),
                          (triangle, [0.0, 1+1e-12], 1e-13, False),
                          (triangle, [0.5+1e-12, 0.5], 1e-13, False),
                          (triangle, [0.5, 0.5+1e-12], 1e-13, False),
                          (quadrilateral, [0.5, 0.5], 0.0, True),
                          (quadrilateral, [0.0, 0.0], 1e-14, True),
                          (quadrilateral, [1.0, 0.0], 1e-14, True),
                          (quadrilateral, [0.0, 1.0], 1e-14, True),
                          (quadrilateral, [1.0, 1.0], 1e-14, True),
                          (quadrilateral, [-1e-12, 0.5], 1e-11, True),
                          (quadrilateral, [1+1e-12, 0.5], 1e-11, True),
                          (quadrilateral, [0.5, -1e-12], 1e-11, True),
                          (quadrilateral, [0.5, 1+1e-12], 1e-11, True),
                          (quadrilateral, [-1e-12, 0.5], 1e-13, False),
                          (quadrilateral, [1+1e-12, 0.5], 1e-13, False),
                          (quadrilateral, [0.5, -1e-12], 1e-13, False),
                          (quadrilateral, [0.5, 1+1e-12], 1e-13, False),
                          (tetrahedron, [0.25, 0.25, 0.25], 0.0, True),
                          (tetrahedron, [1/3, 1/3, 1/3], 1e-14, True),
                          (tetrahedron, [0.0, 0.0, 0.0], 1e-14, True),
                          (tetrahedron, [1.0, 0.0, 0.0], 1e-14, True),
                          (tetrahedron, [0.0, 1.0, 0.0], 1e-14, True),
                          (tetrahedron, [0.0, 0.0, 1.0], 1e-14, True),
                          (tetrahedron, [0.0, 0.5, 0.5], 1e-14, True),
                          (tetrahedron, [0.5, 0.0, 0.5], 1e-14, True),
                          (tetrahedron, [0.5, 0.5, 0.0], 1e-14, True),
                          (tetrahedron, [-1e-12, 0.0, 0.0], 1e-11, True),
                          (tetrahedron, [1+1e-12, 0.0, 0.0], 1e-11, True),
                          (tetrahedron, [0.0, -1e-12, 0.0], 1e-11, True),
                          (tetrahedron, [0.0, 1+1e-12, 0.0], 1e-11, True),
                          (tetrahedron, [0.0, 0.0, -1e-12], 1e-11, True),
                          (tetrahedron, [0.0, 0.0, 1+1e-12], 1e-11, True),
                          (tetrahedron, [-1e-12, 0.0, 0.0], 1e-13, False),
                          (tetrahedron, [1+1e-12, 0.0, 0.0], 1e-13, False),
                          (tetrahedron, [0.0, -1e-12, 0.0], 1e-13, False),
                          (tetrahedron, [0.0, 1+1e-12, 0.0], 1e-13, False),
                          (tetrahedron, [0.0, 0.0, -1e-12], 1e-13, False),
                          (tetrahedron, [0.0, 0.0, 1+1e-12], 1e-13, False),
                          (tetrahedron, [0.5+1e-12, 0.5, 0.5], 1e-13, False),
                          (tetrahedron, [0.5, 0.5+1e-12, 0.5], 1e-13, False),
                          (tetrahedron, [0.5, 0.5, 0.5+1e-12], 1e-13, False),
                          (hexahedron, [0.5, 0.5, 0.5], 0.0, True),
                          (hexahedron, [0.0, 0.0, 0.0], 1e-14, True),
                          (hexahedron, [1.0, 0.0, 0.0], 1e-14, True),
                          (hexahedron, [0.0, 1.0, 0.0], 1e-14, True),
                          (hexahedron, [0.0, 0.0, 1.0], 1e-14, True),
                          (hexahedron, [1.0, 1.0, 0.0], 1e-14, True),
                          (hexahedron, [1.0, 0.0, 1.0], 1e-14, True),
                          (hexahedron, [0.0, 1.0, 1.0], 1e-14, True),
                          (hexahedron, [1.0, 1.0, 1.0], 1e-14, True),
                          (hexahedron, [-1e-12, 0.5, 0.5], 1e-11, True),
                          (hexahedron, [0.5, -1e-12, 0.5], 1e-11, True),
                          (hexahedron, [0.5, 0.5, -1e-12], 1e-11, True),
                          (hexahedron, [1+1e-12, 0.5, 0.5], 1e-11, True),
                          (hexahedron, [0.5, 1+1e-12, 0.5], 1e-11, True),
                          (hexahedron, [0.5, 0.5, 1+1e-12], 1e-11, True),
                          (hexahedron, [-1e-12, 0.5, 0.5], 1e-13, False),
                          (hexahedron, [0.5, -1e-12, 0.5], 1e-13, False),
                          (hexahedron, [0.5, 0.5, -1e-12], 1e-13, False),
                          (hexahedron, [1+1e-12, 0.5, 0.5], 1e-13, False),
                          (hexahedron, [0.5, 1+1e-12, 0.5], 1e-13, False),
                          (hexahedron, [0.5, 0.5, 1+1e-12], 1e-13, False),
                          (interval_x_interval, [0.5, 0.5], 0.0, True),
                          (interval_x_interval, [0.0, 0.0], 1e-14, True),
                          (interval_x_interval, [1.0, 0.0], 1e-14, True),
                          (interval_x_interval, [0.0, 1.0], 1e-14, True),
                          (interval_x_interval, [1.0, 1.0], 1e-14, True),
                          (interval_x_interval, [-1e-12, 0.5], 1e-11, True),
                          (interval_x_interval, [1+1e-12, 0.5], 1e-11, True),
                          (interval_x_interval, [0.5, -1e-12], 1e-11, True),
                          (interval_x_interval, [0.5, 1+1e-12], 1e-11, True),
                          (interval_x_interval, [-1e-12, 0.5], 1e-13, False),
                          (interval_x_interval, [1+1e-12, 0.5], 1e-13, False),
                          (interval_x_interval, [0.5, -1e-12], 1e-13, False),
                          (interval_x_interval, [0.5, 1+1e-12], 1e-13, False),
                          (triangle_x_interval, [0.25, 0.25, 0.5], 0.0, True),
                          (triangle_x_interval, [0.0, 0.0, 0.0], 1e-14, True),
                          (triangle_x_interval, [1.0, 0.0, 0.0], 1e-14, True),
                          (triangle_x_interval, [0.0, 1.0, 0.0], 1e-14, True),
                          (triangle_x_interval, [0.0, 0.0, 1.0], 1e-14, True),
                          (triangle_x_interval, [0.5, 0.5, 0.5], 1e-14, True),
                          (triangle_x_interval, [-1e-12, 0.0, 0.5], 1e-11, True),
                          (triangle_x_interval, [1+1e-12, 0.0, 0.5], 1e-11, True),
                          (triangle_x_interval, [0.0, -1e-12, 0.5], 1e-11, True),
                          (triangle_x_interval, [0.0, 1+1e-12, 0.5], 1e-11, True),
                          (triangle_x_interval, [0.0, 0.0, -1e-12], 1e-11, True),
                          (triangle_x_interval, [0.0, 0.0, 1+1e-12], 1e-11, True),
                          (triangle_x_interval, [-1e-12, 0.0, 0.5], 1e-13, False),
                          (triangle_x_interval, [1+1e-12, 0.0, 0.5], 1e-13, False),
                          (triangle_x_interval, [0.0, -1e-12, 0.5], 1e-13, False),
                          (triangle_x_interval, [0.0, 1+1e-12, 0.5], 1e-13, False),
                          (triangle_x_interval, [0.0, 0.0, -1e-12], 1e-13, False),
                          (triangle_x_interval, [0.0, 0.0, 1+1e-12], 1e-13, False),
                          (triangle_x_interval, [0.5+1e-12, 0.5, 0.5], 1e-13, False),
                          (triangle_x_interval, [0.5, 0.5+1e-12, 0.5], 1e-13, False),
                          (triangle_x_interval, [0.0, 0.0, -1e-12], 1e-13, False),
                          (triangle_x_interval, [0.0, 0.0, 1+1e-12], 1e-13, False),
                          (quadrilateral_x_interval, [0.5, 0.5, 0.5], 0.0, True),
                          (quadrilateral_x_interval, [0.0, 0.0, 0.0], 1e-14, True),
                          (quadrilateral_x_interval, [1.0, 0.0, 0.0], 1e-14, True),
                          (quadrilateral_x_interval, [0.0, 1.0, 0.0], 1e-14, True),
                          (quadrilateral_x_interval, [0.0, 0.0, 1.0], 1e-14, True),
                          (quadrilateral_x_interval, [-1e-12, 0.0, 0.0], 1e-11, True),
                          (quadrilateral_x_interval, [1+1e-12, 0.0, 0.0], 1e-11, True),
                          (quadrilateral_x_interval, [0.0, -1e-12, 0.0], 1e-11, True),
                          (quadrilateral_x_interval, [0.0, 1+1e-12, 0.0], 1e-11, True),
                          (quadrilateral_x_interval, [0.0, 0.0, -1e-12], 1e-11, True),
                          (quadrilateral_x_interval, [0.0, 0.0, 1+1e-12], 1e-11, True),
                          (quadrilateral_x_interval, [-1e-12, 0.0, 0.0], 1e-13, False),
                          (quadrilateral_x_interval, [1+1e-12, 0.0, 0.0], 1e-13, False),
                          (quadrilateral_x_interval, [0.0, -1e-12, 0.0], 1e-13, False),
                          (quadrilateral_x_interval, [0.0, 1+1e-12, 0.0], 1e-13, False),
                          (quadrilateral_x_interval, [0.0, 0.0, -1e-12], 1e-13, False),
                          (quadrilateral_x_interval, [0.0, 0.0, 1+1e-12], 1e-13, False)])
def test_contains_point(cell, point, epsilon, expected):
    assert cell.contains_point(point, epsilon) == expected


@pytest.mark.parametrize(('cell', 'point', 'expected'),
                         [(interval, [0.5], 0.0),
                          (interval, [0.0], 0.0),
                          (interval, [1.0], 0.0),
                          (interval, [-1e-12], 1e-12),
                          (interval, [1+1e-12], 1e-12),
                          (triangle, [0.25, 0.25], 0.0),
                          (triangle, [0.0, 0.0], 0.0),
                          (triangle, [1.0, 0.0], 0.0),
                          (triangle, [0.0, 1.0], 0.0),
                          (triangle, [0.5, 0.5], 0.0),
                          (triangle, [-1e-12, 0.0], 1e-12),
                          (triangle, [1+1e-12, 0.0], 1e-12),
                          (triangle, [0.0, -1e-12], 1e-12),
                          (triangle, [0.0, 1+1e-12], 1e-12),
                          (triangle, [0.5+1e-12, 0.5], 1e-12),
                          (triangle, [0.5, 0.5+1e-12], 1e-12),
                          (quadrilateral, [0.5, 0.5], 0.0),
                          (quadrilateral, [0.0, 0.0], 0.0),
                          (quadrilateral, [1.0, 0.0], 0.0),
                          (quadrilateral, [0.0, 1.0], 0.0),
                          (quadrilateral, [1.0, 1.0], 0.0),
                          (quadrilateral, [-1e-12, 0.5], 1e-12),
                          (quadrilateral, [1+1e-12, 0.5], 1e-12),
                          (quadrilateral, [0.5, -1e-12], 1e-12),
                          (quadrilateral, [0.5, 1+1e-12], 1e-12),
                          (quadrilateral, [-1e-12, 0.5], 1e-12),
                          (quadrilateral, [1+1e-12, 0.5], 1e-12),
                          (quadrilateral, [1+1e-12, 1+1e-12], 2e-12),
                          (tetrahedron, [0.25, 0.25, 0.25], 0.0),
                          (tetrahedron, [1/3, 1/3, 1/3], 0.0),
                          (tetrahedron, [0.0, 0.0, 0.0], 0.0),
                          (tetrahedron, [1.0, 0.0, 0.0], 0.0),
                          (tetrahedron, [0.0, 1.0, 0.0], 0.0),
                          (tetrahedron, [0.0, 0.0, 1.0], 0.0),
                          (tetrahedron, [0.0, 0.5, 0.5], 0.0),
                          (tetrahedron, [0.5, 0.0, 0.5], 0.0),
                          (tetrahedron, [0.5, 0.5, 0.0], 0.0),
                          (tetrahedron, [-1e-12, 0.0, 0.0], 1e-12),
                          (tetrahedron, [1+1e-12, 0.0, 0.0], 1e-12),
                          (tetrahedron, [0.0, -1e-12, 0.0], 1e-12),
                          (tetrahedron, [0.0, 1+1e-12, 0.0], 1e-12),
                          (tetrahedron, [0.0, 0.0, -1e-12], 1e-12),
                          (tetrahedron, [0.0, 0.0, 1+1e-12], 1e-12),
                          (tetrahedron, [1/3+1e-12, 1/3, 1/3], 1e-12),
                          (tetrahedron, [1/3, 1/3+1e-12, 1/3], 1e-12),
                          (tetrahedron, [1/3, 1/3, 1/3+1e-12], 1e-12),
                          (hexahedron, [0.5, 0.5, 0.5], 0.0),
                          (hexahedron, [0.0, 0.0, 0.0], 0.0),
                          (hexahedron, [1.0, 0.0, 0.0], 0.0),
                          (hexahedron, [0.0, 1.0, 0.0], 0.0),
                          (hexahedron, [0.0, 0.0, 1.0], 0.0),
                          (hexahedron, [1.0, 1.0, 0.0], 0.0),
                          (hexahedron, [1.0, 0.0, 1.0], 0.0),
                          (hexahedron, [0.0, 1.0, 1.0], 0.0),
                          (hexahedron, [1.0, 1.0, 1.0], 0.0),
                          (hexahedron, [-1e-12, 0.5, 0.5], 1e-12),
                          (hexahedron, [0.5, -1e-12, 0.5], 1e-12),
                          (hexahedron, [0.5, 0.5, -1e-12], 1e-12),
                          (hexahedron, [1+1e-12, 0.5, 0.5], 1e-12),
                          (hexahedron, [0.5, 1+1e-12, 0.5], 1e-12),
                          (hexahedron, [0.5, 0.5, 1+1e-12], 1e-12),
                          (hexahedron, [-1e-12, -1e-12, -1e-12], 3e-12),
                          (hexahedron, [1.0+1e-12, -1e-12, -1e-12], 3e-12),
                          (hexahedron, [-1e-12, 1.0+1e-12, -1e-12], 3e-12),
                          (hexahedron, [-1e-12, -1e-12, 1.0+1e-12], 3e-12),
                          (hexahedron, [1.0+1e-12, 1.0+1e-12, -1e-12], 3e-12),
                          (hexahedron, [1.0+1e-12, -1e-12, 1.0+1e-12], 3e-12),
                          (hexahedron, [-1e-12, 1.0+1e-12, 1.0+1e-12], 3e-12),
                          (hexahedron, [1.0+1e-12, 1.0+1e-12, 1.0+1e-12], 3e-12),
                          (interval_x_interval, [0.5, 0.5], 0.0),
                          (interval_x_interval, [0.0, 0.0], 0.0),
                          (interval_x_interval, [1.0, 0.0], 0.0),
                          (interval_x_interval, [0.0, 1.0], 0.0),
                          (interval_x_interval, [1.0, 1.0], 0.0),
                          (interval_x_interval, [-1e-12, 0.5], 1e-12),
                          (interval_x_interval, [1+1e-12, 0.5], 1e-12),
                          (interval_x_interval, [0.5, -1e-12], 1e-12),
                          (interval_x_interval, [0.5, 1+1e-12], 1e-12),
                          (interval_x_interval, [-1e-12, 0.5], 1e-12),
                          (interval_x_interval, [1+1e-12, 0.5], 1e-12),
                          (interval_x_interval, [1+1e-12, 1+1e-12], 2e-12),
                          (triangle_x_interval, [0.25, 0.25, 0.5], 0.0),
                          (triangle_x_interval, [0.0, 0.0, 0.0], 0.0),
                          (triangle_x_interval, [1.0, 0.0, 0.0], 0.0),
                          (triangle_x_interval, [0.0, 1.0, 0.0], 0.0),
                          (triangle_x_interval, [0.0, 0.0, 1.0], 0.0),
                          (triangle_x_interval, [0.5, 0.5, 0.5], 0.0),
                          (triangle_x_interval, [-1e-12, 0.0, 0.5], 1e-12),
                          (triangle_x_interval, [1+1e-12, 0.0, 0.5], 1e-12),
                          (triangle_x_interval, [0.0, -1e-12, 0.5], 1e-12),
                          (triangle_x_interval, [0.0, 1+1e-12, 0.5], 1e-12),
                          (triangle_x_interval, [0.0, 0.0, -1e-12], 1e-12),
                          (triangle_x_interval, [0.0, 0.0, 1+1e-12], 1e-12),
                          (quadrilateral_x_interval, [0.5, 0.5, 0.5], 0.0),
                          (quadrilateral_x_interval, [0.0, 0.0, 0.0], 0.0),
                          (quadrilateral_x_interval, [1.0, 0.0, 0.0], 0.0),
                          (quadrilateral_x_interval, [0.0, 1.0, 0.0], 0.0),
                          (quadrilateral_x_interval, [0.0, 0.0, 1.0], 0.0),
                          (quadrilateral_x_interval, [-1e-12, 0.0, 0.0], 1e-12),
                          (quadrilateral_x_interval, [1+1e-12, 0.0, 0.0], 1e-12),
                          (quadrilateral_x_interval, [0.0, -1e-12, 0.0], 1e-12),
                          (quadrilateral_x_interval, [0.0, 1+1e-12, 0.0], 1e-12),
                          (quadrilateral_x_interval, [0.0, 0.0, -1e-12], 1e-12),
                          (quadrilateral_x_interval, [0.0, 0.0, 1+1e-12], 1e-12)])
def test_distance_to_point_l1(cell, point, expected):
    assert np.isclose(cell.distance_to_point_l1(point), expected, rtol=1e-3)


@pytest.mark.parametrize(('cell', 'expected'),
                         [(interval, True),
                          (triangle, True),
                          (quadrilateral, True),
                          (tetrahedron, True),
                          (interval_x_interval, True),
                          (triangle_x_interval, True),
                          (quadrilateral_x_interval, True),
                          (hexahedron, True),
                          (default_interval, False),
                          (default_triangle, False),
                          (default_tetrahedron, False),
                          (default_interval_x_interval, False),
                          (default_hypercube, False),])
def test_is_ufc(cell, expected):
    assert is_ufc(cell) == expected


@pytest.mark.parametrize(('cell', 'expected'),
                         [(interval, True),
                          (triangle, False),
                          (quadrilateral, True),
                          (tetrahedron, False),
                          (interval_x_interval, True),
                          (triangle_x_interval, False),
                          (quadrilateral_x_interval, True),
                          (hexahedron, True),
                          (default_interval, True),
                          (default_triangle, False),
                          (default_tetrahedron, False),
                          (default_interval_x_interval, True),
                          (default_hypercube, True),])
def test_is_hypercube(cell, expected):
    assert is_hypercube(cell) == expected


@pytest.mark.parametrize(('cell'),
                         [interval,
                          quadrilateral,
                          interval_x_interval,
                          triangle_x_interval,
                          quadrilateral_x_interval,
                          hexahedron,
                          default_interval,
                          default_interval_x_interval,
                          default_hypercube])
def test_flatten_maintains_ufc_status(cell):
    ufc_status = is_ufc(cell)
    flat_cell = flatten_reference_cube(cell)
    assert ufc_status == is_ufc(flat_cell)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
