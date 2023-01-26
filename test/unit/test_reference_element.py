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
from math import isclose

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import Point, TensorProductCell, UFCQuadrilateral, UFCHexahedron

point = Point()
interval = UFCInterval()
triangle = UFCTriangle()
quadrilateral = UFCQuadrilateral()
hexahedron = UFCHexahedron()
tetrahedron = UFCTetrahedron()
interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)

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
                         [(interval, [0.5], -0.5),
                          (interval, [0.0], 0.0),
                          (interval, [1.0], 0.0),
                          (interval, [-1e-12], 1e-12),
                          (interval, [1+1e-12], 1e-12),
                          (triangle, [0.25, 0.25], -0.25),
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
                          (quadrilateral, [0.5, 0.5], -0.5),
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
                          (quadrilateral, [1+1e-12, 1+1e-12], 1e-12),
                          (tetrahedron, [0.25, 0.25, 0.25], -0.25),
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
                          (interval_x_interval, [0.5, 0.5], -0.5),
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
                          (interval_x_interval, [1+1e-12, 1+1e-12], 1e-12),
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
                          (quadrilateral_x_interval, [0.5, 0.5, 0.5], -0.5),
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
    if expected < 0:  # Only promise to give negative value
        assert cell.distance_to_point_l1(point) < 0
    else:
        assert isclose(cell.distance_to_point_l1(point), expected, rel_tol=1e-3)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
