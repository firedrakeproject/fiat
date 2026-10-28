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
import itertools

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import Point, TensorProductCell, UFCQuadrilateral, UFCHexahedron
from FIAT.reference_element import is_ufc, is_hypercube, default_simplex, flatten_reference_cube, Hypercube
from FIAT.reference_element import flatten_permutations
from FIAT.orientation_utils import make_entity_permutations_simplex, make_entity_permutations_tensorproduct

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


def _vertex_index(vertices, x):
    """Return the index of vertex ``x`` in the list ``vertices``."""
    index, = np.flatnonzero(np.all(np.isclose(vertices, x), axis=1))
    return index


def _vertex_permutations(cell):
    """Return FIAT's vertex permutation for each orientation of the cell."""
    dim = cell.get_spatial_dimension()
    if cell.is_simplex():
        return make_entity_permutations_simplex(dim, 2)
    factors = cell.product.cells
    factor_dims = [c.get_spatial_dimension() for c in factors]
    permutations = make_entity_permutations_tensorproduct(
        factors, factor_dims, [make_entity_permutations_simplex(d, 2) for d in factor_dims])
    return flatten_permutations({tuple(factor_dims): {0: permutations}})[dim][0]


@pytest.mark.parametrize(('cell', 'num_orientations'),
                         [(point, 1), (interval, 2),
                         (triangle, 6), (tetrahedron, 24),
                         (quadrilateral, 8), (hexahedron, 48),])
def test_orientation_coord_maps(cell, num_orientations):
    """Check that each orientation map takes the vertices of the cell to a
    permutation of its vertices."""
    coord_maps = cell.get_orientation_coord_maps()
    assert sorted(coord_maps) == list(range(num_orientations))

    vertices = np.asarray(cell.get_vertices(), dtype=float)
    permutations = set()
    for o, (Q, t) in coord_maps.items():
        mapped_verts = vertices @ Q.T + t
        p = tuple(_vertex_index(vertices, x) for x in mapped_verts)
        assert sorted(p) == list(range(len(vertices)))
        permutations.add(p)
    assert len(permutations) == num_orientations


@pytest.mark.parametrize('cell', [point, interval, triangle, tetrahedron, quadrilateral, hexahedron])
def test_orientation_coord_maps_follow_vertex_permutations(cell):
    """Check that the map of orientation ``o`` takes vertex ``i`` to vertex ``p[i]``,
    where ``p`` is FIAT's vertex permutation of orientation ``o``."""
    vertices = np.asarray(cell.get_vertices(), dtype=float)
    maps = cell.get_orientation_coord_maps()
    for o, p in _vertex_permutations(cell).items():
        Q, t = maps[o]
        assert np.allclose(vertices @ Q.T + t, vertices[list(p)])


@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron, quadrilateral, hexahedron])
def test_facet_embedding_coord_maps(cell):
    """Check that each facet embedding agrees with get_entity_transform i.e., it takes
    the facet vertices to the cell vertices of that facet."""
    dim = cell.get_spatial_dimension()
    facet_dim = dim - 1
    topology = cell.get_topology()[facet_dim]
    facet_vertices = np.asarray(cell.construct_subelement(facet_dim).get_vertices(), dtype=float)
    A, b, A_inv = cell.facet_coord_embedding_maps
    assert A.shape == (len(topology), dim, facet_dim)
    assert b.shape == (len(topology), dim)
    assert A_inv.shape == (len(topology), facet_dim, dim)

    X_f = np.random.default_rng(0).random((5, facet_dim))
    for lf, facet in topology.items():
        transform = cell.get_entity_transform(facet_dim, lf)
        assert np.allclose(X_f @ A[lf].T + b[lf], transform(X_f))
        assert np.allclose(facet_vertices @ A[lf].T + b[lf], cell.get_vertices_of_subcomplex(facet))
        assert np.allclose(A_inv[lf] @ A[lf], np.eye(facet_dim))


@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron, quadrilateral, hexahedron])
def test_facet_adjacent_cell_maps_vertices(cell):
    """Check that each map takes the vertices of one facet to the vertices of
    the other facet, and that the map in the opposite direction undoes it."""
    dim = cell.get_spatial_dimension()
    vertices = np.asarray(cell.get_vertices(), dtype=float)
    topology = cell.get_topology()[dim - 1]
    A, b = cell.facet_adjacent_cell_maps
    num_facets, _, num_orientations = A.shape[:3]
    combinations = itertools.product(range(num_facets), range(num_facets),
                                     range(num_orientations), range(num_orientations))
    for lf_from, lf_to, o_from, o_to in combinations:
        X = vertices[list(topology[lf_from])]
        X_to = X @ A[lf_from, lf_to, o_from, o_to].T + b[lf_from, lf_to, o_from, o_to]
        assert {_vertex_index(vertices, x) for x in X_to} == set(topology[lf_to])
        X_back = X_to @ A[lf_to, lf_from, o_to, o_from].T + b[lf_to, lf_from, o_to, o_from]
        assert np.allclose(X_back, X)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
