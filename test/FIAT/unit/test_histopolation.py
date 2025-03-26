# Copyright (C) 2025 Imperial College London and others
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
# Authors:
#
# Pablo Brubeck

import pytest
import numpy as np

from FIAT import ufc_simplex, Histopolation, GaussLobattoLegendre
from FIAT.barycentric_interpolation import get_lagrange_points
from FIAT.macro import IsoSplit
from FIAT.quadrature_schemes import create_quadrature


@pytest.fixture
def interval():
    return ufc_simplex(1)


@pytest.mark.parametrize("degree", range(7))
def test_gll_basis_values(interval, degree):
    """Ensure that integrating a simple monomial produces the expected results."""

    s = interval
    q = create_quadrature(s, 2*degree)
    fe = Histopolation(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,)]

    for test_degree in range(degree + 1):
        v = lambda x: sum(x)**test_degree
        coefs = [n(v) for n in fe.dual.nodes]
        integral = np.dot(coefs, np.dot(tab, q.wts))
        reference = q.integrate(v)
        assert np.allclose(integral, reference, rtol=1e-14)


@pytest.mark.parametrize("degree", range(4))
def test_histopolation_dofs(interval, degree):
    """Ensure that basis functions integrate to either 1 or 0 on the GLL subgrid."""

    fe = Histopolation(interval, degree)
    gll = GaussLobattoLegendre(interval, degree+1)
    pts = get_lagrange_points(gll.dual_basis())
    x = np.reshape(pts, (-1, ))

    macrocell = IsoSplit(interval, degree+1, variant="gll")
    assert np.allclose(macrocell.vertices, pts)

    Q = create_quadrature(macrocell, 2*degree)
    qpts = Q.get_points()
    qwts = Q.get_weights().reshape((degree+1, -1))

    tab = fe.tabulate(0, qpts)[(0,)]
    tab = tab.reshape((fe.space_dimension(), degree+1, -1))
    expected = np.eye(degree+1)
    for i in range(degree+1):
        result = np.dot(tab[:, i, :], qwts[i] / (x[i+1] - x[i]))
        assert np.allclose(result, expected[i])
