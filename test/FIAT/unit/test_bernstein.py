# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with FIAT.  If not, see <https://www.gnu.org/licenses/>.

import math

import numpy
import pytest

from FIAT.reference_element import ufc_simplex
from FIAT.bernstein import Bernstein, BernsteinPolynomialSet
from FIAT.macro import AlfeldSplit, CkPolynomialSet
from FIAT.quadrature_schemes import create_quadrature


D02 = numpy.array([
    [0.65423405, 1.39160021, 0.65423405, 3.95416573, 1.39160021, 3.95416573],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [0., 0., 0., 0., 0., 0.],
    [-7.90833147, -7.90833147, -2.78320042, -2.78320042, -1.30846811, -1.30846811],
    [0.0831321, -2.12896637, 2.64569763, -7.25409741, 1.17096531, -6.51673126],
    [-2.12896637, 0.0831321, -7.25409741, 2.64569763, -6.51673126, 1.17096531],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
])

D11 = numpy.array([
    [0.65423405, 1.39160021, 0.65423405, 3.95416573, 1.39160021, 3.95416573],
    [3.29993168, 2.56256552, 0.73736616, -2.56256552, -0.73736616, -3.29993168],
    [-3.95416573, -3.95416573, -1.39160021, -1.39160021, -0.65423405, -0.65423405],
    [-4.69153189, -3.21679958, -4.69153189, 1.90833147, -3.21679958, 1.90833147],
    [0.73736616, -0.73736616, 3.29993168, -3.29993168, 2.56256552, -2.56256552],
    [-1.39160021, -0.65423405, -3.95416573, -0.65423405, -3.95416573, -1.39160021],
    [0., 0., 0., 0., 0., 0.],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
    [0., 0., 0., 0., 0., 0.],
])

D20 = numpy.array([
    [0.65423405, 1.39160021, 0.65423405, 3.95416573, 1.39160021, 3.95416573],
    [2.64569763, 1.17096531, 0.0831321, -6.51673126, -2.12896637, -7.25409741],
    [-7.25409741, -6.51673126, -2.12896637, 1.17096531, 0.0831321, 2.64569763],
    [-2.78320042, -1.30846811, -7.90833147, -1.30846811, -7.90833147, -2.78320042],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
    [0., 0., 0., 0., 0., 0.],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
])


def test_bernstein_2nd_derivatives():
    ref_el = ufc_simplex(2)
    degree = 3

    elem = Bernstein(ref_el, degree)
    rule = create_quadrature(ref_el, degree)
    points = rule.get_points()

    actual = elem.tabulate(2, points)

    assert numpy.allclose(D02, actual[(0, 2)])
    assert numpy.allclose(D11, actual[(1, 1)])
    assert numpy.allclose(D20, actual[(2, 0)])


@pytest.mark.parametrize("dim, degree",
                         [(dim, degree) for dim in (1, 2, 3) for degree in range(1, 2*dim + 1)])
def test_bernstein_dual_basis(dim, degree):
    elem = Bernstein(ufc_simplex(dim), degree)
    poly_set = elem.get_nodal_basis()
    dualmat = elem.get_dual_set().to_riesz(poly_set)

    assert numpy.allclose(dualmat @ poly_set.get_coeffs().T, numpy.eye(elem.space_dimension()))


@pytest.mark.parametrize("dim, degree",
                         [(dim, degree) for dim in (2, 3) for degree in range(1, 2*dim + 1)])
def test_bernstein_c0_alfeld(dim, degree):
    ref_el = AlfeldSplit(ufc_simplex(dim))
    poly_set = BernsteinPolynomialSet(ref_el, degree, order=0)
    points = create_quadrature(ref_el, 2*degree).get_points()

    expected_dimension = sum(
        math.comb(degree - 1, dim) * len(ref_el.get_topology()[dim])
        for dim in ref_el.get_topology()
    )
    values = poly_set.tabulate(points)[(0,) * dim]
    c0_values = CkPolynomialSet(ref_el, degree, order=0).tabulate(points)[(0,) * dim]
    rank = numpy.linalg.matrix_rank

    assert poly_set.get_num_members() == expected_dimension
    assert numpy.allclose(values.sum(axis=0), 1)
    assert rank(values) == rank(c0_values) == rank(numpy.vstack([values, c0_values]))


@pytest.mark.parametrize("dim, variant, degree",
                         [(dim, variant, degree)
                          for dim in (1, 2, 3)
                          for variant in ("alfeld", "iso", "powell-sabin", "worsey-farin")
                          for degree in range(1, 2*dim + 1)
                          if (dim, variant) != (1, "worsey-farin")])
def test_bernstein_macro_variant(dim, variant, degree):
    ref_el = ufc_simplex(dim)
    elem = Bernstein(ref_el, degree, variant=variant)
    ref_complex = elem.get_reference_complex()
    poly_set = elem.get_nodal_basis()
    dualmat = elem.get_dual_set().to_riesz(poly_set)
    assert elem.is_macroelement()
    assert numpy.allclose(dualmat @ poly_set.get_coeffs().T, numpy.eye(elem.space_dimension()))

    points = create_quadrature(ref_complex, 2*degree).get_points()
    values = elem.tabulate(0, points)[(0,) * dim]
    c0_values = CkPolynomialSet(ref_complex, degree, order=0).tabulate(points)[(0,) * dim]
    rank = numpy.linalg.matrix_rank
    assert rank(values) == rank(c0_values) == rank(numpy.vstack([values, c0_values]))

    for facet, dofs in elem.entity_closure_dofs()[dim - 1].items():
        facet_values = elem.tabulate(0, ref_el.make_points(dim - 1, facet, degree + 3))[(0,) * dim]
        assert numpy.allclose(numpy.delete(facet_values, dofs, axis=0), 0)


@pytest.mark.parametrize("degree, expected_dimension", ((1, 3), (2, 6), (3, 12)))
def test_bernstein_c1_alfeld(degree, expected_dimension):
    ref_el = AlfeldSplit(ufc_simplex(2))
    poly_set = BernsteinPolynomialSet(ref_el, degree, order=1)

    assert poly_set.get_num_members() == expected_dimension


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
