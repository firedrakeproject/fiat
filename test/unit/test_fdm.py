# Copyright (C) 2016 Imperial College London and others
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


def make_fdm_element(ref_el, family, degree):
    from FIAT import FDMLagrange, FDMDiscontinuousLagrange, FDMBrokenH1, FDMBrokenL2, FDMQuadrature
    if family == "CG":
        return FDMLagrange(ref_el, degree)
    elif family == "DG":
        return FDMDiscontinuousLagrange(ref_el, degree)
    elif family == "BrokenH1":
        return FDMBrokenH1(ref_el, degree)
    elif family == "BrokenL2":
        return FDMBrokenL2(ref_el, degree)
    elif family == "Quadrature":
        return FDMQuadrature(ref_el, degree)


@pytest.mark.parametrize("family, degree", [(f, degree - 1 if f in {"DG", "BrokenL2"} else degree)
                                            for f in ("CG", "DG", "BrokenH1", "BrokenL2", "Quadrature")
                                            for degree in range(1, 7)])
def test_fdm_basis_values(family, degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import ufc_simplex, make_quadrature

    s = ufc_simplex(1)
    q = make_quadrature(s, degree + 1)
    fe = make_fdm_element(s, family, degree)

    tab = fe.tabulate(0, q.pts)[(0,)]

    for test_degree in range(degree + 1):
        coefs = [float(n(lambda x: x[0]**test_degree)) for n in fe.dual.nodes]
        integral = np.dot(coefs, np.dot(tab, q.wts))
        reference = np.dot([x[0]**test_degree
                            for x in q.pts], q.wts)
        assert np.allclose(integral, reference, rtol=1e-14)


@pytest.mark.parametrize("family, degree", [(f, degree - 1 if f in {"DG", "BrokenL2"} else degree)
                                            for f in ("CG", "DG", "BrokenH1", "BrokenL2", "Quadrature")
                                            for degree in range(1, 7)])
def test_fdm_sparsity(family, degree):
    from FIAT import ufc_simplex, make_quadrature

    s = ufc_simplex(1)
    q = make_quadrature(s, degree+1)
    fe = make_fdm_element(s, family, degree)

    if family == "CG":
        expected = [degree + 3, 5 * degree - 1]
    elif family == "DG":
        expected = [degree + 1]
    elif family == "BrokenH1":
        expected = [degree + 1, degree]
    elif family == "BrokenL2":
        expected = [degree + 1]
    elif family == "Quadrature":
        expected = [degree + 1, 3 * degree - 1 - (degree == 1)]

    nnz = lambda A: A.size - np.sum(np.isclose(A, 0.0E0, rtol=1E-14))
    moments = lambda v, u: np.dot(np.multiply(v, q.get_weights()), u.T)
    tab = fe.tabulate(len(expected)-1, q.get_points())
    for k, ennz in enumerate(expected):
        assert nnz(moments(tab[(k, )], tab[(k, )])) == ennz


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
