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


@pytest.mark.parametrize("degree", range(1, 7))
@pytest.mark.parametrize("family", ["CG", "DG"])
def test_fdm_basis_values(family, degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import ufc_simplex, FDMLagrange, FDMDiscontinuousLagrange, make_quadrature

    s = ufc_simplex(1)
    q = make_quadrature(s, degree + 1)

    if family == "CG":
        fe = FDMLagrange(s, degree)
    else:
        fe = FDMDiscontinuousLagrange(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,)]

    for test_degree in range(degree + 1):
        coefs = [float(n(lambda x: x[0]**test_degree)) for n in fe.dual.nodes]
        integral = np.dot(coefs, np.dot(tab, q.wts))
        reference = np.dot([x[0]**test_degree
                            for x in q.pts], q.wts)
        assert np.allclose(integral, reference, rtol=1e-14)


@pytest.mark.parametrize("degree", range(1, 7))
def test_sparsity(degree):
    from FIAT import ufc_simplex, FDMLagrange, make_quadrature
    cell = ufc_simplex(1)
    fe = FDMLagrange(cell, degree)
    rule = make_quadrature(cell, degree+1)

    basis = fe.tabulate(1, rule.get_points())
    Jhat = basis[(0,)]
    Dhat = basis[(1,)]
    what = rule.get_weights()
    Ahat = np.dot(np.multiply(Dhat, what), Dhat.T)
    Bhat = np.dot(np.multiply(Jhat, what), Jhat.T)
    nnz = lambda A: A.size - np.sum(np.isclose(A, 0.0E0, rtol=1E-14))
    ndof = fe.space_dimension()
    assert nnz(Ahat) == 5*ndof-6
    assert nnz(Bhat) == ndof+2


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
