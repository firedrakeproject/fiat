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


@pytest.mark.parametrize("dim, family, degree", [(dim, f, degree - 1 if f == "DG" else degree)
                                                 for f in ("CG", "DG")
                                                 for dim in range(1, 4)
                                                 for degree in range(1, 7)])
def test_hierarchical_basis_values(dim, family, degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import ufc_simplex, Legendre, IntegratedLegendre, make_quadrature

    s = ufc_simplex(dim)
    q = make_quadrature(s, degree+1)
    if family == "CG":
        fe = IntegratedLegendre(s, degree)
    else:
        fe = Legendre(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,)*dim]

    for test_degree in range(degree + 1):
        v = lambda x: sum(x)**test_degree
        coefs = [n(v) for n in fe.dual.nodes]
        integral = np.dot(coefs, np.dot(tab, q.wts))
        reference = q.integrate(v)
        assert np.allclose(integral, reference, rtol=1e-14)


@pytest.mark.parametrize("family, degree", [(f, degree - 1 if f == "DG" else degree)
                                            for f in ("CG", "DG")
                                            for degree in range(1, 7)])
def test_hierarchical_sparsity(family, degree):
    from FIAT import ufc_simplex, Legendre, IntegratedLegendre, make_quadrature

    s = ufc_simplex(1)
    q = make_quadrature(s, degree+1)
    if family == "CG":
        fe = IntegratedLegendre(s, degree)
        expected = [5 * min(degree, 3) + 3 * max(0, degree-3) - 1, degree + 3]
    else:
        fe = Legendre(s, degree)
        expected = [degree + 1]

    nnz = lambda A: A.size - np.sum(np.isclose(A, 0.0E0, rtol=1E-14))
    moments = lambda v, u: np.dot(np.multiply(v, q.get_weights()), u.T)
    tab = fe.tabulate(len(expected)-1, q.get_points())
    for k, ennz in enumerate(expected):
        A = sum(moments(tab[alpha], tab[alpha]) for alpha in tab if sum(alpha) == k)
        assert nnz(A) == ennz


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
