# Copyright (C) 2024 Pablo Brubeck
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
import numpy
import sympy

from FIAT import expansions, polynomial_set, reference_element
from FIAT.quadrature_schemes import create_quadrature
from itertools import chain


@pytest.fixture(params=(1, 2, 3))
def cell(request):
    dim = request.param
    return reference_element.default_simplex(dim)


@pytest.mark.parametrize("degree", [10])
def test_expansion_values(cell, degree):
    dim = cell.get_spatial_dimension()
    U = expansions.ExpansionSet(cell)
    dpoints = []
    rpoints = []

    numpyoints = 4
    interior = 1
    for alpha in reference_element.lattice_iter(interior, numpyoints+1-interior, dim):
        dpoints.append(tuple(2*numpy.array(alpha, dtype="d")/numpyoints-1))
        rpoints.append(tuple(2*sympy.Rational(a, numpyoints)-1 for a in alpha))

    Uvals = U.tabulate(degree, dpoints)
    idx = (lambda p: p, expansions.morton_index2, expansions.morton_index3)[dim-1]
    eta = sympy.DeferredVector("eta")
    half = sympy.Rational(1, 2)

    def duffy_coords(pt):
        if len(pt) == 1:
            return pt
        elif len(pt) == 2:
            eta0 = 2 * (1 + pt[0]) / (1 - pt[1]) - 1
            eta1 = pt[1]
            return eta0, eta1
        else:
            eta0 = 2 * (1 + pt[0]) / (-pt[1] - pt[2]) - 1
            eta1 = 2 * (1 + pt[1]) / (1 - pt[2]) - 1
            eta2 = pt[2]
            return eta0, eta1, eta2

    def basis(dim, p, q=0, r=0):
        if dim >= 1:
            f = sympy.jacobi(p, 0, 0, eta[0])
            f *= sympy.sqrt(half + p)
        if dim >= 2:
            f *= sympy.jacobi(q, 2*p+1, 0, eta[1]) * ((1 - eta[1])/2) ** p
            f *= sympy.sqrt(1 + p + q)
        if dim >= 3:
            f *= sympy.jacobi(r, 2*p+2*q+2, 0, eta[2]) * ((1 - eta[2])/2) ** (p+q)
            f *= sympy.sqrt(1 + half + p + q + r)
        return f

    def eval_basis(f, pt):
        return float(f.subs(dict(zip(eta, duffy_coords(pt)))))

    for i in range(degree + 1):
        for indices in polynomial_set.mis(dim, i):
            phi = basis(dim, *indices)
            exact = numpy.array([eval_basis(phi, r) for r in rpoints])
            uh = Uvals[idx(*indices)]
            assert numpy.allclose(uh, exact, atol=1E-14)


@pytest.mark.parametrize("degree", [10])
def test_expansion_orthonormality(cell, degree):
    U = expansions.ExpansionSet(cell)
    rule = create_quadrature(cell, 2*degree)
    phi = U.tabulate(degree, rule.pts)
    qwts = rule.get_weights()
    results = numpy.dot(numpy.multiply(phi, qwts), phi.T)
    assert numpy.allclose(results, numpy.diag(numpy.diag(results)))
    assert numpy.allclose(numpy.diag(results), 1.0)


@pytest.mark.parametrize("degree", [10])
def test_bubble_duality(cell, degree):
    sd = cell.get_spatial_dimension()
    B = polynomial_set.make_bubbles(cell, degree)

    Q = create_quadrature(cell, 2*B.degree - sd - 1)
    qpts, qwts = Q.get_points(), Q.get_weights()
    phi = B.tabulate(qpts)[(0,) * sd]
    phi_dual = phi / abs(phi[0])
    scale = 2 ** sd
    results = scale * numpy.dot(numpy.multiply(phi_dual, qwts), phi.T)
    assert numpy.allclose(results, numpy.diag(numpy.diag(results)))
    assert numpy.allclose(numpy.diag(results), 1.0)


@pytest.mark.parametrize("degree", [10])
def test_union_of_polysets(cell, degree):
    """ demonstrates that polysets don't need to have the same degree for union
    using RT space as an example"""

    sd = cell.get_spatial_dimension()
    k = degree
    vecPk = polynomial_set.ONPolynomialSet(cell, degree, (sd,))

    vec_Pkp1 = polynomial_set.ONPolynomialSet(cell, k + 1, (sd,), scale="orthonormal")

    dimPkp1 = expansions.polynomial_dimension(cell, k + 1)
    dimPk = expansions.polynomial_dimension(cell, k)
    dimPkm1 = expansions.polynomial_dimension(cell, k - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk_from_Pkp1 = vec_Pkp1.take(vec_Pk_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(cell, k + 1, scale="orthonormal")
    PkH = Pkp1.take(list(range(dimPkm1, dimPk)))

    Q = create_quadrature(cell, 2 * (k + 1))
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    PkH_at_Qpts = PkH.tabulate(Qpts)[(0,) * sd]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[(0,) * sd]
    x = Qpts.T
    PkHx_at_Qpts = PkH_at_Qpts[:, None, :] * x[None, :, :]
    PkHx_coeffs = numpy.dot(numpy.multiply(PkHx_at_Qpts, Qwts), Pkp1_at_Qpts.T)
    PkHx = polynomial_set.PolynomialSet(cell, k, k + 1, vec_Pkp1.get_expansion_set(), PkHx_coeffs)

    same_deg = polynomial_set.polynomial_set_union_normalized(vec_Pk_from_Pkp1, PkHx)
    different_deg = polynomial_set.polynomial_set_union_normalized(vecPk, PkHx)

    Q = create_quadrature(cell, 2*(degree))
    Qpts, _ = Q.get_points(), Q.get_weights()
    same_vals = same_deg.tabulate(Qpts)[(0,) * sd]
    diff_vals = different_deg.tabulate(Qpts)[(0,) * sd]
    assert numpy.allclose(same_vals - diff_vals, 0)
