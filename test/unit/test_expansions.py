# Copyright (C) 2023 Pablo Brubeck
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
import sympy
import numpy
from FIAT import expansions, quadrature, polynomial_set, reference_element
from FIAT.reference_element import Point, UFCInterval, UFCTriangle, UFCTetrahedron

P = Point()
I = UFCInterval()  # noqa: E741
T = UFCTriangle()
S = UFCTetrahedron()


@pytest.mark.parametrize('cell', [I, T, S])
def test_expansion_orthonormality(cell):
    U = expansions.ExpansionSet(cell)
    degree = 10
    rule = quadrature.make_quadrature(cell, degree + 1)
    phi = U.tabulate(degree, rule.pts)
    w = rule.get_weights()
    scale = 0.5 ** -cell.get_spatial_dimension()
    results = scale * numpy.dot(phi, w[:, None] * phi.T)
    assert numpy.allclose(results, numpy.eye(results.shape[0]))


@pytest.mark.parametrize('dim', range(1, 4))
def test_expansion_values(dim):
    cell = reference_element.default_simplex(dim)
    U = expansions.ExpansionSet(cell)
    dpoints = []
    rpoints = []

    npoints = 4
    interior = 1
    for alpha in reference_element.lattice_iter(interior, npoints+1-interior, dim):
        dpoints.append(tuple(2*numpy.array(alpha, dtype="d")/npoints-1))
        rpoints.append(tuple(2*sympy.Rational(a, npoints)-1 for a in alpha))

    n = 16
    Uvals = U.tabulate(n, dpoints)
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
        fval = f
        for coord, pval in zip(eta, duffy_coords(pt)):
            fval = fval.subs(coord, pval)
        fval = float(fval)
        return fval

    for i in range(n + 1):
        for indices in polynomial_set.mis(dim, i):
            phi = basis(dim, *indices)
            exact = numpy.array([eval_basis(phi, r) for r in rpoints])
            uh = Uvals[idx(*indices)]
            assert numpy.allclose(uh, exact, atol=1E-14)


@pytest.mark.parametrize('cell', [I, T, S])
def test_expansion_derivatives_finite_differences(cell):
    dim = cell.get_spatial_dimension()
    U = expansions.ExpansionSet(cell)

    n = 10
    npoints = 10
    points = reference_element.make_lattice(cell.get_vertices(), npoints, variant="gl")
    points = numpy.array(points)


    vals, grad, hess = U.tabulate_jet(n, points, order=2)
    norm_grad = numpy.sqrt(vals**2 + numpy.linalg.norm(grad, axis=2)**2)
    norm_hess = numpy.sqrt(norm_grad**2 + numpy.linalg.norm(hess, "fro", axis=(2, 3))**2)
    norm_grad = numpy.max(norm_grad)
    norm_hess = numpy.max(norm_hess)
    eps = 1E-6
    print(eps*norm_grad, eps*norm_hess)

    hs = []
    errors_grad = []
    errors_hess = []
    for k in range(4):
        h = (1/n**2) * (0.5**k)
        hs.append(h)
        gradh = numpy.stack([(U.tabulate(n, points + dx[None,:]) -
                              U.tabulate(n, points - dx[None,:])) / h
                             for dx in (0.5*h)*numpy.eye(dim)], axis=2)
        errors_grad.append(numpy.maximum(eps*norm_grad*(h**2), numpy.linalg.norm(grad - gradh, axis=2)))

        hessh = numpy.stack([numpy.stack([
            (U.tabulate(n, points + (dx + dy)[None,:]) -
             U.tabulate(n, points + (dx - dy)[None,:]) -
             U.tabulate(n, points - (dx - dy)[None,:]) +
             U.tabulate(n, points - (dx + dy)[None,:])) / h**2
            for dx in (0.5*h)*numpy.eye(dim)], axis=2)
            for dy in (0.5*h)*numpy.eye(dim)], axis=3)
        errors_hess.append(numpy.maximum(eps*norm_hess*(h**2), numpy.linalg.norm(hess - hessh, "fro", axis=(2, 3))))

    rate_grad = numpy.diff(numpy.log(errors_grad), axis=0) / numpy.diff(numpy.log(hs))[:, None, None]
    assert numpy.all(rate_grad > 1.9)

    rate_hess = numpy.diff(numpy.log(errors_hess), axis=0) / numpy.diff(numpy.log(hs))[:, None, None]
    assert numpy.all(rate_hess > 1.9)
