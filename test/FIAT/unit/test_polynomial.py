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


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("variant", [None, "bubble"])
def test_high_order_expansion_derivatives(dim, variant):
    cell = reference_element.default_simplex(dim)
    degree = 5
    order = 4
    points = reference_element.make_lattice(cell.get_vertices(), 5, interior=1)

    fallback = expansions.ExpansionSet(cell, variant=variant)
    fallback.recurrence_order = 2
    expected = fallback._tabulate(degree, points, order=order)

    recurrence = expansions.ExpansionSet(cell, variant=variant)

    def get_dmats(*args, **kwargs):
        raise AssertionError("high-order derivatives should use recurrence tabulation")

    recurrence.get_dmats = get_dmats
    actual = recurrence._tabulate(degree, points, order=order)

    assert actual.keys() == expected.keys()
    for alpha in actual:
        assert numpy.allclose(actual[alpha], expected[alpha], atol=1E-10, rtol=1E-10)


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


def duffy_points(dim, etas):
    """Collapse a tensor-product grid of collapsed coordinates onto the
    default simplex."""
    grids = numpy.meshgrid(*etas, indexing="ij")
    flat = tuple(grid.ravel() for grid in grids)
    if dim == 2:
        flat = expansions.xi_triangle(flat)
    elif dim == 3:
        flat = expansions.xi_tetrahedron(flat)
    return numpy.stack(flat, axis=-1)


def duffy_term_value(factors, index):
    """Evaluate the outer product of the per-axis factors of a separable
    term for a given lattice multi-index."""
    vals = numpy.ones(())
    m = 0
    for table, i in zip(factors, index):
        vals = numpy.multiply.outer(vals, table[m, i])
        m += i
    return vals.ravel()


@pytest.mark.parametrize("degree", [0, 1, 4])
@pytest.mark.parametrize("variant", [None, "dual"])
@pytest.mark.parametrize("make_cell", [reference_element.default_simplex,
                                       reference_element.ufc_simplex])
def test_tabulate_duffy(make_cell, variant, degree):
    for dim in (1, 2, 3):
        cell = make_cell(dim)
        U = expansions.ExpansionSet(cell, variant=variant)
        # Unequal point counts per axis, including the collapsed vertex eta=1
        etas = [numpy.linspace(-1, 1, 3 + axis) for axis in range(dim)]
        A, b = U.affine_mappings[0]
        pts = numpy.linalg.solve(A, (duffy_points(dim, etas) - b).T).T
        expected = U._tabulate_on_cell(degree, pts, order=1)
        duffy = U.tabulate_duffy(degree, etas, order=1)
        assert expected.keys() == duffy.keys()

        idx = (lambda p: p, expansions.morton_index2, expansions.morton_index3)[dim-1]
        for alpha in expected:
            for index in reference_element.lattice_iter(0, degree+1, dim):
                vals = sum(coeff * duffy_term_value(factors, index)
                           for coeff, factors in duffy[alpha])
                assert numpy.allclose(vals, expected[alpha][idx(*index)], rtol=1E-10, atol=1E-10)


@pytest.mark.parametrize("degree", [0, 1, 4])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_morton_tables(dim, degree):
    """`expansions.morton_forward_table` / `expansions.morton_inverse_table`
    must agree with `expansions.morton_index` on the simplex lattice, and
    be mutual inverses there; this is the Morton dof numbering that
    `tsfc.fem`'s sum-factorized argument/coefficient translation relies
    on to gather/scatter without reordering FIAT's degrees of freedom.
    """
    import math
    fwd = expansions.morton_forward_table(dim, degree)
    inv = expansions.morton_inverse_table(dim, degree)
    ndof = math.comb(degree + dim, dim)
    assert fwd.shape == (degree + 1,) * dim
    assert inv.shape == (ndof, dim)

    seen = numpy.zeros(ndof, dtype=bool)
    for multiindex in reference_element.lattice_iter(0, degree + 1, dim):
        r = expansions.morton_index(dim, degree, *multiindex)
        assert fwd[multiindex] == r
        assert tuple(inv[r]) == multiindex
        seen[r] = True
    assert numpy.all(seen)


@pytest.mark.parametrize("degree", [4])
def test_principal_functions_bubble(cell, degree):
    dim = cell.get_spatial_dimension()
    variant = "bubble"
    etas = [numpy.linspace(-1, 1, 3 + axis) for axis in range(dim)]
    ref_pts = duffy_points(dim, etas).T
    scale = 1.0
    phi, = expansions.dubiner_recurrence(dim, degree, 0, ref_pts,
                                         numpy.eye(dim), scale, variant=variant)
    tables = [expansions.principal_functions(degree, etas[axis], axis + 1, variant=variant)
              for axis in range(dim)]

    idx = (lambda p: p, expansions.morton_index2, expansions.morton_index3)[dim-1]
    for index in reference_element.lattice_iter(0, degree+1, dim):
        vals = -scale * duffy_term_value([table["V"] for table in tables], index)
        assert numpy.allclose(vals, phi[idx(*index)], rtol=1E-10, atol=1E-10)
