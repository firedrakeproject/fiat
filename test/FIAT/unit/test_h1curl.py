"""Polynomial reproduction and exactness of the H1(curl) complex."""

import numpy
import pytest

from FIAT import HuZhangZhang, ReducedFuGuzmanNeilanH1curl, Walkington, ufc_simplex
from FIAT.fu_guzman_neilan import PoincareSpace
from FIAT.guzman_neilan import GuzmanNeilanSpace
from FIAT.polynomial_set import ONPolynomialSet, mis
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


def curl(table):
    return numpy.stack([table[(0, 1, 0)][:, 2] - table[(0, 0, 1)][:, 1],
                        table[(0, 0, 1)][:, 0] - table[(1, 0, 0)][:, 2],
                        table[(1, 0, 0)][:, 1] - table[(0, 1, 0)][:, 0]], axis=1)


def contains(basis, values):
    a = basis.reshape(len(basis), -1).T
    b = values.reshape(len(values), -1).T
    coefficients = numpy.linalg.lstsq(a, b, rcond=None)[0]
    assert numpy.allclose(a @ coefficients, b, atol=2e-9)


@pytest.fixture(params=[False, True])
def cell(request):
    cell = ufc_simplex(3)
    if request.param:
        cell.vertices = ((0, 0, 0), (1., .1, -.37), (.01, .987, -.23), (-.1, -.2, 1.38))
    return cell


@pytest.mark.parametrize("order, dimension, reduced_dimension, scalar_dimension, gradient_rank, curl_rank",
                         [(2, 86, 54, 40, 39, 15), (3, 105, 77, 45, 44, 33)])
def test_reduced_complex(cell, order, dimension, reduced_dimension, scalar_dimension, gradient_rank, curl_rank):
    element = ReducedFuGuzmanNeilanH1curl(cell, order)
    assert element.space_dimension() == dimension
    assert element.degree() == 4
    assert sum(len(v) for v in element.entity_dofs()[0].values()) == 48
    space = element.get_nodal_basis().take(list(range(reduced_dimension)))
    points = create_quadrature(space.ref_el, 8).get_points()
    table = space.tabulate(points, 1)
    values = table[(0, 0, 0)]
    contains(values, ONPolynomialSet(cell, order, shape=(3,)).tabulate(points)[(0, 0, 0)])

    scalar = Walkington(cell, reduced=order == 2).get_nodal_basis().take(list(range(scalar_dimension)))
    scalar_table = scalar.tabulate(points, 1)
    gradient = numpy.stack([scalar_table[alpha] for alpha in mis(3, 1)], axis=1)
    contains(values, gradient)
    assert numpy.linalg.matrix_rank(gradient.reshape(scalar_dimension, -1), tol=1e-8) == gradient_rank

    curls = curl(table)
    gn = GuzmanNeilanSpace(cell, order - 1, reduced=True).tabulate(points)[(0, 0, 0)]
    contains(gn, curls)
    assert numpy.linalg.matrix_rank(curls.reshape(reduced_dimension, -1), tol=1e-8) == curl_rank

    # Face traces of both v and curl(v) depend only on face closure dofs.
    rule = create_quadrature(cell.construct_subelement(2), 8)
    for face, closure in element.entity_closure_dofs()[2].items():
        other = sorted(set(range(reduced_dimension)) - set(closure))
        face_points = FacetQuadratureRule(cell, 2, face, rule).get_points()
        face_table = space.tabulate(face_points, 1)
        assert numpy.allclose(face_table[(0, 0, 0)][other], 0, atol=2e-9)
        assert numpy.allclose(curl(face_table)[other], 0, atol=2e-9)

    # Nodal unisolvence includes all auxiliary constraints used by FInAT.
    riesz = element.dual.to_riesz(element.get_nodal_basis())
    matrix = numpy.tensordot(riesz, element.get_nodal_basis().coeffs, axes=((1, 2), (1, 2)))
    assert numpy.allclose(matrix, numpy.eye(dimension), atol=1e-9)


def test_poincare_space_grad_curl(cell):
    """The lowest order H(grad curl) space has dimension 3 + 15."""
    gn = GuzmanNeilanSpace(cell, 1, reduced=True)
    space = PoincareSpace(ONPolynomialSet(cell, 1), gn, ONPolynomialSet(cell, 0), 4)
    assert len(space) == 18

    points = create_quadrature(space.ref_el, 8).get_points()
    table = space.tabulate(points, 1)
    contains(table[(0, 0, 0)], ONPolynomialSet(cell, 0, shape=(3,)).tabulate(points)[(0, 0, 0)])
    curls = curl(table)
    contains(gn.tabulate(points)[(0, 0, 0)], curls)
    assert numpy.linalg.matrix_rank(curls.reshape(18, -1), tol=1e-8) == 15


def test_hgradcurl_complex(cell):
    element = HuZhangZhang(cell)
    assert element.space_dimension() == 26
    assert element.degree() == 4
    dim = 18
    space = element.get_nodal_basis().take(list(range(dim)))
    points = create_quadrature(space.ref_el, 8).get_points()
    table = space.tabulate(points, 1)
    values = table[(0, 0, 0)]

    # The space contains the lowest order Nedelec space
    contains(values, ONPolynomialSet(cell, 0, shape=(3,)).tabulate(points)[(0, 0, 0)])
    x = points - numpy.mean(cell.get_vertices(), axis=0)
    contains(values, numpy.stack([numpy.cross(e, x).T for e in numpy.eye(3)]))

    curls = curl(table)
    gn = GuzmanNeilanSpace(cell, 1, reduced=True).tabulate(points)[(0, 0, 0)]
    contains(gn, curls)
    assert numpy.linalg.matrix_rank(curls.reshape(dim, -1), tol=1e-8) == 15

    # The tangential trace of v and the trace of curl(v) depend only on the face closure dofs
    rule = create_quadrature(cell.construct_subelement(2), 8)
    for face, closure in element.entity_closure_dofs()[2].items():
        other = sorted(set(range(dim)) - set(closure))
        face_points = FacetQuadratureRule(cell, 2, face, rule).get_points()
        face_table = space.tabulate(face_points, 1)
        n = cell.compute_normal(face)
        tangential = numpy.cross(face_table[(0, 0, 0)][other].transpose(0, 2, 1), n)
        assert numpy.allclose(tangential, 0, atol=2e-9)
        assert numpy.allclose(curl(face_table)[other], 0, atol=2e-9)

    # The tangential component is constant on each edge
    line = create_quadrature(cell.construct_subelement(1), 8)
    for e in cell.get_topology()[1]:
        edge_points = FacetQuadratureRule(cell, 1, e, line).get_points()
        t = cell.compute_edge_tangent(e)
        component = numpy.tensordot(space.tabulate(edge_points)[(0, 0, 0)], t, axes=((1,), (0,)))
        assert numpy.allclose(component, component[:, :1], atol=2e-9)

    # Nodal unisolvence includes all auxiliary constraints used by FInAT
    riesz = element.dual.to_riesz(element.get_nodal_basis())
    matrix = numpy.tensordot(riesz, element.get_nodal_basis().coeffs, axes=((1, 2), (1, 2)))
    assert numpy.allclose(matrix, numpy.eye(26), atol=1e-9)
