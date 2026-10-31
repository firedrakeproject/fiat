"""Polynomial reproduction and exactness of the H1(curl) complex."""

import numpy
import pytest

from FIAT import ReducedFuGuzmanNeilanH1curl, Walkington, ufc_simplex
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


def test_reduced_complex(cell):
    element = ReducedFuGuzmanNeilanH1curl(cell)
    assert element.space_dimension() == 86
    assert sum(len(v) for v in element.entity_dofs()[0].values()) == 48
    space = element.get_nodal_basis().take(list(range(54)))
    points = create_quadrature(space.ref_el, 8).get_points()
    table = space.tabulate(points, 1)
    values = table[(0, 0, 0)]
    contains(values, ONPolynomialSet(cell, 2, shape=(3,)).tabulate(points)[(0, 0, 0)])

    scalar = Walkington(cell, reduced=True).get_nodal_basis().take(list(range(40)))
    scalar_table = scalar.tabulate(points, 1)
    gradient = numpy.stack([scalar_table[alpha] for alpha in mis(3, 1)], axis=1)
    contains(values, gradient)
    assert numpy.linalg.matrix_rank(gradient.reshape(40, -1), tol=1e-8) == 39

    curls = curl(table)
    gn = GuzmanNeilanSpace(cell, 1, reduced=True).tabulate(points)[(0, 0, 0)]
    contains(gn, curls)
    assert numpy.linalg.matrix_rank(curls.reshape(54, -1), tol=1e-8) == 15

    # Face traces of both v and curl(v) depend only on face closure dofs.
    rule = create_quadrature(cell.construct_subelement(2), 8)
    for face, closure in element.entity_closure_dofs()[2].items():
        other = sorted(set(range(54)) - set(closure))
        face_points = FacetQuadratureRule(cell, 2, face, rule).get_points()
        face_table = space.tabulate(face_points, 1)
        assert numpy.allclose(face_table[(0, 0, 0)][other], 0, atol=2e-9)
        assert numpy.allclose(curl(face_table)[other], 0, atol=2e-9)

    # Nodal unisolvence includes all auxiliary constraints used by FInAT.
    riesz = element.dual.to_riesz(element.get_nodal_basis())
    matrix = numpy.tensordot(riesz, element.get_nodal_basis().coeffs, axes=((1, 2), (1, 2)))
    assert numpy.allclose(matrix, numpy.eye(86), atol=1e-9)


def test_invalid_h1curl():
    with pytest.raises(ValueError, match="tetrahedra"):
        ReducedFuGuzmanNeilanH1curl(ufc_simplex(2))
    with pytest.raises(ValueError, match="degree four"):
        ReducedFuGuzmanNeilanH1curl(ufc_simplex(3), 3)
