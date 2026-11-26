import numpy
import pytest

from FIAT import Hermite
from FIAT.polynomial_set import ONPolynomialSet, mis
from FIAT.reference_element import make_affine_mapping, ufc_simplex


@pytest.fixture(params=[2, 3])
def cell(request, geometry):
    cell = ufc_simplex(request.param)
    sd = cell.get_spatial_dimension()
    if geometry != "reference":
        A = numpy.array([[1.2, 0.3, -0.2], [0.1, 0.9, 0.4], [-0.3, 0.2, 1.4]])[:sd, :sd]
        if geometry == "reversed":
            A[[0, 1]] = A[[1, 0]]
        cell.vertices = tuple(map(tuple, numpy.array(cell.get_vertices()) @ A.T + 0.2))
    return cell


@pytest.mark.parametrize("geometry", ["reference", "affine", "reversed"])
def test_reduced_hermite_quadratic_interpolation(cell):
    element = Hermite(cell, reduced=True)
    sd = cell.get_spatial_dimension()
    assert element.space_dimension() == (sd + 1)**2
    assert all(len(ids) == sd + 1 for ids in element.entity_dofs()[0].values())
    assert all(not ids for dim, entities in element.entity_dofs().items()
               if dim > 0 for ids in entities.values())
    basis = element.get_nodal_basis()
    pairing = element.get_dual_set().to_riesz(basis) @ basis.get_coeffs().T
    assert numpy.allclose(pairing, numpy.eye(element.space_dimension()))

    quadratics = ONPolynomialSet(cell, 2)
    coefficients = element.get_dual_set().to_riesz(quadratics) @ quadratics.get_coeffs().T
    points = cell.make_points(sd, 0, sd + 5)
    expected = quadratics.tabulate(points, 1)
    actual = element.tabulate(1, points)
    for alpha in expected:
        assert numpy.allclose(coefficients.T @ actual[alpha], expected[alpha])


@pytest.mark.parametrize("geometry", ["reference", "affine", "reversed"])
def test_reduced_hermite_face_traces(cell):
    element = Hermite(cell, reduced=True)
    sd = cell.get_spatial_dimension()
    face_cell = cell.construct_subelement(2)
    face_element = Hermite(face_cell, reduced=True)
    face_points = face_cell.make_points(2, 0, 7)
    expected = face_element.tabulate(0, face_points)[(0, 0)]
    dofs = element.entity_dofs()[0]

    for face, vertices in cell.get_topology()[2].items():
        face_vertices = numpy.array(cell.get_vertices())[list(vertices)]
        A, b = make_affine_mapping(face_cell.get_vertices(), face_vertices)
        points = numpy.array(face_points) @ A.T + b
        actual = element.tabulate(0, points)[(0,) * sd]
        coefficients, _, rank, _ = numpy.linalg.lstsq(expected.T, actual.T, rcond=None)
        assert rank == 9
        assert numpy.allclose(expected.T @ coefficients, actual.T)
        assert numpy.linalg.matrix_rank(actual) == 9

        for vertex, ids in dofs.items():
            if vertex not in vertices:
                assert numpy.allclose(actual[ids], 0)
            elif sd == 3:
                normal = cell.compute_normal(face)
                directions = numpy.array([normal[alpha.index(1)] for alpha in mis(sd, 1)])
                assert numpy.allclose(directions @ actual[ids[1:]], 0)
