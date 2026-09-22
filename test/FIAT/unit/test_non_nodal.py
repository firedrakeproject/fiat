import numpy

from FIAT import NonNodalElement, ufc_simplex
from FIAT.dual_set import DualSet
from FIAT.functional import PointEvaluation
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.reference_element import make_lattice


def test_non_nodal_element_preserves_polynomial_basis():
    """Recombine the dual while retaining a prescribed polynomial basis."""
    ref_el = ufc_simplex(1)
    poly_set = ONPolynomialSet(ref_el, 2)
    points = make_lattice(ref_el.vertices, 2, variant="gll")
    nodes = [PointEvaluation(ref_el, point) for point in points]
    entity_ids = {0: {0: [0], 1: [1]}, 1: {0: [2]}}
    dual = DualSet(nodes, ref_el, entity_ids)

    raw_riesz = dual.to_riesz(poly_set)
    element = NonNodalElement(poly_set, dual, 2)

    assert all(new is old for new, old in zip(element.dual.nodes, dual.nodes))
    assert numpy.array_equal(element.get_coeffs(), poly_set.get_coeffs())
    assert not numpy.allclose(numpy.dot(raw_riesz, poly_set.get_coeffs().T), numpy.eye(3))
    assert numpy.allclose(element.dual.get_coeffs(), numpy.linalg.inv(
        numpy.dot(raw_riesz, poly_set.get_coeffs().T)))


def test_polynomial_set_recombine():
    """Recombine polynomial members while retaining their metadata."""
    ref_el = ufc_simplex(1)
    poly_set = ONPolynomialSet(ref_el, 2)
    coefficients = numpy.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    new_coeffs = numpy.dot(coefficients, poly_set.get_coeffs())

    recombined = poly_set.recombine(new_coeffs)

    assert len(recombined) == 2
    assert recombined.get_reference_element() is poly_set.get_reference_element()
    assert recombined.get_expansion_set() is poly_set.get_expansion_set()
    assert numpy.array_equal(recombined.get_coeffs(), new_coeffs)
