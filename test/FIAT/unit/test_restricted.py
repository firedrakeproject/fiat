import numpy
import pytest

from FIAT import IntegratedLegendre, Lagrange, RestrictedElement, ufc_simplex
from FIAT.finite_element import FiniteElement
from FIAT.polynomial_set import PolynomialSet


def dual_evaluation(element: FiniteElement,
                    poly_set: PolynomialSet) -> numpy.ndarray:
    return numpy.tensordot(element.get_dual_set().to_riesz(poly_set),
                           poly_set.get_coeffs(), axes=([-1], [-1]))


@pytest.mark.parametrize("domain", ("facet", "interior"))
@pytest.mark.parametrize("family", (Lagrange, IntegratedLegendre))
def test_restricted_dual_evaluation(domain: str, family: type):
    """Restricting a dual evaluation gives the same result as evaluating the restriction."""
    element = family(ufc_simplex(2), 3)
    indices = element.get_dual_set().get_indices(domain)
    restricted = RestrictedElement(element, indices=indices)

    poly_set = element.get_nodal_basis()
    full_evaluation = dual_evaluation(element, poly_set)
    restricted_evaluation = dual_evaluation(restricted, poly_set)

    numpy.testing.assert_allclose(restricted_evaluation,
                                  full_evaluation[indices], atol=1e-14)


@pytest.mark.parametrize("family", (Lagrange, IntegratedLegendre))
def test_restricted_tabulate(family: type):
    """Restricted element tabulation selects the corresponding full basis functions."""
    element = family(ufc_simplex(2), 3)
    indices = element.get_dual_set().get_indices("facet")
    restricted = RestrictedElement(element, indices=indices)
    points = [(0.2, 0.3), (0.1, 0.6)]

    full_values = element.tabulate(0, points)[(0, 0)]
    restricted_values = restricted.tabulate(0, points)[(0, 0)]

    numpy.testing.assert_array_equal(restricted.get_coeffs(), element.get_coeffs()[indices])
    numpy.testing.assert_allclose(restricted_values, full_values[indices], atol=1e-14)


def test_restricted_dual_evaluation_vanishes_on_interior_basis() -> None:
    """Facet duals do not evaluate the interior basis functions."""
    element = IntegratedLegendre(ufc_simplex(1), 5)
    restricted = RestrictedElement(element, restriction_domain="facet")

    evaluation = dual_evaluation(restricted, element.get_nodal_basis())

    numpy.testing.assert_allclose(evaluation[:, 2:], 0.0, atol=1e-14)


@pytest.mark.parametrize("dim", (1, 2, 3))
@pytest.mark.parametrize("family", (Lagrange, IntegratedLegendre))
def test_repeated_restricted_dual_evaluation(dim: int, family: type) -> None:
    """Repeated facet and vertex restrictions preserve the selected duals."""
    element = family(ufc_simplex(dim), dim + 2)
    facets = RestrictedElement(element, restriction_domain="facet")
    vertices = RestrictedElement(facets, restriction_domain="vertex")
    vertices = RestrictedElement(vertices, restriction_domain="vertex")
    indices = element.get_dual_set().get_indices("vertex")
    numpy.testing.assert_allclose(dual_evaluation(vertices, element.get_nodal_basis()),
                                  numpy.eye(element.space_dimension())[indices], atol=1e-13)
