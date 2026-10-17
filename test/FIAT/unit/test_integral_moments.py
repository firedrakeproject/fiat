import numpy
import pytest

from FIAT import dual_set, functional, polynomial_set, quadrature_schemes, ufc_simplex


@pytest.mark.parametrize("sd", [2, 3])
@pytest.mark.parametrize("kind", ["directional", "component", "mixed", "divergence", "tensor divergence"])
def test_derivative_moment(sd, kind):
    cell = ufc_simplex(sd)
    Q = quadrature_schemes.create_quadrature(cell, 6)
    points = Q.get_points()
    f = 1 + points.sum(axis=1)
    a = numpy.arange(1, sd + 1, dtype=float)
    n = a / numpy.linalg.norm(a)
    t = a[::-1]
    if kind in {"directional", "component", "mixed"}:
        directions = (n, t) if kind == "mixed" else (n,)
        shape = (sd,) if kind == "component" else ()
        comp = (sd - 1,) if shape else ()
        node = functional.IntegralMomentOfDerivative(cell, Q, f, *directions, comp=comp, shp=shape)
        order = len(directions)
        expected = f * (6 if order == 2 else 3) * (points @ a)**(3 - order)
        expected *= numpy.prod([a @ direction for direction in directions])

        def derivative(point, alpha):
            value = (6 if order == 2 else 3) * (point @ a)**(3 - order)
            value *= numpy.prod(a**alpha)
            if shape:
                result = numpy.zeros(shape)
                result[comp] = value
                return result
            return numpy.asarray(value)
    elif kind == "divergence":
        node = functional.IntegralMomentOfDivergence(cell, Q, f)
        expected = f * 2 * points.sum(axis=1)

        def derivative(point, alpha):
            return 2 * point * numpy.asarray(alpha)
    else:
        test_values = a[:, None] * f
        node = functional.IntegralMomentOfTensorDivergence(cell, Q, test_values)
        expected = (sd + 1) * (points @ a) * f

        def derivative(point, alpha):
            return numpy.outer(alpha, point) + numpy.outer(point, alpha)

    value = sum(w * derivative(numpy.asarray(point), alpha)[comp]
                for point, entries in node.deriv_dict.items() for w, alpha, comp in entries)
    assert numpy.allclose(value, expected @ Q.get_weights())
    assert isinstance(node, functional.IntegralMomentOfDerivative)

    # Compare quadrature batching against the generic functional contraction.
    ids = {dim: {entity: [] for entity in entities} for dim, entities in cell.get_topology().items()}
    ids[sd][0] = [0]
    dual = dual_set.DualSet([node], cell, ids)
    poly = polynomial_set.ONPolynomialSet(cell, 3, shape=node.target_shape)
    assert numpy.allclose(dual.to_riesz(poly)[0], node.to_riesz(poly))


@pytest.mark.parametrize("sd", [2, 3])
def test_normal_derivative_moment(sd):
    cell = ufc_simplex(sd)
    Q = quadrature_schemes.create_quadrature(cell.construct_subelement(sd - 1), 4)
    a = numpy.arange(1, sd + 1, dtype=float)
    for facet in cell.get_topology()[sd - 1]:
        node = functional.IntegralMomentOfNormalDerivative(cell, facet, Q, numpy.ones(len(Q.pts)))
        value = sum(w * a @ alpha for entries in node.deriv_dict.values() for w, alpha, _ in entries)
        expected = Q.get_weights().sum() * (a @ cell.compute_normal(facet))
        assert numpy.allclose(value, expected)
