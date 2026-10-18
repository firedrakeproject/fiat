from itertools import permutations

import numpy
import pytest

import FIAT
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


@pytest.mark.parametrize("element, degree", [("Argyris", 6), ("HsiehCloughTocher", 4), ("Walkington", 5),
                                             ("ArnoldWinther", 3), ("MardalTaiWinther", 1),
                                             ("Nedelec", 3), ("BrezziDouglasMarini", 2)])
def test_integral_moment_points_and_weights(element, degree):
    """Public point and weight accessors represent the quadrature functional."""
    sd = 3 if element in {"Walkington", "Nedelec"} else 2
    kwargs = {"variant": "integral"} if element in {"Nedelec", "BrezziDouglasMarini"} else {}
    nodes = getattr(FIAT, element)(ufc_simplex(sd), degree, **kwargs).get_dual_set().nodes
    moments = [node for node in nodes if isinstance(node, functional.IntegralMoment)]
    assert moments
    for node in moments:
        points_by_order = node.points_by_order
        weights_by_order = node.weights_by_order
        order = node.max_deriv_order
        points = tuple(map(tuple, node.Q.get_points()))
        assert set(points_by_order) == set(weights_by_order) == {order}
        assert points_by_order[order] == points

        expected = numpy.zeros((*node.target_shape, *(sd,) * order, len(points)))
        expected[node.comp] = node.f_at_qpts * node.Q.get_weights()
        if order > 1:
            rank = len(node.target_shape)
            expected = numpy.mean([expected.transpose((*range(rank), *axes, expected.ndim - 1))
                                   for axes in permutations(range(rank, rank + order))], axis=0)
        assert numpy.allclose(weights_by_order[order], expected)


@pytest.mark.parametrize("sd", [2, 3])
def test_uncontracted_riesz(sd):
    """Inserting a map between the derivative axes changes the direction of the derivative."""
    cell = ufc_simplex(sd)
    Q = quadrature_schemes.create_quadrature(cell, 4)
    f = 1 + Q.get_points().sum(axis=1)
    n = numpy.arange(1, sd + 1, dtype=float)
    G = numpy.random.default_rng(0).random((sd, sd))
    ids = {dim: {entity: [] for entity in entities} for dim, entities in cell.get_topology().items()}
    ids[sd][0] = [0]
    poly = polynomial_set.ONPolynomialSet(cell, 3)

    dual = dual_set.DualSet([functional.IntegralMomentOfDerivative(cell, Q, f, n)], cell, ids)
    R = dual.to_uncontracted_riesz(poly)[1][0]
    mapped = dual_set.DualSet([functional.IntegralMomentOfDerivative(cell, Q, f, G.T @ n)], cell, ids)
    assert numpy.allclose(numpy.einsum("bgk,bg->k", R, G), mapped.to_riesz(poly)[0])
    assert numpy.allclose(numpy.einsum("bbk->k", R), dual.to_riesz(poly)[0])
