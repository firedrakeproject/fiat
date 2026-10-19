import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem.node import traversal
from gem.optimise import aggressive_unroll, cancel_inverses, cancel_reciprocals


@pytest.fixture
def J():
    return gem.Variable("J", (3, 3))


@pytest.fixture
def K():
    return gem.Variable("K", (3, 3))


@pytest.fixture
def X():
    return gem.Variable("X", (3,))


def depends_on(expression, *terminals):
    return any(node in terminals for node in traversal([expression]))


def test_cancel_inverse(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(J[i, k] * K[k, j] * X[j], (k, j))
    result, = cancel_inverses([expr], [(J, K)])
    assert result == X[i]


def test_cancel_inverse_transposed(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(X[j] * K[k, j] * J[i, k], (k, j))
    result, = cancel_inverses([expr], [(J, K)])
    assert result == X[i]


def test_left_inverse_only(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(J[i, k] * K[k, j] * X[j], (k, j))
    result, = cancel_inverses([expr], [(K, J)])
    assert result is expr


def test_cancel_inverse_nested(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(K[i, k] * gem.IndexSum(J[k, j] * X[j], (j,)), (k,))
    result, = cancel_inverses([expr], [(K, J)])
    assert result == X[i]


@pytest.mark.parametrize("a, b, expected", [(0, 1, gem.Zero()), (1, 1, gem.Literal(1.0))])
def test_cancel_inverse_fixed(J, K, a, b, expected):
    k = gem.Index()
    expr = gem.IndexSum(K[a, k] * J[k, b], (k,))
    result, = cancel_inverses([expr], [(K, J)])
    assert result == expected


def test_cancel_inverse_factors_literal_deltas(J, K):
    # The divergence weights delta_{ab} s_c contract a contravariant axis
    # with a covariant axis, leaving s_c times the trace of R.
    rng = numpy.random.default_rng(0)
    s = rng.random(2)
    W = gem.Literal(numpy.multiply.outer(numpy.eye(3), s))
    R = gem.Literal(rng.random((3, 3)))
    a, b, c, d, e = gem.indices(5)
    expr = gem.IndexSum(W[a, b, e] * J[a, c] * K[d, b] * R[c, d], (a, b, c, d))
    result, = cancel_inverses([expr], [(K, J), (J, K)])
    assert not depends_on(result, J, K)
    result = aggressive_unroll(gem.ComponentTensor(result, (e,)))
    assert numpy.allclose(evaluate([result])[0].arr, numpy.trace(R.array) * s)


def test_cancel_inverse_untouched(J, X):
    i, k = gem.indices(2)
    expr = gem.IndexSum(J[i, k] * X[k], (k,))
    result, = cancel_inverses([expr], [(J, J)])
    assert result is expr


def test_cancel_reciprocals():
    d = gem.Variable("d", ())
    x = gem.Variable("x", ())
    assert cancel_reciprocals([d**2 * (1 / d)**2]) == [gem.Literal(1.0)]
    assert cancel_reciprocals([x * d / d]) == [x]
    assert cancel_reciprocals([d * x / d**2]) == [x / d]
    assert cancel_reciprocals([d**0.5 * x * d**0.5 / d]) == [x]


def test_cancel_reciprocals_untouched():
    d = gem.Variable("d", ())
    x = gem.Variable("x", ())
    expr = x * d / (x + d)
    assert cancel_reciprocals([expr]) == [expr]


def test_cancel_reciprocals_keeps_free_indices():
    v = gem.Variable("v", (3,))
    i = gem.Index()
    expr = v[i] / v[i]
    result, = cancel_reciprocals([expr])
    assert result.free_indices == (i,)
