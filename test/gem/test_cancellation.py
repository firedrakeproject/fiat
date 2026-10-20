import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem.node import traversal
from gem.optimise import (aggressive_unroll, cancel_inverses, cancel_reciprocals, common_denominator,
                          fold_constants)


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
    result, = cancel_inverses([expr], {(J, K): gem.Literal(1.0)})
    assert result == X[i]


def test_cancel_inverse_transposed(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(X[j] * K[k, j] * J[i, k], (k, j))
    result, = cancel_inverses([expr], {(J, K): gem.Literal(1.0)})
    assert result == X[i]


def test_left_inverse_only(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(J[i, k] * K[k, j] * X[j], (k, j))
    result, = cancel_inverses([expr], {(K, J): gem.Literal(1.0)})
    assert depends_on(result, J, K)


def test_cancel_inverse_nested(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(K[i, k] * gem.IndexSum(J[k, j] * X[j], (j,)), (k,))
    result, = cancel_inverses([expr], {(K, J): gem.Literal(1.0)})
    assert result == X[i]


def test_cancel_inverse_distributes_over_sums(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(J[i, k] * (K[k, j] + X[k] * X[j]) * X[j], (k, j))
    result, = cancel_inverses([expr], {(J, K): gem.Literal(1.0)})
    assert not depends_on(result, K)
    rng = numpy.random.default_rng(0)
    J_, X_ = rng.random((3, 3)), rng.random(3)
    bindings = {J: J_, K: numpy.linalg.inv(J_), X: X_}
    expected, actual = evaluate([gem.ComponentTensor(e, (i,)) for e in (expr, result)], bindings)
    assert numpy.allclose(expected.arr, actual.arr)


def test_cancel_inverse_sum_without_index(J, K, X):
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(J[i, k] * K[k, j] * (gem.Literal(1.0) + X[j]), (k, j))
    result, = cancel_inverses([expr], {(J, K): gem.Literal(1.0)})
    assert result == gem.Literal(1.0) + X[i]


@pytest.mark.parametrize("a, b, expected", [(0, 1, gem.Zero()), (1, 1, gem.Literal(1.0))])
def test_cancel_inverse_fixed(J, K, a, b, expected):
    k = gem.Index()
    expr = gem.IndexSum(K[a, k] * J[k, b], (k,))
    result, = cancel_inverses([expr], {(K, J): gem.Literal(1.0)})
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
    result, = cancel_inverses([expr], {(K, J): gem.Literal(1.0), (J, K): gem.Literal(1.0)})
    assert not depends_on(result, J, K)
    result = aggressive_unroll(gem.ComponentTensor(result, (e,)))
    assert numpy.allclose(evaluate([result])[0].arr, numpy.trace(R.array) * s)


def test_cancel_scaled_inverse(J, X):
    # adj(J) J = det(J) I
    A = gem.Variable("A", (3, 3))
    d = gem.Variable("d", ())
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(A[i, k] * J[k, j] * X[j], (k, j)) / d
    result, = cancel_reciprocals(cancel_inverses([expr], {(A, J): d}))
    assert result == X[i]


def test_cancel_inverse_factors_deltas_within_tol(J, K):
    R = gem.Literal(numpy.eye(3) + 1e-15 * numpy.diag([0, 1, 2]))
    a, b, c, d = gem.indices(4)
    expr = gem.IndexSum(R[a, b] * J[a, c] * K[d, b] * R[c, d], (a, b, c, d))
    result, = cancel_inverses([expr], {(K, J): gem.Literal(1.0)}, tol=1e-12)
    assert not depends_on(result, J, K)
    result, = cancel_inverses([expr], {(K, J): gem.Literal(1.0)})
    assert depends_on(result, J, K)


def test_cancel_inverse_singleton():
    # The sum over an index of extent 1 is unrolled on construction.
    J = gem.Variable("J", (1, 1))
    K = gem.Variable("K", (1, 1))
    x = gem.Variable("x", ())
    k = gem.Index()
    expr = gem.IndexSum(K[0, k] * J[k, 0], (k,)) * x
    assert not isinstance(expr, gem.IndexSum)
    i = gem.Index()
    result, = cancel_inverses([gem.IndexSum(expr * gem.Variable("y", (2,))[i], (i,))], {(K, J): gem.Literal(1.0)})
    assert not depends_on(result, J, K)


def test_cancel_inverse_untouched(J, X):
    i, k = gem.indices(2)
    expr = gem.IndexSum(J[i, k] * X[k], (k,))
    result, = cancel_inverses([expr], {(J, J): gem.Literal(1.0)})
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


def test_common_denominator():
    d = gem.Variable("d", ())
    s = gem.Variable("s", ())
    x = gem.Variable("x", ())
    y = gem.Variable("y", ())
    numerators, denominator = common_denominator([x / d, x / (d * s) + y / d**2, gem.Zero()])
    assert denominator == gem.Product(d**2, s)
    assert numerators == [x * d * s, x * d + y * s, gem.Zero()]
    assert not any(isinstance(node, gem.Division) for node in traversal(numerators))


def test_common_denominator_untouched():
    x = gem.Variable("x", ())
    assert common_denominator([x]) == ([x], gem.Literal(1.0))


def test_fold_constants():
    rng = numpy.random.default_rng(0)
    A, B = (gem.Literal(rng.random((3, 3))) for _ in range(2))
    i, j, k = gem.indices(3)
    expr = gem.IndexSum(A[i, k] * B[k, j] + A[i, 0] * gem.Delta(k, j), (k,))
    result, = fold_constants([expr])
    C, = result.children
    assert isinstance(result, gem.Indexed) and isinstance(C, gem.Literal)
    assert result.multiindex in {(i, j), (j, i)}
    expected = A.array @ B.array + A.array[:, :1]
    assert numpy.allclose(C.array if result.multiindex == (i, j) else C.array.T, expected)


def test_fold_constants_trace():
    A = gem.Literal(numpy.arange(9.0).reshape(3, 3))
    k = gem.Index()
    result, = fold_constants([gem.IndexSum(A[k, k], (k,))])
    assert result == gem.Literal(12.0)


def test_fold_constants_keeps_variables():
    A = gem.Literal(numpy.arange(9.0).reshape(3, 3))
    X = gem.Variable("X", (3,))
    i, k = gem.indices(2)
    expr = gem.IndexSum(X[k] * gem.IndexSum(A[i, k] * A[k, i], (i,)), (k,))
    result, = fold_constants([expr])
    assert sum(isinstance(node, gem.IndexSum) for node in traversal([result])) == 1
    bindings = {X: numpy.arange(3.0)}
    assert numpy.allclose(*(r.arr for r in evaluate([expr, result], bindings)))
