import math

import gem
import numpy
import pytest
from gem.interpreter import evaluate
from gem.node import traversal

from FIAT import DiscontinuousLagrange
from FIAT.precision import calibrate_tolerance
from FIAT.reference_element import ufc_simplex


@pytest.mark.parametrize("dtype,expected", [
    (None, 1E-12),
    (numpy.float64, 1E-12),
    (numpy.float32, math.sqrt(1E-12)),
    ("float32", math.sqrt(1E-12)),
    (numpy.complex128, 1E-12),
    (numpy.complex64, math.sqrt(1E-12)),
])
def test_calibrate_tolerance(dtype, expected):
    assert calibrate_tolerance(1E-12, dtype) == expected


def macro_element(dtype):
    """A macro element whose symbolic tabulation exercises
    FIAT.expansions.compute_partition_of_unity, where `dtype` is used."""
    K = ufc_simplex(1, dtype=dtype)
    return DiscontinuousLagrange(K, 1, variant="iso")


def literals(expr):
    """Return the set of gem.Literal values appearing in a gem expression."""
    return {node.value for node in traversal([expr]) if isinstance(node, gem.Literal)}


def evaluate_at(exprs, x0, value):
    """Evaluate an iterable of gem expressions at x0 = value."""
    results = evaluate(list(exprs), bindings={x0: numpy.asarray(value)})
    return numpy.array([result.arr for result in results])


def test_dtype_propagates_into_symbolic_tabulation():
    """The `dtype`-adjusted tolerance should appear verbatim in the
    gem expression tree produced by tabulating a macro element,
    confirming it reaches FIAT.expansions.compute_partition_of_unity."""
    fe64 = macro_element(dtype=numpy.float64)
    fe32 = macro_element(dtype=numpy.float32)
    x0 = gem.Variable("x0", ())

    tab64 = fe64.tabulate(0, (x0,))[(0,)][0]
    tab32 = fe32.tabulate(0, (x0,))[(0,)][0]

    assert 1E-12 in literals(tab64)
    assert math.sqrt(1E-12) in literals(tab32)


def test_dtype_changes_macro_tabulation_near_subcell_boundary():
    """Near a macro subcell boundary, a `float32`-adjusted tolerance
    should classify a point differently than the unadjusted `float64`
    tolerance, giving different tabulated values; away from the
    boundary both dtypes should agree."""
    fe64 = macro_element(dtype=numpy.float64)
    fe32 = macro_element(dtype=numpy.float32)
    x0 = gem.Variable("x0", ())

    tab64 = fe64.tabulate(0, (x0,))[(0,)]
    tab32 = fe32.tabulate(0, (x0,))[(0,)]

    # The macro element splits the interval at its midpoint (x0 = 0.5).
    # 1e-9 lies between the float64 tolerance (1e-12) and the
    # float32-adjusted tolerance (sqrt(1e-12) = 1e-6), so the two
    # dtypes classify this point into different subcells.
    near_boundary = 0.5 + 1e-9
    assert not numpy.allclose(evaluate_at(tab64, x0, near_boundary),
                              evaluate_at(tab32, x0, near_boundary))

    # Far from the boundary, both dtypes classify the point the same way.
    away_from_boundary = 0.5 + 1e-3
    assert numpy.allclose(evaluate_at(tab64, x0, away_from_boundary),
                          evaluate_at(tab32, x0, away_from_boundary))
