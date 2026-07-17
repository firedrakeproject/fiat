import math

import numpy
import sympy

from FIAT import DiscontinuousLagrange
from FIAT.precision import prec
from FIAT.reference_element import ufc_simplex


def test_prec_none_returns_tol_unchanged():
    assert prec(1E-12, None) == 1E-12


def test_prec_double_precision_returns_tol_unchanged():
    assert prec(1E-12, numpy.float64) == 1E-12


def test_prec_single_precision_returns_sqrt_tol():
    assert prec(1E-12, numpy.float32) == math.sqrt(1E-12)


def test_prec_accepts_dtype_convertible_string():
    assert prec(1E-12, "float32") == math.sqrt(1E-12)


def test_prec_complex_double_returns_tol_unchanged():
    assert prec(1E-12, numpy.complex128) == 1E-12


def test_prec_complex_single_returns_sqrt_tol():
    assert prec(1E-12, numpy.complex64) == math.sqrt(1E-12)


def macro_element(dtype):
    """A macro element whose symbolic tabulation exercises
    FIAT.expansions.compute_partition_of_unity, where `dtype` is used."""
    K = ufc_simplex(1)
    K.vertices = tuple(map(tuple, numpy.array(K.vertices, dtype=dtype)))
    return DiscontinuousLagrange(K, 1, variant="iso")


def test_dtype_propagates_into_symbolic_tabulation():
    """The `dtype`-adjusted tolerance should appear verbatim in the
    symbolic expression tree produced by tabulating a macro element,
    confirming it reaches FIAT.expansions.compute_partition_of_unity."""
    fe = macro_element()
    x0 = sympy.Symbol("x0")

    tab64 = fe.tabulate(0, (x0,), dtype=numpy.float64)[(0,)][0]
    tab32 = fe.tabulate(0, (x0,), dtype=numpy.float32)[(0,)][0]

    assert sympy.Float(1E-12) in tab64.atoms(sympy.Float)
    assert sympy.Float(math.sqrt(1E-12)) in tab32.atoms(sympy.Float)


def test_dtype_changes_macro_tabulation_near_subcell_boundary():
    """Near a macro subcell boundary, a `float32`-adjusted tolerance
    should classify a point differently than the unadjusted `float64`
    tolerance, giving different tabulated values; away from the
    boundary both dtypes should agree."""
    fe = macro_element()
    x0 = sympy.Symbol("x0")

    tab64 = fe.tabulate(0, (x0,), dtype=numpy.float64)[(0,)]
    tab32 = fe.tabulate(0, (x0,), dtype=numpy.float32)[(0,)]
    f64 = sympy.lambdify(x0, list(tab64))
    f32 = sympy.lambdify(x0, list(tab32))

    # The macro element splits the interval at its midpoint (x0 = 0.5).
    # 1e-9 lies between the float64 tolerance (1e-12) and the
    # float32-adjusted tolerance (sqrt(1e-12) = 1e-6), so the two
    # dtypes classify this point into different subcells.
    near_boundary = 0.5 + 1e-9
    assert not numpy.allclose(f64(near_boundary), f32(near_boundary))

    # Far from the boundary, both dtypes classify the point the same way.
    away_from_boundary = 0.5 + 1e-3
    assert numpy.allclose(f64(away_from_boundary), f32(away_from_boundary))
