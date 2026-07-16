import math

import numpy

from FIAT.precision import prec


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
