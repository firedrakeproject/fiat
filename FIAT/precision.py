"""Tolerance adjustment for the caller's floating-point precision."""
import math

import numpy

#: Working precision assumed when the caller does not specify one.
DEFAULT_SCALAR_DTYPE = numpy.dtype("float64")


def calibrate_tolerance(tol: float, dtype=DEFAULT_SCALAR_DTYPE) -> float:
    """Relax `tol` to ``sqrt(tol)`` if `dtype` is single-precision,
    since float32's unit roundoff (~1.2e-7) is much larger than
    float64's (~2.2e-16). Otherwise `tol` is returned unchanged.

    :arg tol: the tolerance appropriate for double precision.
    :arg dtype: the caller's working precision. `None` behaves like
        double precision.
    """
    is_single = numpy.dtype(dtype) in (numpy.dtype(numpy.float32), numpy.dtype(numpy.complex32))
    return math.sqrt(tol) if is_single else tol
