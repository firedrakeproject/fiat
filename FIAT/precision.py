"""Tolerance adjustment for the caller's floating-point precision."""
import math

import numpy

#: Working precision assumed when the caller does not specify one.
DEFAULT_SCALAR_DTYPE = numpy.dtype("float64")


def prec(tol: float, dtype=DEFAULT_SCALAR_DTYPE) -> float:
    """Return ``sqrt(tol)`` if `dtype` has single-precision (4-byte)
    real components, else `tol` unchanged.

    float32 has a much larger unit roundoff than float64 (machine
    epsilon ~1.2e-7 versus ~2.2e-16), so a `tol` calibrated for double
    precision is often violated by single-precision rounding error.
    Taking `sqrt(tol)` is a standard heuristic for relaxing an
    absolute tolerance to a coarser working precision.

    :arg tol: the tolerance appropriate for double precision.
    :arg dtype: a numpy dtype (or dtype-convertible object) describing
        the caller's working precision, e.g. `form_compiler_parameters
        ["scalar_type"]`. Defaults to `DEFAULT_SCALAR_DTYPE` (float64), for
        which `tol` is returned unchanged. `None` is also accepted and
        is treated the same as `DEFAULT_SCALAR_DTYPE`.
    """
    is_single = numpy.zeros((), dtype=numpy.dtype(dtype)).real.dtype == numpy.dtype(numpy.float32)
    return math.sqrt(tol) if is_single else tol
