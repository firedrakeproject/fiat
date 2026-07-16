"""Tolerance adjustment for the caller's floating-point precision."""
import math

import numpy


def prec(tol: float, scalar_type=None) -> float:
    """Return ``sqrt(tol)`` if `scalar_type` has single-precision (4-byte)
    real components, else `tol` unchanged.

    :arg tol: the tolerance appropriate for double precision.
    :arg scalar_type: a numpy dtype (or dtype-convertible object) describing
        the caller's working precision, e.g. `form_compiler_parameters
        ["scalar_type"]`. If `None`, `tol` is returned unchanged.
    """
    if scalar_type is None:
        return tol
    is_single = numpy.zeros((), dtype=numpy.dtype(scalar_type)).real.dtype == numpy.dtype(numpy.float32)
    return math.sqrt(tol) if is_single else tol
