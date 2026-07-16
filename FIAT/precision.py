"""Tolerance adjustment for PETSc's active floating-point precision."""
import math

import numpy

try:
    from petsc4py.PETSc import ScalarType as _PETScScalarType
except ImportError:
    _PETScScalarType = numpy.float64

# Whether PETSc's scalar type has single-precision (4-byte) real components
# (covers both real single and single-complex builds).
single_precision = numpy.zeros((), dtype=_PETScScalarType).real.dtype == numpy.dtype(numpy.float32)


def prec(tol: float) -> float:
    """Return ``sqrt(tol)`` in single precision, else ``tol`` unchanged."""
    return math.sqrt(tol) if single_precision else tol
