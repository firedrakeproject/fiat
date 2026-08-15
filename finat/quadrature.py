import hashlib
from abc import ABCMeta, abstractmethod
from functools import cached_property
from math import factorial

import gem
import numpy
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from FIAT.quadrature_schemes import create_quadrature as fiat_scheme
from FIAT.reference_element import LINE, QUADRILATERAL, TENSORPRODUCT
from gem.utils import safe_repr
from recursivenodes.quadrature import gaussjacobi

from finat.point_set import (CollapsedTensorProductPointSet,
                             GaussLegendrePointSet, GaussLobattoLegendrePointSet,
                             KMVPointSet, PointSet, TensorPointSet)


def make_quadrature(ref_el, degree, scheme="default"):
    """Create a quadrature rule for a reference element.

    Parameters
    ----------
    ref_el : FIAT.reference_element.Cell
        Reference cell.
    degree : int or tuple of int
        Exact polynomial degree.
    scheme : str, optional
        Quadrature scheme.

    Returns
    -------
    QuadratureRule
        Quadrature points, weights, and reference cell.

    Notes
    -----
    The ``default`` scheme uses tabulated simplex rules at low degree. It
    uses collapsed Gauss rules at high degree. The ``canonical`` scheme uses
    collapsed Gauss rules with flat points. The ``KMV`` scheme uses spectral
    lumped rules. The ``collapsed`` scheme retains the tensor-product Duffy
    structure for sum factorization.

    """
    if ref_el.get_shape() == TENSORPRODUCT:
        try:
            degree = tuple(degree)
        except TypeError:
            degree = (degree,) * len(ref_el.cells)

        assert len(ref_el.cells) == len(degree)
        quad_rules = [make_quadrature(c, d, scheme)
                      for c, d in zip(ref_el.cells, degree)]
        return TensorProductQuadratureRule(quad_rules, ref_el=ref_el)

    if ref_el.get_shape() == QUADRILATERAL:
        return make_quadrature(ref_el.product, degree, scheme)

    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)

    if scheme.lower() == "collapsed":
        return collapsed_gauss_jacobi_quadrature(ref_el, degree)

    if scheme.lower() in {"kmv", "lump"}:
        fiat_rule = fiat_scheme(ref_el, degree, "KMV")
        if ref_el.get_shape() == LINE:
            point_set = GaussLobattoLegendrePointSet(fiat_rule.get_points())
        else:
            point_set = KMVPointSet(fiat_rule.get_points())
    elif ref_el.get_shape() == LINE and not ref_el.is_macrocell():
        # FIAT uses Gauss-Legendre line quadature, however, since we
        # symbolically label it as such, we wish not to risk attaching
        # the wrong label in case FIAT changes.  So we explicitly ask
        # for Gauss-Legendre line quadature.
        num_points = (degree + 1 + 1) // 2  # exact integration
        fiat_rule = GaussLegendreQuadratureLineRule(ref_el, num_points)
        point_set = GaussLegendrePointSet(fiat_rule.get_points())
    else:
        fiat_rule = fiat_scheme(ref_el, degree, scheme)
        point_set = PointSet(fiat_rule.get_points())

    return QuadratureRule(point_set, fiat_rule.get_weights(), ref_el=ref_el, io_ornt_map_tuple=fiat_rule._intrinsic_orientation_permutation_map_tuple)


def collapsed_gauss_jacobi_quadrature(ref_el, degree):
    """Create a structured collapsed Gauss-Jacobi quadrature rule.

    Parameters
    ----------
    ref_el : FIAT.reference_element.Cell
        The simplex to create the quadrature rule on.
    degree : int
        The degree of polynomial that the rule should integrate exactly.

    Returns
    -------
    CollapsedTensorProductQuadratureRule
        The structured quadrature rule.

    Notes
    -----
    The Duffy map is
    ``x_t = eta_t * prod(u > t) (1 - eta_u)``. The Jacobi weight
    ``(1 - eta_u)**u`` absorbs the Duffy Jacobian on axis ``u``. The simplex
    weights are products of the one-dimensional weights.

    """
    if ref_el.is_macrocell():
        raise NotImplementedError("Collapsed quadrature is not supported on split cells")
    dim = ref_el.get_spatial_dimension()
    num_points = (degree + 1 + 1) // 2  # exact integration
    factors = []
    for axis in range(dim):
        xs, ws = gaussjacobi(num_points, axis, 0.0)
        # Map from the biunit to the unit interval, folding the change of
        # measure of the Jacobi weight into the quadrature weights
        xs = (1.0 + xs) / 2.0
        ws = ws / 2.0 ** (axis + 1)
        if axis == 0:
            # The Duffy map produces points on the unit simplex
            ws = ws * (ref_el.volume() * factorial(dim))
        factors.append(QuadratureRule(PointSet(xs[:, None]), ws))
    return CollapsedTensorProductQuadratureRule(factors, ref_el=ref_el)


class AbstractQuadratureRule(metaclass=ABCMeta):
    """Abstract class representing a quadrature rule as point set and a
    corresponding set of weights."""

    def __hash__(self):
        return int.from_bytes(hashlib.md5(repr(self).encode()).digest(), byteorder="big")

    def __eq__(self, other):
        return type(other) is type(self) and repr(other) == repr(self)

    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def point_set(self):
        """Point set object representing the quadrature points."""

    @property
    @abstractmethod
    def weight_expression(self):
        """GEM expression describing the weights, with the same free indices
        as the point set."""

    @cached_property
    def extrinsic_orientation_permutation_map(self):
        """A map from extrinsic orientations to corresponding axis permutation matrices.

        Notes
        -----
        result[eo] gives the physical axis-reference axis permutation matrix corresponding to
        eo (extrinsic orientation).

        """
        if self.ref_el is None:
            raise ValueError("Must set ref_el")
        return self.ref_el.extrinsic_orientation_permutation_map

    @cached_property
    def intrinsic_orientation_permutation_map_tuple(self):
        """A tuple of maps from intrinsic orientations to corresponding point permutations for each reference cell axis.

        Notes
        -----
        result[axis][io] gives the physical point-reference point permutation array corresponding to
        io (intrinsic orientation) on ``axis``.

        """
        if any(m is None for m in self._intrinsic_orientation_permutation_map_tuple):
            raise ValueError("Must set _intrinsic_orientation_permutation_map_tuple")
        return self._intrinsic_orientation_permutation_map_tuple


class QuadratureRule(AbstractQuadratureRule):
    """Generic quadrature rule with no internal structure."""

    def __init__(self, point_set, weights, ref_el=None, io_ornt_map_tuple=(None, )):
        weights = numpy.asarray(weights)
        assert len(point_set.points) == len(weights)

        self.ref_el = ref_el
        self.point_set = point_set
        self.weights = numpy.asarray(weights)
        self._intrinsic_orientation_permutation_map_tuple = io_ornt_map_tuple

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"{self.point_set!r}, "
            f"{safe_repr(self.weights)}, "
            f"{self.ref_el!r}, "
            f"{self._intrinsic_orientation_permutation_map_tuple!r}"
            ")"
        )

    @cached_property
    def point_set(self):
        pass  # set at initialisation

    @cached_property
    def weight_expression(self):
        return gem.Indexed(gem.Literal(self.weights), self.point_set.indices)


class TensorProductQuadratureRule(AbstractQuadratureRule):
    """Quadrature rule which is a tensor product of other rules."""

    def __init__(self, factors, ref_el=None):
        self.ref_el = ref_el
        self.factors = tuple(factors)
        self._intrinsic_orientation_permutation_map_tuple = tuple(
            m
            for factor in factors
            for m in factor._intrinsic_orientation_permutation_map_tuple
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.factors!r}, {self.ref_el!r})"

    @cached_property
    def point_set(self):
        return TensorPointSet(q.point_set for q in self.factors)

    @cached_property
    def weight_expression(self):
        return gem.Product(*(q.weight_expression for q in self.factors))


class CollapsedTensorProductQuadratureRule(AbstractQuadratureRule):
    """Simplex quadrature rule with tensor-product structure in collapsed
    coordinates, following Karniadakis & Sherwin.

    Parameters
    ----------
    factors : tuple of QuadratureRule
        One-dimensional quadrature rules of collapsed coordinates on the
        unit interval, one per spatial dimension of the simplex, with the
        Duffy Jacobian folded into the weights.
    ref_el : FIAT.reference_element.Cell
        The simplex the quadrature rule integrates over.

    """

    def __init__(self, factors, ref_el=None):
        self.ref_el = ref_el
        self.factors = tuple(factors)
        self._intrinsic_orientation_permutation_map_tuple = (None,)

    def __repr__(self):
        return f"{type(self).__name__}({self.factors!r}, {self.ref_el!r})"

    @cached_property
    def point_set(self):
        return CollapsedTensorProductPointSet([q.point_set for q in self.factors])

    @cached_property
    def weight_expression(self):
        return gem.Product(*(q.weight_expression for q in self.factors))
