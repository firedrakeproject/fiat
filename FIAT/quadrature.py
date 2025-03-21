# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Marie E. Rognes (meg@simula.no), 2012
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2015

import itertools
from math import factorial
import numpy
from recursivenodes.quadrature import gaussjacobi, lobattogaussjacobi, simplexgausslegendre

from FIAT import reference_element
from FIAT.orientation_utils import make_entity_permutations_simplex


def pseudo_determinant(A):
    return numpy.sqrt(abs(numpy.linalg.det(numpy.dot(A.T, A))))


def map_quadrature(pts_ref, wts_ref, source_cell, target_cell, jacobian=False):
    """Map quadrature points and weights defined on source_cell to target_cell.
    """
    while source_cell.get_parent():
        source_cell = source_cell.get_parent()
    A, b = reference_element.make_affine_mapping(source_cell.get_vertices(),
                                                 target_cell.get_vertices())
    if len(pts_ref.shape) != 2:
        pts_ref = pts_ref.reshape(-1, A.shape[1])
    scale = pseudo_determinant(A)
    pts = numpy.dot(pts_ref, A.T)
    pts = numpy.add(pts, b, out=pts)
    wts = scale * wts_ref

    # return immutable types
    pts = tuple(map(tuple, pts))
    wts = tuple(wts.flat)
    if jacobian:
        return pts, wts, A
    return pts, wts


class QuadratureRule(object):
    """General class that models integration over a reference element
    as the weighted sum of a function evaluated at a set of points.
    """
    def __init__(self, ref_el, pts, wts):
        if len(wts) != len(pts):
            raise ValueError("Have %d weights, but %d points" % (len(wts), len(pts)))

        self.ref_el = ref_el
        self.pts = pts
        self.wts = wts
        self._intrinsic_orientation_permutation_map_tuple = (None, )

    def get_points(self):
        return numpy.array(self.pts)

    def get_weights(self):
        return numpy.array(self.wts)

    def integrate(self, f):
        return sum(w * f(x) for x, w in zip(self.pts, self.wts))

    @property
    def extrinsic_orientation_permutation_map(self):
        """A map from extrinsic orientations to corresponding axis permutation matrices.

        Notes
        -----
        result[eo] gives the physical axis-reference axis permutation matrix corresponding to
        eo (extrinsic orientation).

        """
        return self.ref_el.extrinsic_orientation_permutation_map

    @property
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


class GaussJacobiQuadratureLineRule(QuadratureRule):
    """Gauss-Jacobi quadature rule determined by Jacobi weights a and b
    using m roots of m:th order Jacobi polynomial."""

    def __init__(self, ref_el, m, a=0, b=0):
        Ref1 = reference_element.DefaultLine()
        pts_ref, wts_ref = gaussjacobi(m, a, b)
        pts, wts = map_quadrature(pts_ref, wts_ref, Ref1, ref_el)
        QuadratureRule.__init__(self, ref_el, pts, wts)
        # Set _intrinsic_orientation_permutation_map_tuple.
        dim = 1
        a = numpy.zeros((factorial(dim + 1), m), dtype=int)
        for io, perm in make_entity_permutations_simplex(dim, m).items():
            a[io, perm] = range(m)
        self._intrinsic_orientation_permutation_map_tuple = (a, )


class GaussLobattoLegendreQuadratureLineRule(QuadratureRule):
    """Gauss-Lobatto-Legendre quadrature rule on the interval.

    The quadrature rule uses m points for a degree of precision of 2m-3.
    """
    def __init__(self, ref_el, m):
        if m < 2:
            raise ValueError(
                "Gauss-Labotto-Legendre quadrature invalid for fewer than 2 points")
        Ref1 = reference_element.DefaultLine()
        pts_ref, wts_ref = lobattogaussjacobi(m, 0, 0)
        pts, wts = map_quadrature(pts_ref, wts_ref, Ref1, ref_el)
        QuadratureRule.__init__(self, ref_el, pts, wts)


class GaussLegendreQuadratureLineRule(GaussJacobiQuadratureLineRule):
    """Gauss--Legendre quadrature rule on the interval.

    The quadrature rule uses m points for a degree of precision of 2m-1.
    """
    def __init__(self, ref_el, m):
        super().__init__(ref_el, m)


class RadauQuadratureLineRule(QuadratureRule):
    """Gauss--Radau quadrature rule on the interval.

    The quadrature rule uses m points for a degree of precision of 2m-1.
    """
    def __init__(self, ref_el, m, right=True):
        if m < 1:
            raise ValueError(
                "Gauss-Radau quadrature invalid for fewer than 1 points")

        right = int(right)
        x0 = ref_el.vertices[right]
        volume = ref_el.volume()
        if m > 1:
            # Make the interior points and weights
            rule = GaussJacobiQuadratureLineRule(ref_el, m-1, right, 1-right)
            # Remove the hat weight from the quadrature weights
            x = rule.get_points().reshape((-1,))
            hat = (2.0 / volume) * abs(x0[0] - x)
            wts = rule.get_weights() / hat
            pts = rule.pts
        else:
            # Special case for lowest order.
            wts = ()
            pts = ()

        # Get the weight at the endpoint via sum(ws) == volume
        w0 = volume - sum(wts)
        xs = (*pts, x0) if right else (x0, *pts)
        ws = (*wts, w0) if right else (w0, *wts)

        QuadratureRule.__init__(self, ref_el, xs, ws)


class CollapsedQuadratureSimplexRule(QuadratureRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the hypercube to the simplex."""

    def __init__(self, ref_el, m):
        dim = ref_el.get_spatial_dimension()
        Ref1 = reference_element.default_simplex(dim)
        pts_ref, wts_ref = simplexgausslegendre(dim, m)
        pts, wts = map_quadrature(pts_ref, wts_ref, Ref1, ref_el)
        QuadratureRule.__init__(self, ref_el, pts, wts)


class CollapsedQuadratureTriangleRule(CollapsedQuadratureSimplexRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the square to the triangle."""
    pass


class CollapsedQuadratureTetrahedronRule(CollapsedQuadratureSimplexRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the cube to the tetrahedron."""
    pass


class FacetQuadratureRule(QuadratureRule):
    """A quadrature rule on a facet mapped from a reference quadrature rule.
    """
    def __init__(self, ref_el, entity_dim, entity_id, Q_ref):
        # Construct the facet of interest
        facet = ref_el.construct_subelement(entity_dim)
        facet_topology = ref_el.get_topology()[entity_dim][entity_id]
        facet.vertices = ref_el.get_vertices_of_subcomplex(facet_topology)

        # Map reference points and weights on the appropriate facet
        pts_ref = Q_ref.get_points()
        wts_ref = Q_ref.get_weights()
        pts, wts, J = map_quadrature(pts_ref, wts_ref, Q_ref.ref_el, facet, jacobian=True)

        # Initialize super class with new points and weights
        QuadratureRule.__init__(self, facet, pts, wts)
        self._J = J
        self._reference_rule = Q_ref

    def reference_rule(self):
        return self._reference_rule

    def jacobian(self):
        return self._J

    def jacobian_determinant(self):
        return pseudo_determinant(self._J)


def make_quadrature(ref_el, m):
    """Returns the collapsed quadrature rule using m points per
    direction on the given reference element. In the tensor product
    case, m is a tuple."""

    if isinstance(m, tuple):
        min_m = min(m)
    else:
        min_m = m

    msg = "Expecting at least one (not %d) quadrature point per direction" % min_m
    assert (min_m > 0), msg

    if ref_el.get_shape() == reference_element.POINT:
        return QuadratureRule(ref_el, [()], [1])
    elif ref_el.get_shape() == reference_element.LINE:
        return GaussJacobiQuadratureLineRule(ref_el, m)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return CollapsedQuadratureTriangleRule(ref_el, m)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return CollapsedQuadratureTetrahedronRule(ref_el, m)
    elif ref_el.get_shape() == reference_element.QUADRILATERAL:
        line_rule = GaussJacobiQuadratureLineRule(ref_el.construct_subelement(1), m)
        return make_tensor_product_quadrature(line_rule, line_rule)
    elif ref_el.get_shape() == reference_element.HEXAHEDRON:
        line_rule = GaussJacobiQuadratureLineRule(ref_el.construct_subelement(1), m)
        return make_tensor_product_quadrature(line_rule, line_rule, line_rule)
    else:
        raise ValueError("Unable to make quadrature for cell: %s" % ref_el)


def make_tensor_product_quadrature(*quad_rules):
    """Returns the quadrature rule for a TensorProduct cell, by combining
    the quadrature rules of the components."""
    ref_el = reference_element.TensorProductCell(*[q.ref_el
                                                   for q in quad_rules])
    # Coordinates are "concatenated", weights are multiplied
    pts = [list(itertools.chain(*pt_tuple))
           for pt_tuple in itertools.product(*[q.pts for q in quad_rules])]
    wts = [numpy.prod(wt_tuple)
           for wt_tuple in itertools.product(*[q.wts for q in quad_rules])]
    return QuadratureRule(ref_el, pts, wts)
