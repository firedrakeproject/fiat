# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import numpy

from FIAT.expansions import ExpansionSet
from FIAT.finite_element import CiarletElement
from FIAT.lagrange import LagrangeDualSet
from FIAT.polynomial_set import PolynomialSet, mis
from FIAT.reference_element import multiindex_equal


class BernsteinExpansionSet(ExpansionSet):
    """Bernstein polynomial expansion set on a simplex."""

    def __init__(self, ref_el):
        if not ref_el.is_simplex():
            raise ValueError("Bernstein expansion sets require a simplex")
        super().__init__(ref_el, scale=1.0)

    def _tabulate_on_cell(self, n, pts, order=0, cell=0, direction=None):
        """Tabulate the expansion set and its derivatives on one cell."""
        if direction is not None:
            raise NotImplementedError("directional Bernstein tabulation is not implemented")

        ref_el = self.ref_el
        dim = ref_el.get_spatial_dimension()
        topology = ref_el.get_topology()
        vertices = ref_el.get_vertices_of_subcomplex(topology[dim][cell])

        B2R = numpy.vstack([numpy.asarray(vertices).T, numpy.ones(len(vertices))])
        R2B = numpy.linalg.inv(B2R)
        points = numpy.asarray(pts)
        B = numpy.concatenate([points, numpy.ones((*points.shape[:-1], 1))],
                              axis=-1).dot(R2B.T)

        raw_result = {
            (derivative, i): vec
            for i, alpha in enumerate(multiindex_equal(dim+1, n))
            for o in range(order + 1)
            for derivative, vec in bernstein_Dx(
                B, alpha, o, R2B
            ).items()
        }
        num_members = math.comb(n + dim, dim)
        dtype = numpy.array(list(raw_result.values())).dtype
        result = {
            alpha: numpy.zeros((num_members, *points.shape[:-1]), dtype=dtype)
            for o in range(order + 1)
            for alpha in mis(dim, o)
        }
        for (alpha, i), vec in raw_result.items():
            result[alpha][i] = vec
        return result


class BernsteinPolynomialSet(PolynomialSet):
    """The Bernstein polynomials of a given degree on a simplex.

    :arg ref_el: The simplex.
    :arg degree: The polynomial degree.
    :kwarg ordering: The ordering of the Bernstein polynomials, either
        None for the ordering of the expansion set, or "topological"
        for decreasing support entity dimension,
        which orders the polynomials by their barycentric
        exponents sorted in decreasing order.
    """
    def __init__(self, ref_el, degree, ordering=None):
        sd = ref_el.get_spatial_dimension()
        alphas = list(multiindex_equal(sd + 1, degree))
        if ordering is None:
            order = Ellipsis
        elif ordering == "topological":
            order = sorted(range(len(alphas)), key=lambda i: sorted(alphas[i], reverse=True))
        else:
            raise ValueError(f"Invalid ordering {ordering}")
        coeffs = numpy.eye(len(alphas))[order]
        super().__init__(ref_el, degree, degree, BernsteinExpansionSet(ref_el), coeffs)


class Bernstein(CiarletElement):
    """A finite element with Bernstein polynomials as basis functions.

    The nodes are linear combinations of point evaluations at GLL points.
    """

    def __init__(self, ref_el, degree):
        poly_set = BernsteinPolynomialSet(ref_el, degree)
        dual = LagrangeDualSet(ref_el, degree, point_variant="gll", sort_entities=True)
        super().__init__(poly_set, dual, degree, formdegree=0, recombine_dual=True)


def bernstein_db(points, ks, alpha=None):
    """Evaluates Bernstein polynomials or its derivative at barycentric
    points.

    :arg points: array of points in barycentric coordinates
    :arg ks: exponents defining the Bernstein polynomial
    :arg alpha: derivative tuple

    :returns: array of Bernstein polynomial values at given points.
    """
    points = numpy.asarray(points)
    ks = numpy.array(tuple(ks))

    *shp, d_1 = points.shape
    assert d_1 == len(ks)

    if alpha is None:
        alpha = numpy.zeros(d_1)
    else:
        alpha = numpy.array(tuple(alpha))
        assert d_1 == len(alpha)

    ls = ks - alpha
    if any(k < 0 for k in ls):
        return numpy.zeros(len(points))
    elif all(k == 0 for k in ls):
        return numpy.ones(len(points))
    else:
        # Calculate coefficient
        coeff = math.factorial(ks.sum())
        for k in ls:
            coeff //= math.factorial(k)
        return coeff * numpy.prod(points**ls, axis=-1)


def bernstein_Dx(points, ks, order, R2B):
    """Evaluates Bernstein polynomials or its derivatives according to
    reference coordinates.

    :arg points: array of points in BARYCENTRIC COORDINATES
    :arg ks: exponents defining the Bernstein polynomial
    :arg alpha: derivative order (returns all derivatives of this
                specified order)
    :arg R2B: linear mapping from reference to barycentric coordinates

    :returns: dictionary mapping from derivative tuples to arrays of
              Bernstein polynomial values at given points.
    """
    points = numpy.asarray(points)
    ks = tuple(ks)

    *shp, d_1 = points.shape
    assert d_1 == len(ks)

    # Collect derivatives according to barycentric coordinates
    Db_map = {alpha: bernstein_db(points, ks, alpha)
              for alpha in mis(d_1, order)}

    # Arrange derivative tensor (barycentric coordinates)
    dtype = numpy.array(list(Db_map.values())).dtype
    Db_shape = (d_1,) * order
    Db_tensor = numpy.empty(Db_shape + tuple(shp), dtype=dtype)
    for ds in numpy.ndindex(Db_shape):
        alpha = tuple(map(ds.count, range(d_1)))
        Db_tensor[ds] = Db_map[alpha]

    # Coordinate transformation: barycentric -> reference
    result = {}
    for alpha in mis(d_1 - 1, order):
        values = Db_tensor
        for d, k in enumerate(alpha):
            for _ in range(k):
                values = R2B[:, d].dot(values)
        result[alpha] = values
    return result
