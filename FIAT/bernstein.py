# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import numpy

from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import mis
from FIAT.pointwise_dual import compute_pointwise_dual
from FIAT.reference_element import make_lattice


class BernsteinDualSet(DualSet):
    """The dual basis for Bernstein elements."""

    def __init__(self, ref_el, degree):
        # Initialise data structures
        topology = ref_el.get_topology()
        entity_ids = {dim: {entity_i: []
                            for entity_i in entities}
                      for dim, entities in topology.items()}

        # Calculate inverse topology
        inverse_topology = {vertices: (dim, entity_i)
                            for dim, entities in topology.items()
                            for entity_i, vertices in entities.items()}

        # Generate triangular barycentric indices
        dim = ref_el.get_spatial_dimension()
        kss = mis(dim + 1, degree)

        # Fill data structures
        nodes = []
        for i, ks in enumerate(kss):
            vertices, = numpy.nonzero(ks)
            entity_dim, entity_i = inverse_topology[tuple(vertices)]
            entity_ids[entity_dim][entity_i].append(i)

            # Leave nodes unimplemented for now
            nodes.append(None)

        super().__init__(nodes, ref_el, entity_ids)


class Bernstein(FiniteElement):
    """A finite element with Bernstein polynomials as basis functions."""

    def __init__(self, ref_el, degree):
        dual = BernsteinDualSet(ref_el, degree)
        k = 0  # 0-form
        super().__init__(ref_el, dual, degree, k)
        pts = make_lattice(ref_el.vertices, degree, variant="gll")
        newdual = compute_pointwise_dual(self, pts)
        self.dual = newdual

    def degree(self):
        """The degree of the polynomial space."""
        return self.get_order()

    def value_shape(self):
        """The value shape of the finite element functions."""
        return ()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        # Transform points to reference cell coordinates
        ref_el = self.get_reference_element()
        dim = ref_el.get_spatial_dimension()
        if entity is None:
            entity = (dim, 0)

        entity_dim, entity_id = entity
        entity_transform = ref_el.get_entity_transform(entity_dim, entity_id)

        points = numpy.asarray(points)
        cell_points = entity_transform(points)

        # Construct Cartesian to Barycentric coordinate mapping
        vs = numpy.asarray(ref_el.get_vertices())
        B2R = numpy.vstack([vs.T, numpy.ones(len(vs))])
        R2B = numpy.linalg.inv(B2R)

        B = numpy.concatenate([cell_points, numpy.ones((*cell_points.shape[:-1], 1))
                               ], axis=-1).dot(R2B.T)

        # Evaluate everything
        deg = self.degree()
        raw_result = {(alpha, i): vec
                      for i, ks in enumerate(mis(dim + 1, deg))
                      for o in range(order + 1)
                      for alpha, vec in bernstein_Dx(B, ks, o, R2B).items()}

        # Rearrange result
        space_dim = self.space_dimension()
        dtype = numpy.array(list(raw_result.values())).dtype
        result = {alpha: numpy.zeros((space_dim, *points.shape[:-1]), dtype=dtype)
                  for o in range(order + 1)
                  for alpha in mis(dim, o)}
        for (alpha, i), vec in raw_result.items():
            result[alpha][i] = vec
        return result


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
