# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import numpy

from FIAT import expansions, polynomial_set
from FIAT.finite_element import NonNodalElement
from FIAT.dual_set import DualSet
from FIAT.functional import PointEvaluation
from FIAT.polynomial_set import mis
from FIAT.reference_element import default_simplex, make_lattice


class BernsteinExpansionSet(expansions.ExpansionSet):
    """Expansion set for Bernstein polynomials on a simplex."""

    def get_num_members(self, n: int) -> int:
        """Return the number of Bernstein polynomials of degree ``n``."""
        dim = self.ref_el.get_spatial_dimension()
        return math.comb(n + dim, dim)

    def _tabulate_on_cell(self, n: int, pts: object, order: int = 0,
                          cell: int = 0, direction: object | None = None) -> dict:
        """Tabulate Bernstein polynomials and their derivatives."""
        dim = self.ref_el.get_spatial_dimension()
        pts = numpy.asarray(pts)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts[None, :]
        A, b = self.affine_mappings[cell]
        ref_pts = numpy.add(numpy.dot(pts, A.T), b).T

        vertices = numpy.asarray(default_simplex(dim).get_vertices())
        barycentric_to_reference = numpy.vstack([vertices.T, numpy.ones(dim + 1)])
        reference_to_barycentric = numpy.linalg.inv(barycentric_to_reference)
        barycentric = numpy.concatenate(
            [ref_pts.T, numpy.ones((ref_pts.shape[1], 1))], axis=-1)
        barycentric = numpy.dot(barycentric, reference_to_barycentric.T)

        multiindices = mis(dim + 1, n)
        values = {
            derivative_order: [bernstein_Dx(
                barycentric, ks, derivative_order, reference_to_barycentric)
                for ks in multiindices]
            for derivative_order in range(order + 1)
        }
        result = {}
        for derivative_order in range(order + 1):
            for alpha in mis(dim, derivative_order):
                derivative = numpy.stack([
                    value[alpha] for value in values[derivative_order]])
                if derivative_order == 0:
                    result[alpha] = derivative
                    continue

                components = []
                directions = tuple(
                    direction for direction, count in enumerate(alpha)
                    for _ in range(count))
                for index in numpy.ndindex((dim,) * derivative_order):
                    beta = tuple(index.count(i) for i in range(dim))
                    coefficient = numpy.prod([
                        A[coordinate, direction]
                        for coordinate, direction in zip(index, directions)
                    ])
                    components.append(coefficient * numpy.stack(
                        [value[beta] for value in values[derivative_order]]))
                result[alpha] = sum(components)
        if single_point:
            result = {alpha: values[:, 0] for alpha, values in result.items()}
        return result


class BernsteinDualSet(DualSet):
    """The dual basis for Bernstein elements."""

    def __init__(self, ref_el: object, degree: int) -> None:
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

            nodes.append(PointEvaluation(ref_el, make_lattice(
                ref_el.vertices, degree, variant="gll")[i]))

        super().__init__(nodes, ref_el, entity_ids)


class Bernstein(NonNodalElement):
    """A finite element with Bernstein polynomials as basis functions."""

    def __init__(self, ref_el: object, degree: int) -> None:
        expansion_set = BernsteinExpansionSet(ref_el)
        poly_set = polynomial_set.PolynomialSet(
            ref_el, degree, degree, expansion_set,
            numpy.eye(expansion_set.get_num_members(degree)))
        dual = BernsteinDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree, formdegree=0)


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
