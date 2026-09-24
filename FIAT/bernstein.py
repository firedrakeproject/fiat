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
        if not (ref_el.is_simplex() or ref_el.is_macrocell()):
            raise ValueError("Bernstein expansion sets require a simplex or macrocell")
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
    """The Bernstein polynomials of a given degree on a simplex or macrocell.

    :arg ref_el: The simplex or macrocell.
    :arg degree: The polynomial degree.
    :kwarg order: The continuity order for a macrocell. Only C0 is currently
        supported.
    :kwarg entity_ids: An optional entity-to-basis-index map defining the
        ordering of the macrocell basis.
    """
    def __init__(self, ref_el, degree, order=0, entity_ids=None):
        expansion_set = BernsteinExpansionSet(ref_el)
        if order != 0:
            raise NotImplementedError("Only C0 Bernstein polynomial sets are implemented")
        if degree < 0:
            raise ValueError("Bernstein polynomial sets require a nonnegative degree")
        coeffs = _c0_coefficients(ref_el, degree, entity_ids)
        super().__init__(ref_el, degree, degree, expansion_set, coeffs)


class Bernstein(CiarletElement):
    """A finite element with Bernstein polynomials as basis functions.

    The nodes are linear combinations of point evaluations at GLL points.
    """

    def __init__(self, ref_el, degree):
        dual = LagrangeDualSet(ref_el, degree, point_variant="gll", sort_entities=True)
        poly_set = BernsteinPolynomialSet(ref_el, degree, entity_ids=dual.get_entity_ids())
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


def _default_entity_ids(ref_el, degree):
    """Construct the default Bernstein basis entity ordering."""
    topology = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()
    local_alphas = list(multiindex_equal(sd + 1, degree))
    inverse = {
        vertices: (entity_dim, entity)
        for entity_dim, entities in topology.items()
        for entity, vertices in entities.items()
    }
    candidates = {}
    for cell_vertices in topology[sd].values():
        for alpha in local_alphas:
            support = tuple(vertex for vertex, exponent in zip(cell_vertices, alpha) if exponent)
            entity_dim, entity = inverse[support]
            entity_vertices = topology[entity_dim][entity]
            beta = tuple(alpha[cell_vertices.index(vertex)] for vertex in entity_vertices)
            candidates.setdefault((entity_dim, entity, beta), alpha)

    candidates = list(candidates.items())
    order = sorted(
        range(len(candidates)),
        key=lambda i: tuple(sorted(candidates[i][1], reverse=True)),
    )
    row_of_candidate = {
        candidates[candidate][0]: row for row, candidate in enumerate(order)
    }
    entity_ids = {dim: {entity: [] for entity in topology[dim]} for dim in topology}
    for entity_dim, entities in topology.items():
        for entity in entities:
            for beta in multiindex_equal(entity_dim + 1, degree, imin=1):
                candidate = (entity_dim, entity, beta)
                entity_ids[entity_dim][entity].append(row_of_candidate[candidate])
    return entity_ids


def _c0_coefficients(ref_el, degree, entity_ids=None):
    """Construct C0 Bernstein coefficients from entity multiindices."""
    sd = ref_el.get_spatial_dimension()
    topology = ref_el.get_topology()
    if degree == 0:
        coeffs = numpy.zeros((1, len(topology[sd])))
        coeffs[0] = 1
        return coeffs
    local_alphas = list(multiindex_equal(sd + 1, degree))
    local_alpha_ids = {alpha: i for i, alpha in enumerate(local_alphas)}
    num_cells = len(topology[sd])
    num_local = len(local_alphas)
    if entity_ids is None:
        entity_ids = _default_entity_ids(ref_el, degree)

    num_members = sum(len(ids) for entities in entity_ids.values() for ids in entities.values())
    coeffs = numpy.zeros((num_members, num_cells * num_local))

    for dim, entities in topology.items():
        entity_alphas = list(multiindex_equal(dim + 1, degree, imin=1))
        for entity, entity_vertices in entities.items():
            ids = entity_ids[dim][entity]
            for row, alpha in zip(ids, entity_alphas):
                for cell, cell_vertices in topology[sd].items():
                    if not set(entity_vertices).issubset(cell_vertices):
                        continue
                    local_alpha = tuple(
                        alpha[entity_vertices.index(vertex)] if vertex in entity_vertices else 0
                        for vertex in cell_vertices
                    )
                    coeffs[row, cell * num_local + local_alpha_ids[local_alpha]] = 1
    return coeffs
