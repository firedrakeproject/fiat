# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import itertools
import math
import numpy
import scipy.linalg

from FIAT.check_format_variant import parse_lagrange_variant
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
    :kwarg order: The order of continuity across the interior facets of a
        macrocell, either 0 or 1.
    :kwarg vorder: The order of super-smoothness at the interior vertex of a
        macrocell.
    :kwarg entity_ids: An optional entity-to-basis-index map defining the
        ordering of the basis. Each entity carries the basis functions
        associated with the Bernstein coefficients on that entity that lie in
        the minimal determining set.
    """
    def __init__(self, ref_el, degree, order=0, vorder=None, entity_ids=None):
        expansion_set = BernsteinExpansionSet(ref_el)
        if degree < 0:
            raise ValueError("Bernstein polynomial sets require a nonnegative degree")
        if order not in {0, 1}:
            raise NotImplementedError("Only C0 and C1 Bernstein polynomial sets are implemented")
        if order == 0 and vorder is None:
            coeffs = _c0_coefficients(ref_el, degree, entity_ids)
        else:
            coeffs = _ck_coefficients(ref_el, degree, order, vorder, entity_ids)
        super().__init__(ref_el, degree, degree, expansion_set, coeffs)


class Bernstein(CiarletElement):
    """A finite element with Bernstein polynomials as basis functions.

    The nodes are linear combinations of Lagrange point evaluations.

    :arg ref_el: The reference simplex.
    :arg degree: The polynomial degree.
    :kwarg variant: A comma-separated string that may specify the type of
        point distribution and the splitting strategy if a macro element is
        desired, as in :class:`FIAT.lagrange.Lagrange`.
        Example: variant='alfeld' gives C0 piecewise Bernstein polynomials
        on the barycentric refinement.
    """

    def __init__(self, ref_el, degree, variant="gll"):
        splitting, point_variant = parse_lagrange_variant(variant)
        if splitting is not None:
            ref_el = splitting(ref_el)
        dual = LagrangeDualSet(ref_el, degree, point_variant=point_variant, sort_entities=True)
        poly_set = BernsteinPolynomialSet(ref_el, degree, entity_ids=dual._macro_entity_ids)
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
        alpha = numpy.zeros(d_1, dtype=int)
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


def _domain_points(ref_el, degree):
    """Collect the local Bernstein multiindices of each C0 Bernstein function.

    :returns: A dict mapping the C0 Bernstein function supported on an entity
        with a restricted multiindex, keyed as ``(entity_dim, entity, beta)``,
        to the list of ``(cell, alpha)`` local multiindices on the cells
        containing that entity.
    """
    topology = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()
    inverse = {
        vertices: (entity_dim, entity)
        for entity_dim, entities in topology.items()
        for entity, vertices in entities.items()
    }
    points = {}
    for cell, cell_vertices in topology[sd].items():
        for alpha in multiindex_equal(sd + 1, degree):
            support = tuple(vertex for vertex, exponent in zip(cell_vertices, alpha) if exponent)
            entity_dim, entity = inverse[support]
            entity_vertices = topology[entity_dim][entity]
            beta = tuple(alpha[cell_vertices.index(vertex)] for vertex in entity_vertices)
            points.setdefault((entity_dim, entity, beta), []).append((cell, alpha))
    return points


def _default_entity_ids(ref_el, degree):
    """Construct the default Bernstein basis entity ordering."""
    topology = ref_el.get_topology()
    candidates = [(key, local_points[0][1])
                  for key, local_points in _domain_points(ref_el, degree).items()]
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
            if len(ids) != len(entity_alphas):
                raise ValueError(f"Expected {len(entity_alphas)} basis functions on entity {(dim, entity)}, "
                                 f"but got {len(ids)}")
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


def _ck_coefficients(ref_el, degree, order, vorder, entity_ids=None):
    """Construct smooth Bernstein coefficients from a minimal determining set.

    The spline space consists of the C0 piecewise polynomials that are
    C^order across the interior facets and C^vorder at the interior vertex of
    a simplex split around a single interior vertex. Following Lai and
    Schumaker (2007), Theorems 8.5 and 18.6, the minimal determining set
    consists of the Bernstein coefficients

    - on the boundary of the parent simplex,
    - at distance at least ``order`` from the interior facets and at least
      ``vorder`` from the interior vertex, and
    - in the ball of radius ``vorder - 1`` around the interior vertex, those
      at distance ``vorder - k (c + 1)`` from the interior vertex and greater
      than ``k`` from the interior facets not containing them, where ``c`` is
      the codimension of the smallest face of the split containing them.

    On the ball of radius ``vorder`` the spline is a polynomial of degree
    ``vorder``, which we write as the sum of ``b^k q_k`` over ``k``, where
    ``b`` is the product of the barycentric coordinates of the parent simplex.
    The coefficients of the third kind with a given ``k`` are multiples of the
    Bernstein coefficients of ``q_k`` on the boundary of the parent simplex,
    plus contributions from ``q_j`` with ``j < k``.

    The remaining coefficients are the unique solution of the smoothness
    conditions.
    """
    sd = ref_el.get_spatial_dimension()
    topology = ref_el.get_topology()
    interior_vertices = ref_el.get_interior_facets(0)
    if len(interior_vertices) > 1 or len(topology[0]) != sd + 1 + len(interior_vertices):
        raise NotImplementedError("Smooth Bernstein polynomial sets are only implemented "
                                  "on simplices split around a single interior vertex")
    vorder = order if vorder is None else max(order, vorder)

    def distance_to_interior_vertex(cell, alpha):
        return degree - sum(exponent for vertex, exponent in zip(topology[sd][cell], alpha)
                            if vertex in interior_vertices)

    def distance_to_interior_facets(cell, alpha):
        return min(exponent for vertex, exponent in zip(topology[sd][cell], alpha)
                   if vertex not in interior_vertices)

    def in_ball_determining_set(cell, alpha):
        exponents = [exponent for vertex, exponent in zip(topology[sd][cell], alpha)
                     if vertex not in interior_vertices]
        codim = exponents.count(0)
        k, remainder = divmod(vorder - distance_to_interior_vertex(cell, alpha), codim + 1)
        return k > 0 and remainder == 0 and all(exponent > k for exponent in exponents if exponent)

    def in_determining_set(cell, alpha):
        distance = distance_to_interior_vertex(cell, alpha)
        return (distance == degree
                or (distance >= vorder and distance_to_interior_facets(cell, alpha) >= order)
                or in_ball_determining_set(cell, alpha))

    domain_points = _domain_points(ref_el, degree)
    free = {key for key, local_points in domain_points.items()
            if any(in_determining_set(*point) for point in local_points)}

    def entity_multiindices(dim):
        return list(multiindex_equal(dim + 1, degree, imin=1))

    if entity_ids is None:
        c0_ids = _default_entity_ids(ref_el, degree)
        free_rows = sorted(row for dim, entities in c0_ids.items()
                           for entity, ids in entities.items()
                           for beta, row in zip(entity_multiindices(dim), ids)
                           if (dim, entity, beta) in free)
        renumbering = {row: i for i, row in enumerate(free_rows)}
        entity_ids = {dim: {entity: [renumbering[row]
                                     for beta, row in zip(entity_multiindices(dim), ids)
                                     if (dim, entity, beta) in free]
                            for entity, ids in entities.items()}
                      for dim, entities in c0_ids.items()}

    # Number the C0 functions outside the determining set after the determining set
    num_free = sum(len(ids) for entities in entity_ids.values() for ids in entities.values())
    dependent_rows = itertools.count(num_free)
    c0_ids = {dim: {} for dim in topology}
    for dim, entities in topology.items():
        betas = entity_multiindices(dim)
        for entity in entities:
            ids = entity_ids[dim][entity]
            num_dofs = sum((dim, entity, beta) in free for beta in betas)
            if len(ids) != num_dofs:
                raise ValueError(f"Expected {num_dofs} basis functions on entity {(dim, entity)}, "
                                 f"but got {len(ids)}")
            ids = iter(ids)
            c0_ids[dim][entity] = [next(ids) if (dim, entity, beta) in free else next(dependent_rows)
                                   for beta in betas]
    c0_coeffs = _c0_coefficients(ref_el, degree, c0_ids)

    row_of_point = {}
    for (dim, entity, beta), local_points in domain_points.items():
        row = c0_ids[dim][entity][entity_multiindices(dim).index(beta)]
        row_of_point.update(dict.fromkeys(local_points, row))

    # On the ball of radius vorder around the interior vertex, the spline is a
    # polynomial of degree vorder. Its Bernstein coefficients on each cell are
    # convex combinations of its Bernstein coefficients on the parent simplex,
    # which enter the conditions as auxiliary unknowns.
    parent_vertices = [vertex for vertex in topology[0] if vertex not in interior_vertices]
    parent_alphas = {alpha: i for i, alpha in enumerate(multiindex_equal(sd + 1, vorder))}
    B2R = numpy.vstack([numpy.transpose(ref_el.get_vertices_of_subcomplex(parent_vertices)),
                        numpy.ones(sd + 1)])
    num_columns = len(c0_coeffs) + len(interior_vertices) * len(parent_alphas)

    conditions = []
    for interior_vertex in ref_el.get_vertices_of_subcomplex(interior_vertices):
        bary = numpy.linalg.solve(B2R, [*interior_vertex, 1])[None, :]
        for m in range(vorder + 1):
            gammas = list(multiindex_equal(sd + 1, vorder - m))
            weights = [bernstein_db(bary, gamma)[0] for gamma in gammas]
            for local_points in domain_points.values():
                cell, alpha = local_points[0]
                if distance_to_interior_vertex(cell, alpha) != m:
                    continue
                exponents = dict(zip(topology[sd][cell], alpha))
                condition = numpy.zeros(num_columns)
                condition[row_of_point[cell, alpha]] += 1
                for gamma, weight in zip(gammas, weights):
                    parent_alpha = tuple(exponents.get(vertex, 0) + g
                                         for vertex, g in zip(parent_vertices, gamma))
                    condition[len(c0_coeffs) + parent_alphas[parent_alpha]] -= weight
                conditions.append(condition)

    # The C^m condition across the facet shared by cells a and b equates the
    # coefficients in cell b at distance m from the facet to the m-th
    # de Casteljau step in cell a towards the vertex of cell b opposite to the facet.
    for facet in ref_el.get_interior_facets(sd - 1):
        facet_vertices = topology[sd - 1][facet]
        cell_a, cell_b = ref_el.connectivity[(sd - 1, sd)][facet]
        vertex_b, = set(topology[sd][cell_b]) - set(facet_vertices)
        bary = ref_el.compute_barycentric_coordinates(
            ref_el.get_vertices_of_subcomplex((vertex_b,)), entity=(sd, cell_a))
        for m in range(1, order + 1):
            gammas = list(multiindex_equal(sd + 1, m))
            weights = [bernstein_db(bary, gamma)[0] for gamma in gammas]
            for beta in multiindex_equal(sd, degree - m):
                exponents = dict(zip(facet_vertices, beta))
                alpha_b = tuple(m if vertex == vertex_b else exponents.get(vertex, 0)
                                for vertex in topology[sd][cell_b])
                condition = numpy.zeros(num_columns)
                condition[row_of_point[cell_b, alpha_b]] += 1
                for gamma, weight in zip(gammas, weights):
                    alpha_a = tuple(exponents.get(vertex, 0) + g
                                    for vertex, g in zip(topology[sd][cell_a], gamma))
                    condition[row_of_point[cell_a, alpha_a]] -= weight
                conditions.append(condition)

    A = numpy.reshape(conditions, (-1, num_columns))
    dependent, *_ = scipy.linalg.lstsq(A[:, num_free:], -A[:, :num_free], cond=None,
                                       lapack_driver="gelsd")
    dependent = dependent[:len(c0_coeffs) - num_free]
    return c0_coeffs[:num_free] + dependent.T @ c0_coeffs[num_free:]
