# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
        macrocell, or a dict mapping the dimension (facets or vertices) and
        the entity id to the order of smoothness at that entity.
    :kwarg vorder: The order of super-smoothness at the interior vertices of a
        macrocell.  Only allowed when ``order`` is an integer.
    :kwarg entity_ids: An optional entity-to-basis-index map defining the
        ordering of the basis. Each entity carries the basis functions
        associated with the Bernstein coefficients on that entity that lie in
        the minimal determining set.
    """
    def __init__(self, ref_el, degree, order=0, vorder=None, entity_ids=None):
        if degree < 0:
            raise ValueError("Bernstein polynomial sets require a nonnegative degree")
        coeffs = _ck_coefficients(ref_el, degree, order, vorder, entity_ids)
        expansion_set = BernsteinExpansionSet(ref_el)
        super().__init__(ref_el, degree, degree, expansion_set, coeffs)


class Bernstein(CiarletElement):
    """A C0 finite element with Bernstein polynomials as basis functions.

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


def _entity_support_mis(ref_el, degree):
    """Collect the local Bernstein multiindices supporting each C0 function.

    :returns: A dict mapping the C0 Bernstein function associated with an
        entity and restricted multiindex, keyed as ``(entity_dim, entity_id,
        beta)``, to the list of ``(cell_id, alpha)`` pairs locating its local
        Bernstein multiindices.
    """
    topology = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()
    connectivity = ref_el.get_connectivity()

    betas = {dim: tuple(multiindex_equal(dim + 1, degree, imin=1))
             for dim in topology}
    entity_support_mis = {}
    for cell, cell_vertices in topology[sd].items():
        for entity_dim, entities in topology.items():
            for entity_id in connectivity[(sd, entity_dim)][cell]:
                entity_vertices = entities[entity_id]
                for beta in betas[entity_dim]:
                    # A C0 Bernstein function is identified by its entity and
                    # its Bernstein multiindex on that entity.  Record every
                    # cell-local multiindex that contributes to that function.
                    alpha = [0] * (sd + 1)
                    for vertex, exponent in zip(entity_vertices, beta):
                        alpha[cell_vertices.index(vertex)] = exponent
                    entity_support = (entity_dim, entity_id, beta)
                    entity_support_mis.setdefault(entity_support, []).append(
                        (cell, tuple(alpha)))
    return entity_support_mis


def _entity_support_rows(ref_el, degree, entity_support_mis, entity_ids=None):
    """Number entity-associated Bernstein functions in entity order."""
    # The row number is the global C0 Bernstein-function number.  Preserve the
    # caller's macroelement ordering when it is supplied; otherwise use the
    # usual entity/multiindex ordering.
    if entity_ids is None:
        ordered = sorted(entity_support_mis,
                         key=lambda entity_support: tuple(
                             sorted(entity_support_mis[entity_support][0][1], reverse=True)))
        return {entity_support: row for row, entity_support in enumerate(ordered)}

    rows = {}
    for dim, entities in ref_el.get_topology().items():
        betas = list(multiindex_equal(dim + 1, degree, imin=1))
        for entity in entities:
            entity_supports = [(dim, entity, beta) for beta in betas
                               if (dim, entity, beta) in entity_support_mis]
            ids = entity_ids[dim][entity]
            if len(ids) != len(entity_supports):
                raise ValueError(f"Expected {len(entity_supports)} basis functions on entity "
                                 f"{(dim, entity)}, "
                                 f"but got {len(ids)}")
            rows.update(zip(entity_supports, ids))
    return rows


def _assemble_c0_coefficients(ref_el, degree, entity_support_mis, rows):
    """Assemble the C0 basis from its local Bernstein supports."""
    sd = ref_el.get_spatial_dimension()
    local_alpha_ids = {alpha: i for i, alpha in enumerate(multiindex_equal(sd + 1, degree))}
    num_local = len(local_alpha_ids)
    num_cells = len(ref_el.get_topology()[sd])
    coeffs = numpy.zeros((len(rows), num_cells * num_local))
    # Rows are global C0 functions and columns are cell-local Bernstein
    # coefficients.  A function is the sum of its local supports, so each
    # support contributes a single one in its corresponding column.
    for entity_support, local_support in entity_support_mis.items():
        for cell, alpha in local_support:
            coeffs[rows[entity_support], cell * num_local + local_alpha_ids[alpha]] = 1
    return coeffs


def _smoothness_relations(ref_el, degree, order):
    """Collect the Bernstein smoothness relations implied by the orders.

    :returns: A set of ``(facet, m, beta)`` triples.  Each triple is the
        relation matching the coefficient at normal distance ``m`` from the
        interior ``facet`` and facet multiindex ``beta`` with the ``m``-th de
        Casteljau step of the coefficients on the other side.
    """
    sd = ref_el.get_spatial_dimension()
    facets = ref_el.get_topology()[sd - 1]
    interior_facets = set(ref_el.get_interior_facets(sd - 1))
    relations = set()
    # C^r across a facet matches the first r normal layers.
    for facet, forder in order.get(sd - 1, {}).items():
        for m in range(1, min(forder, degree) + 1):
            relations.update((facet, m, beta) for beta in multiindex_equal(sd, degree - m))
    # C^rho at a vertex v matches the coefficients in the disk D_rho(v) across
    # every interior facet attached to v (Lai and Schumaker, Lemma 5.10).
    for vertex, vorder in order.get(0, {}).items():
        for facet in ref_el.connectivity[(0, sd - 1)][vertex]:
            if facet not in interior_facets:
                continue
            local = facets[facet].index(vertex)
            for m in range(1, min(vorder, degree) + 1):
                relations.update((facet, m, beta) for beta in multiindex_equal(sd, degree - m)
                                 if degree - beta[local] <= vorder)
    return relations


def _independent_columns(A, classes, tol=1e-8):
    """Select a column basis of ``A``, one priority class at a time.

    :arg A: The matrix.
    :arg classes: A list of lists of column indices, in decreasing priority.
    :kwarg tol: The relative residual below which a column is considered to
        be in the span of the columns already selected.
    :returns: The indices of the selected columns.

    Within each class, the column with the largest residual relative to its
    norm is selected first (column pivoting), so that the selected columns are
    well conditioned.  The class is exhausted when every remaining column lies
    in the span of the columns selected so far.
    """
    norms = numpy.linalg.norm(A, axis=0)
    R = A.astype(float).copy()
    selected = []
    for columns in classes:
        columns = [j for j in columns if norms[j] > 0]
        while columns:
            ratios = numpy.linalg.norm(R[:, columns], axis=0) / norms[columns]
            k = numpy.argmax(ratios)
            if ratios[k] <= tol:
                break
            j = columns.pop(k)
            q = R[:, j] / numpy.linalg.norm(R[:, j])
            # Two passes of Gram-Schmidt keep the residuals orthogonal.
            for _ in range(2):
                R -= numpy.outer(q, q @ R)
            selected.append(j)
    return selected


def _ck_coefficients(ref_el, degree, order, vorder, entity_ids=None):
    """Construct smooth Bernstein coefficients from a minimal determining set.

    The spline space consists of C0 piecewise polynomials with the requested
    smoothness on the specified facets and vertices.  The smoothness relations
    on the C0 Bernstein coefficients form a matrix ``A``, whose null space is
    the spline space.  A minimal determining set (MDS) is selected from that
    null space, and the remaining coefficients are solved for.
    """
    sd = ref_el.get_spatial_dimension()
    topology = ref_el.get_topology()
    if degree == 0:
        return numpy.ones((1, len(topology[sd])))

    # Convert the scalar API to the entity-wise representation used below:
    # one scalar order applies to every interior facet and interior vertex.
    if not isinstance(order, dict):
        vorder = order if vorder is None else max(order, vorder)
        order = {
            sd - 1: dict.fromkeys(ref_el.get_interior_facets(sd - 1), order),
            0: dict.fromkeys(ref_el.get_interior_facets(0), vorder),
        }
    elif vorder is not None:
        raise ValueError("vorder cannot be specified with entity-specific orders")

    if not set(order).issubset({0, sd - 1}):
        raise NotImplementedError("Only facet and vertex constraints have been implemented")
    if not set(order.get(sd - 1, {})).issubset(topology[sd - 1]):
        raise ValueError("Facet orders must be specified on facets of the reference element")
    if not set(order.get(0, {})).issubset(topology[0]):
        raise ValueError("Vertex orders must be specified on vertices of the reference element")

    entity_support_mis = _entity_support_mis(ref_el, degree)
    relations = _smoothness_relations(ref_el, degree, order)
    if not relations:
        rows = _entity_support_rows(ref_el, degree, entity_support_mis, entity_ids)
        return _assemble_c0_coefficients(ref_el, degree, entity_support_mis, rows)

    supports = list(entity_support_mis)
    column = {support: j for j, entity_support in enumerate(supports)
              for support in entity_support_mis[entity_support]}

    # Bernstein relations keep the smoothness constraints algebraic and local.
    # For a coefficient on cell_b at normal distance m from the shared facet,
    # the matching coefficient on cell_a is the m-th de Casteljau step toward
    # cell_b's opposite vertex.
    A = numpy.zeros((len(relations), len(supports)))
    for i, (facet, m, beta) in enumerate(sorted(relations)):
        facet_vertices = topology[sd - 1][facet]
        cell_a, cell_b = ref_el.connectivity[(sd - 1, sd)][facet]
        vertex_b, = set(topology[sd][cell_b]) - set(facet_vertices)
        bary = ref_el.compute_barycentric_coordinates(
            ref_el.get_vertices_of_subcomplex((vertex_b,)), entity=(sd, cell_a))
        exponents = dict(zip(facet_vertices, beta))
        alpha_b = tuple(m if vertex == vertex_b else exponents.get(vertex, 0)
                        for vertex in topology[sd][cell_b])
        A[i, column[cell_b, alpha_b]] += 1
        for gamma in multiindex_equal(sd + 1, m):
            alpha_a = tuple(exponents.get(vertex, 0) + g
                            for vertex, g in zip(topology[sd][cell_a], gamma))
            A[i, column[cell_a, alpha_a]] -= bernstein_db(bary, gamma)[0]
    A /= abs(A).max(axis=1, keepdims=True)

    # The smooth splines are the null space of A.  A set M of coefficients is
    # determining if and only if the null space basis N restricted to M is
    # invertible (Lai and Schumaker, Section 5.6).  Select M by column
    # pivoting on N.T so that the dependent coefficients, N[D] @ inv(N[M]),
    # stay bounded.  Boundary coefficients are selected first, and then
    # interior coefficients from the highest to the lowest entity dimension.
    _, sigma, Vt = numpy.linalg.svd(A)
    rank = numpy.count_nonzero(sigma > 1e-10 * sigma[0])
    interior_entities = {dim: set(ref_el.get_interior_facets(dim)) for dim in topology}
    classes = {}
    for j, (dim, entity, _) in enumerate(supports):
        classes.setdefault((entity not in interior_entities[dim], dim), []).append(j)
    free = _independent_columns(Vt[rank:], [classes[c] for c in sorted(classes, reverse=True)])
    if len(free) != len(supports) - rank:
        raise ValueError("Smoothness constraints do not determine all coefficients")

    rows = _entity_support_rows(
        ref_el, degree, {supports[j]: entity_support_mis[supports[j]] for j in free}, entity_ids)
    num_determining = len(rows)
    # Keep determining rows first and append all dependent C0 supports.
    for entity_support in supports:
        rows.setdefault(entity_support, len(rows))
    c0_coeffs = _assemble_c0_coefficients(ref_el, degree, entity_support_mis, rows)

    # Solve the smoothness relations for the dependent coefficients.
    A = A[:, numpy.argsort([rows[entity_support] for entity_support in supports])]
    A_free, A_dependent = A[:, :num_determining], A[:, num_determining:]
    X, *_ = scipy.linalg.lstsq(A_dependent, -A_free, lapack_driver="gelsy")
    if not numpy.allclose(A_free + A_dependent @ X, 0, atol=1e-10):
        raise ValueError("Smoothness constraints are inconsistent with the determining set")
    coeffs = c0_coeffs[:num_determining] + X.T @ c0_coeffs[num_determining:]
    # The Bernstein coefficients of constants are all one, so the basis
    # functions sum to one exactly.
    coeffs[-1] = 1 - coeffs[:-1].sum(axis=0)
    return coeffs
