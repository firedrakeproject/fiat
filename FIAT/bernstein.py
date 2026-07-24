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
from FIAT.expansions import ExpansionSet
from FIAT.polynomial_set import PolynomialSet, mis
from FIAT.pointwise_dual import compute_pointwise_dual
from FIAT.reference_element import (SimplicialComplex, lexicographical_iter,
                                    make_lattice)


def _bernstein_factors(n: int, eta: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    """Tabulate univariate Bernstein factors for a collapsed simplex axis.

    Parameters
    ----------
    n : int
        The total polynomial degree.
    eta : numpy.ndarray
        Points on ``[-1, 1]``.

    Returns
    -------
    tuple
        Value, degree-lowered, and degree-and-index-lowered tables.

    """
    eta = numpy.asarray(eta)
    z = 0.5 * (1.0 + eta)
    shape = (n + 1, n + 1, len(eta))
    values = numpy.zeros(shape, dtype=z.dtype)
    lower = numpy.zeros(shape, dtype=z.dtype)
    shifted = numpy.zeros(shape, dtype=z.dtype)
    for m in range(n + 1):
        degree = n - m
        for i in range(degree + 1):
            values[m, i] = math.comb(degree, i) * z**i * (1.0 - z)**(degree-i)
        for i in range(degree):
            lower[m, i] = math.comb(degree - 1, i) * z**i * (1.0 - z)**(degree-1-i)
            shifted[m, i+1] = lower[m, i]
    return values, lower, shifted


class BernsteinExpansionSet(ExpansionSet):
    """Bernstein polynomial expansion set on a simplex."""

    def __init__(self, ref_el: SimplicialComplex) -> None:
        if not ref_el.is_simplex():
            raise ValueError("Bernstein expansion sets require a simplex")
        super().__init__(ref_el, scale=1.0)

    def get_duffy_permutation(self, n: int) -> numpy.ndarray:
        """Map Duffy lattice positions to Bernstein expansion members."""
        dim = self.ref_el.get_spatial_dimension()
        return numpy.arange(math.comb(n + dim, dim))

    def _tabulate_on_cell(self, n: int, pts: numpy.ndarray, order: int = 0,
                          cell: int = 0, direction: numpy.ndarray | None = None) -> dict:
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
            for i, lattice_index in enumerate(lexicographical_iter(dim, n))
            for o in range(order + 1)
            for derivative, vec in bernstein_Dx(
                B, (n - sum(lattice_index), *reversed(lattice_index)), o, R2B
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

    def tabulate_duffy(self, n: int, eta_pts: tuple, order: int = 0,
                       cell: int = 0) -> dict:
        """Tabulate Bernstein polynomials in separable collapsed form.

        The raw lattice multi-index ``j`` represents the barycentric
        exponent tuple ``(n - sum(j), *reversed(j))``. Reversing the
        collapsed axes gives the product from Ainsworth et al.,
        ``prod_t B[j_t, n - sum(j[:t])]``.
        """
        if order > 1:
            raise NotImplementedError("tabulate_duffy is limited to first derivatives")

        dim = self.ref_el.get_spatial_dimension()
        assert len(eta_pts) == dim
        A, _ = self.affine_mappings[cell]
        axes = tuple(reversed(range(dim)))
        tables = tuple(_bernstein_factors(n, eta_pts[axis]) for axis in axes)
        values = tuple(table[0] for table in tables)
        result = {(0,) * dim: [(1.0, tuple(zip(axes, values)))]}

        if order:
            lower = tuple(table[1] for table in tables)
            # On the default (-1, 1) simplex,
            # d/dxi_l B_alpha^n = n/2 * (B_{alpha-e_{l+1}}^{n-1}
            #                            - B_{alpha-e_0}^{n-1}).
            # Both degree-lowered terms retain the separable product form.
            for k in range(dim):
                terms = []
                for ell in range(dim):
                    coeff = 0.5 * n * A[ell, k]
                    if coeff == 0.0:
                        continue
                    s = dim - 1 - ell
                    shifted = tuple(table[1] if t < s else
                                    table[2] if t == s else table[0]
                                    for t, table in enumerate(tables))
                    terms.extend(((coeff, tuple(zip(axes, shifted))),
                                  (-coeff, tuple(zip(axes, lower)))))
                if not terms:
                    terms.append((0.0, tuple(zip(axes, values))))
                alpha = tuple(int(i == k) for i in range(dim))
                result[alpha] = terms
        return result


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
        kss = [(degree - sum(alpha), *reversed(alpha))
               for alpha in lexicographical_iter(dim, degree)]

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

    def __init__(self, ref_el: SimplicialComplex, degree: int) -> None:
        dual = BernsteinDualSet(ref_el, degree)
        k = 0  # 0-form
        super().__init__(ref_el, dual, degree, k)

        expansion_set = BernsteinExpansionSet(ref_el)
        size = math.comb(degree + ref_el.get_spatial_dimension(),
                         ref_el.get_spatial_dimension())
        coeffs = numpy.eye(size)
        self.poly_set = PolynomialSet(ref_el, degree, degree, expansion_set, coeffs)

        pts = make_lattice(ref_el.vertices, degree, variant="gll")
        newdual = compute_pointwise_dual(self, pts)
        self.dual = newdual

    def degree(self):
        """The degree of the polynomial space."""
        return self.get_order()

    def get_nodal_basis(self) -> PolynomialSet:
        """Return the Bernstein basis encoded as a polynomial set."""
        return self.poly_set

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

        return self.poly_set.get_expansion_set()._tabulate(
            self.degree(), cell_points, order=order)


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
        return numpy.zeros(points.shape[:-1])
    elif all(k == 0 for k in ls):
        return numpy.ones(points.shape[:-1])
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
