from functools import reduce

import math
import numpy

import gem
from gem.node import MemoizerArg
from gem.optimise import filtered_replace_indices

from FIAT.reference_element import lattice_iter, lexicographical_iter
from FIAT.expansions import (lexicographic_permutation,
                             morton_index, C0_basis)
from finat.point_set import CollapsedTensorProductPointSet


class DuffyElement:
    """Mixin for simplicial elements whose nodal basis coincides with the
    Dubiner expansion set, enabling O(p^d) sum-factorized tabulation on
    collapsed (Duffy) tensor-product point sets.

    The basis functions are enumerated by a lattice multi-index rather
    than the flat degree-of-freedom index. For continuity=None elements
    (`finat.spectral.Legendre`) the flat index of a lattice point is its
    lattice-lexicographic index (`FIAT.expansions.lexicographic_permutation`),
    the dof order `FIAT.hierarchical.LegendreDual` uses. For continuity="C0"
    elements (`finat.spectral.IntegratedLegendre`) each flat dof index
    corresponds to a small, fixed number of lattice points, combined
    per `FIAT.expansions.c0_recombination_matrix` (see `get_sparse_coeffs`).
    """

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        """Return code for evaluating the element at known points on the
        reference element, using `duffy_evaluation` when ``ps`` has
        collapsed tensor-product structure, and falling back to the
        standard dense tabulation otherwise (``coordinate_mapping`` is
        ignored; `duffy_evaluation` only supports reference tabulation).

        Since `duffy_evaluation` already returns a flat-dof-indexed
        tabulation, no separate scatter step is needed here, and no
        special-cased path is needed for `Coefficient` evaluation either
        (`tsfc.fem.translate_coefficient`'s generic dense contraction
        applies uniformly) -- both compose transparently with
        `finat.tensorfiniteelement.TensorFiniteElement`
        (`Vector`/`TensorElement`), which only ever delegates
        `basis_evaluation` to the scalar base element.

        :returns: a dict mapping each derivative multi-index alpha to a
            `gem.ComponentTensor` of shape ``(self.space_dimension(),)``,
            matching the standard (non-factorized) tabulation's convention.
        """
        sd = self.cell.get_dimension()
        if not (isinstance(ps, CollapsedTensorProductPointSet)
                and (entity is None or entity == (sd, 0))
                and not self.complex.is_macrocell()):
            return super().basis_evaluation(order, ps, entity=entity, coordinate_mapping=coordinate_mapping)
        return self.duffy_evaluation(order, ps, entity)

    def get_sparse_coeffs(self):
        """Sparse recombination of this element's nodal basis in terms of
        its expansion set's raw (continuity=None) lattice tabulation: dof
        ``r`` reads ``sum_k row_coeff[r, k] * phi(row_multiindex[r, k])``,
        where ``phi`` is the raw per-lattice-multi-index tabulation
        `duffy_evaluation` computes.

        * continuity=None (`finat.spectral.Legendre`): each dof reads
          exactly one lattice point (``k`` has extent 1), at its
          lattice-lexicographic position
          (`FIAT.expansions.lexicographic_multiindices`); the expansion
          set's native order is raw Morton order, so the per-dof weight
          also undoes the lattice-lexicographic permutation
          (`FIAT.expansions.lexicographic_permutation`).
        * continuity="C0" (`finat.spectral.IntegratedLegendre`): each dof
          combines at most ``dim + 1`` lattice points
          (`FIAT.expansions.c0_recombination_matrix`), padded with
          zero-weight terms to a common extent; the expansion set's own
          `tabulate` is already entity-ordered by `FIAT.expansions.
          C0_basis`, so no further reordering applies.

        In both cases the per-dof weight also rescales the expansion set's
        native tabulation to this element's actual nodal basis, and is
        only valid when the nodal basis coincides with the expansion set
        (up to that reordering/rescaling) -- checked explicitly below.

        :returns: ``(row_multiindex, row_coeff)``, integer/float arrays of
            shape ``(ndof, k, dim)``/``(ndof, k)``.
        """
        degree = self.degree
        sd = self.cell.get_spatial_dimension()
        ndof = self.space_dimension()
        poly_set = self._element.get_nodal_basis()
        coeffs = poly_set.get_coeffs()
        expansion_set = poly_set.get_expansion_set()
        continuity_c0 = expansion_set.continuity == "C0"
        col = numpy.arange(ndof) if continuity_c0 else lexicographic_permutation(sd, degree)
        scale = coeffs[numpy.arange(ndof), col]
        expected = numpy.zeros_like(coeffs)
        expected[numpy.arange(ndof), col] = scale
        if not numpy.allclose(coeffs, expected):
            raise NotImplementedError("duffy_evaluation requires the element basis "
                                      "to coincide with the expansion set")

        if continuity_c0:
            row_multiindex, row_coeff = c0_recombination_matrix(sd, degree)
        else:
            row_multiindex = lexicographic_multiindices(sd, degree).reshape(ndof, 1, sd)
            row_coeff = numpy.ones((ndof, 1))
        return row_multiindex, row_coeff * scale[:, None]

    def duffy_evaluation(self, order, ps, entity=None):
        """Return the sum-factorized tabulation of the element on a
        collapsed tensor-product point set, indexed by the flat dof
        index (matching `self.get_indices()`'s convention).

        Internally the raw expansion set is tabulated per lattice
        multi-index ``(i_1, ..., i_d)``, ranging over the rectangular
        bounding box of the simplex lattice (entries outside the simplex
        lattice, ``i_1 + ... + i_d > degree``, tabulate to zero); each dof
        then reads the (small, fixed number of) lattice multi-index terms
        `get_sparse_coeffs` assigns it -- i.e. the returned tabulation
        is ``expansion_tabulation * coeffs``, with the recombination
        coefficients folded in here rather than left for callers to
        reapply.

        :returns: a dict mapping each derivative multi-index alpha to a
            `gem.ComponentTensor` of shape ``(self.space_dimension(),)``.
        """
        assert isinstance(ps, CollapsedTensorProductPointSet)
        cell_dim = self.cell.get_dimension()
        if entity is not None and entity != (cell_dim, 0):
            raise NotImplementedError("duffy_evaluation is only supported on the cell interior")
        if self.complex.is_macrocell():
            raise NotImplementedError("duffy_evaluation is not supported on split cells")

        degree = self.degree
        sd = self.cell.get_spatial_dimension()
        poly_set = self._element.get_nodal_basis()
        expansion_set = poly_set.get_expansion_set()
        etas = tuple(2.0 * f.points.ravel() - 1.0 for f in ps.factors)
        duffy = expansion_set.tabulate_duffy(degree, etas, order=order)

        multiindex = []
        for _ in range(sd):
            multiindex.append(gem.JaggedIndex(extent=degree + 1, parents=tuple(multiindex)))
        multiindex = tuple(multiindex)
        # Index expressions for the weight exponents m_t = i_1 + ... + i_{t-1}:
        # the first two are affine, the rest are looked up in an index table.
        m_indices = [0, *multiindex[:1]]
        for t in range(2, sd):
            # Clamp the exponent to stay in bounds on the rectangular bounding
            # box of the lattice: out-of-lattice entries are already zeroed by
            # the factors of the previous axes.
            table = reduce(numpy.add.outer, (numpy.arange(degree + 1),) * t)
            table = numpy.minimum(table, degree)
            m_indices.append(gem.VariableIndex(
                gem.Indexed(gem.Literal(table, dtype=gem.uint_type), multiindex[:t])))

        literals = {}

        def as_gem(table):
            key = id(table)
            try:
                return literals[key]
            except KeyError:
                return literals.setdefault(key, gem.Literal(table))

        result = {}
        for alpha, terms in duffy.items():
            exprs = []
            for coeff, factors in terms:
                expr = gem.Product(*(gem.Indexed(as_gem(table), (m, i, pt))
                                     for table, m, i, pt
                                     in zip(factors, m_indices, multiindex, ps.indices)))
                if coeff != 1.0:
                    expr = gem.Product(gem.Literal(coeff), expr)
                exprs.append(expr)
            result[alpha] = gem.Sum(*exprs)

        # Substitute each dof r's k lattice-multi-index terms (row_multiindex)
        # for `multiindex` in `result`, and contract with their weights
        # (row_coeff) over k: result[alpha][r] = sum_k row_coeff[r, k] *
        # result[alpha](multiindex=row_multiindex[r, k]).
        row_multiindex, row_coeff = self.get_sparse_coeffs()
        r = gem.Index(extent=self.space_dimension())
        k = gem.Index(extent=row_coeff.shape[1])
        coeff_expr = gem.Indexed(gem.Literal(row_coeff), (r, k))
        subst = tuple(
            (axis, gem.VariableIndex(gem.Indexed(gem.Literal(row_multiindex[..., t], dtype=gem.uint_type), (r, k))))
            for t, axis in enumerate(multiindex))
        mapper = MemoizerArg(filtered_replace_indices)
        return {alpha: gem.ComponentTensor(gem.IndexSum(gem.Product(coeff_expr, mapper(expr, subst)), (k,)), (r,))
                for alpha, expr in result.items()}


def c0_recombination_matrix(dim, n):
    """The dense matrix form of `C0_basis`'s row recombination + entity
    reorder, and the raw (continuity=None) lattice multi-index of each of
    its columns.

    `C0_basis` operates by pure row operations on whatever array it is
    given, so feeding it the identity recovers the combination it applies
    to any other tabulation as an explicit matrix: ``C0_basis(dim, n,
    [phi])[0] == R @ phi`` for the ``R`` returned here.
    """
    ndof = math.comb(n + dim, dim)
    R, = C0_basis(dim, n, [numpy.eye(ndof)])
    # `C0_basis` indexes its rows/columns by Morton position (via
    # `morton_index`), not by `lattice_iter`'s enumeration order -- invert
    # `morton_index` to get each Morton position's lattice multi-index.
    raw_multiindices = [None] * ndof
    for multiindex in lattice_iter(0, n + 1, dim):
        raw_multiindices[morton_index(dim, n, *multiindex)] = multiindex

    rows = [numpy.nonzero(R[r])[0] for r in range(ndof)]
    k = max(len(cols) for cols in rows)
    row_multiindex = numpy.zeros((ndof, k, dim), dtype=int)
    row_coeff = numpy.zeros((ndof, k))
    for r, cols in enumerate(rows):
        pad = cols[-1]
        for t in range(k):
            m = cols[t] if t < len(cols) else pad
            row_multiindex[r, t] = raw_multiindices[m]
            if t < len(cols):
                row_coeff[r, t] = R[r, m]
    return row_multiindex, row_coeff


def lexicographic_multiindices(dim, n):
    """Lattice multi-index of each lattice-lexicographic dof position.

    :returns: an integer array of shape ``(ndof, dim)`` such that row ``j``
        is the lattice multi-index of dof ``j`` in lattice-lexicographic
        order (see `lexicographic_permutation`).
    """
    ndof = math.comb(n + dim, dim)
    return numpy.fromiter((i for multiindex in lexicographical_iter(dim, n) for i in multiindex),
                          dtype=int, count=ndof * dim).reshape(ndof, dim)


def lexicographic_offsets(dim, n):
    """Per-axis-prefix offset tables for lattice-lexicographic order: for
    dof index ``j`` with lattice multi-index ``(i_1, ..., i_dim)``,
    ``j == offsets[dim - 2][i_1, ..., i_{dim - 1}] + i_dim`` (the flat
    index is affine, slope 1, in the innermost coordinate).

    Prefixes off the simplex lattice are set to ``ndof``, not ``0``, so
    they compare as unreachable by any real flat index rather than as a
    valid (wrong) one.

    :returns: ``dim - 1`` integer arrays, the ``t``-th of shape
        ``(n + 1,) * (t + 1)``. Empty for ``dim == 1``.
    """
    ndof = math.comb(n + dim, dim)
    multiindices = lexicographic_multiindices(dim, n)
    offsets = []
    for t in range(dim - 1):
        shape = (n + 1,) * (t + 1)
        table = numpy.full(shape, ndof, dtype=int)
        prefixes = multiindices[:, :t + 1]
        seen = numpy.zeros(shape, dtype=bool)
        for j, prefix in enumerate(map(tuple, prefixes)):
            if not seen[prefix]:
                table[prefix] = j
                seen[prefix] = True
        offsets.append(table)
    return tuple(offsets)
