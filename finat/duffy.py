from functools import reduce

import numpy

import gem
from gem.node import MemoizerArg
from gem.optimise import contraction, ffc_rounding, filtered_replace_indices
from FIAT.expansions import (lexicographic_offsets, lexicographic_permutation,
                             lexicographic_multiindices,
                             c0_recombination_row_terms, c0_recombination_col_terms)
from finat.point_set import CollapsedTensorProductPointSet


def _index_value(index):
    """GEM scalar expression for the numeric value of a `gem.Index`."""
    # FIXME write proper codegen for this hack
    return gem.Indexed(gem.Literal(numpy.arange(index.extent, dtype=gem.uint_type),
                                   dtype=gem.uint_type), (index,))


def _as_gem_uint(n):
    return gem.Literal(n, dtype=gem.uint_type)


def _flat_index_expr(multiindex, ndof, offsets):
    """Flat lattice-lexicographic index of a lattice multi-index: one small
    table lookup (`FIAT.expansions.lexicographic_offsets`) over the outer
    coordinates, plus the innermost coordinate.

    ``multiindex`` ranges over the rectangular bounding box of the simplex
    lattice, including out-of-lattice points, for which the table already
    returns ``ndof``; the result is clamped to ``ndof - 1`` there since the
    corresponding tabulation is zero and the meaningless index is only
    ever multiplied by zero.
    """
    if len(multiindex) == 1:
        expr = _index_value(multiindex[0])
    else:
        offset = gem.Indexed(gem.Literal(offsets[-1], dtype=gem.uint_type), multiindex[:-1])
        expr = gem.Sum(offset, _index_value(multiindex[-1]))
    return gem.MinValue(expr, _as_gem_uint(ndof - 1))


def _element_scale(element):
    """Per-dof rescaling relating ``element``'s nodal basis to its
    expansion set's native `tabulate` ordering: dof ``r`` reads
    expansion-set member ``col[r]`` with weight ``scale[r]``.

    * continuity=None (`finat.spectral.Legendre`): the expansion set's
      native order is raw Morton order, and the permutation reorders it
      to lattice-lexicographic (`FIAT.expansions.lexicographic_permutation`).
    * continuity="C0" (`finat.spectral.IntegratedLegendre`): the expansion
      set's own `tabulate` is already entity-ordered by `FIAT.expansions.
      C0_basis`, so the permutation is the identity.

    :returns: ``(continuity_c0, scale)``.
    """
    degree = element.degree
    sd = element.cell.get_spatial_dimension()
    poly_set = element._element.get_nodal_basis()
    coeffs = poly_set.get_coeffs()
    expansion_set = poly_set.get_expansion_set()
    continuity_c0 = expansion_set.continuity == "C0"
    col = numpy.arange(len(coeffs)) if continuity_c0 else lexicographic_permutation(sd, degree)
    scale = coeffs[numpy.arange(len(coeffs)), col]
    expected = numpy.zeros_like(coeffs)
    expected[numpy.arange(len(coeffs)), col] = scale
    if not numpy.allclose(coeffs, expected):
        raise NotImplementedError("duffy_evaluation requires the element basis "
                                  "to coincide with the expansion set")
    return continuity_c0, scale


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
    per `FIAT.expansions.c0_recombination_row_terms`/
    `c0_recombination_col_terms` (see `_element_scale`).
    """

    def duffy_evaluation(self, order, ps, entity=None):
        """Return the sum-factorized tabulation of the element on a
        collapsed tensor-product point set.

        The basis functions are enumerated by a lattice multi-index
        ``(i_1, ..., i_d)`` rather than the flat dof index, ranging over
        the rectangular bounding box of the simplex lattice; entries
        outside the simplex lattice (``i_1 + ... + i_d > degree``)
        tabulate to zero.

        :returns: ``(multiindex, result)``, where ``multiindex`` is a
            tuple of d `gem.JaggedIndex` of extent ``degree + 1``, each
            with the preceding indices as parents, and ``result`` maps
            each derivative multi-index alpha to a scalar gem expression
            free in ``multiindex + ps.indices``.
        """
        assert isinstance(ps, CollapsedTensorProductPointSet)
        cell_dim = self.cell.get_dimension()
        if entity is not None and entity != (cell_dim, 0):
            raise NotImplementedError("duffy_evaluation is only supported on the cell interior")
        if self.complex.is_macrocell():
            raise NotImplementedError("duffy_evaluation is not supported on split cells")

        degree = self.degree
        sd = self.cell.get_spatial_dimension()
        continuity_c0, scale = _element_scale(self)
        poly_set = self._element.get_nodal_basis()
        expansion_set = poly_set.get_expansion_set()
        etas = tuple(2.0 * f.points.ravel() - 1.0 for f in ps.factors)
        duffy = expansion_set.tabulate_duffy(degree, etas, order=order)
        if not continuity_c0:
            # continuity=None: the flat dof index directly *is* a raw
            # lattice multi-index (up to the lattice-lexicographic
            # reorder), so the per-dof scale -- uniform here, see
            # `_element_scale` -- can be folded in once, right away.
            unit = scale[0]
            duffy = {alpha: [(unit * coeff, factors) for coeff, factors in terms]
                     for alpha, terms in duffy.items()}

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
        return multiindex, result

    def _duffy_applies(self, ps, entity):
        """Whether `duffy_evaluation` can tabulate on ``ps``/``entity``."""
        cell_dim = self.cell.get_dimension()
        return (isinstance(ps, CollapsedTensorProductPointSet)
                and (entity is None or entity == (cell_dim, 0))
                and not self.complex.is_macrocell())

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        """Return code for evaluating the element at known points on the
        reference element, using `duffy_evaluation` when ``ps`` has
        collapsed tensor-product structure, and falling back to the
        standard dense tabulation otherwise (``coordinate_mapping`` is
        ignored; `duffy_evaluation` only supports reference tabulation).

        :returns: a dict mapping each derivative multi-index alpha to a
            `gem.ComponentTensor` of shape ``(self.space_dimension(),)``,
            matching the standard (non-factorized) tabulation's convention.
        """
        if not self._duffy_applies(ps, entity):
            return super().basis_evaluation(order, ps, entity=entity, coordinate_mapping=coordinate_mapping)
        multiindex, result = self.duffy_evaluation(order, ps, entity)
        return _scatter_to_dof_index(multiindex, result, self)

    def duffy_contraction(self, order, ps, entity, vec, epsilon):
        """Contract a sum-factorized tabulation against a coefficient
        vector (of shape ``(self.space_dimension(),)``), only for
        derivative multi-indices alpha with ``sum(alpha) == order``.

        The sum over the flat dof index is rewritten as a sum over the
        lattice multi-index, gathering the coefficient vector at each
        lattice point: a small table lookup (`_flat_index_expr`) for
        continuity=None, or at most 2 dofs per lattice point
        (`FIAT.expansions.c0_recombination_col_terms`) for continuity="C0".
        `gem.optimise.contraction` then sum-factorizes the resulting
        nested sum, exploiting the same axis-separable structure that
        makes `duffy_evaluation` itself O(p^d).

        :returns: a dict mapping alpha to a `gem.ComponentTensor` over
            ``self.get_value_indices()``, free in the point indices only.
        """
        multiindex, result = self.duffy_evaluation(order, ps, entity)
        result = {alpha: ffc_rounding(table, epsilon)
                  for alpha, table in result.items()
                  if sum(alpha) == order}

        continuity_c0, scale = _element_scale(self)
        if continuity_c0:
            col_dof, col_coeff = c0_recombination_col_terms(len(multiindex), self.degree)
            col_coeff = col_coeff * scale[col_dof]
            k = gem.Index(extent=col_coeff.shape[-1])
            dof_index = gem.VariableIndex(gem.Indexed(gem.Literal(col_dof, dtype=gem.uint_type), multiindex + (k,)))
            coeff_expr = gem.Indexed(gem.Literal(col_coeff), multiindex + (k,))
            gathered = gem.Product(coeff_expr, gem.Indexed(vec, (dof_index,)))
            vec_r, = gem.optimise.remove_componenttensors([gem.IndexSum(gathered, (k,))])
        else:
            ndof = self.space_dimension()
            offsets = lexicographic_offsets(len(multiindex), self.degree)
            r_index = gem.VariableIndex(_flat_index_expr(multiindex, ndof, offsets))
            vec_r, = gem.optimise.remove_componenttensors([gem.Indexed(vec, (r_index,))])

        zeta = self.get_value_indices()
        value_dict = {}
        for alpha, expr in result.items():
            value = gem.IndexSum(gem.Product(expr, vec_r), multiindex)
            value_dict[alpha] = gem.ComponentTensor(contraction(value), zeta)
        return value_dict


def _scatter_to_dof_index(multiindex, result, element):
    """Reshape a lattice-indexed tabulation (``multiindex``/``result``, as
    returned by `DuffyElement.duffy_evaluation`) into a flat-dof-indexed
    one: a dict mapping each derivative multi-index alpha to a dense
    `gem.ComponentTensor` of shape ``(element.space_dimension(),)``,
    matching the standard (non-factorized) tabulation's convention.

    For continuity=None, each dof reads exactly one lattice point, looked
    up from a precomputed ``(ndof, dim)`` table (`FIAT.expansions.
    lexicographic_multiindices`). For continuity="C0", each dof is a fixed
    linear combination of at most ``dim + 1`` lattice points (`FIAT.
    expansions.c0_recombination_row_terms`): the same substitution is
    applied once per combined term (a small, fixed extent ``k`` index) and
    summed with the combination coefficients.

    Both cases substitute the lattice axes with a `gem.VariableIndex`
    table lookup expressing the lattice multi-index as a function of the
    flat index ``r``, keeping ``r`` a genuine flat index throughout
    (matching `element.get_indices()`'s convention). A plain table lookup,
    rather than computing the inverse index via arithmetic, is deliberate:
    arithmetic here made loopy's scheduler fail for bilinear forms where
    both the test and trial bases are scattered simultaneously (see
    `tsfc/AGENTS.md`); the table lookup is both schedulable and slightly
    cheaper in flops.
    """
    continuity_c0, scale = _element_scale(element)
    ndof = element.space_dimension()
    r = gem.Index(extent=ndof)
    mapper = MemoizerArg(filtered_replace_indices)
    if continuity_c0:
        row_multiindex, row_coeff = c0_recombination_row_terms(len(multiindex), element.degree)
        row_coeff = row_coeff * scale[:, None]
        k = gem.Index(extent=row_coeff.shape[-1])
        coeff_expr = gem.Indexed(gem.Literal(row_coeff), (r, k))
        subst = tuple(
            (axis, gem.VariableIndex(gem.Indexed(gem.Literal(row_multiindex[..., t], dtype=gem.uint_type), (r, k))))
            for t, axis in enumerate(multiindex))
        return {alpha: gem.ComponentTensor(gem.IndexSum(gem.Product(coeff_expr, mapper(expr, subst)), (k,)), (r,))
                for alpha, expr in result.items()}
    if len(multiindex) == 1:
        # 1D: the flat index already *is* the (only) lattice coordinate.
        subst = ((multiindex[0], r),)
    else:
        table = lexicographic_multiindices(len(multiindex), element.degree)
        lit = gem.Literal(table, dtype=gem.uint_type)
        subst = tuple((axis, gem.VariableIndex(gem.Indexed(lit, (r, t))))
                      for t, axis in enumerate(multiindex))
    return {alpha: gem.ComponentTensor(mapper(expr, subst), (r,))
            for alpha, expr in result.items()}
