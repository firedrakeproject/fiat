from functools import reduce

import numpy

import gem
from gem.node import MemoizerArg
from gem.optimise import contraction, ffc_rounding, filtered_replace_indices
from FIAT.expansions import lexicographic_offsets, lexicographic_permutation
from finat.point_set import CollapsedTensorProductPointSet


def _index_value(index):
    """GEM scalar expression for the numeric value of a `gem.Index`."""
    return gem.Indexed(gem.Literal(numpy.arange(index.extent, dtype=gem.uint_type),
                                   dtype=gem.uint_type), (index,))


def _one(n):
    return gem.Literal(n, dtype=gem.uint_type)


def _flat_index_expr(multiindex, ndof, offsets):
    """Flat lattice-lexicographic index of a lattice multi-index, as index
    arithmetic against a small table.

    `FIAT.expansions.lexicographic_offsets`' tables make the dof order
    FIAT now uses (`FIAT.hierarchical.LegendreDual`) affine in the
    innermost lattice coordinate for each fixed value of the outer
    coordinates, so the flat index is just one small table lookup (over
    the outer coordinates only) plus the innermost coordinate -- unlike
    the old Morton (total-degree-major) index, whose formula mixes all
    coordinates non-separably and needed genuine arithmetic.

    ``multiindex`` ranges over the rectangular bounding box of the
    simplex lattice (each `gem.JaggedIndex` has extent ``degree + 1``),
    including points outside the simplex lattice (``sum(multiindex) >
    degree``), for which the table lookup already returns ``ndof`` (see
    `FIAT.expansions.lexicographic_offsets`), overshooting ``ndof - 1``.
    Like the old Morton table, the result is clamped to ``ndof - 1`` for
    those points: the corresponding tabulation is exactly zero there, so
    the clamped (but otherwise meaningless) index is only ever multiplied
    by zero.

    Parameters
    ----------
    multiindex : tuple of gem.Index
        The lattice multi-index, of length 1, 2, or 3.
    ndof : int
        The element's space dimension; the result is clamped to
        ``ndof - 1``.
    offsets : tuple of numpy.ndarray
        `FIAT.expansions.lexicographic_offsets`'s per-axis-prefix tables.

    Returns
    -------
    gem.Node
        A scalar expression of dtype `gem.uint_type`, free in
        ``multiindex``, equal to the flat lattice-lexicographic index,
        clamped to ``ndof - 1``.

    """
    if len(multiindex) == 1:
        expr = _index_value(multiindex[0])
    else:
        offset = gem.Indexed(gem.Literal(offsets[-1], dtype=gem.uint_type), multiindex[:-1])
        expr = gem.Sum(offset, _index_value(multiindex[-1]))
    return gem.MinValue(expr, _one(ndof - 1))


def _step_ge(a, b):
    """1 if a >= b else 0, for gem uint scalar expressions a, b."""
    return gem.Conditional(gem.Comparison(">=", a, b), _one(1), _one(0))


def _step_le(a, b):
    """1 if a <= b else 0, for gem uint scalar expressions a, b."""
    return gem.Conditional(gem.Comparison("<=", a, b), _one(1), _one(0))


def _bounded_sub(a, b, bound):
    """``a - b`` as a gem uint expression, given ``0 <= b <= a <= bound``.

    Unsigned index arithmetic has no subtraction (subtracting would
    underflow whenever ``b > a``, which cannot be ruled out for `Node`
    dtypes without a signed type), so this counts, for each candidate
    ``0 < k <= bound``, whether ``b + k`` is still ``<= a``: exactly
    ``a - b`` of them are.
    """
    terms = [_step_le(gem.Sum(b, _one(k)), a) for k in range(1, bound + 1)]
    return reduce(gem.Sum, terms, _one(0))


def _inverse_lex_index_exprs(r, sd, degree, offsets):
    """Lattice multi-index of a flat lattice-lexicographic dof index.

    Inverts `_flat_index_expr`: given the flat degree-of-freedom index
    ``r`` (assumed on the simplex lattice, i.e. not a clamped
    out-of-lattice value), returns the lattice multi-index
    ``(i_1, ..., i_sd)`` such that `_flat_index_expr` maps it back to
    ``r``. Each outer coordinate is recovered by counting, via
    `_step_ge`, how many of the (at most ``degree``) candidate thresholds
    from `FIAT.expansions.lexicographic_offsets`'s tables ``r`` clears;
    the innermost coordinate is recovered via `_bounded_sub` against the
    resulting offset.

    Parameters
    ----------
    r : gem.Node
        Scalar expression of dtype `gem.uint_type`, the value of the
        flat index (e.g. `_index_value` of the `gem.Index` itself).
    sd : int
        Number of lattice axes (2 or 3; 1 is the identity map and needs
        no inversion).
    degree : int
        The element's polynomial degree.
    offsets : tuple of numpy.ndarray
        `FIAT.expansions.lexicographic_offsets`'s per-axis-prefix tables.

    Returns
    -------
    tuple of gem.Node
        The ``sd`` lattice coordinates, each a scalar expression of
        dtype `gem.uint_type`, free in whatever ``r`` is free in.

    """
    if sd == 2:
        table, = offsets
        p = reduce(gem.Sum, (_step_ge(r, _one(int(table[k]))) for k in range(1, degree + 1)), _one(0))
        offset_p = gem.Indexed(gem.Literal(table, dtype=gem.uint_type), (gem.VariableIndex(p),))
        q = _bounded_sub(r, offset_p, degree)
        return p, q
    elif sd == 3:
        table1, table2 = offsets
        p = reduce(gem.Sum, (_step_ge(r, _one(int(table1[k]))) for k in range(1, degree + 1)), _one(0))
        p_idx = gem.VariableIndex(p)
        q = reduce(gem.Sum, (_step_ge(r, gem.Indexed(gem.Literal(table2, dtype=gem.uint_type), (p_idx, k)))
                             for k in range(1, degree + 1)), _one(0))
        offset_pq = gem.Indexed(gem.Literal(table2, dtype=gem.uint_type), (p_idx, gem.VariableIndex(q)))
        last = _bounded_sub(r, offset_pq, degree)
        return p, q, last
    else:
        raise NotImplementedError("Lattice-lexicographic index arithmetic is only implemented up to dimension 3")


class DuffyElement:
    """Mixin for simplicial elements whose nodal basis coincides with the
    Dubiner expansion set, enabling O(p^d) sum-factorized tabulation on
    collapsed (Duffy) tensor-product point sets.

    The basis functions are enumerated by a lattice multi-index rather
    than the flat degree-of-freedom index; the flat index of a lattice
    point is its lattice-lexicographic index
    (`FIAT.expansions.lexicographic_permutation`), the dof order
    `FIAT.hierarchical.LegendreDual` uses.
    """

    def duffy_evaluation(self, order, ps, entity=None):
        """Return the sum-factorized tabulation of the element on a
        collapsed tensor-product point set.

        The basis functions are enumerated by a lattice multi-index
        ``(i_1, ..., i_d)`` rather than the flat basis index; the flat index
        of a member is its lattice-lexicographic index
        (`FIAT.expansions.lexicographic_permutation`).
        The lattice indices range over the rectangular bounding box of the
        simplex lattice; entries outside the simplex lattice
        (``i_1 + ... + i_d > degree``) tabulate to zero.

        Parameters
        ----------
        order : int
            The maximum order of differentiation, currently up to 1.
        ps : CollapsedTensorProductPointSet
            The point set with collapsed tensor-product structure.
        entity : tuple or None
            The cell entity on which to tabulate; only the cell itself is
            supported.

        Returns
        -------
        tuple
            ``(multiindex, result)``, where ``multiindex`` is a tuple of d
            `gem.JaggedIndex` of extent ``degree + 1``, each with the
            preceding indices as parents, enumerating the basis lattice,
            and ``result`` maps each derivative multi-index alpha
            to a scalar gem expression with free indices
            ``multiindex + ps.indices``.

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
        coeffs = poly_set.get_coeffs()
        # The element basis must be a uniform rescaling of the expansion
        # set, permuted from Morton to lattice-lexicographic dof order
        # (FIAT.hierarchical.LegendreDual).
        unit = coeffs[(0,) * coeffs.ndim]
        perm = lexicographic_permutation(sd, degree)
        expected = numpy.zeros_like(coeffs)
        expected[numpy.arange(len(coeffs)), perm] = unit
        if not numpy.allclose(coeffs, expected):
            raise NotImplementedError("duffy_evaluation requires the element basis "
                                      "to coincide with the expansion set")
        expansion_set = poly_set.get_expansion_set()
        etas = tuple(2.0 * f.points.ravel() - 1.0 for f in ps.factors)
        duffy = expansion_set.tabulate_duffy(degree, etas, order=order)
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
                expr = reduce(gem.Product,
                              (gem.Indexed(as_gem(table), (m, i, pt))
                               for table, m, i, pt in zip(factors, m_indices,
                                                          multiindex, ps.indices)))
                if coeff != 1.0:
                    expr = gem.Product(gem.Literal(coeff), expr)
                exprs.append(expr)
            result[alpha] = reduce(gem.Sum, exprs)
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
        standard dense tabulation otherwise.

        Parameters
        ----------
        order : int
            Return derivatives up to this order.
        ps : AbstractPointSet
            The point set.
        entity : tuple or None
            The cell entity on which to tabulate.
        coordinate_mapping : PhysicalGeometry or None
            Ignored; `duffy_evaluation` only supports reference tabulation.

        Returns
        -------
        dict
            Mapping each derivative multi-index alpha to a
            `gem.ComponentTensor` of shape ``(self.space_dimension(),)``,
            matching the convention of the standard (non-factorized)
            tabulation.

        """
        if not self._duffy_applies(ps, entity):
            return super().basis_evaluation(order, ps, entity=entity, coordinate_mapping=coordinate_mapping)
        multiindex, result = self.duffy_evaluation(order, ps, entity)
        return _scatter_to_dof_index(multiindex, result, self)

    def duffy_contraction(self, order, ps, entity, vec, epsilon):
        """Contract a sum-factorized tabulation against a coefficient vector.

        The sum over the flat degree-of-freedom index is rewritten as a sum
        over the lattice multi-index, gathering the coefficient vector
        through the same lattice-lexicographic dof numbering FIAT now uses
        (`FIAT.hierarchical.LegendreDual`), computed as index arithmetic
        against a small table (`_flat_index_expr`,
        `FIAT.expansions.lexicographic_offsets`) rather than a full
        ``ndof``-sized lookup.  `gem.optimise.contraction` sum-factorizes
        the resulting nested sum over the lattice multi-index, exploiting
        the same axis-separable structure that makes `duffy_evaluation`
        itself O(p^d).

        Parameters
        ----------
        order : int
            The derivative order to contract; only derivative
            multi-indices alpha with ``sum(alpha) == order`` are returned.
        ps : CollapsedTensorProductPointSet
            The point set with collapsed tensor-product structure.
        entity : tuple or None
            The cell entity on which to tabulate; only the cell itself is
            supported.
        vec : gem.Node
            The coefficient's local dof vector, of shape
            ``(self.space_dimension(),)``.
        epsilon : float
            Tolerance for `gem.optimise.ffc_rounding` of the tabulation.

        Returns
        -------
        dict
            Mapping alpha to a `gem.ComponentTensor` over
            ``self.get_value_indices()`` (empty for scalar elements),
            free in the point indices only.

        """
        multiindex, result = self.duffy_evaluation(order, ps, entity)
        result = {alpha: ffc_rounding(table, epsilon)
                  for alpha, table in result.items()
                  if sum(alpha) == order}

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
    """Reshape a lattice-indexed tabulation into a flat-dof-indexed one.

    Builds, for each derivative multi-index alpha, a dense
    `gem.ComponentTensor` of shape ``(element.space_dimension(),)``
    indexed by the flat degree-of-freedom index, matching the shape
    convention of the standard (non-factorized) tabulation.  The flat
    index of a lattice point is its lattice-lexicographic index
    (`FIAT.expansions.lexicographic_permutation`), the dof order
    `FIAT.hierarchical.LegendreDual` uses.

    Substitutes each lattice axis with `_inverse_lex_index_exprs`'s index
    arithmetic against a small table (`FIAT.expansions.
    lexicographic_offsets`, not a full ``ndof``-sized inverse table),
    expressing the lattice multi-index as a function of the new flat
    index ``r`` -- exactly the substitution `translate_argument`/
    `translate_coefficient` already use for canonical quadrature-point
    reordering (`gem.node.MemoizerArg(gem.optimise.
    filtered_replace_indices)`).  This keeps ``r`` a genuine flat index
    throughout (matching `element.get_indices()`'s convention), unlike
    scattering via an `IndexSum`-`gem.Delta` construction over the full
    lattice bounding box, which is also correct but was measured to
    regress the downstream quadrature-contraction flop count (the outer
    loop over the lattice bounding box is asymptotically larger than the
    flat ``ndof`` loop it replaces).

    Parameters
    ----------
    multiindex : tuple of gem.JaggedIndex
        The lattice multi-index free in each entry of ``result``, as
        returned by `DuffyElement.duffy_evaluation`.
    result : dict
        Mapping alpha to a scalar GEM expression free in ``multiindex``
        (and point indices).
    element : DuffyElement
        The element being tabulated.

    Returns
    -------
    dict
        Mapping alpha to a `gem.ComponentTensor` of shape
        ``(element.space_dimension(),)``.

    """
    ndof = element.space_dimension()
    r = gem.Index(extent=ndof)
    if len(multiindex) == 1:
        # 1D: the flat index already *is* the (only) lattice coordinate.
        subst = ((multiindex[0], r),)
    else:
        offsets = lexicographic_offsets(len(multiindex), element.degree)
        inverse = _inverse_lex_index_exprs(_index_value(r), len(multiindex), element.degree, offsets)
        subst = tuple((axis, gem.VariableIndex(expr)) for axis, expr in zip(multiindex, inverse))
    mapper = MemoizerArg(filtered_replace_indices)
    return {alpha: gem.ComponentTensor(mapper(expr, subst), (r,))
            for alpha, expr in result.items()}
