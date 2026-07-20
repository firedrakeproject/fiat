from functools import reduce

import numpy

import gem
from FIAT.expansions import morton_inverse_table
from gem.node import MemoizerArg
from gem.optimise import contraction, ffc_rounding, filtered_replace_indices
from finat.point_set import CollapsedTensorProductPointSet


def _index_value(index):
    """GEM scalar expression for the numeric value of a `gem.Index`."""
    return gem.Indexed(gem.Literal(numpy.arange(index.extent, dtype=gem.uint_type),
                                   dtype=gem.uint_type), (index,))


def _one(n):
    return gem.Literal(n, dtype=gem.uint_type)


def _morton_index_expr(multiindex, ndof):
    """Flat Morton index of a lattice multi-index, as index arithmetic.

    Computes `FIAT.expansions.morton_index2` / `morton_index3`'s formula
    directly as a `gem.Node` built from `gem.Sum`/`gem.Product`/
    `gem.FloorDiv` on the multi-index components, rather than a table
    lookup.

    ``multiindex`` ranges over the rectangular bounding box of the
    simplex lattice (each `gem.JaggedIndex` has extent ``degree + 1``),
    including points outside the simplex lattice (``sum(multiindex) >
    degree``), for which the raw formula overshoots ``ndof - 1``. Like
    `FIAT.expansions.morton_forward_table`, the result is clamped to
    ``ndof - 1`` for those points: the corresponding tabulation is exactly
    zero there, so the clamped (but otherwise meaningless) index is only
    ever multiplied by zero.

    Parameters
    ----------
    multiindex : tuple of gem.Index
        The lattice multi-index, of length 1, 2, or 3.
    ndof : int
        The element's space dimension, i.e. the number of points on the
        simplex lattice; the result is clamped to ``ndof - 1``.

    Returns
    -------
    gem.Node
        A scalar expression of dtype `gem.uint_type`, free in
        ``multiindex``, equal to the flat Morton index, clamped to
        ``ndof - 1``.

    """
    values = [_index_value(index) for index in multiindex]
    if len(values) == 1:
        p, = values
        expr = p
    elif len(values) == 2:
        p, q = values
        s = gem.Sum(p, q)
        shell = gem.FloorDiv(gem.Product(s, gem.Sum(s, _one(1))), _one(2))
        expr = gem.Sum(shell, q)
    elif len(values) == 3:
        p, q, r = values
        s = gem.Sum(gem.Sum(p, q), r)
        qr = gem.Sum(q, r)
        shell = gem.FloorDiv(gem.Product(gem.Product(s, gem.Sum(s, _one(1))), gem.Sum(s, _one(2))), _one(6))
        layer = gem.FloorDiv(gem.Product(qr, gem.Sum(qr, _one(1))), _one(2))
        expr = gem.Sum(gem.Sum(shell, layer), r)
    else:
        raise NotImplementedError("Morton index arithmetic is only implemented up to dimension 3")
    return gem.MinValue(expr, _one(ndof - 1))


class DuffyElement:
    """Mixin for simplicial elements whose nodal basis coincides with the
    Dubiner expansion set, enabling O(p^d) sum-factorized tabulation on
    collapsed (Duffy) tensor-product point sets.

    The basis functions are enumerated by a lattice multi-index rather
    than the flat degree-of-freedom index; the flat index of a lattice
    point is its Morton index (`FIAT.expansions.morton_index`), the same
    enumeration FIAT already uses for the element's degrees of freedom,
    so no reordering of the element's dof numbering is involved.
    """

    def duffy_evaluation(self, order, ps, entity=None):
        """Return the sum-factorized tabulation of the element on a
        collapsed tensor-product point set.

        The basis functions are enumerated by a lattice multi-index
        ``(i_1, ..., i_d)`` rather than the flat basis index; the flat index
        of a member is its Morton index (`FIAT.expansions.morton_index`).
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
        poly_set = self._element.get_nodal_basis()
        coeffs = poly_set.get_coeffs()
        # The element basis must be a uniform rescaling of the expansion set
        unit = coeffs[(0,) * coeffs.ndim]
        if not numpy.allclose(coeffs, unit * numpy.eye(len(coeffs))):
            raise NotImplementedError("duffy_evaluation requires the element basis "
                                      "to coincide with the expansion set")
        expansion_set = poly_set.get_expansion_set()
        etas = tuple(2.0 * f.points.ravel() - 1.0 for f in ps.factors)
        duffy = expansion_set.tabulate_duffy(degree, etas, order=order)
        duffy = {alpha: [(unit * coeff, factors) for coeff, factors in terms]
                 for alpha, terms in duffy.items()}

        sd = self.cell.get_spatial_dimension()
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
        through the same Morton dof numbering FIAT already uses
        (`FIAT.expansions.morton_index2` / `morton_index3`), computed as
        index arithmetic (`_morton_index_expr`) rather than a table
        lookup.  `gem.optimise.contraction` sum-factorizes the resulting
        nested sum over the lattice multi-index, exploiting the same
        axis-separable structure that makes `duffy_evaluation` itself
        O(p^d).

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

        r_index = gem.VariableIndex(_morton_index_expr(multiindex, self.space_dimension()))
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
    index of a lattice point is its Morton index
    (`FIAT.expansions.morton_index`), the same enumeration FIAT already
    uses for the element's degrees of freedom, so no reordering of the
    element's dof numbering is involved.

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
    sd = len(multiindex)
    ndof = element.space_dimension()
    r = gem.Index(extent=ndof)
    inv_table = morton_inverse_table(sd, element.degree)
    subst = tuple(
        (axis, gem.VariableIndex(gem.Indexed(
            gem.Literal(numpy.ascontiguousarray(inv_table[:, t]), dtype=gem.uint_type), (r,))))
        for t, axis in enumerate(multiindex)
    )
    mapper = MemoizerArg(filtered_replace_indices)
    return {alpha: gem.ComponentTensor(mapper(expr, subst), (r,))
            for alpha, expr in result.items()}
