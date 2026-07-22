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
    """Mixin enabling O(p^d) sum-factorized tabulation on collapsed (Duffy)
    coordinates, for simplicial elements whose nodal basis coincides with
    the Dubiner expansion set.

    Dofs read off a lattice multi-index, not their flat index. continuity=None:
    one lattice point per dof, lattice-lexicographic order. continuity="C0":
    up to dim+1 points per dof, per `C0_basis`. See `get_sparse_coeffs`.
    """

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        """Dispatch to `duffy_evaluation` on collapsed point sets, else the
        dense fallback. Both return flat-dof-indexed tabulations, so this
        composes with `TensorFiniteElement` for free.
        """
        sd = self.cell.get_dimension()
        if not (isinstance(ps, CollapsedTensorProductPointSet)
                and (entity is None or entity == (sd, 0))
                and not self.complex.is_macrocell()):
            return super().basis_evaluation(order, ps, entity=entity, coordinate_mapping=coordinate_mapping)
        return self.duffy_evaluation(order, ps, entity)

    def get_sparse_coeffs(self):
        """Sparse recombination weights from the raw (continuity=None)
        lattice tabulation into this element's nodal basis:
        dof r = sum_k row_coeff[r, k] * phi(row_multiindex[r, k]).

        continuity=None: k=1, at the dof's lattice-lexicographic point.
        continuity="C0": up to dim+1 points per dof, from
        `c0_recombination_matrix`, zero-padded to a common k.

        Raises if the nodal basis doesn't coincide with the expansion set
        up to this reordering/rescaling.

        :returns: (row_multiindex, row_coeff), shape (ndof, k, dim) / (ndof, k).
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
        """Sum-factorized tabulation on a collapsed point set, flat-dof-indexed
        (matching `get_indices()`).

        Tabulates the raw expansion set over the rectangular lattice
        bounding box (zero outside the simplex), then reads off each dof's
        row from `get_sparse_coeffs`.

        No `gem.JaggedIndex`/jagged loop bound is needed: every index here
        is a plain `gem.Index` of constant extent. Jagged *access* -- not
        jagged *loop bounds* -- comes from `gem.VariableIndex`: wrapping a
        table lookup as a `VariableIndex` makes the index itself an
        expression of other free indices, so a constant-extent loop can
        still land on a data-dependent table position each iteration.
        `m_indices` and the row_multiindex substitution below both use this
        to gather, never a jagged loop.

        :returns: dict alpha -> `gem.ComponentTensor` of shape (ndof,).
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

        def lookup_index(index_table, multiindex):
            index_table = gem.Literal(index_table, dtype=gem.uint_type)
            return gem.VariableIndex(gem.Indexed(index_table, multiindex))

        multiindex = tuple(gem.Index(extent=degree + 1) for _ in range(sd))
        # m_t = i_1 + ... + i_{t-1}: affine for t <= 1, else a VariableIndex
        # gather (jagged access, not a jagged loop) on a clamped table --
        # clamping is safe since out-of-lattice entries already tabulate to
        # zero via the factors of the previous axes.
        duffy_indices = [0, *multiindex[:1]]
        for t in range(2, sd):
            index_table = reduce(numpy.add.outer, (numpy.arange(degree + 1),) * t)
            index_table = numpy.minimum(index_table, degree)
            duffy_indices.append(lookup_index(index_table, multiindex[:t]))

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
                expr = gem.Product(*(gem.Indexed(as_gem(table), (index_expr, i, pt))
                                     for table, index_expr, i, pt
                                     in zip(factors, duffy_indices, multiindex, ps.indices)))
                if coeff != 1.0:
                    expr = gem.Product(gem.Literal(coeff), expr)
                exprs.append(expr)
            result[alpha] = gem.Sum(*exprs)

        # Each dof r contracts k lattice points: substitute every multiindex
        # axis with a VariableIndex gather on (r, k) -- again jagged access,
        # not a jagged loop -- then sum over k.
        row_multiindex, row_coeff = self.get_sparse_coeffs()
        r = gem.Index(extent=self.space_dimension())
        k = gem.Index(extent=row_coeff.shape[1])
        coeff_expr = gem.Indexed(gem.Literal(row_coeff), (r, k))
        subst = tuple(
            (axis, lookup_index(row_multiindex[..., t], (r, k)))
            for t, axis in enumerate(multiindex))
        mapper = MemoizerArg(filtered_replace_indices)
        return {alpha: gem.ComponentTensor(gem.IndexSum(gem.Product(coeff_expr, mapper(expr, subst)), (k,)), (r,))
                for alpha, expr in result.items()}


def c0_recombination_matrix(dim, n):
    """Dense `C0_basis` row recombination, decomposed into per-dof sparse
    terms.

    Feeding `C0_basis` the identity recovers its row operations as an
    explicit matrix R: C0_basis(dim, n, [phi])[0] == R @ phi. `C0_basis`
    indexes rows/columns by Morton position, not `lattice_iter`'s order, so
    raw_multiindices inverts `morton_index` to recover each column's
    lattice point.

    :returns: (row_multiindex, row_coeff), shape (ndof, k, dim) / (ndof, k),
        zero-padded to the largest row's nonzero count k.
    """
    ndof = math.comb(n + dim, dim)
    R, = C0_basis(dim, n, [numpy.eye(ndof)])
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
    """Lattice multi-index of each lattice-lexicographic dof (see
    `lexicographic_permutation`), shape (ndof, dim).
    """
    ndof = math.comb(n + dim, dim)
    return numpy.fromiter((i for multiindex in lexicographical_iter(dim, n) for i in multiindex),
                          dtype=int, count=ndof * dim).reshape(ndof, dim)
