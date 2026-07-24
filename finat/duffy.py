from functools import reduce

import math
import numpy

import gem

from FIAT.reference_element import lattice_iter
from FIAT.expansions import morton_index, C0_basis
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

    def get_sparse_coeffs(self) -> tuple:
        """Return sparse recombination weights for the element basis.

        Each row is represented as a linear combination of raw expansion
        members addressed by simplex lattice multi-indices. Discontinuous
        expansion sets have one member per row, in any bijective ordering;
        C0 expansion sets use the sparse recombination from `C0_basis`.

        Returns
        -------
        tuple
            ``(row_multiindex, row_coeff)`` with shapes ``(ndof, k, dim)``
            and ``(ndof, k)``.

        """
        degree = self.degree
        sd = self.cell.get_spatial_dimension()
        ndof = self.space_dimension()
        poly_set = self._element.get_nodal_basis()
        coeffs = poly_set.get_coeffs()
        expansion_set = poly_set.get_expansion_set()
        continuity_c0 = expansion_set.continuity == "C0"

        if continuity_c0:
            col = numpy.arange(ndof)
        else:
            col = numpy.argmax(numpy.abs(coeffs), axis=1)
        scale = coeffs[numpy.arange(ndof), col]
        expected = numpy.zeros_like(coeffs)
        expected[numpy.arange(ndof), col] = scale
        if (not numpy.allclose(coeffs, expected)
                or (not continuity_c0 and len(numpy.unique(col)) != ndof)):
            raise NotImplementedError("duffy_evaluation requires the element basis "
                                      "to coincide with the expansion set")

        if continuity_c0:
            row_multiindex, row_coeff = c0_recombination_matrix(sd, degree)
        else:
            raw_multiindices = morton_multiindices(sd, degree)
            row_multiindex = raw_multiindices[col, None, :]
            row_coeff = numpy.ones((ndof, 1))
        return row_multiindex, row_coeff * scale[:, None]

    def duffy_evaluation(self, order, ps, entity=None):
        """Sum-factorized tabulation on a collapsed point set, flat-dof-indexed
        (matching `get_indices()`).

        Tabulates the raw expansion set over the jagged lattice
        (`gem.JaggedIndex`, zero outside the simplex), applies the diagonal
        (home-point) weight, and flattens to the dof index with
        `gem.FlattenedTensor`: coefficient contractions over the flat index
        turn back into jagged lattice loops (`gem.optimise.unflatten`),
        keeping the per-axis factors separable for sum factorisation; any
        other use lowers to per-dof gather tables
        (`gem.optimise.replace_flattened`).  C0 corrections (k >= 1 in
        `get_sparse_coeffs`) are expressed as a `gem.SparseMatrix` mat-vec
        against the diagonal tabulation: its delta expansion cancels
        through the ordinary delta-elimination machinery into its own
        genuinely small (nnz-sized) loop, since only an O(degree)-sized
        subset of dofs need any correction.

        :returns: dict alpha -> rank-1 gem expression of shape (ndof,).
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
        axis_permutation = expansion_set.duffy_axis_permutation
        assert sorted(axis_permutation) == list(range(sd))
        etas = tuple(2.0 * ps.factors[t].points.ravel() - 1.0 for t in axis_permutation)
        point_indices = tuple(ps.indices[t] for t in axis_permutation)
        duffy = expansion_set.tabulate_duffy(degree, etas, order=order)

        def lookup_index(index_table, multiindex):
            index_table = gem.Literal(index_table, dtype=gem.uint_type)
            return gem.VariableIndex(gem.Indexed(index_table, multiindex))

        multiindex = []
        for _ in range(sd):
            multiindex.append(gem.JaggedIndex(extent=degree + 1, parents=tuple(multiindex)))
        multiindex = tuple(multiindex)
        # m_t = i_1 + ... + i_{t-1}: affine for t <= 1, else a VariableIndex
        # gather on a clamped table -- clamping only acts outside the jagged
        # bounds, where the factors of the previous axes already tabulate to
        # zero.
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
                                     in zip(factors, duffy_indices, multiindex, point_indices)))
                if coeff != 1.0:
                    expr = gem.Product(gem.Literal(coeff), expr)
                exprs.append(expr)
            result[alpha] = gem.Sum(*exprs)

        # Diagonal (home-point) part: per-dof weight and flat dof position,
        # both as lattice tables (weight is zero off the simplex).
        row_multiindex, row_coeff = self.get_sparse_coeffs()
        ndof = self.space_dimension()
        home = tuple(row_multiindex[:, 0, :].T)
        weight = numpy.zeros((degree + 1,) * sd)
        weight[home] = row_coeff[:, 0]
        ordering_array = numpy.zeros((degree + 1,) * sd, dtype=int)
        ordering_array[home] = numpy.arange(ndof)
        ordering = gem.Literal(ordering_array, dtype=gem.uint_type)
        weight = gem.Indexed(gem.Literal(weight), multiindex)
        tables = {alpha: gem.FlattenedTensor(gem.Product(weight, expr), multiindex, ordering)
                  for alpha, expr in result.items()}

        if row_coeff.shape[1] > 1:
            # C0 corrections: each further column of a corrected row is
            # *some other dof's home point* (`home` is a bijection over all
            # ndof columns of the recombination matrix), so no
            # re-tabulation is needed -- correcting dof r is a sparse
            # linear combination of the already-tabulated diagonal entries
            # of the dofs owning those columns.  Only an O(degree)-sized
            # subset of dofs need any correction at all; the SparseMatrix
            # delta expansion cancels into a genuine nnz-sized loop rather
            # than an O(ndof) or O(ndof * npts) one.
            rows, cols, data = [], [], []
            for r in range(ndof):
                for t in range(1, row_coeff.shape[1]):
                    coeff = row_coeff[r, t]
                    if coeff == 0.0:
                        continue
                    j = ordering_array[tuple(row_multiindex[r, t, :])]
                    rows.append(r)
                    cols.append(j)
                    data.append(coeff / row_coeff[j, 0])
            if rows:
                sparse = gem.SparseMatrix((ndof, ndof), rows, cols, data)
                r = gem.Index(extent=ndof)
                j = gem.Index(extent=ndof)
                tables = {alpha: gem.ComponentTensor(
                    gem.Sum(gem.Indexed(tables[alpha], (r,)),
                            gem.IndexSum(gem.Product(gem.Indexed(sparse, (r, j)),
                                                     gem.Indexed(tables[alpha], (j,))), (j,))),
                    (r,))
                    for alpha in result}
        return tables


def morton_multiindices(dim: int, n: int) -> numpy.ndarray:
    """Return each Morton-ordered simplex lattice multi-index."""
    ndof = math.comb(n + dim, dim)
    multiindices = [None] * ndof
    for multiindex in lattice_iter(0, n + 1, dim):
        multiindices[morton_index(dim, n, *multiindex)] = multiindex
    return numpy.asarray(multiindices, dtype=int)


def c0_recombination_matrix(dim, n):
    """Dense `C0_basis` row recombination, decomposed into per-dof sparse
    terms.

    Feeding `C0_basis` the identity recovers its row operations as an
    explicit matrix R: C0_basis(dim, n, [phi])[0] == R @ phi. `C0_basis`
    indexes rows/columns by Morton position, not `lattice_iter`'s order, so
    raw_multiindices inverts `morton_index` to recover each column's
    lattice point.

    :returns: (row_multiindex, row_coeff), shape (ndof, k, dim) / (ndof, k),
        zero-padded to the largest row's nonzero count k.  Each row's home
        column (a bijective matching dof <-> lattice point with nonzero
        weight) comes first.
    """
    ndof = math.comb(n + dim, dim)
    R, = C0_basis(dim, n, [numpy.eye(ndof)])
    raw_multiindices = morton_multiindices(dim, n)

    # Home column of each dof, by iteratively matching rows with a single
    # remaining candidate.
    candidates = {r: set(numpy.nonzero(R[r])[0]) for r in range(ndof)}
    home = {}
    while candidates:
        for r, cols in candidates.items():
            if len(cols) == 1:
                break
        else:
            raise NotImplementedError("no bijective dof <-> lattice point matching")
        c, = candidates.pop(r)
        home[r] = c
        for cols in candidates.values():
            cols.discard(c)

    rows = [[home[r], *(c for c in numpy.nonzero(R[r])[0] if c != home[r])]
            for r in range(ndof)]
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
