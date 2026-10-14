from functools import reduce

import numpy

import gem

from FIAT.expansions import C0_basis
from finat.physically_mapped import MappedTabulation
from finat.point_set import CollapsedTensorProductPointSet


class DuffyElement:
    """Mixin for sum-factorized tabulation on collapsed simplex coordinates."""

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        """Tabulate on a collapsed point set when the element supports it."""
        sd = self.cell.get_dimension()
        if not (isinstance(ps, CollapsedTensorProductPointSet)
                and order <= 1
                and (entity is None or entity == (sd, 0))
                and not self.complex.is_macrocell()):
            return super().basis_evaluation(
                order, ps, entity=entity,
                coordinate_mapping=coordinate_mapping)
        return self.duffy_evaluation(order, ps, entity)

    def get_coefficient_matrix(self) -> gem.Literal:
        """Return the map from lattice-ordered expansions to dofs.

        Returns
        -------
        gem.Literal
            Coefficient matrix in Duffy lattice order.

        """
        degree = self.degree
        sd = self.cell.get_spatial_dimension()
        poly_set = self._element.get_nodal_basis()
        coeffs = numpy.array(poly_set.get_coeffs(), copy=True)
        expansion_set = poly_set.get_expansion_set()
        if expansion_set.continuity == "C0":
            recombination, = C0_basis(sd, degree,
                                      [numpy.eye(coeffs.shape[1])])
            coeffs = coeffs @ recombination

        coeffs = coeffs[:, expansion_set.get_duffy_permutation(degree)]
        if numpy.allclose(coeffs, 0.0):
            raise ValueError("empty Duffy coefficient matrix")
        return gem.Literal(coeffs)

    def duffy_evaluation(self, order, ps, entity=None):
        """Return a sum-factorized, dof-indexed tabulation."""
        assert isinstance(ps, CollapsedTensorProductPointSet)
        cell_dim = self.cell.get_dimension()
        if entity is not None and entity != (cell_dim, 0):
            raise NotImplementedError(
                "duffy_evaluation is only supported on the cell interior")
        if self.complex.is_macrocell():
            raise NotImplementedError("duffy_evaluation is not supported on split cells")

        degree = self.degree
        sd = self.cell.get_spatial_dimension()
        poly_set = self._element.get_nodal_basis()
        expansion_set = poly_set.get_expansion_set()
        etas = tuple(2.0 * factor.points.ravel() - 1.0
                     for factor in ps.factors)
        duffy = expansion_set.tabulate_duffy(degree, etas, order=order)

        def lookup_index(index_table, multiindex):
            index_table = gem.Literal(index_table, dtype=gem.uint_type)
            return gem.VariableIndex(gem.Indexed(index_table, multiindex))

        multiindex = []
        for _ in range(sd):
            multiindex.append(gem.JaggedIndex(extent=degree + 1, parents=tuple(multiindex)))
        multiindex = tuple(multiindex)
        # The first table axis is the sum of all preceding lattice indices,
        # not an independent iteration axis, so it is an indirect index.
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
                expr = gem.Product(*(
                    gem.Indexed(as_gem(table),
                                (index_expr, i, ps.indices[axis]))
                    for (axis, table), index_expr, i
                    in zip(factors, duffy_indices, multiindex)))
                if coeff != 1.0:
                    expr = gem.Product(gem.Literal(coeff), expr)
                exprs.append(expr)
            result[alpha] = gem.Sum(*exprs)

        coefficients = self.get_coefficient_matrix()
        tabulation = {
            alpha: gem.FlattenedTensor(expr, multiindex)
            for alpha, expr in result.items()}
        return MappedTabulation(coefficients, tabulation)
