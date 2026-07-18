import FIAT
import numpy

import gem
from abc import ABCMeta, abstractmethod
from functools import reduce

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement, Lagrange, DiscontinuousLagrange
from finat.point_set import (GaussLobattoLegendrePointSet, GaussLegendrePointSet,
                             KMVPointSet, CollapsedTensorProductPointSet)


class SpectralElement(metaclass=ABCMeta):
    """Base class to implement spectral elements."""

    @property
    @abstractmethod
    def point_set_family(self):
        """The PointSet subclass on which this element tabulates to a Delta."""
        pass

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''
        result = super().basis_evaluation(order, ps, entity=entity, coordinate_mapping=coordinate_mapping)
        cell_dimension = self.cell.get_dimension()
        if entity is None or entity == (cell_dimension, 0):  # on cell interior
            space_dim = self.space_dimension()
            if isinstance(ps, self.point_set_family) and len(ps.points) == space_dim:
                # Bingo: evaluation points match node locations!
                spatial_dim = self.cell.get_spatial_dimension()
                q, = ps.indices
                r, = self.get_indices()
                result[(0,) * spatial_dim] = gem.ComponentTensor(gem.Delta(q, r), (r,))
        return result


class GaussLobattoLegendre(SpectralElement, Lagrange):
    """1D continuous element with nodes at the Gauss-Lobatto points."""
    point_set_family = GaussLobattoLegendrePointSet

    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(FIAT.GaussLobattoLegendre(cell, degree))


class GaussLegendre(SpectralElement, DiscontinuousLagrange):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""
    point_set_family = GaussLegendrePointSet

    def __init__(self, cell, degree):
        super(DiscontinuousLagrange, self).__init__(FIAT.GaussLegendre(cell, degree))


class KongMulderVeldhuizen(SpectralElement, ScalarFiatElement):
    """Simplicial continuous element with nodes at the KMV points."""
    point_set_family = KMVPointSet

    def __init__(self, cell, degree):
        super(ScalarFiatElement, self).__init__(FIAT.KongMulderVeldhuizen(cell, degree))
        cite("Chin1999higher")
        cite("Geevers2018new")


class Legendre(ScalarFiatElement):
    """DG element with Legendre polynomials."""

    def __init__(self, cell, degree, variant=None):
        fiat_element = FIAT.Legendre(cell, degree, variant=variant)
        super().__init__(fiat_element)

    def duffy_evaluation(self, order, ps, entity=None):
        """Return the sum-factorized tabulation of the element on a
        collapsed tensor-product point set.

        The basis functions are enumerated by a lattice multi-index
        ``(i_1, ..., i_d)`` rather than the flat basis index; the flat index
        of a member is its Morton index (`FIAT.expansions.morton_index2`,
        `FIAT.expansions.morton_index3`).  The lattice indices range over the
        rectangular bounding box of the simplex lattice; entries outside the
        simplex lattice (``i_1 + ... + i_d > degree``) tabulate to zero.

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


class IntegratedLegendre(ScalarFiatElement):
    """CG element with integrated Legendre polynomials."""

    def __init__(self, cell, degree, variant=None):
        fiat_element = FIAT.IntegratedLegendre(cell, degree, variant=variant)
        super().__init__(fiat_element)


class FDMLagrange(ScalarFiatElement):
    """1D CG element with FDM shape functions and point evaluation BCs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMLagrange(cell, degree)
        super().__init__(fiat_element)


class FDMDiscontinuousLagrange(ScalarFiatElement):
    """1D DG element with derivatives of FDM shape functions with point evaluation Bcs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMDiscontinuousLagrange(cell, degree)
        super().__init__(fiat_element)


class FDMQuadrature(ScalarFiatElement):
    """1D CG element with FDM shape functions and orthogonalized vertex modes."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMQuadrature(cell, degree)
        super().__init__(fiat_element)


class FDMBrokenH1(ScalarFiatElement):
    """1D Broken CG element with FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMBrokenH1(cell, degree)
        super().__init__(fiat_element)


class FDMBrokenL2(ScalarFiatElement):
    """1D DG element with derivatives of FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMBrokenL2(cell, degree)
        super().__init__(fiat_element)


class FDMHermite(ScalarFiatElement):
    """1D CG element with FDM shape functions, point evaluation BCs and derivative BCs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMHermite(cell, degree)
        super().__init__(fiat_element)
