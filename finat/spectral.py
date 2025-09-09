import FIAT

import gem
from abc import ABC, abstractmethod

from finat.fiat_elements import ScalarFiatElement, Lagrange, DiscontinuousLagrange
from finat.point_set import GaussLobattoLegendrePointSet, GaussLegendrePointSet, KMVPointSet

try:
    from firedrake_citations import Citations
    Citations().add("Geevers2018new", """
@article{Geevers2018new,
 title={New higher-order mass-lumped tetrahedral elements for wave propagation modelling},
 author={Geevers, Sjoerd and Mulder, Wim A and van der Vegt, Jaap JW},
 journal={SIAM journal on scientific computing},
 volume={40},
 number={5},
 pages={A2830--A2857},
 year={2018},
 publisher={SIAM},
 doi={https://doi.org/10.1137/18M1175549},
}
""")
    Citations().add("Chin1999higher", """
@article{chin1999higher,
 title={Higher-order triangular and tetrahedral finite elements with mass lumping for solving the wave equation},
 author={Chin-Joe-Kong, MJS and Mulder, Wim A and Van Veldhuizen, M},
 journal={Journal of Engineering Mathematics},
 volume={35},
 number={4},
 pages={405--426},
 year={1999},
 publisher={Springer},
 doi={https://doi.org/10.1023/A:1004420829610},
}
""")
except ImportError:
    Citations = None


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
        if Citations is not None:
            sd = cell.get_spatial_dimension()
            Citations().register("Chin1999higher" if sd == 2 else "Geevers2018new")


class Legendre(ScalarFiatElement):
    """DG element with Legendre polynomials."""

    def __init__(self, cell, degree, variant=None):
        fiat_element = FIAT.Legendre(cell, degree, variant=variant)
        super().__init__(fiat_element)


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
