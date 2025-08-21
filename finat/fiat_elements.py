import FIAT
import gem
import numpy as np
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import PointSet, PointSingleton

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


class FiatElement(FiniteElementBase):
    """Base class for finite elements for which the tabulation is provided
    by FIAT."""
    def __init__(self, fiat_element):
        super().__init__()
        self._element = fiat_element

    @property
    def cell(self):
        return self._element.get_reference_element()

    @property
    def complex(self):
        return self._element.get_reference_complex()

    @property
    def degree(self):
        # Requires FIAT.CiarletElement
        return self._element.degree()

    @property
    def formdegree(self):
        return self._element.get_formdegree()

    def entity_dofs(self):
        return self._element.entity_dofs()

    def entity_closure_dofs(self):
        return self._element.entity_closure_dofs()

    @property
    def entity_permutations(self):
        return self._element.entity_permutations()

    def space_dimension(self):
        return self._element.space_dimension()

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    @property
    def value_shape(self):
        return self._element.value_shape()

    @property
    def fiat_equivalent(self):
        # Just return the underlying FIAT element
        return self._element

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''
        fiat_element = self._element
        fiat_result = fiat_element.tabulate(order, ps.points, entity)

        value_shape = self.value_shape
        value_size = np.prod(value_shape, dtype=int)
        space_dimension = fiat_element.space_dimension()
        # In almost all cases, we have
        # self.space_dimension() == self._element.space_dimension()
        # But for Bell, FIAT reports 21 basis functions,
        # but FInAT only 18 (because there are actually 18
        # basis functions, and the additional 3 are for
        # dealing with transformations between physical
        # and reference space).
        if self.space_dimension() == space_dimension:
            beta = self.get_indices()
            index_shape = tuple(index.extent for index in beta)
        else:
            index_shape = (space_dimension,)
            beta = tuple(gem.Index(extent=i) for i in index_shape)
            assert len(beta) == len(self.get_indices())

        zeta = self.get_value_indices()
        basis_indices = beta + zeta

        result = {}
        for alpha, fiat_table in fiat_result.items():
            if isinstance(fiat_table, Exception):
                shape = ps.points.shape[:-1] + index_shape + value_shape
                result[alpha] = gem.Failure(shape, fiat_table)
                continue

            derivative = sum(alpha)
            fiat_table = fiat_table.reshape(space_dimension, value_size, -1)

            point_indices = ()
            if derivative == self.degree and not self.complex.is_macrocell():
                # Make sure numerics satisfies theory
                if fiat_table.dtype == object:
                    bindings = {X: np.zeros(X.shape)
                                for pt in ps.points
                                for X in gem.extract_type(pt, gem.Variable)}
                    gem_table = gem.as_gem(fiat_table)
                    val, = gem.interpreter.evaluate((gem_table,), bindings=bindings)
                    fiat_table = val.arr.T

                fiat_table = fiat_table[..., 0]
            elif derivative > self.degree:
                # Make sure numerics satisfies theory
                if fiat_table.dtype != object:
                    assert np.allclose(fiat_table, 0.0)
                fiat_table = np.zeros(fiat_table.shape[:-1])
            else:
                point_indices = ps.indices

            point_shape = tuple(index.extent for index in point_indices)
            table_shape = index_shape + value_shape + point_shape
            table_indices = basis_indices + point_indices

            fiat_table = fiat_table.reshape(table_shape)
            gem_table = gem.as_gem(fiat_table)

            expr = gem.Indexed(gem_table, table_indices)
            expr = gem.ComponentTensor(expr, basis_indices)
            result[alpha] = expr

        return result

    def point_evaluation(self, order, refcoords, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at an arbitrary points on
        the reference element.

        :param order: return derivatives up to this order.
        :param refcoords: GEM expression representing the coordinates
                          on the reference entity.  Its shape must be
                          a vector with the correct dimension, its
                          free indices are arbitrary.
        :param entity: the cell entity on which to tabulate.
        :param coordinate_mapping: a
           :class:`~.physically_mapped.PhysicalGeometry` object that
           provides physical geometry callbacks (may be None).
        '''
        if entity is None:
            entity = (self.cell.get_dimension(), 0)
        entity_dim, entity_i = entity

        # Spatial dimension of the entity
        esd = self.cell.construct_subelement(entity_dim).get_spatial_dimension()
        assert isinstance(refcoords, gem.Node) and refcoords.shape == (esd,)

        # Coordinates on the reference entity (GEM)
        Xi = tuple(gem.Indexed(refcoords, i) for i in np.ndindex(refcoords.shape))
        ps = PointSingleton(Xi)
        return self.basis_evaluation(order, ps, entity=entity,
                                     coordinate_mapping=coordinate_mapping)

    @cached_property
    def _dual_basis(self):
        # Return the numerical part of the dual basis, this split is
        # needed because the dual_basis itself can't produce the same
        # point set over and over in case it is used multiple times
        # (in for example a tensorproductelement).
        fiat_dual_basis = self._element.dual_basis()
        seen = dict()
        allpts = []
        # Find the unique points to evaluate at.
        # We might be able to make this a smaller set by treating each
        # point one by one, but most of the redundancy comes from
        # multiple functionals using the same quadrature rule.
        for dual in fiat_dual_basis:
            if len(dual.deriv_dict) != 0:
                raise NotImplementedError("FIAT dual bases with derivative nodes represented via a ``Functional.deriv_dict`` property do not currently have a FInAT dual basis")
            pts = dual.get_point_dict().keys()
            pts = tuple(sorted(pts))  # need this for determinism
            if pts not in seen:
                # k are indices into Q (see below) for the seen points
                kstart = len(allpts)
                kend = kstart + len(pts)
                seen[pts] = kstart, kend
                allpts.extend(pts)
        # Build Q.
        # Q is a tensor of weights (of total rank R) to contract with a unique
        # vector of points to evaluate at, giving a tensor (of total rank R-1)
        # where the first indices (rows) correspond to a basis functional
        # (node).
        # Q is a DOK Sparse matrix in (row, col, higher,..)=>value pairs (to
        # become a gem.SparseLiteral when implemented).
        # Rows (i) are number of nodes/dual functionals.
        # Columns (k) are unique points to evaluate.
        # Higher indices (*cmp) are tensor indices of the weights when weights
        # are tensor valued.
        Q = {}
        for i, dual in enumerate(fiat_dual_basis):
            point_dict = dual.get_point_dict()
            pts = tuple(sorted(point_dict.keys()))
            kstart, kend = seen[pts]
            for p, k in zip(pts, range(kstart, kend)):
                for weight, cmp in point_dict[p]:
                    Q[(i, k, *cmp)] = weight
        if all(len(set(key)) == 1 and np.isclose(weight, 1) and len(key) == 2
               for key, weight in Q.items()):
            # Identity matrix Q can be expressed symbolically
            extents = tuple(map(max, zip(*Q.keys())))
            js = tuple(gem.Index(extent=e+1) for e in extents)
            assert len(js) == 2
            Q = gem.ComponentTensor(gem.Delta(*js), js)
        else:
            # temporary until sparse literals are implemented in GEM which will
            # automatically convert a dictionary of keys internally.
            # TODO the below is unnecessarily slow and would be sped up
            # significantly by building Q in a COO format rather than DOK (i.e.
            # storing coords and associated data in (nonzeros, entries) shaped
            # numpy arrays) to take advantage of numpy multiindexing
            if len(Q) == 1:
                Qshape = tuple(s + 1 for s in tuple(Q)[0])
            else:
                Qshape = tuple(s + 1 for s in map(max, *Q))
            Qdense = np.zeros(Qshape, dtype=np.float64)
            for idx, value in Q.items():
                Qdense[idx] = value
            Q = gem.Literal(Qdense)
        return Q, np.asarray(allpts)

    @property
    def dual_basis(self):
        # Return Q with x.indices already a free index for the
        # consumer to use
        # expensive numerical extraction is done once per element
        # instance, but the point set must be created every time we
        # build the dual.
        Q, pts = self._dual_basis
        x = PointSet(pts)
        assert len(x.indices) == 1
        assert Q.shape[1] == x.indices[0].extent
        i, *js = gem.indices(len(Q.shape) - 1)
        Q = gem.ComponentTensor(gem.Indexed(Q, (i, *x.indices, *js)), (i, *js))
        return Q, x

    @property
    def mapping(self):
        mappings = set(self._element.mapping())
        if len(mappings) != 1:
            return None
        else:
            result, = mappings
            return result


class Regge(FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.Regge(cell, degree, variant=variant))


class HellanHerrmannJohnson(FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.HellanHerrmannJohnson(cell, degree, variant=variant))


class GopalakrishnanLedererSchoberlFirstKind(FiatElement):  # traceless matrix valued
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.GopalakrishnanLedererSchoberlFirstKind(cell, degree, variant=variant))


class GopalakrishnanLedererSchoberlSecondKind(FiatElement):  # traceless matrix valued
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.GopalakrishnanLedererSchoberlSecondKind(cell, degree, variant=variant))


class ScalarFiatElement(FiatElement):
    @property
    def value_shape(self):
        return ()


class Bernstein(ScalarFiatElement):
    # TODO: Replace this with a smarter implementation
    def __init__(self, cell, degree):
        super().__init__(FIAT.Bernstein(cell, degree))


class Bubble(ScalarFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.Bubble(cell, degree, variant=variant))


class FacetBubble(ScalarFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.FacetBubble(cell, degree, variant=variant))


class CrouzeixRaviart(ScalarFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.CrouzeixRaviart(cell, degree, variant=variant))


class Lagrange(ScalarFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.Lagrange(cell, degree, variant=variant))


class KongMulderVeldhuizen(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.KongMulderVeldhuizen(cell, degree))
        if Citations is not None:
            Citations().register("Chin1999higher")
            Citations().register("Geevers2018new")


class DiscontinuousLagrange(ScalarFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.DiscontinuousLagrange(cell, degree, variant=variant))


class Histopolation(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.Histopolation(cell, degree))


class Real(DiscontinuousLagrange):
    ...


class Serendipity(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.Serendipity(cell, degree))


class DPC(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.DPC(cell, degree))


class DiscontinuousTaylor(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.DiscontinuousTaylor(cell, degree))


class HDivTrace(ScalarFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.HDivTrace(cell, degree, variant=variant))


class VectorFiatElement(FiatElement):
    @property
    def value_shape(self):
        return (self.cell.get_spatial_dimension(),)


class RaviartThomas(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.RaviartThomas(cell, degree, variant=variant))


class TrimmedSerendipityFace(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.TrimmedSerendipityFace(cell, degree))

    @property
    def entity_permutations(self):
        raise NotImplementedError(f"entity_permutations not yet implemented for {type(self)}")


class TrimmedSerendipityDiv(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.TrimmedSerendipityDiv(cell, degree))

    @property
    def entity_permutations(self):
        raise NotImplementedError(f"entity_permutations not yet implemented for {type(self)}")


class TrimmedSerendipityEdge(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.TrimmedSerendipityEdge(cell, degree))

    @property
    def entity_permutations(self):
        raise NotImplementedError(f"entity_permutations not yet implemented for {type(self)}")


class TrimmedSerendipityCurl(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.TrimmedSerendipityCurl(cell, degree))

    @property
    def entity_permutations(self):
        raise NotImplementedError(f"entity_permutations not yet implemented for {type(self)}")


class BrezziDouglasMarini(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.BrezziDouglasMarini(cell, degree, variant=variant))


class BrezziDouglasMariniCubeEdge(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.BrezziDouglasMariniCubeEdge(cell, degree))

    @property
    def entity_permutations(self):
        raise NotImplementedError(f"entity_permutations not yet implemented for {type(self)}")


class BrezziDouglasMariniCubeFace(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.BrezziDouglasMariniCubeFace(cell, degree))

    @property
    def entity_permutations(self):
        raise NotImplementedError(f"entity_permutations not yet implemented for {type(self)}")


class BrezziDouglasFortinMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.BrezziDouglasFortinMarini(cell, degree))


class Nedelec(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.Nedelec(cell, degree, variant=variant))


class NedelecSecondKind(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super().__init__(FIAT.NedelecSecondKind(cell, degree, variant=variant))
