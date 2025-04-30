from finat.point_set import UnknownPointSet, FacetPointSet

import numpy

import FIAT

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.quadrature import make_quadrature, AbstractQuadratureRule


def make_quadrature_element(fiat_ref_cell, degree, scheme="default", codim=0):
    """Construct a :class:`QuadratureElement` from a given a reference
    element, degree and scheme.

    :param fiat_ref_cell: The FIAT reference cell to build the
        :class:`QuadratureElement` on.
    :param degree: The degree of polynomial that the rule should
        integrate exactly.
    :param scheme: The quadrature scheme to use - e.g. "default",
        "canonical" or "KMV".
    :param codim: The codimension of the quadrature scheme.
    :returns: The appropriate :class:`QuadratureElement`
    """
    if codim > 0:
        sd = fiat_ref_cell.get_spatial_dimension()
        rule_ref_cell = fiat_ref_cell.construct_subelement(sd - codim)
    else:
        rule_ref_cell = fiat_ref_cell

    rule = make_quadrature(rule_ref_cell, degree, scheme=scheme)
    return QuadratureElement(fiat_ref_cell, rule)


class QuadratureElement(FiniteElementBase):
    """A set of quadrature points pretending to be a finite element."""

    def __init__(self, fiat_ref_cell, rule):
        """Construct a :class:`QuadratureElement`.

        :param fiat_ref_cell: The FIAT reference cell to build the
            :class:`QuadratureElement` on
        :param rule: A :class:`AbstractQuadratureRule` to use
        """
        self.cell = fiat_ref_cell
        if not isinstance(rule, AbstractQuadratureRule):
            raise TypeError("rule is not an AbstractQuadratureRule")
        self._rule = rule

    @cached_property
    def cell(self):
        pass  # set at initialisation

    @property
    def complex(self):
        return self.cell

    @property
    def degree(self):
        raise NotImplementedError("QuadratureElement does not represent a polynomial space.")

    @property
    def formdegree(self):
        return None

    @cached_property
    def _entity_dofs(self):
        top = self.cell.get_topology()
        entity_dofs = {dim: {entity: [] for entity in entities}
                       for dim, entities in top.items()}
        ps = self._rule.point_set
        num_pts = len(ps.points)
        to_int = lambda x: sum(x) if isinstance(x, tuple) else x
        cur = 0
        for dim in sorted(top):
            if to_int(dim) == ps.dimension:
                for entity in sorted(top[dim]):
                    entity_dofs[dim][entity].extend(range(cur, cur + num_pts))
                    cur += num_pts
        return entity_dofs

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return numpy.prod(self.index_shape, dtype=int)

    @cached_property
    def _point_set(self):
        ps = self._rule.point_set
        sd = self.cell.get_spatial_dimension()
        return ps if ps.dimension == sd else FacetPointSet(self.cell, ps)

    @property
    def index_shape(self):
        ps = self._point_set
        return tuple(index.extent for index in ps.indices)

    @property
    def value_shape(self):
        return ()

    @cached_property
    def fiat_equivalent(self):
        ps = self._point_set
        if isinstance(ps, UnknownPointSet):
            raise ValueError("A quadrature element with rule with runtime points has no fiat equivalent!")
        weights = getattr(self._rule, 'weights', None)
        if weights is None:
            # we need the weights.
            weights, = evaluate([self._rule.weight_expression])
            weights = weights.arr.flatten()
            self._rule.weights = weights

        return FIAT.QuadratureElement(self.cell, ps.points, weights)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''
        rule_dim = self._rule.point_set.dimension
        if entity is None:
            entity = (rule_dim, 0)
        entity_dim, entity_id = entity
        if entity_dim != rule_dim:
            raise ValueError(f"Cannot tabulate QuadratureElement of dimension {rule_dim}"
                             f" on subentities of dimension {entity_dim}.")

        if order:
            raise ValueError("Derivatives are not defined on a QuadratureElement.")

        if not self._rule.point_set.almost_equal(ps):
            raise ValueError("Mismatch of quadrature points!")

        # Return an outer product of identity matrices
        basis_indices = self.get_indices()
        point_indices = ps.indices
        if len(basis_indices) > len(point_indices):
            point_indices = (entity_id, *point_indices)
        delta = gem.Delta(point_indices, basis_indices)

        sd = self.cell.get_spatial_dimension()
        return {(0,) * sd: gem.ComponentTensor(delta, basis_indices)}

    def point_evaluation(self, order, refcoords, entity=None, coordinate_mapping=None):
        raise NotImplementedError("QuadratureElement cannot do point evaluation!")

    @property
    def dual_basis(self):
        ps = self._point_set
        multiindex = self.get_indices()
        # Evaluation matrix is just an outer product of identity
        # matrices, evaluation points are just the quadrature points.
        Q = gem.Delta(ps.indices, multiindex)
        Q = gem.ComponentTensor(Q, multiindex)
        return Q, ps

    @property
    def mapping(self):
        return "affine"
