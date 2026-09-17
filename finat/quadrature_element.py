from finat.point_set import UnknownPointSet, FacetPointSet

import numpy

import FIAT

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.quadrature import make_quadrature, AbstractQuadratureRule, QuadratureRule


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

    if isinstance(scheme, AbstractQuadratureRule):
        rule = scheme
        assert rule.ref_el >= rule_ref_cell
    else:
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
        ps = self._rule.point_set
        sd = self.cell.get_spatial_dimension()
        if not isinstance(ps, UnknownPointSet) and ps.dimension == sd:
            return self.cell.point_entity_ids(ps.points)

        top = self.cell.get_topology()
        entity_dofs = {dim: {entity: [] for entity in entities}
                       for dim, entities in top.items()}
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
    def _weights(self):
        if isinstance(self._rule, QuadratureRule):
            return self._rule.weights
        weights, = evaluate([self._rule.weight_expression])
        return weights.arr.flatten()

    @cached_property
    def fiat_equivalent(self):
        ps = self._point_set
        if isinstance(ps, UnknownPointSet):
            raise ValueError("A quadrature element with rule with runtime points has no fiat equivalent!")
        return FIAT.QuadratureElement(self.cell, ps.points, self._weights)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''
        if entity is None:
            entity = (self.cell.get_dimension(), 0)
        entity_dim, entity_id = entity
        if isinstance(entity_dim, tuple):
            entity_dim = sum(entity_dim)

        rule_dim = self._rule.point_set.dimension
        if entity_dim != rule_dim:
            raise ValueError(f"Cannot tabulate QuadratureElement of dimension {rule_dim}"
                             f" on subentities of dimension {entity_dim}.")

        if order:
            raise ValueError("Derivatives are not defined on a QuadratureElement.")

        # A union of points has no structure of its own to tabulate on.
        if len(ps.point_sets) > 1:
            return self._stack_tabulations(order, ps, entity, coordinate_mapping=coordinate_mapping)

        basis_indices = self.get_indices()
        ps_indices = ps.indices
        if isinstance(self._point_set, FacetPointSet):
            # A FacetPointSet carries a facet index, absent from the rule's.
            ps_indices = (entity_id, *ps_indices)

        rule_ps = self._rule.point_set
        blocks = rule_ps.point_sets
        matches = [k for k, block in enumerate(blocks) if block.almost_equal(ps)]
        if not matches:
            raise ValueError("Mismatch of quadrature points!")
        k, = matches

        if len(blocks) > 1:
            # `ps` is one point set of the union: tabulate onto the rows of the
            # identity it owns, and zero onto the others.  Concatenating along
            # the basis index lets the contraction with a coefficient split.
            beta = tuple(gem.Index(extent=index.extent) for index in ps_indices)
            own = gem.ComponentTensor(gem.Delta(ps_indices, beta), beta)
            branches = [own if j == k else gem.Zero(tuple(i.extent for i in block.indices))
                        for j, block in enumerate(blocks)]
            delta = gem.Indexed(gem.Concatenate(*branches), basis_indices)
        else:
            # Return an outer product of identity matrices
            delta = gem.Delta(ps_indices, basis_indices)

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

    @cached_property
    def _summand_rules(self):
        """The rules of the summands this element is a direct sum of.

        :returns: one :class:`~finat.quadrature.QuadratureRule` for each
            point set of a :class:`~finat.point_set.UnionPointSet` rule, in
            the order they are stacked, or an empty tuple if the rule has a
            single point set.

        A union of point sets is how the points of a direct sum are stacked,
        so splitting it back up recovers a rule for each summand, each on
        the points its own functionals evaluate on.
        """
        rule_ps = self._rule.point_set
        if len(rule_ps.point_sets) == 1:
            return ()

        rules = []
        offset = 0
        for ps in rule_ps.point_sets:
            n = len(ps.points)
            rules.append(QuadratureRule(ps, self._weights[offset:offset + n],
                                        ref_el=self._rule.ref_el))
            offset += n
        return tuple(rules)

    @property
    def mapping(self):
        return "affine"
