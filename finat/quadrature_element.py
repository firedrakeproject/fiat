from finat.point_set import UnknownPointSet, FacetPointSet, UnionPointSet

import numpy

import FIAT

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase, broadcast_tensor
from finat.quadrature import (make_quadrature, AbstractQuadratureRule,
                              QuadratureRule, TensorProductQuadratureRule)


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

    @cached_property
    def _facet_factorisation(self):
        """The products this element is the direct sum of, one per direction.

        Returns
        -------
        tuple or None
            A :class:`~finat.tensor_product.TensorProductElement` for each
            direction the cell's facets lie in, in facet numbering order, or
            ``None`` if the points are not on the facets of a product cell.

        Notes
        -----
        The facets of a product lie in the direction of one factor at a time:
        ``d(A x B) = dA x B  u  A x dB``.  One direction's facets are a
        product, and only then do the points factor, which is what dual
        evaluation needs of them; all the facets at once are not.

        So the rule on a facet is a product of rules on all but one of the
        factors, and the factor left out contributes its two vertices, which
        are the whole of its own boundary.

        The sum cannot be brought outermost as
        :func:`~finat.enriched.as_enriched` does, because the summands number
        their degrees of freedom in each product's order rather than this
        element's; :meth:`dual_evaluation` renumbers them.

        """
        # Avoid circular import dependency
        from finat.tensor_product import TensorProductElement

        product = getattr(self.cell, "product", None)
        if product is None or not isinstance(self._point_set, FacetPointSet):
            return None

        rule = self._rule
        factors = rule.factors if isinstance(rule, TensorProductQuadratureRule) else (rule,)
        assert len(factors) + 1 == len(product.cells)

        summands = []
        for k, cell in enumerate(product.cells):
            ends = make_quadrature(cell.construct_subelement(0), 0)
            rules = (*factors[:k], ends, *factors[k:])
            summands.append(TensorProductElement(
                [QuadratureElement(c, r) for c, r in zip(product.cells, rules)]))
        return tuple(summands)

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

        sd = self.cell.get_spatial_dimension()

        # Tabulate each summand of a union of points on its own points.
        if isinstance(ps, UnionPointSet):
            return self._stack_tabulations(order, ps, entity, coordinate_mapping=coordinate_mapping)

        basis_indices = self.get_indices()
        ps_indices = ps.indices
        if isinstance(self._point_set, FacetPointSet):
            # A FacetPointSet carries a facet index, absent from the rule's.
            ps_indices = (entity_id, *ps_indices)

        rule_ps = self._rule.point_set
        rule_blocks = rule_ps.point_sets if isinstance(rule_ps, UnionPointSet) else (rule_ps,)
        matches = [k for k, block in enumerate(rule_blocks) if block.almost_equal(ps)]
        if not matches:
            raise ValueError("Mismatch of quadrature points!")

        if isinstance(rule_ps, UnionPointSet):
            # `ps` is one summand of the union: tabulate onto the rows of the
            # identity it owns, and zero onto the others.  Concatenating along
            # the basis index lets the contraction with a coefficient split.
            k, = matches
            beta = tuple(gem.Index(extent=index.extent) for index in ps_indices)
            own = gem.ComponentTensor(gem.Delta(ps_indices, beta), beta)
            branches = [own if j == k else gem.Zero(tuple(index.extent for index in other.indices))
                        for j, other in enumerate(rule_ps.point_sets)]
            delta = gem.Indexed(gem.Concatenate(*branches), basis_indices)
        else:
            # Return an outer product of identity matrices
            delta = gem.Delta(ps_indices, basis_indices)

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

    def dual_evaluation(self, fn, coordinate_mapping=None):
        """Dual evaluate on points that factor, a summand at a time."""
        summands = self._facet_factorisation
        if summands is not None:
            branches = []
            for k, summand in enumerate(summands):
                expr, beta = summand.dual_evaluation(
                    fn, coordinate_mapping=coordinate_mapping)
                # Factor k numbers the facets of its direction and the others
                # the points of the facet's own rule, in the same order.
                # Bring the facet index outermost, as this element has it.
                branches.append(gem.ComponentTensor(expr, (beta[k], *beta[:k], *beta[k+1:])))

            # The facets of one direction are numbered consecutively, so the
            # summands stack in order along the flattened basis index.
            beta = gem.Index(extent=self.space_dimension())
            return gem.Indexed(gem.Concatenate(*branches), (beta,)), (beta,)

        rule_ps = self._rule.point_set
        if not isinstance(rule_ps, UnionPointSet):
            return super().dual_evaluation(fn, coordinate_mapping=coordinate_mapping)

        weights = getattr(self._rule, 'weights', None)
        if weights is None:
            weights, = evaluate([self._rule.weight_expression])
            weights = weights.arr.flatten()

        evals = []
        offset = 0
        for summand in rule_ps.point_sets:
            n = len(summand.points)
            sub_rule = QuadratureRule(summand, weights[offset:offset + n], ref_el=self._rule.ref_el)
            sub = QuadratureElement(self.cell, sub_rule)
            expr, sub_indices = sub.dual_evaluation(fn, coordinate_mapping=coordinate_mapping)
            evals.append(broadcast_tensor(expr, sub_indices))
            offset += n

        basis_indices = self.get_indices()
        return gem.Indexed(gem.Concatenate(*evals), basis_indices), basis_indices

    @property
    def mapping(self):
        return "affine"
