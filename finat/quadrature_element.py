from finat.point_set import UnknownPointSet, FacetPointSet, UnionPointSet

import numpy

import FIAT
from FIAT.reference_element import Hypercube

import gem
from gem.interpreter import evaluate
from gem.optimise import remove_componenttensors
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
            ``None`` if the quadrature points do not factor by facet direction.

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

        if not isinstance(self.cell, Hypercube):
            return None

        rule = self._rule
        if rule.point_set.dimension != self.cell.get_spatial_dimension() - 1:
            return None

        cells = self.cell.product.cells
        factors = rule.factors if isinstance(rule, TensorProductQuadratureRule) else (rule,)
        if len(factors) + 1 != len(cells):
            return None

        facet_products = []
        for k, cell in enumerate(cells):
            ends = make_quadrature(cell.construct_subelement(0), 0)
            rules = (*factors[:k], ends, *factors[k:])
            facet_products.append(TensorProductElement(
                [QuadratureElement(c, r) for c, r in zip(cells, rules)]))
        return tuple(facet_products)

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
        # Distinct summands can use identical points. Identify the block by
        # object identity to preserve its offset in the coefficient vector.
        matches = [k for k, block in enumerate(blocks) if block is ps]
        if not matches:
            matches = [k for k, block in enumerate(blocks) if block.almost_equal(ps)]
        if not matches:
            raise ValueError("Mismatch of quadrature points!")
        k = matches[0]

        if len(blocks) > 1:
            # `ps` is one point set of the union: tabulate onto the rows of the
            # identity it owns, and zero onto the others.  Concatenating along
            # the basis index lets the contraction with a coefficient split.
            beta = tuple(gem.Index(extent=index.extent) for index in ps_indices)
            own = gem.ComponentTensor(gem.Delta(ps_indices, beta), beta)
            branches = [own if j == k else gem.Zero(tuple(i.extent for i in block.indices))
                        for j, block in enumerate(blocks)]
            delta = gem.Indexed(gem.Concatenate(*branches), basis_indices)
        elif (facet_products := self._facet_factorisation) is not None:
            facet_index, *point_indices = basis_indices
            facet_values = numpy.zeros(facet_index.extent)
            facet_values[entity_id] = 1
            # Group the facet selector by the same directions as dual evaluation.
            facet_blocks = facet_values.reshape(len(facet_products), -1)
            facet_selector = gem.Concatenate(*(gem.Literal(block) for block in facet_blocks))
            delta = (gem.Indexed(facet_selector, (facet_index,))
                     * gem.Delta(ps.indices, tuple(point_indices)))
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

    def dual_evaluation(self, fn, coordinate_mapping=None):
        """Dual evaluate a facet rule one product direction at a time."""
        facet_products = self._facet_factorisation
        if facet_products is not None:
            facet_index, *facet_point_indices = self.get_indices()
            branches = []
            for k, facet_product in enumerate(facet_products):
                expr, point_indices, beta = facet_product.dual_evaluation(
                    fn, coordinate_mapping=coordinate_mapping)
                expr = gem.IndexSum(expr, point_indices)
                point_subst = tuple(zip(beta[:k] + beta[k + 1:], facet_point_indices, strict=True))
                expr, = remove_componenttensors([expr], point_subst)
                branches.append(broadcast_tensor(expr, (beta[k],)))

            # Facets of each direction are consecutive in the facet numbering.
            # Keep the quadrature point axes shared across the directions.
            expr = gem.Indexed(gem.Concatenate(*branches), (facet_index,))
            return expr, (), (facet_index, *facet_point_indices)

        rule_ps = self._rule.point_set
        if not isinstance(rule_ps, UnionPointSet):
            return super().dual_evaluation(fn, coordinate_mapping=coordinate_mapping)

        evals = []
        offset = 0
        for ps in rule_ps.point_sets:
            n = len(ps.points)
            rule = QuadratureRule(ps, self._weights[offset:offset + n], ref_el=self._rule.ref_el)
            element = QuadratureElement(self.cell, rule)
            expr, point_indices, indices = element.dual_evaluation(
                fn, coordinate_mapping=coordinate_mapping)
            evals.append(broadcast_tensor(
                gem.IndexSum(expr, point_indices), indices))
            offset += n

        basis_indices = self.get_indices()
        return gem.Indexed(gem.Concatenate(*evals), basis_indices), (), basis_indices

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
