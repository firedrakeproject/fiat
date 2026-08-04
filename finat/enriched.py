from functools import partial, singledispatch
from itertools import chain
from operator import add, methodcaller

import FIAT
import gem
import numpy
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.cube import FlattenedDimensions
from finat.discontinuous import DiscontinuousElement
from finat.finiteelementbase import FiniteElementBase, broadcast_tensor
from finat.hdivcurl import HCurlElement, HDivElement, WrapperElementBase
from finat.point_set import UnionPointSet
from finat.tensor_product import TensorProductElement
from finat.tensorfiniteelement import TensorFiniteElement


class EnrichedElement(FiniteElementBase):
    """A finite element whose basis functions are the union of the
    basis functions of several other finite elements."""

    def __new__(cls, elements, is_nodal_enriched=None):
        elements = tuple(chain.from_iterable(e.elements if isinstance(e, EnrichedElement) else (e,) for e in elements))
        if len(elements) == 1:
            return elements[0]
        else:
            self = super().__new__(cls)
            self.elements = elements

            if is_nodal_enriched is None:
                is_nodal_enriched = all(
                    is_orthogonal(elements[i], elements[j])
                    for i in range(len(elements))
                    for j in range(i+1, len(elements))
                )

            self.is_nodal_enriched = is_nodal_enriched
            return self

    @cached_property
    def cell(self):
        result, = set(elem.cell for elem in self.elements)
        return result

    @cached_property
    def complex(self):
        return FIAT.reference_element.max_complex(set(elem.complex for elem in self.elements))

    @cached_property
    def degree(self):
        return tree_map(max, *[elem.degree for elem in self.elements])

    @cached_property
    def formdegree(self):
        ks = set(elem.formdegree for elem in self.elements)
        if None in ks:
            return None
        else:
            return max(ks)

    def entity_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom for the finite element.'''
        return concatenate_entity_dofs(self.cell, self.elements,
                                       methodcaller("entity_dofs"))

    @cached_property
    def entity_permutations(self):
        '''Return the map of topological entities to the map of
        orientations to permutation lists for the finite element'''
        return concatenate_entity_permutations(self.elements)

    @cached_property
    def _entity_support_dofs(self):
        return concatenate_entity_dofs(self.cell, self.elements,
                                       methodcaller("entity_support_dofs"))

    def space_dimension(self):
        '''Return the dimension of the finite element space.'''
        return sum(elem.space_dimension() for elem in self.elements)

    @cached_property
    def index_shape(self):
        return (self.space_dimension(),)

    @cached_property
    def value_shape(self):
        '''A tuple indicating the shape of the element.'''
        shape, = set(elem.value_shape for elem in self.elements)
        return shape

    @cached_property
    def fiat_equivalent(self):
        if self.is_mixed:
            # EnrichedElement is actually a MixedElement
            return FIAT.MixedElement([e.element.fiat_equivalent
                                      for e in self.elements], ref_el=self.cell)
        else:
            return FIAT.EnrichedElement(*(e.fiat_equivalent
                                          for e in self.elements))

    @cached_property
    def is_mixed(self):
        # Avoid circular import dependency
        from finat.mixed import MixedSubElement

        return all(isinstance(e, MixedSubElement) for e in self.elements)

    def _compose_evaluations(self, results):
        keys, = set(map(frozenset, results))

        def merge(tables):
            tables = tuple(tables)
            zeta = self.get_value_indices()
            tensors = []
            for elem, table in zip(self.elements, tables):
                beta_i = elem.get_indices()
                tensors.append(gem.ComponentTensor(
                    gem.Indexed(table, beta_i + zeta),
                    beta_i
                ))
            beta = self.get_indices()
            return gem.ComponentTensor(
                gem.Indexed(gem.Concatenate(*tensors), beta),
                beta + zeta
            )
        return {key: merge(result[key] for result in results)
                for key in keys}

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''
        results = [element.basis_evaluation(order, ps, entity, coordinate_mapping=coordinate_mapping)
                   for element in self.elements]
        return self._compose_evaluations(results)

    def point_evaluation(self, order, refcoords, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at an arbitrary points on
        the reference element.

        :param order: return derivatives up to this order.
        :param refcoords: GEM expression representing the coordinates
                          on the reference entity.  Its shape must be
                          a vector with the correct dimension, its
                          free indices are arbitrary.
        :param entity: the cell entity on which to tabulate.
        '''
        results = [element.point_evaluation(order, refcoords, entity, coordinate_mapping)
                   for element in self.elements]
        return self._compose_evaluations(results)

    @property
    def mapping(self):
        mappings = set(elem.mapping for elem in self.elements)
        if len(mappings) != 1:
            return None
        else:
            result, = mappings
            return result

    @cached_property
    def _summands(self):
        """The summands that are not themselves direct sums, in basis order.

        An element is rewritten as a direct sum one level at a time, so an
        element this one enriches may be a sum in turn.  The summands that are
        not are the ones that evaluate on their own points, and so the ones
        whose points make up the union that :attr:`dual_basis` and
        :meth:`dual_evaluation` both work against.
        """
        summands = []
        for element in self.elements:
            expanded = as_enriched(element)
            summands.extend(expanded._summands if expanded is not None
                            else [element])
        return tuple(summands)

    @property
    def dual_basis(self):
        """The weights of the dual basis, on the union of the summands' points.

        Returns
        -------
        tuple
            A ``(Q, x)`` pair, as
            :attr:`~finat.finiteelementbase.FiniteElementBase.dual_basis`.

        Notes
        -----
        A summand's functionals evaluate only on that summand's points, so the
        weights are block diagonal: they concatenate along the basis index, and
        along the point index each summand's weights sit at its own offset in
        the union, with zeros against every other summand's points.

        The concatenation over the points cannot be split, as the point index
        is contracted away, so the zeros off the diagonal are real entries.
        :meth:`dual_evaluation` contracts each summand on its own points
        rather than use these weights; they are here for the callers that
        want a single weight tensor.

        """
        duals = [element.dual_basis for element in self._summands]
        x = UnionPointSet([xk for _, xk in duals])
        p, = x.indices
        zeta = self.get_value_indices()
        # The natural shape of each summand's own points, in the order they
        # occupy the union.
        shapes = [tuple(i.extent for i in xk.indices) for _, xk in duals]

        blocks = []
        for k, (element, (Q, xk)) in enumerate(zip(self._summands, duals)):
            alpha = element.get_indices()
            # Turn this summand's point indices into a shape, so that its
            # weights can be embedded at its own offset in the union.
            own = gem.ComponentTensor(gem.Indexed(Q, alpha + zeta), xk.indices)
            pieces = [own if j == k else gem.Zero(shape)
                      for j, shape in enumerate(shapes)]
            weights = gem.Indexed(gem.Concatenate(*pieces), (p,))
            blocks.append(gem.ComponentTensor(weights, alpha))

        beta, = self.get_indices()
        Q = gem.Indexed(gem.Concatenate(*blocks), (beta,))
        return gem.ComponentTensor(Q, (beta,) + zeta), x

    def dual_evaluation(self, fn, coordinate_mapping=None):
        """Dual evaluate each element on its own points.

        The summands do not all evaluate on the same points, so contracting
        :attr:`dual_basis` would contract against the zeros it carries off its
        diagonal: each element contracts on its own points instead, and the
        results stack along the basis index the sum occupies, which is the one
        :meth:`_compose_evaluations` also concatenates over.

        Concatenating over the basis index is what
        :func:`~gem.unconcatenate.unconcatenate` is for: the index stays free
        in the assignment, so it can be split downstream.  A concatenation
        over the points could not be, as the points are contracted here.

        Parameters
        ----------
        fn : Callable
            As :meth:`~finat.finiteelementbase.FiniteElementBase.dual_evaluation`.
        coordinate_mapping : PhysicalGeometry or None
            As :meth:`~finat.finiteelementbase.FiniteElementBase.dual_evaluation`.

        Returns
        -------
        tuple
            An ``(expression, point_indices, basis_indices)`` triple, as
            :meth:`~finat.finiteelementbase.FiniteElementBase.dual_evaluation`
            returns.  The points are contracted here, so none is left free.

        """
        if not self.is_nodal_enriched:
            raise NotImplementedError(
                f"Dual evaluation not defined for non-nodal {type(self).__name__}"
            )
        evals = []
        for element in self.elements:
            expr, point_indices, indices = element.dual_evaluation(
                fn, coordinate_mapping=coordinate_mapping)
            evals.append(broadcast_tensor(gem.IndexSum(expr, point_indices), indices))

        beta = self.get_indices()
        return gem.Indexed(gem.Concatenate(*evals), beta), (), beta


@singledispatch
def as_enriched(element):
    """Rewrite an element as a direct sum, bringing the sum outermost.

    Parameters
    ----------
    element : FiniteElementBase
        The element to rewrite.

    Returns
    -------
    EnrichedElement or None
        An element with the same basis functions in the same order as
        ``element``, whose summands each evaluate their dual basis on their
        own points, or ``None`` if ``element`` is not a direct sum.

    Notes
    -----
    Only a direct sum can be dual evaluated summand by summand, so an element
    with a summed part has to be rewritten with the sum outermost before its
    dual basis can be evaluated.  The sum commutes with everything that can
    hold it:

    * a pullback is linear, so it distributes over the sum;
    * a tensor product distributes over a sum in its first factor;
    * the wrappers that leave the dual basis alone are transparent.

    """
    return None


@as_enriched.register(EnrichedElement)
def as_enriched_enriched(element):
    return element


@as_enriched.register(FlattenedDimensions)
def as_enriched_flattened(element):
    return as_enriched(element.product)


@as_enriched.register(DiscontinuousElement)
def as_enriched_discontinuous(element):
    return as_enriched(element.element)


@as_enriched.register(WrapperElementBase)
def as_enriched_wrapper(element):
    """Distribute the pullback over the sum the wrapped element is."""
    summands = as_enriched(element.wrappee)
    if summands is None:
        return None
    # The pullback is chosen from the mapping and form degree, which the
    # summands share with their sum, so every summand takes the same one.
    return EnrichedElement([type(element)(e) for e in summands.elements],
                           is_nodal_enriched=summands.is_nodal_enriched)


@as_enriched.register(TensorFiniteElement)
def as_enriched_tensor_finite_element(element):
    """Distribute the vector/tensor wrapper over the sum its base element is."""
    summands = as_enriched(element.base_element)
    if summands is None:
        return None
    return EnrichedElement(
        [TensorFiniteElement(e, element._shape, element._transpose) for e in summands.elements],
        is_nodal_enriched=summands.is_nodal_enriched)


@as_enriched.register(TensorProductElement)
def as_enriched_tensor_product(element):
    """Distribute the product over the sum its first factor is.

    Only the first factor may be summed.  The summands of a sum in the first
    factor own a contiguous range of the flat basis index, so they stack in
    the order the product already numbers them; a sum in any later factor
    would interleave with the factors before it, and stacking would renumber
    the degrees of freedom.
    """
    first, *rest = element.factors
    if any(as_enriched(factor) is not None for factor in rest):
        raise NotImplementedError(
            "Only the first factor of a TensorProductElement may be a direct"
            " sum, as the degrees of freedom of a later one do not stack"
        )
    summands = as_enriched(first)
    if summands is None:
        return None
    return EnrichedElement(
        [TensorProductElement((e, *rest)) for e in summands.elements],
        is_nodal_enriched=summands.is_nodal_enriched)


def tree_map(f, *args):
    """Like the built-in :py:func:`map`, but applies to a tuple tree."""
    nonleaf, = set(isinstance(arg, tuple) for arg in args)
    if nonleaf:
        ndim, = set(map(len, args))  # asserts equal arity of all args
        return tuple(tree_map(f, *subargs) for subargs in zip(*args))
    else:
        return f(*args)


def concatenate_entity_dofs(ref_el, elements, method):
    """Combine the entity DoFs from a list of elements into a combined
    dict containing the information for the concatenated DoFs of all
    the elements.

    :arg ref_el: the reference cell
    :arg elements: subelement whose DoFs are concatenated
    :arg method: method to obtain the entity DoFs dict
    :returns: concatenated entity DoFs dict
    """
    entity_dofs = {dim: {i: [] for i in entities}
                   for dim, entities in ref_el.get_topology().items()}
    offsets = numpy.cumsum([0] + list(e.space_dimension()
                                      for e in elements), dtype=int)
    for i, d in enumerate(map(method, elements)):
        for dim, dofs in d.items():
            for ent, off in dofs.items():
                entity_dofs[dim][ent] += list(map(partial(add, offsets[i]), off))
    return entity_dofs


def concatenate_entity_permutations(elements):
    """For each dimension, for each entity, and for each possible
    entity orientation, collect the DoF permutation lists from
    entity_permutations dicts of elements and concatenate them.

    :arg elements: subelements whose DoF permutation lists are concatenated
    :returns: entity_permutation dict of the :class:`EnrichedElement` object
        composed of elements.
    """
    permutations = {}
    for element in elements:
        for dim, e_o_p_map in element.entity_permutations.items():
            dim_permutations = permutations.setdefault(dim, {})
            for e, o_p_map in e_o_p_map.items():
                e_dim_permutations = dim_permutations.setdefault(e, {})
                for o, p in o_p_map.items():
                    o_e_dim_permutations = e_dim_permutations.setdefault(o, [])
                    offset = len(o_e_dim_permutations)
                    o_e_dim_permutations += list(offset + q for q in p)
    return permutations


def is_orthogonal(A, B):
    """Test whether two elements map into orthogonal components.

    .. todo::

       Read the components off the products of :class:`gem.Delta` that the
       transforms select with, rather than evaluating the contraction
       numerically in an element constructor.
    """
    if isinstance(A, (HCurlElement, HDivElement)) and isinstance(B, (HCurlElement, HDivElement)):
        dim, = A.value_shape
        zeta = gem.Index(extent=dim)
        a = A.transform(gem.Literal(numpy.ones(A.wrappee.value_shape)), zeta)
        b = B.transform(gem.Literal(numpy.ones(B.wrappee.value_shape)), zeta)
        contraction = gem.ComponentTensor(gem.Product(a, b), (zeta,))
        result, = evaluate([contraction])
        return not result.arr.any()
    return False
