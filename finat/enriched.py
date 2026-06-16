from functools import partial
from itertools import chain
from operator import add, methodcaller

import FIAT
import gem
import numpy
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.hdivcurl import HCurlElement, HDivElement


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

    def dual_evaluation(self, argument, coordinate_mapping=None):
        if not self.is_nodal_enriched:
            raise NotImplementedError(
                f"Dual evaluation not defined for element {type(self).__name__}"
            )
        # Gather results from all sub-elements
        # Each sub_result is (eval_expr, local_indices)
        sub_results = [sub.dual_evaluation(argument, coordinate_mapping=coordinate_mapping)
                       for sub in self.elements]

        # Extract the evaluation sub-expressions
        # We must ensure that all subindices are in the free indices of subexpr
        # before wrapping in ComponentTensor. If some are missing (e.g. if the
        # expression simplified to a constant), we multiply by a dummy ones tensor.
        evals = []
        for sub, (subexpr, subindices) in zip(self.elements, sub_results):
            missing_indices = tuple(idx for idx in subindices if idx not in subexpr.free_indices)
            if missing_indices:
                shape = tuple(idx.extent for idx in missing_indices)
                ones = gem.Literal(numpy.ones(shape))
                dummy = gem.Indexed(ones, missing_indices)
                subexpr = gem.Product(subexpr, dummy)
            evals.append(gem.ComponentTensor(subexpr, subindices))

        beta = self.get_indices()
        expr = gem.Indexed(gem.Concatenate(*evals), beta)
        return expr, beta


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
    """Test whether two elements are orthogonal."""
    if isinstance(A, (HCurlElement, HDivElement)) and isinstance(B, (HCurlElement, HDivElement)):
        Amap = A.transform(gem.Literal(numpy.ones(A.wrappee.value_shape)))
        Bmap = B.transform(gem.Literal(numpy.ones(B.wrappee.value_shape)))
        return sum(a * b for a, b in zip(Amap, Bmap)) == gem.Literal(0.0)
    return False
