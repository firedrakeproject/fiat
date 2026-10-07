"""Composite optimisation pipelines over GEM.

The passes in `gem.optimise`, `gem.coffee`, `gem.refactorise` and
`gem.jagged` are primitives; each rewrites a DAG in one way.  This module is
the layer above them, where a pipeline may use all four.  Keeping the
pipelines here is what lets the primitive modules stay free of each other.
"""

from collections import OrderedDict
from collections.abc import Iterable
from itertools import zip_longest

import numpy

from gem.coffee import optimise_monomial_sum
from gem.gem import (ComponentTensor, Delta, FlattenedTensor, Indexed,
                     IndexSum, ListTensor, Node, Product)
from gem.jagged import unflatten_free_indices, unflatten
from gem.node import MemoizerArg, traversal
from gem.optimise import (cancel_nested_deltas, delta_elimination,
                          filtered_replace_indices,
                          pull_back_indirect_delta, repeated_contractions,
                          sum_factorise, traverse_product)
from gem.refactorise import (ATOMIC, COMPOUND, OTHER, FactorisationError,
                             collect_monomials)


def contraction(expression):
    """Optimise the contractions of the tensor product at the root of
    the expression, including:

    - IndexSum-Delta cancellation
    - Sum factorisation

    This routine was designed with finite element coefficient
    evaluation in mind.
    """

    # Common memoizer to remove ComponentTensors
    index_replacer = MemoizerArg(filtered_replace_indices)

    # Eliminate annoying ComponentTensors
    expression = index_replacer(expression, ())

    def flatten(root):
        """Break the product tree at ``root`` up and cancel its Deltas."""
        # The contraction at the root is always broken up, as that is the
        # one being optimised
        keep = repeated_contractions(root)
        sum_indices, factors = traverse_product(
            root, index_replacer=index_replacer,
            stop_at=lambda e: e is not root and e in keep)
        sum_indices, factors = pull_back_indirect_delta(
            sum_indices, factors, index_replacer)
        sum_indices, factors = delta_elimination(
            sum_indices, factors, index_replacer=index_replacer)
        return sum_indices, [index_replacer(f, ()) for f in factors]

    def rebuild(expression):
        sum_indices, factors = flatten(expression)
        flattened = IndexSum(Product(*factors), sum_indices)
        unflattened = unflatten(flattened)
        if unflattened is not flattened:
            sum_indices, factors = flatten(unflattened)
        return sum_factorise(sum_indices, factors)

    # Sometimes the value shape is composed as a ListTensor, which
    # could get in the way of decomposing factors.  In particular,
    # this is the case for H(div) and H(curl) conforming tensor
    # product elements.  So if ListTensors are used, they are pulled
    # out to be outermost, so we can straightforwardly factorise each
    # of its entries.
    lt_fis = OrderedDict()  # ListTensor free indices
    for node in traversal((expression,)):
        if isinstance(node, Indexed):
            child, = node.children
            if isinstance(child, ListTensor):
                lt_fis.update(zip_longest(node.multiindex, ()))
    lt_fis = tuple(index for index in lt_fis if index in expression.free_indices)

    if lt_fis:
        # Rebuild each split component
        tensor = ComponentTensor(expression, lt_fis)
        entries = [Indexed(tensor, zeta) for zeta in numpy.ndindex(tensor.shape)]
        entries = [index_replacer(e, ()) for e in entries]
        return Indexed(ListTensor(
            numpy.array(list(map(rebuild, entries))).reshape(tensor.shape)
        ), lt_fis)
    else:
        # Rebuild whole expression at once
        return rebuild(expression)


def _refactor_unflattened_outputs(
        outputs: list[tuple[Node, Node]]) -> list[tuple[Node, Node]] | None:
    r"""Recover sum factorisation after exposing every argument lattice.

    Consider a bilinear form whose argument tabulations are transformed by
    sparse matrices,

    .. math::

        A_{ij} = \sum_q (S B(q))_i\,G(q)\,(T C(q))_j.

    Sparse-delta cancellation moves rows of ``S`` and ``T`` into the return
    scatter.  Both flat argument indices must then be replaced by their
    jagged lattice multiindices *before* expanding derivative sums.  Expanding
    after only one replacement duplicates the second lattice once per
    summand, producing many equivalent loop nests.

    The refactorisation is valid when the non-tabulation part of every
    monomial is independent of the argument indices.  This condition says
    exactly that the sparse transforms have been absorbed by the scatter.
    Dense or otherwise residual transforms use the conservative legacy path.
    """
    candidates = []
    has_nontrivial_sum = False
    for variable, expression in outputs:
        argument_indices = frozenset(variable.free_indices)
        contraction_indices = frozenset(
            index
            for node in traversal((expression,))
            if isinstance(node, IndexSum)
            for index in node.multiindex)

        def classify(node):
            involved = argument_indices.intersection(node.free_indices)
            if not involved:
                return OTHER
            if isinstance(node, Indexed):
                return ATOMIC if contraction_indices.intersection(
                    node.free_indices) else OTHER
            return COMPOUND

        try:
            monomial_sum, = collect_monomials([expression], classify)
        except FactorisationError:
            return None

        monomials = tuple(monomial_sum)
        if any(argument_indices.intersection(monomial.rest.free_indices)
               for monomial in monomials):
            return None
        has_nontrivial_sum |= len(monomials) > 1
        sum_indices = tuple(OrderedDict.fromkeys(
            index
            for monomial in monomials
            for index in monomial.sum_indices))
        candidates.append((variable, monomial_sum, sum_indices))

    if not has_nontrivial_sum:
        return None
    return [
        (variable, optimise_monomial_sum(
            monomial_sum, variable.index_ordering(), sum_indices))
        for variable, monomial_sum, sum_indices in candidates
    ]


def unflatten_returns(
        pairs: Iterable[tuple[Node, Node]]) -> list[tuple[Node, Node]]:
    """Unflatten free argument indices in assignment pairs.

    Every compatible argument lattice is exposed jointly.  Bilinear
    expressions are then refactorised as monomial sums so that sparse
    argument transforms remain in the output scatter while quadrature
    contractions recover their one-dimensional factors.  Expressions with
    residual argument-dependent transforms retain the established local
    distribution strategy.
    """
    pairs = list(pairs)
    if not any(isinstance(node, FlattenedTensor)
               for _, expression in pairs
               for node in traversal((expression,))):
        return pairs

    result = []
    for variable, expression in pairs:
        expression = cancel_nested_deltas(expression)
        joint_outputs, changed = unflatten_free_indices(
            variable, expression, split_separable_sums=False)
        refactored = _refactor_unflattened_outputs(
            joint_outputs) if changed else None
        if refactored is not None:
            result.extend(refactored)
            continue

        outputs, changed = unflatten_free_indices(
            variable, expression, split_separable_sums=True)
        if changed or any(isinstance(node, Delta)
                          for _, output in outputs
                          for node in traversal((output,))):
            outputs = [(output_variable, contraction(output))
                       for output_variable, output in outputs]
        result.extend(outputs)
    return result
