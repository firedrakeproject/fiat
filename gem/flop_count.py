"""
This file contains all the necessary functions to accurately count the
total number of floating point operations for a given script.
"""

import gem.gem as gem
import gem.impero as imp
from contextvars import ContextVar
from functools import singledispatch
import numpy

from gem.cost import node_cost


@singledispatch
def statement(tree, temporaries):
    raise NotImplementedError


@statement.register(imp.Block)
def statement_block(tree, temporaries):
    flops = sum(statement(child, temporaries) for child in tree.children)
    return flops


@statement.register(imp.For)
def statement_for(tree, temporaries):
    extent = tree.index.extent
    assert extent is not None
    active_token = _active_indices.set(
        _active_indices.get() | {tree.index})
    try:
        index_values = _index_values.get()
        if getattr(tree.index, "parents", ()) and all(
                parent in index_values for parent in tree.index.parents):
            extent = tree.index.iteration_extent(index_values)
        child, = tree.children
        if tree.index in _control_indices.get():
            flops = 0
            for value in range(extent):
                token = _index_values.set(
                    index_values | {tree.index: value})
                try:
                    flops += statement(child, temporaries)
                finally:
                    _index_values.reset(token)
            return flops
        flops = statement(child, temporaries)
        return flops * extent
    finally:
        _active_indices.reset(active_token)


@statement.register(imp.Initialise)
def statement_initialise(tree, temporaries):
    return 0


@statement.register(imp.Accumulate)
def statement_accumulate(tree, temporaries):
    flops = expression_flops(tree.indexsum.children[0], temporaries)
    return flops + 1


@statement.register(imp.Return)
def statement_return(tree, temporaries):
    flops = expression_flops(tree.expression, temporaries)
    return flops + 1


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(tree, temporaries):
    flops = expression_flops(tree.indexsum.children[0], temporaries)
    return flops + 1


@statement.register(imp.Evaluate)
def statement_evaluate(tree, temporaries):
    flops = expression_flops(tree.expression, temporaries, top=True)
    return flops


@singledispatch
def flops(expr, temporaries):
    raise NotImplementedError(f"Don't know how to count flops of {type(expr)}")


@flops.register(gem.Failure)
def flops_failure(expr, temporaries):
    raise ValueError("Not expecting a Failure node")


@flops.register(gem.Variable)
@flops.register(gem.Identity)
@flops.register(gem.Delta)
@flops.register(gem.Zero)
@flops.register(gem.Literal)
@flops.register(gem.Index)
@flops.register(gem.VariableIndex)
def flops_zero(expr, temporaries):
    # Initial set up of these Gem nodes are of 0 floating point operations.
    return 0


@flops.register(gem.LogicalNot)
@flops.register(gem.LogicalAnd)
@flops.register(gem.LogicalOr)
@flops.register(gem.ListTensor)
def flops_zeroplus(expr, temporaries):
    # These nodes contribute 0 floating point operations, but their children may not.
    return 0 + sum(expression_flops(child, temporaries)
                   for child in expr.children)


@flops.register(gem.Product)
@flops.register(gem.Sum)
@flops.register(gem.Division)
@flops.register(gem.Comparison)
@flops.register(gem.MathFunction)
@flops.register(gem.MinValue)
@flops.register(gem.MaxValue)
@flops.register(gem.Power)
def flops_arithmetic(expr, temporaries):
    # The cost of the operation itself is shared with gem.cost, which weighs
    # it by free indices rather than by enclosing loops.
    return node_cost(expr) + sum(expression_flops(child, temporaries)
                                 for child in expr.children)


@flops.register(gem.Conditional)
def flops_conditional(expr, temporaries):
    condition, then, else_ = (expression_flops(child, temporaries)
                              for child in expr.children)
    return condition + max(then, else_)


@flops.register(gem.Indexed)
@flops.register(gem.FlexiblyIndexed)
def flops_indexed(expr, temporaries):
    aggregate = sum(expression_flops(child, temporaries)
                    for child in expr.children)
    # Average flops per entry
    return aggregate / numpy.prod(expr.children[0].shape, dtype=int)


@flops.register(gem.IndexSum)
def flops_indexsum(expr, temporaries):
    raise ValueError("Not expecting IndexSum")


@flops.register(gem.Inverse)
@flops.register(gem.Solve)
def flops_dense_linear_algebra(expr, temporaries):
    return node_cost(expr) + sum(expression_flops(child, temporaries)
                                 for child in expr.children)


@flops.register(gem.ComponentTensor)
def flops_componenttensor(expr, temporaries):
    body, = expr.children
    implicit_indices = tuple(
        index for index in expr.multiindex
        if index not in _active_indices.get())
    if not implicit_indices:
        return expression_flops(body, temporaries)
    control = _control_indices.get().intersection(implicit_indices)
    if not control and not any(
            getattr(index, "parents", ()) for index in implicit_indices):
        extent = numpy.prod(
            [index.extent for index in implicit_indices], dtype=int)
        return extent * expression_flops(body, temporaries)

    def count(position):
        if position == len(implicit_indices):
            return expression_flops(body, temporaries)
        index = implicit_indices[position]
        values = _index_values.get()
        extent = index.extent
        if getattr(index, "parents", ()):
            extent = index.iteration_extent(values)
        total = 0
        for value in range(extent):
            token = _index_values.set(values | {index: value})
            try:
                total += count(position + 1)
            finally:
                _index_values.reset(token)
        return total

    return count(0)


def expression_flops(expression, temporaries, top=False):
    """An approximation to flops required for each expression.

    :arg expression: GEM expression.
    :arg temporaries: Expressions that are assigned to temporaries
    :arg top: are we at the root?
    :returns: flop count for the expression
    """
    if not top and expression in temporaries:
        return 0
    else:
        return flops(expression, temporaries)


def count_flops(impero_c):
    """An approximation to flops required for a scheduled impero_c tree.

    :arg impero_c: a :class:`~.Impero_C` object.
    :returns: approximate flop count for the tree.
    """
    try:
        control_token = _control_indices.set(
            frozenset(_find_control_indices(impero_c.tree)))
        index_token = _index_values.set({})
        active_token = _active_indices.set(frozenset())
        try:
            return statement(impero_c.tree, set(impero_c.temporaries))
        finally:
            _active_indices.reset(active_token)
            _index_values.reset(index_token)
            _control_indices.reset(control_token)
    except (ValueError, NotImplementedError):
        return 0


_index_values = ContextVar("flop_count_index_values", default={})
_active_indices = ContextVar("flop_count_active_indices",
                             default=frozenset())
_control_indices = ContextVar("flop_count_control_indices",
                              default=frozenset())


def _find_control_indices(tree):
    """Find loop indices controlling dependent loop bounds."""
    result = set()
    if isinstance(tree, imp.For):
        if getattr(tree.index, "parents", ()):
            result.update(tree.index.parents)
        result.update(_find_control_indices(tree.children[0]))
    elif isinstance(tree, imp.Block):
        for child in tree.children:
            result.update(_find_control_indices(child))
    return result
