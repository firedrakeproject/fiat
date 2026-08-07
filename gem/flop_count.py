"""
This file contains all the necessary functions to accurately count the
total number of floating point operations for a given script.
"""

import gem.gem as gem
import gem.impero as imp
from contextvars import ContextVar
from functools import singledispatch
from gem.node import traversal
import numpy
import math


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
    index_values = _index_values.get()
    if getattr(tree.index, "parents", ()) and all(
            parent in index_values for parent in tree.index.parents):
        extent = tree.index.iteration_extent(index_values)
    child, = tree.children
    if tree.index in _control_indices.get():
        flops = 0
        for value in range(extent):
            token = _index_values.set(index_values | {tree.index: value})
            try:
                flops += statement(child, temporaries)
            finally:
                _index_values.reset(token)
        return flops
    flops = statement(child, temporaries)
    return flops * extent


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
@flops.register(gem.Comparison)
def flops_zeroplus(expr, temporaries):
    # These nodes contribute 0 floating point operations, but their children may not.
    return 0 + sum(expression_flops(child, temporaries)
                   for child in expr.children)


@flops.register(gem.Product)
def flops_product(expr, temporaries):
    # Multiplication by -1 is not a flop.
    a, b = expr.children
    if isinstance(a, gem.Literal) and a.value == -1:
        return expression_flops(b, temporaries)
    elif isinstance(b, gem.Literal) and b.value == -1:
        return expression_flops(a, temporaries)
    else:
        return 1 + sum(expression_flops(child, temporaries)
                       for child in expr.children)


@flops.register(gem.Sum)
@flops.register(gem.Division)
@flops.register(gem.MathFunction)
@flops.register(gem.MinValue)
@flops.register(gem.MaxValue)
def flops_oneplus(expr, temporaries):
    return 1 + sum(expression_flops(child, temporaries)
                   for child in expr.children)


@flops.register(gem.Power)
def flops_power(expr, temporaries):
    base, exponent = expr.children
    base_flops = expression_flops(base, temporaries)
    if isinstance(exponent, gem.Literal):
        exponent = exponent.value
        if exponent > 0 and exponent == math.floor(exponent):
            return base_flops + int(math.ceil(math.log2(exponent)))
        else:
            return base_flops + 5  # heuristic
    else:
        return base_flops + 5   # heuristic


@flops.register(gem.Conditional)
def flops_conditional(expr, temporaries):
    condition, then, else_ = expr.children
    condition_flops = expression_flops(condition, temporaries)
    value = _static_value(condition)
    if value is not _UNKNOWN:
        branch = then if value else else_
        return condition_flops + expression_flops(branch, temporaries)
    then_flops = expression_flops(then, temporaries)
    else_flops = expression_flops(else_, temporaries)
    return condition_flops + max(then_flops, else_flops)


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
def flops_inverse(expr, temporaries):
    n, _ = expr.shape
    # 2n^3 + child flop count
    return 2*n**3 + sum(expression_flops(child, temporaries)
                        for child in expr.children)


@flops.register(gem.Solve)
def flops_solve(expr, temporaries):
    n, m = expr.shape
    # 2mn + inversion cost of A + children flop count
    return 2*n*m + 2*n**3 + sum(expression_flops(child, temporaries)
                                for child in expr.children)


@flops.register(gem.ComponentTensor)
def flops_componenttensor(expr, temporaries):
    body, = expr.children
    multiindex = expr.multiindex
    control = _control_indices.get().intersection(multiindex)
    if not control and not any(
            getattr(index, "parents", ()) for index in multiindex):
        return numpy.prod(expr.shape, dtype=int) * expression_flops(
            body, temporaries)

    def count(position):
        if position == len(multiindex):
            return expression_flops(body, temporaries)
        index = multiindex[position]
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
        try:
            return statement(impero_c.tree, set(impero_c.temporaries))
        finally:
            _index_values.reset(index_token)
            _control_indices.reset(control_token)
    except (ValueError, NotImplementedError):
        return 0

_UNKNOWN = object()
_index_values = ContextVar("flop_count_index_values", default={})
_control_indices = ContextVar("flop_count_control_indices",
                              default=frozenset())


def _static_index(index):
    """Evaluate an index from the current statically known loop values."""
    if isinstance(index, (int, numpy.integer)):
        return int(index)
    if isinstance(index, gem.Index):
        return _index_values.get().get(index, _UNKNOWN)
    if isinstance(index, gem.VariableIndex):
        value = _static_value(index.expression)
        return _UNKNOWN if value is _UNKNOWN else int(value)
    return _UNKNOWN


def _static_value(expression):
    """Partially evaluate compile-time data in a scalar expression."""
    if isinstance(expression, gem.Zero):
        return 0
    if isinstance(expression, gem.Literal) and not expression.shape:
        return expression.value
    if isinstance(expression, gem.Index):
        return _static_index(expression)
    if isinstance(expression, gem.Indexed):
        aggregate, = expression.children
        multiindex = tuple(map(_static_index, expression.multiindex))
        if any(index is _UNKNOWN for index in multiindex):
            return _UNKNOWN
        if isinstance(aggregate, gem.Literal):
            return aggregate.array[multiindex]
        if isinstance(aggregate, gem.ListTensor):
            return _static_value(aggregate.array[multiindex])
        return _UNKNOWN
    if isinstance(expression, gem.Comparison):
        left, right = map(_static_value, expression.children)
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        if expression.operator == "==":
            return left == right
        if expression.operator == "!=":
            return left != right
        if expression.operator == "<":
            return left < right
        if expression.operator == "<=":
            return left <= right
        if expression.operator == ">":
            return left > right
        if expression.operator == ">=":
            return left >= right
    if isinstance(expression, gem.LogicalNot):
        value = _static_value(expression.children[0])
        return _UNKNOWN if value is _UNKNOWN else not value
    if isinstance(expression, (gem.LogicalAnd, gem.LogicalOr)):
        left, right = map(_static_value, expression.children)
        if isinstance(expression, gem.LogicalAnd):
            if left is False or right is False:
                return False
            return _UNKNOWN if left is _UNKNOWN or right is _UNKNOWN else True
        if left is True or right is True:
            return True
        return _UNKNOWN if left is _UNKNOWN or right is _UNKNOWN else False
    return _UNKNOWN


def _terminal_expressions(tree):
    """Return GEM expressions evaluated by an Impero terminal."""
    if isinstance(tree, imp.Evaluate):
        return (tree.expression,)
    if isinstance(tree, imp.Accumulate):
        return tree.indexsum.children
    if isinstance(tree, imp.Return):
        return (tree.expression,)
    if isinstance(tree, imp.ReturnAccumulate):
        return tree.indexsum.children
    if isinstance(tree, imp.Noop):
        return (tree.expression,)
    return ()


def _find_control_indices(tree):
    """Find loop indices controlling jagged bounds or static branches."""
    result = set()
    if isinstance(tree, imp.For):
        if getattr(tree.index, "parents", ()):
            result.update(tree.index.parents)
        result.update(_find_control_indices(tree.children[0]))
    elif isinstance(tree, imp.Block):
        for child in tree.children:
            result.update(_find_control_indices(child))
    else:
        for expression in _terminal_expressions(tree):
            for node in traversal((expression,)):
                if isinstance(node, gem.Conditional):
                    result.update(node.children[0].free_indices)
    return result
