import functools
import numbers
from collections.abc import Mapping

import gem.gem
import gem.node


def replace(expr, replace_map: Mapping[gem.gem.Node, gem.gem.Node]):
    """Replace each subexpression of ``expr`` that is a key of ``replace_map``, also inside index expressions.

    A replacement is not rewritten again.
    """
    cache = {}

    def mapper(node):
        if node in replace_map:
            return replace_map[node]
        # Literal equality ignores the dtype, so 0 and 0.0 must not share an entry
        key = (node, node.dtype)
        if key not in cache:
            cache[key] = _replace(node, mapper)
        return cache[key]

    return mapper(expr)


@functools.singledispatch
def _replace(node, self):
    return gem.node.reuse_if_untouched(node, self)


@_replace.register
def _(node: gem.gem.Indexed, self):
    multiindex = tuple(_replace_index(index, self) for index in node.multiindex)
    aggregate = self(node.aggregate)
    if aggregate is node.aggregate and multiindex == node.multiindex:
        return node
    return gem.gem.Indexed(aggregate, multiindex)


@_replace.register
def _(node: gem.gem.VariableIndex, self):
    # The expression is not a child, so reuse_if_untouched would skip it
    expression = self(node.expression)
    return node if expression is node.expression else gem.gem.VariableIndex(expression)


def _replace_index(index, self):
    if isinstance(index, numbers.Integral):
        return index
    expression = index.expression if isinstance(index, gem.gem.VariableIndex) else index
    new = self(expression)
    if new == expression:
        return index
    if isinstance(new, gem.gem.Constant) and not new.shape:
        return int(new.value)
    return gem.gem.VariableIndex(new)  # returns an Index unchanged


# replace_by_name?
def replace_variables(expr, replace_map: Mapping[str, gem.gem.Node]):
    mapper = gem.node.Memoizer(_replace_variables)
    mapper.replace_map = replace_map
    return mapper(expr)


@functools.singledispatch
def _replace_variables(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_replace_variables.register(gem.gem.Node)(gem.node.reuse_if_untouched)


@_replace_variables.register
def _(var: gem.gem.Variable, self):
    new_var = self.replace_map.get(var.name, var)
    assert var.shape == new_var.shape
    return new_var
