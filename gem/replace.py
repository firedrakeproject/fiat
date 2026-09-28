import functools
from collections.abc import Mapping

import gem.gem
import gem.node



def replace(expr, replace_map):
    raise NotImplementedError
    for node in gem.node.traversal([expr]):
        print(node)
    breakpoint()


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
