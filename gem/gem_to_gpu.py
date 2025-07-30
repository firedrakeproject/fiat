from gem import gem
import numpy as np
from functools import singledispatch


def construct_einsum_str(expr, index):
    idx_dict = {idx: chr(65 + i) for i, idx in enumerate(set(sum(index + [expr.free_indices], tuple())))}
    idx_str = ""
    for idx in index:
        for sub in idx:
            idx_str += idx_dict[sub]
        idx_str += ","
    idx_str = idx_str[:-1]
    idx_str += "->"
    idx_str += "".join([idx_dict[free] for free in expr.free_indices])
    return idx_str


def to_cupy(assignments):

    args = {}
    declare = {"counter": 0}

    @singledispatch
    def recurse(expr):
        """Visit an gem expression to convert it to a cupy function..

        :arg expr: GEM expression
        """
        raise AssertionError("unsupported expression type %s" % type(expr))

    @recurse.register(gem.Product)
    @recurse.register(gem.IndexSum)
    def recurse_indexsum(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        idx_str = construct_einsum_str(expr, index)
        # put sub expressions into temporary if they are over an arbitary number of characters (61)
        if any([len(command) > 60 for command in commands]):
            for i in range(len(expr.children)):
                if expr.children[i] not in declare.keys():
                    declare[expr.children[i]] = (f"is{declare["counter"]}", commands[i].replace("\n", ""))
                    declare["counter"] += 1
            operands = ",".join([declare[expr.children[i]][0] for i in range(len(expr.children))])
            return f"cp.einsum(\"{idx_str}\", {operands})", expr.free_indices
        return f"cp.einsum(\"{idx_str}\", {",".join(commands)})", expr.free_indices

    @recurse.register(gem.Sum)
    def recurse_sum(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.add({commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.Division)
    def recurse_div(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.divide({commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.FloorDiv)
    def recurse_floor_div(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.floor_divide({commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.Remainder)
    def recurse_remainder(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.remainder({commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.Power)
    def recurse_power(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.power{commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.MathFunction)
    def recurse_fn(expr):
        chld, idx = recurse(expr.children[0])
        name = expr.name
        if name != "abs":
            name = "cp." + name
        return name + "(" + chld + ")", idx

    @recurse.register(gem.MaxValue)
    def recurse_max(expr):
        chld, idx = recurse(expr.children[0])
        return f"cp.max({chld})", idx

    @recurse.register(gem.MinValue)
    def recurse_min(expr):
        chld, idx = recurse(expr.children[0])
        return f"cp.min({chld})", idx

    @recurse.register(gem.Comparison)
    def recurse_compare(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"({commands[0]} {expr.operator} {commands[1]})", idx

    @recurse.register(gem.LogicalNot)
    def recurse_not(expr):
        chld, idx = recurse(expr.children[0])
        return f"cp.logical_not({chld})", idx

    @recurse.register(gem.LogicalAnd)
    def recurse_and(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.logical_and{commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.LogicalOr)
    def recurse_or(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.logical_or{commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.Conditional)
    def recurse_cond(expr):
        # children are ordered as (condition, then, else)
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index[1:])) == 1
        return f"(commands[1] if commands[0] else commands[2])", index[1]

    @recurse.register(gem.ListTensor)
    def recurse_list_tensor(expr):
        str_array = repr(np.empty_like(expr.array)).replace('None', "{}")
        str_array = "cp." + str_array.replace("object", "cp.float64")
        chld_list = []
        idx_list = []
        for chld_expr in expr.array.flatten():
            chld, idx = recurse(chld_expr)
            chld_list += [chld]
            idx_list += [idx]
        assert len(set(idx_list)) == 1
        return str_array.format(*chld_list), idx_list[0]

    @recurse.register(gem.Indexed)
    def recurse_indexed(expr):
        chld, idx = recurse(expr.children[0])
        chld += "["
        for i in expr.multiindex:
            if isinstance(i, gem.Index):
                chld += ":,"
            else:
                chld += f"{i},"
        chld = chld[:-1] + "]"
        return chld, expr.index_ordering() + idx

    @recurse.register(gem.FlexiblyIndexed)
    def recurse_findexed(expr):
        # TODO this doesn't encapsulate the detail dim2idx
        chld, idx = recurse(expr.children[0])
        chld += "["
        for (off, var) in expr.dim2idxs:
            if len(var) == 0:
                chld += f"{off},"
            else:
                for (i, stride) in var:
                    if isinstance(i, gem.Index):
                        # chld += f"{off},:,"
                        chld += ":,"
                    else:
                        breakpoint()
        chld = chld[:-1] + "]"
        return chld, expr.index_ordering()

    @recurse.register(gem.Variable)
    def recurse_variable(expr):
        args[expr.name] = 1
        return expr.name, tuple()

    @recurse.register(gem.Zero)
    def recurse_identity(expr):
        return f"cp.zeros({expr.shape},dtype=cp.float64)", tuple()

    @recurse.register(gem.Identity)
    def recurse_identity(expr):
        return f"cp.eye({expr.shape},dtype={expr.dtype})", tuple()

    @recurse.register(gem.Literal)
    def recurse_literal(expr):
        if expr not in declare.keys():
            declare[expr] = (f"t{declare["counter"]}", expr.array)
            declare["counter"] += 1
        return declare[expr][0], tuple()

    def func_decl(*args):
        return [f"def cupy_kernel({", ".join(args)}):"]

    strs = []
    for var, expr in assignments:
        e, e_idx = recurse(expr)
        v, v_idx = recurse(var)
        assert v_idx == e_idx
        strs += [f"\t{v}+=cp.array({e})"]

    temp_vars = []
    for key, val in declare.items():
        if key != "counter" and val[0][0] == "t":
            temp_vars += [f"\t{val[0]} = cp.{repr(val[1])}"]
        elif key != "counter":
            temp_vars += [f"\t{val[0]} = {repr(val[1])[1:-1]}"]
    
    arg_list = list(args.keys())
    # this ordering probably needs work
    if "A" in arg_list:
        a_idx = arg_list.index("A")
        a = arg_list.pop(a_idx)
        arg_list = [a] + arg_list
    res = "\n".join(func_decl(*arg_list) + temp_vars + strs)

    return res, arg_list
