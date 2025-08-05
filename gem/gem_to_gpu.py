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

def to_triton(assignments, temporaries):

    args = {}
    temps = {}
    declare = {"counter": 0}
    arrays = {"counter": 0} 

    def recurse(expr):
        if expr not in temps:
            return _recurse(expr)
        return temps[expr][0], temps[expr][1][1]

    @singledispatch
    def _recurse(expr):
        """Visit an gem expression to convert it to a cupy function..

        :arg expr: GEM expression
        """
        raise AssertionError("unsupported expression type %s" % type(expr))

    @_recurse.register(gem.Product)
    def _recurse_product(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        if not all([i == index[0] or len(i) == 0 for i in index]):
            breakpoint()
        return "(" + " * ".join(commands) + ")", index[0]
    
    @_recurse.register(gem.IndexSum)
    def _recurse_indexsum(expr):
        raise NotImplementedError("IndexSum")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        prod = "*".join(commands)
        return f"tl.sum({prod}, {sum_idx})"

    @_recurse.register(gem.Sum)
    def _recurse_sum(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        if not all([i == index[0] for i in index]):
            breakpoint()
        return "(" + " + ".join(commands) + ")" , index[0] 
    @_recurse.register(gem.Division)
    def _recurse_div(expr):
        raise NotImplementedError("Div")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.divide({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.FloorDiv)
    def _recurse_floor_div(expr):
        raise NotImplementedError("FloorDiv")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.floor_divide({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Remainder)
    def _recurse_remainder(expr):
        raise NotImplementedError("Remainder")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.remainder({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Power)
    def _recurse_power(expr):
        raise NotImplementedError("Power")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.power{commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.MathFunction)
    def _recurse_fn(expr):
        chld, idx = recurse(expr.children[0])
        name = expr.name
        name = "tl." + name
        return name + "(" + chld + ")", idx

    @_recurse.register(gem.MaxValue)
    def _recurse_max(expr):
        raise NotImplementedError("Max")
        chld, idx = recurse(expr.children[0])
        return f"cp.max({chld})", idx

    @_recurse.register(gem.MinValue)
    def _recurse_min(expr):
        raise NotImplementedError("Min")
        chld, idx = recurse(expr.children[0])
        return f"cp.min({chld})", idx

    @_recurse.register(gem.Comparison)
    def _recurse_compare(expr):
        raise NotImplementedError("Compare")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"({commands[0]} {expr.operator} {commands[1]})", idx

    @_recurse.register(gem.LogicalNot)
    def _recurse_not(expr):
        raise NotImplementedError("Logical Not")
        chld, idx = recurse(expr.children[0])
        return f"cp.logical_not({chld})", idx

    @_recurse.register(gem.LogicalAnd)
    def _recurse_and(expr):
        raise NotImplementedError("Logical And")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.logical_and{commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.LogicalOr)
    def _recurse_or(expr):
        raise NotImplementedError("Logical Or")
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.logical_or{commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Conditional)
    def _recurse_cond(expr):
        raise NotImplementedError("Cond")
        # children are ordered as (condition, then, else)
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index[1:])) == 1
        return f"(commands[1] if commands[0] else commands[2])", index[1]

    @_recurse.register(gem.ListTensor)
    def _recurse_list_tensor(expr):
        raise NotImplementedError("ListTensor")
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

    @_recurse.register(gem.Indexed)
    def _recurse_indexed(expr):
        chld, idx = recurse(expr.children[0])
        dim = 0
        index = []
        for i in expr.multiindex:
            if not isinstance(i, gem.Index):
                index += [(dim,i)]
            dim += 1
        if len(index) > 0:
            prev_index = ""
            counter = 0
            for i in index:
                temps[expr] = (f"{chld}_{i[0]}",(f"_take_slice_({chld}, len({chld}.shape), {i[0]}, {i[1]}, {chld}.shape[{dim - i[0] - 1}]).reshape({chld}.shape[{i[0]}])", expr.index_ordering() + idx))
                prev_index = str(index[0])
                counter += 1
            res_str = f"{chld}_{i[0]}" 
        else:
            res_str = chld
        return res_str, expr.index_ordering() + idx

    @_recurse.register(gem.FlexiblyIndexed)
    def _recurse_findexed(expr):
        # TODO this doesn't encapsulate the detail dim2idx
        chld, idx = recurse(expr.children[0])
        index = None
        for (off, var) in expr.dim2idxs:
            if len(var) == 0:
                index = f"{off}"
            else:
                for (i, stride) in var:
                    if isinstance(i, gem.Index):
                        pass
                    else:
                        raise NotImplementedError("Flexibly indexed Strides")
        if index is not None:
            temps[expr] = (f"{chld}{index}",(f"_take_slice_({chld}, len({chld}.shape), 0, {index}, {chld}.shape[1]).reshape({chld}.shape[0])", expr.index_ordering()))
            res_str = f"{chld}{index}" 
        else:
            res_str = chld
        return res_str, expr.index_ordering()

    @_recurse.register(gem.Variable)
    def _recurse_variable(expr):
        args[expr.name] = expr.shape 
        return expr.name, tuple()

    @_recurse.register(gem.Zero)
    def _recurse_identity(expr):
        raise NotImplementedError("Zero")
        return f"cp.zeros({expr.shape},dtype=cp.float64)", tuple()

    @_recurse.register(gem.Identity)
    def _recurse_identity(expr):
        raise NotImplementedError("Id")
        return f"cp.eye({expr.shape},dtype={expr.dtype})", tuple()

    @_recurse.register(gem.Literal)
    def _recurse_literal(expr):
        if len(expr.array.shape) == 0:
            return str(expr.array.item()), tuple()
        if expr not in arrays.keys():
            val = arrays["counter"]
            arrays[expr] = (f"t{val}", expr.array)
            arrays["counter"] += 1
        return arrays[expr][0], tuple()

    def next_pow2(val):
        return np.power(2, np.ceil(np.log2(val)))

    def func_decl(*args):
        return ["@triton.jit", f"def triton_kernel({", ".join(args)}):"]

    counter = 0
    temp_assigns = []
    strs = []
    for t in temporaries:
        temps[t] = (f"inter{counter}", recurse(t))
        counter += 1

    for var, expr in assignments:
        e, e_idx = recurse(expr)
        v, v_idx = recurse(var)
        assert v_idx == e_idx
        strs += [f"\t{v}_res={e}"]
    kernel_args = {"arrays": [], "sizes_pow2":[], "sizes_actual":[], "strides":[], "blocks":[]}
    
    temp_vars = ["\tpid = tl.program_id(axis=0)"]
    for name, size in args.items():
        kernel_args["arrays"] += [(name, None)]
        size = ("BLOCK_SIZE_C",) + size
        stride_acc = 1
        for i, dim in enumerate(reversed(size)):
            kernel_args["strides"] += [(name + f"_stride{i}", stride_acc)]
            if not isinstance(dim, str):
                kernel_args["sizes_pow2"] += [(name + f"_dim{i}", next_pow2(dim))]
                kernel_args["sizes_actual"] += [(name + f"_size{i}", dim)]
                stride_acc *= dim
                temp_vars += [f"\t{name}_offsets{i} = tl.arange(0, {name}_dim{i})"]
            else:
                temp_vars += [f"\t{name}_offsets{i} = pid * {dim} + tl.arange(0, {dim})"]
        # TODO put in case where dimension has blocks over it
        offsets = f"\t{name}_offsets ="
        mask = f"\t{name}_mask = {name}_offsets < "
        block = f"\t{name}"
        for i in range(0, len(size)):
            slicing = ",".join(["None" if j == i else ":" for j in range(len(size))])
            offsets += f" {name}_offsets{i}[{slicing}]*{name}_stride{i} +"
            if 0 < i < len(size):
                mask += f" ({name}_offsets{i}*{name}_stride{i} + {name}_size{i-1})[{slicing}] + "
        ptr = f"\t{name}_ptr = {name}"
        load = f"\t{name} = tl.load({name} + {name}_offsets, mask={name}_mask, other=0)"


        temp_vars += [offsets[:-2], mask[:-2], ptr, load]

    for val in arrays.values():
        if isinstance(val, int):
            continue
        name, array = val
        kernel_args["arrays"] += [(name, array)]
        strides = []
        stride_acc = 1
        for i, dim in enumerate(reversed(array.shape)):
            kernel_args["sizes_pow2"] += [(name + f"_dim{i}", next_pow2(dim))]
            kernel_args["sizes_actual"] += [(name + f"_size{i}", dim)]
            kernel_args["strides"] += [(name + f"_stride{i}", stride_acc)]
            stride_acc *= dim
            temp_vars += [f"\t{name}_offsets{i} = tl.arange(0, {name}_dim{i})"]
        offsets = f"\t{name}_offsets ="
        mask = f"\t{name}_mask = {name}_offsets < "
        for i in range(len(array.shape)):
            slicing = ",".join(["None" if j == i else ":" for j in range(len(array.shape))])
            offsets += f" {name}_offsets{i}[{slicing}]*{name}_stride{i} +"
            if i < len(size):
                mask += f" ({name}_offsets{i}*{name}_stride{i} + {name}_size{i})[{slicing}] + "
        ptr = f"\t{name}_ptr = {name}"
        load = f"\t{name} = tl.load({name} + {name}_offsets, mask={name}_mask, other=0)"

        temp_vars += [offsets[:-2], mask[:-2], ptr, load]
            

    for key, val in temps.items():
        if key != "counter":
            temp_vars += [f"\t{val[0]} = {val[1][0]}"]
    #strs += [f"tl.store({output} + offsets, {v}, mask=mask"]

    temp_vars += [f"\tbreakpoint()"]

    for key, val in declare.items():
        if key != "counter" and val[0][0] == "t":
            temp_vars += [f"\t{val[0]} = {repr(val[1])}"]
        elif key != "counter":
            temp_vars += [f"\t{val[0]} = {repr(val[1])[1:-1]}"]

    const_exprs =kernel_args["sizes_pow2"] + kernel_args["sizes_actual"] + kernel_args["strides"] + [("BLOCK_SIZE_C", None)] 
    const_exprs = [exp[0] + ": tl.constexpr" for exp in const_exprs]
    array_exprs = [exp[0] for exp in kernel_args["arrays"]]
    arg_list = array_exprs + const_exprs 
    # this ordering probably needs work
    if "A" in arg_list:
        a_idx = arg_list.index("A")
        a_idx2 = kernel_args["arrays"].index(("A", None))
        a = arg_list.pop(a_idx)
        a2 = kernel_args["arrays"].pop(a_idx2)
        arg_list = [a] + arg_list
        kernel_args["arrays"] = [a2] + kernel_args["arrays"]
        strs += [f"\ttl.store(A_ptr + A_offsets, A_res, mask = A_mask)"] 
    res = "\n".join(func_decl(*arg_list) + temp_vars + strs)
    return res, kernel_args
