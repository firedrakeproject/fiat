import numbers
from gem import gem
import itertools
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


def mlir_contraction(expr, index):
    idx_dict = {idx: chr(65 + i) for i, idx in enumerate(set(sum(index + [expr.free_indices], tuple())))}
    all_indices = idx_dict.values()
    per_index = []
    for idx in index:
        per_index.append([idx_dict[sub] for sub in idx])
    out = [idx_dict[free] for free in expr.free_indices]

    nice = lambda ix: ", ".join(ix)

    per_index_str = ", ".join(
        f"affine_map<({nice(all_indices)}) -> ({nice(X)})>" for X in per_index
    )

    return f"""linalg.contract indexing_maps = [{per_index_str}, affine_map<({nice(all_indices)}) -> ({nice(out)})>]"""


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
        if len(index[1]) != 0:
            raise NotImplementedError("Power: exponent must be scalar")
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
        chld, child_insns, idx = recurse(expr.children[0])

        insns = list(child_insns)
        offsets = []
        sizes = []
        strides = []
        breakpoint()
        for (off, var) in expr.dim2idxs:
            offset_expr = off
            for (i, stride) in var:
                offset_expr += i*stride
            offset_ssa = new_var()
            offsets.append(offset_expr)
            insns.append(offset_insn)

        ssa = new_var()
        return ssa, tuple(insns), expr.index_ordering()

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
    temp_vars += [f"\tprint(is7)"] 
    strs += [f"\tprint(A)"] 
    arg_list = list(args.keys())
    # this ordering probably needs work
    if "A" in arg_list:
        a_idx = arg_list.index("A")
        a = arg_list.pop(a_idx)
        arg_list = [a] + arg_list
    res = "\n".join(func_decl(*arg_list) + temp_vars + strs)
    return res, arg_list

def to_triton(func_name, assignments, temporaries, blocks, temps=None, prev_block_indices=None):

    args = {}
    if temps is None:
        temps = {}
    arrays = {"counter": 0} 
    used_temps = {}
    if prev_block_indices is None:
        prev_block_indices = {}
    subbed_indices = {}

    mycounter = itertools.count()
    new_var = lambda: f"%myvar{next(mycounter)}"

    def recurse(expr, outer_idx=tuple()):
        if expr not in temps:
            return _recurse(expr, outer_idx)
        if expr not in used_temps:
            used_temps[expr] = temps[expr]
        # shape = temps[expr][1][1]
        #removed_indices = [prev_block_indices[s] for s in prev_block_indices if prev_block_indices[s] not in outer_idx["vars"] + outer_idx["temps"]]
        #shape = [s if s not in removed_indices else list(prev_block_indices.keys())[list(prev_block_indices.values()).index(s)] for s in shape]
        #shape = [s for s in shape if s in outer_idx["vars"] + outer_idx["temps"] or not isinstance(s, str)] 
        # print(temps[expr][0], shape, temps[expr][1][1])
        # return temps[expr][0], tuple(shape)
        return temps[expr]

    @singledispatch
    def _recurse(expr, outer_idx):
        """Visit an gem expression to convert it to a cupy function..

        :arg expr: GEM expression
        """
        raise AssertionError("unsupported expression type %s" % type(expr))

    @_recurse.register(gem.Product)
    def _recurse_product(expr, outer_idx):
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        index_without_empty = list(set(index) - set([tuple()]))
        if len(index_without_empty) != 1:
            # case where the shapes are not exactly the same
            outer = []
            inner = []
            for i, (command, idx) in enumerate(summands):
                # need to ensure that added block index remains the first index
                #block_indices =  [v for vs in outer_idx.values() for v in vs]
                block_indices =  outer_idx["vars"]
                #if any(outer in idx for outer in block_indices):
                slicing = "["
                for axis in idx:
                    if axis.name == "cell":
                        if i > 0:
                            # if cell index is on the right child, create extra axis to enable product
                            slicing +=  ":, None, "
                        if axis not in outer:
                            outer += [axis]
                        idx = list(idx)
                        idx.remove(axis)
                        idx = tuple(idx)
                    elif axis in outer_idx["temps"]:
                        if axis not in inner:
                            inner += [axis]
                        idx = list(idx)
                        idx.remove(axis)
                        idx = tuple(idx)
                if len(slicing) > 1:
                    command += slicing[:-2] + "]"
                summands[i] = (command, idx)
            commands = [s[0] for s in summands]
            index = [s[1] for s in summands]
            if "inter4" in commands:
                breakpoint()
            if any([i in subbed_indices.keys() or i in subbed_indices.values() for i in idx for idx in index]):
                breakpoint()
            # integer reps of the indices
            new_shape0 = list([ord(i[-1]) if isinstance(i, str) else i.count for i in index[0]])
            new_shape1 = list([ord(i[-1]) if isinstance(i, str) else i.count for i in index[1]])
            if not all([i0 == i1 or i0 == tuple() or i1 == tuple() for i0, i1 in zip(index[0][::-1], index[1][::-1])]):
                # Empty axis need to be added to allow broadcasting
                idx0 = len(index[0]) - 1
                idx1 = len(index[1]) - 1
                slices0 = "]"
                slices1 = "]"

                while 0 <= idx0 or 0 <= idx1:
                    if index[0][idx0] == index[1][idx1]:
                        idx0 -= 1
                        idx1 -= 1
                        slices0 = ",:" + slices0
                        slices1 = ",:" + slices1
                    elif len(index[0]) > len(index[1]):
                        new_shape1 = new_shape1[:idx1+1] + [1] + new_shape1[idx1+1:]
                        slices1 = ",None" + slices1
                        idx0 -= 1
                    elif len(index[1]) > len(index[0]):
                        new_shape0 = new_shape1[:idx0+1] + [1] + new_shape1[idx0+1:]
                        slices0 = ",None" + slices0
                        idx1 -= 1
                    else:
                        raise NotImplementedError("Equal length broadcasting")
                for i, slices in enumerate([slices0, slices1]):
                    if "None" in slices:
                        if commands[i][-1] == "]":
                            commands[i] = commands[i][:-7] + slices
                        else:
                            commands[i] = commands[i] + "[" + slices[2:]
                        
            result_shape = np.broadcast_shapes(tuple(new_shape0), tuple(new_shape1))
            all_indices = set([i for s in summands for i in s[1]])
            result = tuple([i for i_count in result_shape for i in all_indices if i.count == i_count])
            index = tuple(outer + inner) + result 
        else:
            index = index_without_empty[0]
        
        return "(" + " * ".join(commands) + ")", tuple(index)
    
    @_recurse.register(gem.IndexSum)
    def _recurse_indexsum(expr, outer_idx):
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        prod = "*".join(commands)
        if len(index) > 1:
            raise NotImplementedError("IndexSum with multiple children")
        sum_index = list(expr.multiindex)
        #sum_index = list(set(index[0]) - set(expr.free_indices + tuple([idx for val in outer_idx.values() for idx in val])))
        if len(sum_index) > 1:
            raise NotImplementedError("Index Summing over multiple indices at once")
        if sum_index[0] in subbed_indices:
            sum_index[0] = subbed_indices[sum_index[0]]
        index_list = list(index[0])
        sum_axis = index_list.index(sum_index[0])
        index_list.pop(sum_axis)
        breakpoint()
        return f"tl.sum({prod}, {sum_axis})", tuple(index_list)

    @_recurse.register(gem.Sum)
    def _recurse_sum(expr, outer_idx):
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        breakpoint()
        if len(commands) != 2:
            breakpoint()
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        breakpoint()
        ssa = new_var()
        return ssa, [f"{ssa} = arith.add {lhs}, {rhs}: {mytype}"], index[0] 

    @_recurse.register(gem.Division)
    def _recurse_div(expr, outer_idx):
        raise NotImplementedError("Div")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        breakpoint()
        return f"cp.divide({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.FloorDiv)
    def _recurse_floor_div(expr, outer_idx):
        raise NotImplementedError("FloorDiv")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        breakpoint()
        return f"cp.floor_divide({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Remainder)
    def _recurse_remainder(expr, outer_idx):
        raise NotImplementedError("Remainder")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        breakpoint()
        return f"cp.remainder({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Power)
    def _recurse_power(expr, outer_idx):
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        if len(index[1]) != 0:
            raise NotImplementedError("Power: exponent must be scalar")
        breakpoint()
        return f"libdevice.pow({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.MathFunction)
    def _recurse_fn(expr, outer_idx):
        chld, idx = recurse(expr.children[0], outer_idx)
        name = expr.name
        name = "tl." + name
        breakpoint()
        return name + "(" + chld + ")", idx

    @_recurse.register(gem.MaxValue)
    def _recurse_max(expr, outer_idx):
        raise NotImplementedError("Max")
        chld, idx = recurse(expr.children[0], outer_idx)
        breakpoint()
        return f"cp.max({chld})", idx

    @_recurse.register(gem.MinValue)
    def _recurse_min(expr, outer_idx):
        raise NotImplementedError("Min")
        chld, idx = recurse(expr.children[0], outer_idx)
        breakpoint()
        return f"cp.min({chld})", idx

    @_recurse.register(gem.Comparison)
    def _recurse_compare(expr, outer_idx):
        raise NotImplementedError("Compare")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        breakpoint()
        return f"({commands[0]} {expr.operator} {commands[1]})", idx

    @_recurse.register(gem.LogicalNot)
    def _recurse_not(expr, outer_idx):
        raise NotImplementedError("Logical Not")
        chld, idx = recurse(expr.children[0])
        breakpoint()
        return f"cp.logical_not({chld})", idx

    @_recurse.register(gem.LogicalAnd)
    def _recurse_and(expr, outer_idx):
        raise NotImplementedError("Logical And")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.logical_and{commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.LogicalOr)
    def _recurse_or(expr, outer_idx):
        raise NotImplementedError("Logical Or")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.logical_or{commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Conditional)
    def _recurse_cond(expr, outer_idx):
        raise NotImplementedError("Cond")
        # children are ordered as (condition, then, else)
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index[1:])) == 1
        return f"(commands[1] if commands[0] else commands[2])", index[1]

    @_recurse.register(gem.ListTensor)
    def _recurse_list_tensor(expr, outer_idx):
        raise NotImplementedError("ListTensor")
        str_array = repr(np.empty_like(expr.array)).replace('None', "{}")
        str_array = "cp." + str_array.replace("object", "cp.float64")
        chld_list = []
        idx_list = []
        for chld_expr in expr.array.flatten():
            chld, idx = recurse(chld_expr, outer_idx)
            chld_list += [chld]
            idx_list += [idx]
        assert len(set(idx_list)) == 1
        breakpoint()
        return str_array.format(*chld_list), idx_list[0]

    @_recurse.register(gem.Indexed)
    def _recurse_indexed(expr, outer_idx):
        # chld, idx = recurse(expr.children[0], outer_idx)
        recursed = recurse(expr.children[0], outer_idx)
        dim = 0
        index = []
        slice_dims = []
        for i in expr.multiindex:
            if not isinstance(i, gem.Index):
                index += [(dim, i)]
            else:
                slice_dims += [dim]
            dim += 1
        if len(index) > 0:
            raise NotImplementedError
            prev_index = ""
            counter = 0
            if len(slice_dims) > 1:
                raise NotImplementedError("Slicing accross multiple dims not implmented")
            for i in index:
                temps[expr] = (f"{chld}_{i[0]}",(f"_take_slice_({chld}, len({chld}.shape), {slice_dims[0]}, {i[1]}, {chld}.shape[{dim - i[0] - 1}]).reshape({chld}.shape[{dim - i[0] - 1}])", expr.index_ordering() + idx))
                prev_index = str(index[0])
                counter += 1
            res_str = f"{chld}_{i[0]}" 
        else:
            return recursed[0], recursed[1], recursed[2] + expr.index_ordering()
            res_str = chld
        if any([i in outer_idx['temps'] for i in idx]):
            replace_num = sum([i in outer_idx['temps'] for i in idx])
            #index = idx + expr.index_ordering()[sum([i in outer_idx['temps'] for i in idx]):]
            index = idx[replace_num:] + expr.index_ordering()
            for i in range(replace_num):
                subbed_indices[expr.index_ordering()[i]] = idx[i] 
            return res_str, index 
        return res_str, idx + expr.index_ordering()

    @_recurse.register(gem.FlexiblyIndexed)
    def _recurse_findexed(expr, outer_idx):
        # TODO this doesn't encapsulate the detail dim2idx
        if len(outer_idx) < 1:
            breakpoint()
        chld, insns, idx = recurse(expr.children[0], outer_idx)
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
            raise NotImplementedError
            temps[expr] = (f"{chld}{index}",(f"_take_slice_({chld}, len({chld}.shape), 0, {index}, {chld}.shape[1]).reshape({chld}.shape[0])", idx + expr.index_ordering() ))
            res_str = f"{chld}{index}" 
        else:
            return chld, insns, idx+expr.index_ordering()
            res_str = chld

        breakpoint()
        return res_str, idx + expr.index_ordering()

    @_recurse.register(gem.Variable)
    def _recurse_variable(expr, outer_idx):
        args[expr.name] = expr.shape 
        return f"%{expr.name}", (), ()

    @_recurse.register(gem.Zero)
    def _recurse_identity(expr, outer_idx):
        raise NotImplementedError("Zero")
        return f"cp.zeros({expr.shape},dtype=cp.float64)", tuple()

    @_recurse.register(gem.Identity)
    def _recurse_identity(expr, outer_idx):
        raise NotImplementedError("Id")
        return f"cp.eye({expr.shape},dtype={expr.dtype})", tuple()

    @_recurse.register(gem.Literal)
    def _recurse_literal(expr, outer_idx):
        if expr.array.dtype != np.float64:
            raise NotImplementedError

        ssa = new_var()
        if len(expr.array.shape) == 0:
            return ssa, [f"{ssa} = arith.constant {expr.array.item()} : f64"], ()

        if expr not in arrays.keys():
            val = arrays["counter"]
            arrays[expr] = (f"t{val}", expr.array)
            arrays["counter"] += 1
        # outer_idx["temps"]
        tensor_insn = f"tensor.from_elements {', '.join(map(str, expr.array.flatten()))} : {tensor_shape(expr.shape, expr.dtype)}"
        return ssa, [f"{ssa} = {tensor_insn}"], () 



    block_dims = {"vars":[],
                  "temps":[]} 
    #block_dims = {"vars":[block[0] for block in blocks["vars"]],
    #             "temps":[block[0] for block in blocks["temps"]]} 
    counter = 0
    temp_assigns = []
    stores = []
    for t in temporaries:
        # temps[t] = (f"inter{counter}", recurse(t, block_dims))
        temps[t] = recurse(t, block_dims)
        counter += 1

    # block_dims_vars = {"vars": block_dims["vars"], "temps":[]}
    for var, expr in assignments:
        e, e_idx = recurse(expr, block_dims)
        v, v_idx = recurse(var, block_dims)
        #assert v_idx == e_idx
        stores += [f"\t{v}_res={e}"]
    breakpoint()
    kernel_data = {"entrypoint": func_name, "arrays": [], "sizes_pow2":[], "sizes_actual":[], "strides":[], "insns": [], "pids" : [], "extra_blocks":[]}
    kernel_data["blocks"] = blocks 
     
    for i, block in enumerate(block_dims['vars'] + block_dims['temps']): #
        kernel_data["pids"] += [f"\tpid_{block[-1]} = tl.program_id(axis={i})"]

    
    for name, size in args.items():
        size = tuple([block for block in block_dims["vars"]]) + size
        res, kernel_data = construct_array_arguments(name, None, size, kernel_data)
        kernel_data["insns"] += res

    for val in arrays.values():
        if isinstance(val, int):
            continue
        name, array = val
        replaced_dims = len(block_dims["temps"])
        size = tuple([block for block in block_dims["temps"]]) + array.shape[replaced_dims:]
        res, kernel_data = construct_array_arguments(name, array, size, kernel_data)
        kernel_data["insns"] += res        

    temp_vars = []
    if len(temporaries) != 0:
        for key, val in temps.items():
            if key != "counter":
                kernel_data["insns"] += [f"\t{val[0]} = {val[1][0]}"]
    else:
        for expr in used_temps:
            name, (_, shape) = used_temps[expr]
            shape = [s if isinstance(s, str) else s.extent for s in list(shape)]
            replaced_dims = sum([isinstance(s, str) and s not in block_dims["vars"] for s in shape]) 
            if len(expr.shape) > 0:
                shape = shape +  list(expr.shape[replaced_dims:])
            for i, s in enumerate(shape):
                if isinstance(s, str) and s not in block_dims["vars"] + block_dims["temps"]:
                   shape[i] = f"size_{s[-1]}"
                   if (s, None, None) not in kernel_data["extra_blocks"]:
                       extra_pid = [f"\tpid_{s[-1]} = 0"]
                       kernel_data["pids"] += extra_pid
                       kernel_data["extra_blocks"] += [(s, None, None)]
        
            res, kernel_data = construct_array_arguments(name, None, tuple(shape), kernel_data)
            kernel_data["insns"] = res + kernel_data["insns"]

    kernel_data["insns"] +=  stores
    res, kernel_data = kernel_data_to_str(kernel_data)
    return res, kernel_data, temps, subbed_indices

def next_pow2(val):
    return int(np.power(2, np.ceil(np.log2(val))))

def func_decl(func_name, *args, jit=True):
    res = []
    if jit:
        res += ["@triton.jit"]
    return res + [f"def {func_name}({", ".join(args)}):"]

def construct_array_arguments(name, value, shape, kernel_data, load=True):
    kernel_data["arrays"] += [(name, value)]
    blocks = kernel_data["blocks"]["vars"] + kernel_data["blocks"]["temps"]
    res = []
    stride_acc = []
    for dim, i in reversed(list(zip(shape, range(len(shape))))):
        if all([isinstance(s, int) for s in stride_acc]):
            stride = int(np.prod(stride_acc))
        else:
            stride = "*".join([str(s) for s in stride_acc])
        kernel_data["strides"] += [(name + f"_stride{i}", stride)]
        if not isinstance(dim, str):
            kernel_data["sizes_pow2"] += [(name + f"_dim{i}", next_pow2(dim))]
            kernel_data["sizes_actual"] += [(name + f"_size{i}", int(dim))]
            stride_acc += [dim]
            res += [f"\t{name}_offsets{i} = tl.arange(0, {name}_dim{i})"]
        else:
            res += [f"\t{name}_offsets{i} = (pid_{dim[-1]} * {dim} + tl.arange(0, {name}_dim{i})) "]
            kernel_data["sizes_actual"] += [(name + f"_size{i}", f"{dim} * (pid_{dim[-1]}+1) if {dim} * (pid_{dim[-1]}+1) < size_{dim[-1]} else size_{dim[-1]}")]
            if dim in [block[0] for block in blocks]:
                kernel_data["sizes_pow2"] += [(name + f"_dim{i}", f"{dim}")]
            else:
                kernel_data["sizes_pow2"] += [(name + f"_dim{i}", f"dim_{dim[-1]}")]
            stride_acc += [f"size_{dim[-1]}"]
    offsets, mask = construct_offsets(name, len(shape))
    ptr = f"\t{name}_ptr = {name}"
    res += [offsets, mask, ptr]
    if load:
        res += [f"\t{name} = tl.load({name} + {name}_offsets, mask={name}_mask, other=0)"]

    return res, kernel_data

def kernel_data_to_str(kernel_data):
    block_dims = kernel_data["blocks"]
    extra_blocks = kernel_data["extra_blocks"]
    const_exprs =  kernel_data["sizes_pow2"] + kernel_data["sizes_actual"] + kernel_data["strides"]

    const_exprs = [f"\t{exp[0]}:tl.constexpr = {exp[1]}" for exp in const_exprs]
    array_exprs = [exp[0] for exp in kernel_data["arrays"]]
    pow2_extra = [f"dim_{block[0][-1]}:tl.constexpr" for block in extra_blocks]
    block_dim_args = [f"size_{block[0][-1]}:tl.constexpr" for block in block_dims['vars'] + block_dims['temps'] + extra_blocks] 
    arg_list = array_exprs + block_dim_args + pow2_extra + [f"{block}:tl.constexpr" for block, _, _ in block_dims['vars'] + block_dims['temps']] 

    # this ordering probably needs work
    if "A" in arg_list:
        a_idx = arg_list.index("A")
        a_idx2 = kernel_data["arrays"].index(("A", None))
        a = arg_list.pop(a_idx)
        a2 = kernel_data["arrays"].pop(a_idx2)
        arg_list = [a] + arg_list
        kernel_data["arrays"] = [a2] + kernel_data["arrays"]
        kernel_data["insns"] += [f"\ttl.atomic_add(A_ptr + A_offsets, A_res, mask = A_mask)"] 
    res = "\n".join(func_decl(kernel_data["entrypoint"], *arg_list) + kernel_data["pids"] + const_exprs + kernel_data["insns"])
    return res + "\n", kernel_data

def construct_offsets(name, num_dims):
    # TODO put in case where dimension has blocks over it
    offsets = f"\t{name}_offsets ="
    mask = f"\t{name}_mask = {name}_offsets < "
    block = f"\t{name}"
    for i in range(0, num_dims):
        slicing = ",".join([":" if j == i else "None" for j in range(num_dims)])
        offsets += f" {name}_offsets{i}[{slicing}]*{name}_stride{i} +"
        if i < num_dims - 1:
            mask += f" ({name}_offsets{i}*{name}_stride{i})[{slicing}] + "
        else:
            mask += f"{name}_size{i}"
    return offsets[:-2], mask


def to_triton_wrapper(assignments, temporaries):
    from firedrake.device import compute_device
    vars_blocks = {"vars": compute_device.blocks["vars"], "temps":[]}
    str1, data1, temps1, subs1 = to_triton("subkernel1", [], temporaries, compute_device.blocks)
    str2, data2, temps2, subs2 = to_triton("subkernel2", assignments, [], vars_blocks, temps1, subs1)
    args = [exp[0] for exp in data1["arrays"]]
    args1 = args.copy()
    array_decl = []
    for array in data2['arrays']:
        full_arr_sizes = [(size, int(name[-1])) if not isinstance(size, str) else (f"size_{size[-1]}", int(name[-1])) for name, size in data2["sizes_actual"] if name[:-6] == array[0]]
        arr_sizes = [(size, int(name[-1])) if not isinstance(size,str) else (f"BLOCK_SIZE_{size[-1]}", int(name[-1])) for name, size in data2["sizes_actual"] if name[:-6] == array[0]]
        full_arr_sizes.sort(key=lambda a: a[1])
        arr_sizes.sort(key=lambda a: a[1])
        #block sizes should be replaced by overall dims
        if array[0] != "A":
            offsets, data1 = construct_array_arguments(f"{array[0]}", None, tuple([a[0] for a in arr_sizes]), data1, load=False)
            args1 += [f"{array[0]}"]
            data1["insns"] = offsets + data1["insns"]
            data1["insns"] += [f"\ttl.store({array[0]}_ptr + {array[0]}_offsets, {array[0]}, mask = {array[0]}_mask)"]
            #data1["insns"] += ["\tbreakpoint()"]
            array_decl += [f"\t{array[0]} = torch.from_numpy(np.zeros(({','.join([str(a[0]) for a in full_arr_sizes])}))).float().to(DEVICE)"]
    new_str1, data1 = kernel_data_to_str(data1)

    res_str = [new_str1, str2]
    grid = ""
    blocks = data1["blocks"]
    grid_sizes = []
    block_size = []
    block_args = []
    output_var = data2["arrays"][0]
    for name, _, _ in blocks['vars'] + blocks['temps']:
        block_size += [f"size_{name[-1]}"]
        block_args += [f"{name}"]
        grid_sizes += [f"triton.cdiv(size_{name[-1]}, meta['{name}'])"]
    wrapper_func = func_decl("triton_kernel", *([output_var[0]] + args + block_size + block_args), jit=False) + array_decl
    wrapper_func += [f"\tgrid = lambda meta: ({','.join(grid_sizes)}, )"]
    wrapper_func += [f"\tsubkernel1[grid]({','.join(args1 + block_size + block_args)})"]
    wrapper_func += [f"\ttorch.cuda.current_stream().synchronize()"]
    wrapper_func += [f"\tprint(inter6)"]
    wrapper_func += [f"\tprint(inter4)"]
    #wrapper_func += [f"\tbreakpoint()"]
    grid_sizes = []
    block_args = []
    for name, _, _ in blocks['vars']:
        block_args += [f"{name}"]
        grid_sizes += [f"triton.cdiv(size_{name[-1]}, meta['{name}'])"]
    for name, _, _ in data2["extra_blocks"]:
        wrapper_func += [f"\tdim_{name[-1]} = int(np.power(2, np.ceil(np.log2(size_{name[-1]}))))"]
        block_size += [f"dim_{name[-1]}"]
    wrapper_func += [f"\tgrid = lambda meta: ({','.join(grid_sizes)}, )"]
    wrapper_func += [f"\tsubkernel2[grid]({','.join([a[0] for a in data2['arrays']] + block_size + block_args)})"]
    wrapper_func += [f"\tprint(A)"]
    wrapper_func += [f"\tbreakpoint()"]
    res_data = {"arrays": [output_var] + [a for a in data1["arrays"] if a[0] in args], "sizes_pow2":[], "sizes_actual":[]}
    res = "\n".join([new_str1, str2] + wrapper_func) 
    breakpoint()
    return res, res_data
    
    
def to_mlir(assignments):
    args = {}
    declare = {"counter": 0}

    mycounter = itertools.count()
    # map varname to MLIR type
    type_registry = {}

    def new_name():
        name = f"%myvar{next(mycounter)}"
        # if name == "%myvar10":
        #     breakpoint()
        return name

    def new_var(dtype):
        name = new_name()
        type_registry[name] = MLIR_DTYPES[dtype]
        return name


    def new_tensor(shape, dtype):
        name = new_name()
        type_registry[name] = tensor_shape(shape, dtype)
        return name


    def register_constant(value, dtype):
        if isinstance(value, gem.Index):
            type_registry["%{value.name}"] = "i32"
            return f"%{value.name}", ()
        elif isinstance(value, numbers.Number):
            ssa = new_var(dtype)
            return ssa, (f"{ssa} = arith.constant {value} : {MLIR_DTYPES[dtype]}",)
        else:
            return value, ()

    def mlir_mul(vars, dtype):
        if dtype == np.int32:
            arith_insn = "arith.muli"
        else:
            assert dtype == np.float64
            arith_insn = "arith.mulf"

        varname, varinsns = register_constant(vars[0], dtype)

        if len(vars) == 1:
            return varname, varinsns
        else:
            ssa = new_var(dtype)
            subvar, subinsns = mlir_mul(vars[1:], dtype)
            return ssa, subinsns + varinsns + (f"{ssa} = {arith_insn} {varname}, {subvar} : {MLIR_DTYPES[dtype]}",)

    def mlir_add(vars, dtype):
        if dtype == np.int32:
            arith_insn = "arith.addi"
        else:
            assert dtype == np.float64
            arith_insn = "arith.addf"

        varname, varinsns = register_constant(vars[0], dtype)

        if len(vars) == 1:
            return varname, varinsns
        else:
            ssa = new_var(dtype)
            subvar, subinsns = mlir_add(vars[1:], dtype)
            return ssa, subinsns + varinsns + (f"{ssa} = {arith_insn} {varname}, {subvar} : {MLIR_DTYPES[dtype]}",)

    @singledispatch
    def recurse(expr):
        """Visit an gem expression to convert it to a cupy function..

        :arg expr: GEM expression
        """
        raise AssertionError("unsupported expression type %s" % type(expr))

    @recurse.register(gem.IndexSum)
    def _(expr):
        assert len(expr.children) == 1
        child_var, child_insns, child_indices = recurse(expr.children[0])

        if len(expr.multiindex) > 1:
            raise NotImplementedError

        contracted_index = expr.multiindex[0]
        contracted_index_loc = child_indices.index(contracted_index)

        shape = tuple(index.extent for i, index in enumerate(child_indices) if i != contracted_index_loc)
        ssa = new_tensor(shape, "f64")
        contract_insn = (
            f"{ssa} = linalg.reduce {{arith.addf}} "
            f"ins({child_var} : {type_registry[child_var]}) "
            f"outs(%todo : {type_registry[ssa]}) "
            f"dimensions = [{contracted_index_loc}]"
        )

        return ssa, child_insns + (contract_insn,), expr.free_indices


    @recurse.register(gem.Product)
    def recurse_indexsum(expr):
        summands = [recurse(e) for e in expr.children]
        commands = [s[0] for s in summands]
        assert len(commands) == 2
        subinsns = []
        for s in summands:
            subinsns.extend(s[1])
        subinsns = tuple(subinsns)
        index = [s[2] for s in summands]
        contract_expr = mlir_contraction(expr, index)

        shape = tuple(index.extent for index in expr.free_indices)
        ssa = new_tensor(shape, "f64")
        dtype = type_registry[ssa]

        contract_insn = (
            f"{contract_expr} "
            f"ins({', '.join(commands)} : {', '.join([type_registry[c] for c in commands])}) "
            f"outs({ssa} : {dtype})"
        )

        return ssa, subinsns+(contract_insn,), expr.free_indices

    @recurse.register(gem.Sum)
    def recurse_sum(expr):
        vars = []
        insns = []
        indices = []
        for e in expr.children:
            var, child_insns, child_indices = recurse(e)
            vars.append(var)
            insns.extend(child_insns)
            indices.append(child_indices)

        assert len(set(indices)) == 1
        ssa, add_insns = mlir_add(vars, np.float64)
        return ssa, tuple(insns) + add_insns, indices[0]

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
        if len(index[1]) != 0:
            raise NotImplementedError("Power: exponent must be scalar")
        return f"cp.power{commands[0]}, {commands[1]})", index[0]

    @recurse.register(gem.MathFunction)
    def _(expr):
        child_var, child_insns, idx = recurse(expr.children[0])
        if expr.name != "abs":
            raise NotImplementedError
        ssa = new_name()
        dtype = type_registry[child_var]
        type_registry[ssa] = dtype
        math_insn = f"{ssa} = linalg.abs {child_var} : {dtype}"
        return ssa, child_insns + (math_insn,), idx

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
        chld, insns, idx = recurse(expr.children[0])
        ssa = new_name()
        offsets = []
        sizes = []
        strides = []
        for i, size in zip(expr.multiindex, expr.children[0].shape, strict=True):
            if isinstance(i, gem.Index):
                offsets.append(0)
                sizes.append(size)
            else:
                offsets.append(i)
                sizes.append(1)
            strides.append(1)

        nice = lambda iterable: f"[{', '.join(map(str, iterable))}]"

        in_shape = type_registry[chld]
        out_shape = tensor_shape(sizes, "f64")  # FIXME: hardcoded type
        type_registry[ssa] = out_shape
        insns = insns + (f"{ssa} = tensor.extract_slice {chld}{nice(offsets)}{nice(sizes)}{nice(strides)} : {in_shape} to {out_shape}",)
        return ssa, insns, expr.index_ordering() + idx

    @recurse.register(gem.FlexiblyIndexed)
    def recurse_findexed(expr):
        # TODO this doesn't encapsulate the detail dim2idx
        chld, child_insns, idx = recurse(expr.children[0])

        insns = list(child_insns)
        offsets = []
        sizes = expr.children[0].shape
        strides = []
        for (off, var) in expr.dim2idxs:
            offset_expr = [off]
            for (i, stride) in var:
                offset_ssa, offset_insns = mlir_mul([i, stride], np.int32)
                offset_expr.append(offset_ssa)
                insns.extend(offset_insns)
            offset_ssa, moreinsns = mlir_add(offset_expr, np.int32)
            offsets.append(offset_ssa)
            insns.extend(moreinsns)

            strides.append("1")

        ssa = new_name()
        dtype = tensor_shape(sizes, "f64")
        type_registry[ssa] = dtype
        newinsn = f"{ssa} = tensor.extract_slice {chld}{offsets}{sizes}{strides} : in to out"
        return ssa, tuple(insns), expr.index_ordering()

    @recurse.register(gem.Variable)
    def recurse_variable(expr):
        # awful hack, dtype needs propagating
        if expr.name in {"coords", "w_0", "A"}:
            dtype = np.float64
        else:
            assert expr.dtype is not None
            dtype = expr.dtype

        name = "%"+expr.name

        args[expr.name] = 1
        type_registry[name] = MLIR_DTYPES[dtype]
        return name, (), tuple()

    @recurse.register(gem.Zero)
    def recurse_identity(expr):
        raise NotImplementedError
        return f"cp.zeros({expr.shape},dtype=cp.float64)", tuple()

    @recurse.register(gem.Identity)
    def recurse_identity(expr):
        raise NotImplementedError
        return f"cp.eye({expr.shape},dtype={expr.dtype})", tuple()

    @recurse.register(gem.Literal)
    def recurse_literal(expr):
        name = f"%t{declare["counter"]}"
        declare["counter"] += 1

        if expr.shape:
            dtype = tensor_shape(expr.shape, expr.dtype)
        else:
            dtype = MLIR_DTYPES[expr.dtype]

        type_registry[name] = dtype
        if expr not in declare.keys():
            declare[expr] = (name, expr.array)
        return name, (), tuple()

    def func_decl(*args):
        return [f"def cupy_kernel({", ".join(args)}):"]

    insns = []

    temp_vars = []
    for key, val in declare.items():
        if key != "counter" and val[0][0] == "t":
            # temp_vars += [f"\t{val[0]} = cp.{repr(val[1])}"]
            insns.append(f"%{declare[expr][0]} : tensor.put {repr(val)}")
        elif key != "counter":
            insns.append(f"%{declare[expr][0]}")
            # temp_vars += [f"\t{val[0]} = {repr(val[1])[1:-1]}"]

    for var, expr in assignments:
        e, e_insns, e_idx = recurse(expr)
        v, v_insns, v_idx = recurse(var)
        assert set(v_idx) == set(e_idx)
        insns.extend(e_insns)
        insns.extend(v_insns)
        insns.append(f"{v} = arith.addf {v}, {e}")

    # arg_list = list(args.keys())
    # # this ordering probably needs work
    # if "A" in arg_list:
    #     a_idx = arg_list.index("A")
    #     a = arg_list.pop(a_idx)
    #     arg_list = [a] + arg_list
    # res = "\n".join(func_decl(*arg_list) + temp_vars + strs)
    return "\n".join(insns)


MLIR_DTYPES = {
    np.int32: "i32",
    np.float64: "f64",
    "i32": "i32",
    "f64": "f64",
    np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32",
}


def tensor_shape(shape, dtype):
    return f"tensor<{'x'.join(map(str, shape))}x{MLIR_DTYPES[dtype]}>"
