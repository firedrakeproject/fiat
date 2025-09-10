from gem import gem
import numpy as np
from functools import singledispatch


def to_triton(func_name, assignments, temporaries, blocks, temps=None, prev_block_indices=None, block_names=None):
    "This function takes in a series of GEM expressions containing named indices, and constructs the appropriate triton code"

    args = {}
    if temps is None:
        temps = {}
    arrays = {"counter": 0} 
    used_temps = {}
    if prev_block_indices is None:
        prev_block_indices = {}
    subbed_indices = {}

    def recurse(expr, outer_idx=tuple()):
        if expr not in temps:
            return _recurse(expr, outer_idx)
        if expr not in used_temps:
            used_temps[expr] = temps[expr]
        shape = temps[expr][1][1]
        return temps[expr][0], tuple(shape)

    @singledispatch
    def _recurse(expr, outer_idx):
        """Visit an gem expression to convert it to a triton function..

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
                        slices1 = "a,None" + slices1
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
        return f"tl.sum({prod}, {sum_axis})", tuple(index_list)

    @_recurse.register(gem.Sum)
    def _recurse_sum(expr, outer_idx):
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return "(" + " + ".join(commands) + ")" , index[0] 

    @_recurse.register(gem.Division)
    def _recurse_div(expr, outer_idx):
        raise NotImplementedError("Div")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.divide({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.FloorDiv)
    def _recurse_floor_div(expr, outer_idx):
        raise NotImplementedError("FloorDiv")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.floor_divide({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Remainder)
    def _recurse_remainder(expr, outer_idx):
        raise NotImplementedError("Remainder")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"cp.remainder({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.Power)
    def _recurse_power(expr, outer_idx):
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        if len(index[1]) != 0:
            raise NotImplementedError("Power: exponent must be scalar")
        return f"libdevice.pow({commands[0]}, {commands[1]})", index[0]

    @_recurse.register(gem.MathFunction)
    def _recurse_fn(expr, outer_idx):
        chld, idx = recurse(expr.children[0], outer_idx)
        name = expr.name
        name = "tl." + name
        return name + "(" + chld + ")", idx

    @_recurse.register(gem.MaxValue)
    def _recurse_max(expr, outer_idx):
        raise NotImplementedError("Max")
        chld, idx = recurse(expr.children[0], outer_idx)
        return f"cp.max({chld})", idx

    @_recurse.register(gem.MinValue)
    def _recurse_min(expr, outer_idx):
        raise NotImplementedError("Min")
        chld, idx = recurse(expr.children[0], outer_idx)
        return f"cp.min({chld})", idx

    @_recurse.register(gem.Comparison)
    def _recurse_compare(expr, outer_idx):
        raise NotImplementedError("Compare")
        summands = [recurse(e, outer_idx) for e in expr.children]
        commands = [s[0] for s in summands]
        index = [s[1] for s in summands]
        assert len(set(index)) == 1
        return f"({commands[0]} {expr.operator} {commands[1]})", idx

    @_recurse.register(gem.LogicalNot)
    def _recurse_not(expr, outer_idx):
        raise NotImplementedError("Logical Not")
        chld, idx = recurse(expr.children[0])
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
        # Should be done by creating two temporary variables for each operation 
        # on the list tensor
        # tl.cat might be worth considering here
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
        return str_array.format(*chld_list), idx_list[0]

    @_recurse.register(gem.Indexed)
    def _recurse_indexed(expr, outer_idx):
        chld, idx = recurse(expr.children[0], outer_idx)
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
            prev_index = ""
            counter = 0
            if len(slice_dims) > 1:
                raise NotImplementedError("Indexed: Slicing accross multiple dims not implmented")
            for i in index:
                temps[expr] = (f"{chld}_{i[0]}",(f"_take_slice_({chld}, len({chld}.shape), {slice_dims[0]}, {i[1]}, {chld}.shape[{dim - i[0] - 1}]).reshape({chld}.shape[{dim - i[0] - 1}])", expr.index_ordering() + idx))
                prev_index = str(index[0])
                counter += 1
            res_str = f"{chld}_{i[0]}" 
        else:
            res_str = chld
        return res_str, idx + expr.index_ordering()

    @_recurse.register(gem.FlexiblyIndexed)
    def _recurse_findexed(expr, outer_idx):
        # Indexing the variables along any axis
        # Shape of dim2idxs: (offset, ((index,stride), (...), ...)
        # Currently restricted to the slicing case of only 1 dimension, ie dim2idx = (n, ())
        # Slicing in Triton is difficult - in order to do this it may require recursive joining of subarrays
        chld, idx = recurse(expr.children[0], outer_idx)
        index = None
        for (off, var) in expr.dim2idxs:
            if len(var) == 0:
                index = f"{off}"
                index_dim = len(var)
            elif off != 0:
                # reshape, slice, reshape
                #raise NotImplementedError("Flexibly Indexed: Too many dimensions, limit: 1")
                index = f"{off}"
                index_dim = len(var)
                temps[expr] = (f"{chld}{off}",(f"{str(var)}", idx + expr.index_ordering() ))
                res_str=f"{chld}{off}"
            else:
                for (i, stride) in var:
                    if isinstance(i, gem.Index):
                        pass
                    else:
                        raise NotImplementedError("Flexibly indexed: index is not a gem.Index")
        if index is not None:
            temps[expr] = (f"{chld}{index}",(f"_take_slice_({chld}, len({chld}.shape), {index_dim}, {index}, {chld}.shape[1]).reshape({chld}.shape[0])", idx + expr.index_ordering() ))
            res_str = f"{chld}{index}" 
        else:
            res_str = chld

        return res_str, idx + expr.index_ordering()

    @_recurse.register(gem.Variable)
    def _recurse_variable(expr, outer_idx):
        args[expr.name] = expr.shape 
        breakpoint()
        return expr.name, tuple() 

    @_recurse.register(gem.Zero)
    def _recurse_zero(expr, outer_idx):
        return f"tl.zeros({expr.shape}, dtype=tl.float64)", tuple()

    @_recurse.register(gem.Identity)
    def _recurse_identity(expr, outer_idx):
        # Triton doesn't have a built in identity function
        # One could be written using something along the lines of:
        # y = tl.arange(0, num_elements)
        # rows = tl.arange(0, num_rows)
        # stride = tl.full(stride)
        # y = tl.where(y in rows*stride, 0, 1)
        # then reshape to right shape
        # this is untested so not sure if it works
        # lower to identity literal
        raise NotImplementedError("Identity")

    @_recurse.register(gem.Literal)
    def _recurse_literal(expr, outer_idx):
        #
        if len(expr.array.shape) == 0:
            return str(expr.array.item()), tuple()
        if expr not in arrays.keys():
            val = arrays["counter"]
            arrays[expr] = (f"t{val}", expr.array)
            arrays["counter"] += 1
        return arrays[expr][0], tuple() 



    block_dims = {"vars":[],
                  "temps":[]} 
    counter = 0
    temp_assigns = []
    existing_blocks = []
    stores = []
    for t in temporaries:
        code_str, free_vars = recurse(t, block_dims)
        temps[t] = (f"inter{counter}", (code_str, free_vars))
        for v in free_vars:
            # FRAGILE requires no two blocks that have the same first initial
            if v.name is not None and all([v.name[0] != b[-1].lower() for b in existing_blocks]):
                existing_blocks += [block_names[v.name]]
        counter += 1

    for var, expr in assignments:
        e, e_idx = recurse(expr, block_dims)
        v, v_idx = recurse(var, block_dims)
        #assert v_idx == e_idx
        stores += [f"\t{v}_res={e}"]
    #breakpoint()
    kernel_data = {"entrypoint": func_name, "arrays": [], "sizes_pow2":[], "sizes_actual":[], "strides":[], "insns": [], "pids" : [], "extra_blocks":[]}
    kernel_data["blocks"] = blocks 
     
    #block_dims['vars'] + block_dims['temps']
    for i, block in enumerate(existing_blocks):
        kernel_data["pids"] += [f"\tpid_{block[-1]} = tl.program_id(axis={i})"]

    
    for name, size in args.items():
        size = tuple([block for block in existing_blocks]) + size
        res, kernel_data = construct_array_arguments(name, None, size, kernel_data)
        kernel_data["insns"] += res
    #breakpoint()

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
                kernel_data["insns"] += [f"\tbreakpoint()"]
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
    # Given information about the arrays, construct the string to load the array and add its information to kernel data
    kernel_data["arrays"] += [(name, value)]
    blocks = kernel_data["blocks"]["vars"] + kernel_data["blocks"]["temps"]
    res = []
    stride_acc = []
    if name == "coords":
        breakpoint()
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
    # Converts the information stored in kernel data (arrays, their sizes, instructions) into the correct string
    block_dims = kernel_data["blocks"]
    extra_blocks = kernel_data["extra_blocks"]
    const_exprs =  kernel_data["sizes_pow2"] + kernel_data["sizes_actual"] + kernel_data["strides"]

    const_exprs = [f"\t{exp[0]}:tl.constexpr = {exp[1]}" for exp in const_exprs]
    array_exprs = [exp[0] for exp in kernel_data["arrays"]]
    pow2_extra = [f"dim_{block[0][-1]}:tl.constexpr" for block in extra_blocks]
    block_dim_args = [f"size_{block[0][-1]}:tl.constexpr" for block in block_dims['vars'] + block_dims['temps'] + extra_blocks] 
    arg_list = array_exprs + block_dim_args + pow2_extra + [f"{block}:tl.constexpr" for block, _, _ in block_dims['vars'] + block_dims['temps']] 

    # A is the output array so needes to be treated separately and will be passed first
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
    # Construct string containing definition of offsets and masks for a particular array
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
    # Produces a string containing two kernels, and a wrapper function that constructs their grid and calls them
    # The first kernel will process the instructions designated as temporaries by GEM
    # The second will produce the final result
    # The two launches ensure synchronisation of data between the two kernels. 
    from firedrake.device import compute_device
    vars_blocks = {"vars": compute_device.blocks["vars"], "temps":[]}
    str1, data1, temps1, subs1 = to_triton("subkernel1", [], temporaries, compute_device.blocks, block_names=compute_device.block_names)
    str2, data2, temps2, subs2 = to_triton("subkernel2", assignments, [], vars_blocks, temps1, subs1,block_names=compute_device.block_names)
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
    return res, res_data
    
    
