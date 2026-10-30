"""
An interpreter for GEM trees.
"""
import numpy
import operator
from collections import OrderedDict
from functools import singledispatch
import itertools

from gem import gem, node
from gem.optimise import replace_delta

__all__ = ("evaluate", )


class Result(object):
    """An array object that tracks which axes of the array correspond to
    gem free indices (and what those free indices are).

    :arg arr: The array.
    :arg fids: The free indices.

    The first ``len(fids)`` axes of the provided array correspond to
    the free indices, the remaining axes are the shape of each entry.
    """
    def __init__(self, arr, fids=None):
        self.arr = arr
        self.fids = fids if fids is not None else ()

    def broadcast(self, fids):
        """Given some free indices, return a broadcasted array which
        contains extra dimensions that correspond to indices in fids
        that are not in ``self.fids``.

        Note that inserted dimensions will have length one.

        :arg fids: The free indices for broadcasting.
        """
        # Select free indices
        axes = tuple(self.fids.index(fi) for fi in fids if fi in self.fids)
        assert len(axes) == len(self.fids)
        # Add shape
        axes += tuple(range(len(self.fids), self.arr.ndim))
        # Move axes, insert extra axes
        arr = numpy.transpose(self.arr, axes)
        for i, fi in enumerate(fids):
            if fi not in self.fids:
                arr = numpy.expand_dims(arr, axis=i)
        return arr

    def filter(self, idx, fids):
        """Given an index tuple and some free indices, return a
        "filtered" index tuple which removes entries that correspond
        to indices in fids that are not in ``self.fids``.

        :arg idx: The index tuple to filter.
        :arg fids: The free indices for the index tuple.
        """
        return tuple(idx[fids.index(i)] for i in self.fids) + idx[len(fids):]

    def __getitem__(self, idx):
        return self.arr[tuple(idx)]

    def __setitem__(self, idx, val):
        self.arr[idx] = val

    @property
    def tshape(self):
        """The total shape of the result array."""
        return self.arr.shape

    @property
    def fshape(self):
        """The shape of the free index part of the result array."""
        return self.tshape[:len(self.fids)]

    @property
    def shape(self):
        """The shape of the shape part of the result array."""
        return self.tshape[len(self.fids):]

    def __repr__(self):
        return "Result(%r, %r)" % (self.arr, self.fids)

    def __str__(self):
        return repr(self)

    @classmethod
    def empty(cls, *children, **kwargs):
        """Build an empty Result object.

        :arg children: The children used to determine the shape and
            free indices.
        :kwarg dtype: The data type of the result array.
        """
        dtype = kwargs.get("dtype", float)
        assert all(children[0].shape == c.shape for c in children)
        fids = []
        for f in itertools.chain(*(c.fids for c in children)):
            if f not in fids:
                fids.append(f)
        shape = tuple(i.extent for i in fids) + children[0].shape
        return cls(numpy.empty(shape, dtype=dtype), tuple(fids))


@singledispatch
def _evaluate(expression, self):
    """Evaluate an expression using a provided callback handler.

    :arg expression: The expression to evaluation.
    :arg self: The callback handler (should provide bindings).
    """
    raise ValueError("Unhandled node type %s" % type(expression))


@_evaluate.register(gem.Zero)
def _evaluate_zero(e, self):
    """Zeros produce an array of zeros."""
    return Result(numpy.zeros(e.shape, dtype=float))


@_evaluate.register(gem.Failure)
def _evaluate_failure(e, self):
    """Failure nodes produce NaNs."""
    return Result(numpy.full(e.shape, numpy.nan, dtype=float))


@_evaluate.register(gem.Constant)
def _evaluate_constant(e, self):
    """Constants return their array."""
    return Result(e.array)


@_evaluate.register(gem.Delta)
def _evaluate_delta(e, self):
    """Lower delta and evaluate."""
    e, = replace_delta((e,))
    return self(e)


@_evaluate.register(gem.Variable)
def _evaluate_variable(e, self):
    """Look up variables in the provided bindings."""
    try:
        val = self.bindings[e]
    except KeyError:
        raise ValueError("Binding for %s not found" % e)
    if val.shape != e.shape:
        raise ValueError("Binding for %s has wrong shape.  %s, not %s." %
                         (e, val.shape, e.shape))
    return Result(val)


@_evaluate.register(gem.Power)
@_evaluate.register(gem.Division)
@_evaluate.register(gem.Product)
@_evaluate.register(gem.Sum)
def _evaluate_operator(e, self):
    op = {gem.Product: operator.mul,
          gem.Division: operator.truediv,
          gem.Sum: operator.add,
          gem.Power: operator.pow}[type(e)]

    a, b = [self(o) for o in e.children]
    result = Result.empty(a, b)
    fids = result.fids
    result.arr = op(a.broadcast(fids), b.broadcast(fids))
    return result


@_evaluate.register(gem.MathFunction)
def _evaluate_mathfunction(e, self):
    ops = [self(o) for o in e.children]
    result = Result.empty(*ops)
    names = {
        "abs": abs,
        "log": numpy.log,
        "real": operator.attrgetter("real"),
        "imag": operator.attrgetter("imag"),
        "conj": operator.methodcaller("conjugate"),
    }
    op = names[e.name]
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(*(o[o.filter(idx, result.fids)] for o in ops))
    return result


@_evaluate.register(gem.MaxValue)
@_evaluate.register(gem.MinValue)
def _evaluate_minmaxvalue(e, self):
    ops = [self(o) for o in e.children]
    result = Result.empty(*ops)
    op = {gem.MinValue: min,
          gem.MaxValue: max}[type(e)]
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(*(o[o.filter(idx, result.fids)] for o in ops))
    return result


@_evaluate.register(gem.Comparison)
def _evaluate_comparison(e, self):
    ops = [self(o) for o in e.children]
    op = {">": operator.gt,
          ">=": operator.ge,
          "==": operator.eq,
          "!=": operator.ne,
          "<": operator.lt,
          "<=": operator.le}[e.operator]
    result = Result.empty(*ops, dtype=bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(*(o[o.filter(idx, result.fids)] for o in ops))
    return result


@_evaluate.register(gem.LogicalNot)
def _evaluate_logicalnot(e, self):
    val = self(e.children[0])
    assert val.arr.dtype == numpy.dtype("bool")
    result = Result.empty(val, bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = not val[val.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.LogicalAnd)
def _evaluate_logicaland(e, self):
    a, b = [self(o) for o in e.children]
    assert a.arr.dtype == numpy.dtype("bool")
    assert b.arr.dtype == numpy.dtype("bool")
    result = Result.empty(a, b, bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = a[a.filter(idx, result.fids)] and \
            b[b.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.LogicalOr)
def _evaluate_logicalor(e, self):
    a, b = [self(o) for o in e.children]
    assert a.arr.dtype == numpy.dtype("bool")
    assert b.arr.dtype == numpy.dtype("bool")
    result = Result.empty(a, b, dtype=bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = a[a.filter(idx, result.fids)] or \
            b[b.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.Conditional)
def _evaluate_conditional(e, self):
    cond, then, else_ = [self(o) for o in e.children]
    assert cond.arr.dtype == numpy.dtype("bool")
    result = Result.empty(cond, then, else_)
    for idx in numpy.ndindex(result.tshape):
        if cond[cond.filter(idx, result.fids)]:
            result[idx] = then[then.filter(idx, result.fids)]
        else:
            result[idx] = else_[else_.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.Indexed)
def _evaluate_indexed(e, self):
    """Indexing maps shape to free indices"""
    val = self(e.children[0]) # tensor being indexed
    # fids = tuple(i for i in e.multiindex if isinstance(i, (gem.Index, gem.ListIndex))) # free indices in the multiindex
    fids = tuple(
        i.free_index if isinstance(i, gem.ListIndex) else i
        for i in e.multiindex
        if isinstance(i, (gem.Index, gem.ListIndex))
    )


    all_fids = val.fids + fids

    idx = []

    # idx has one entry per axis but val.arr is a single array with shape (*val.fids extents, *tensor_shape)
    # and the inner loop consumes the tensor_shape axes one by ones through indexing.
    # When val.arr is indexed with idx as val[idx], NumPy broadcasts each entry of idx to operate
    # on all axes simultaneously so we need to reshape each entry of idx appropriately.

    # First pick up all the free indices that's already part of the tensor
    for fid in val.fids:
        # idx.append(slice(None))
        shape = tuple(fid.extent if f is fid else 1 for f in all_fids)
        idx.append(numpy.arange(fid.extent).reshape(shape))

    # Now grab the shape axes
    for i in e.multiindex:
        if isinstance(i, gem.ListIndex):
            # ListIndex, use the index array
            shape = tuple(i.free_index.extent if f is i.free_index else 1 for f in all_fids)
            idx.append(i.index_array.reshape(shape))
        elif isinstance(i, gem.Index):
            # Free index, want entire extent
            # idx.append(slice(None))
            shape = tuple(i.extent if f is i else 1 for f in all_fids)
            idx.append(numpy.arange(i.extent).reshape(shape))
        elif isinstance(i, gem.VariableIndex):
            # Variable index, evaluate inner expression
            result, = self(i.expression)
            assert not result.tshape # int
            idx.append(result[()])
        else:
            # Fixed index, just pick that value
            idx.append(i)
    assert len(idx) == len(val.tshape)
    return Result(val[idx], all_fids)


@_evaluate.register(gem.FlexiblyIndexed)
def _evaluate_flexiblyindexed(e, self):
    val = self(e.children[0])

    all_fids = val.fids + e.free_indices

    idx = []

    # Collect existing free indices
    for fid in val.fids:
        shape = tuple(fid.extent if f is fid else 1 for f in all_fids)
        fidx_array = numpy.arange(fid.extent).reshape(shape)
        idx.append(fidx_array)

    # Compute a flat index for each dim from dim2idxs
    for dim, (offset, idxs) in zip(e.children[0].shape, e.dim2idxs):
        if isinstance(offset, gem.Node):
            offset_result = self(offset)
            assert not offset_result.tshape
            offset_val = offset_result[()]
        else:
            offset_val = offset

        dim_flat_idx_components = []
        for i, s in idxs:
            # Index i may be one of: Index, VariableIndex, int
            if isinstance(i, gem.Index):
                # Free index contributes an array index given by the extent
                shape = tuple(i.extent if f is i else 1 for f in all_fids)
                i_vals = numpy.arange(i.extent).reshape(shape) * s
                dim_flat_idx_components.append(i_vals)
            elif isinstance(i, gem.VariableIndex):
                # Variable index gives a single integer index
                # obtained by evaluating its inner expression
                result, = self(i.expression)
                assert not result.tshape
                dim_flat_idx_components.append(result[()]*s)
            else:
                # Fixed index is just an integer so use that
                dim_flat_idx_components.append(i*s)

        # From dim_flat_idx_components compute the flat index into dim
        dim_idx_array = offset_val + sum(flat_index_component for flat_index_component in dim_flat_idx_components)
        idx.append(dim_idx_array)

    return Result(val[idx], all_fids)


@_evaluate.register(gem.ComponentTensor)
def _evaluate_componenttensor(e, self):
    """Component tensors map free indices to shape."""
    val = self(e.children[0])
    axes = []
    fids = []
    # First grab the free indices that aren't bound
    for a, f in enumerate(val.fids):
        if f not in e.multiindex:
            axes.append(a)
            fids.append(f)
    # Now the bound free indices
    for i in e.multiindex:
        axes.append(val.fids.index(i))
    # Now the existing shape
    axes.extend(range(len(val.fshape), len(val.tshape)))
    return Result(numpy.transpose(val.arr, axes=axes),
                  tuple(fids))


@_evaluate.register(gem.IndexSum)
def _evaluate_indexsum(e, self):
    """Index sums reduce over the given axis."""
    val = self(e.children[0])
    idx = tuple(map(val.fids.index, e.multiindex))
    rfids = tuple(fi for fi in val.fids if fi not in e.multiindex)
    return Result(val.arr.sum(axis=idx), rfids)


@_evaluate.register(gem.ListTensor)
def _evaluate_listtensor(e, self):
    """List tensors just turn into arrays."""
    ops = [self(o) for o in e.children]
    tmp = Result.empty(*ops)
    arrs = [numpy.broadcast_to(o.broadcast(tmp.fids), tmp.fshape) for o in ops]
    arrs = numpy.moveaxis(numpy.asarray(arrs), 0, -1).reshape(tmp.fshape + e.shape)
    return Result(arrs, tmp.fids)


@_evaluate.register(gem.Concatenate)
def _evaluate_concatenate(e, self):
    """Concatenate nodes flatten and concatenate shapes."""
    ops = [self(o) for o in e.children]
    fids = tuple(OrderedDict.fromkeys(itertools.chain(*(o.fids for o in ops))))
    fshape = tuple(i.extent for i in fids)
    arrs = []
    for o in ops:
        # Create temporary with correct shape
        arr = numpy.empty(fshape + o.shape)
        # Broadcast for extra free indices
        arr[:] = o.broadcast(fids)
        # Flatten shape
        arr = arr.reshape(arr.shape[:arr.ndim-len(o.shape)] + (-1,))
        arrs.append(arr)
    arrs = numpy.concatenate(arrs, axis=-1)
    return Result(arrs, fids)


def evaluate(expressions, bindings=None):
    """Evaluate some GEM expressions given variable bindings.

    :arg expressions: A single GEM expression, or iterable of
        expressions to evaluate.
    :kwarg bindings: An optional dict mapping GEM :class:`gem.Variable`
        nodes to data.
    :returns: a list of the evaluated expressions.
    """
    try:
        exprs = tuple(expressions)
    except TypeError:
        exprs = (expressions, )
    mapper = node.Memoizer(_evaluate)
    mapper.bindings = bindings if bindings is not None else {}
    return list(map(mapper, exprs))
