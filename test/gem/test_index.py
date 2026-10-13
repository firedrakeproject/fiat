import numpy
import pytest

import gem
from gem.gem import unique
from gem.interpreter import evaluate


def test_unique_sorts_by_creation_order():
    """Free indices sort by creation order, whatever their addresses are."""
    indices = [gem.Index() for _ in range(64)]
    shuffled = indices[1::2] + indices[0::2]
    assert unique(shuffled) == tuple(indices)


def test_free_indices_survive_address_reuse():
    """A node orders its free indices the same way whatever preceded it."""
    def build(padding):
        scratch = [gem.Index() for _ in range(padding)]
        del scratch
        i, j = gem.indices(2)
        A = gem.Variable("A", (2, 3))
        return gem.Indexed(A, (j, i)).free_indices == (i, j)

    assert all(build(padding) for padding in range(32))


@pytest.mark.parametrize("literal_index", [0, 1])
def test_literal_index_with_symbolic_stride_gives_integer_offset(literal_index):
    """A literal index times a symbolic stride gives an unsigned integer offset."""
    i = gem.Index(extent=2)
    strides = gem.Literal(numpy.array([1, 2]), dtype=gem.uint_type)
    stride = gem.Indexed(strides, (i,))
    A = gem.Variable("A", (4,))
    node = gem.FlexiblyIndexed(A, ((1, ((literal_index, stride),)),))

    (offset, idxs), = node.dim2idxs
    assert idxs == ()
    assert node.free_indices == (i,)
    assert offset.dtype == gem.uint_type
    result, = evaluate([offset])
    assert result.fids == (i,)
    assert result.arr.dtype.kind == "u"
    assert numpy.array_equal(result.arr, 1 + literal_index * numpy.array([1, 2]))


def test_literal_index_with_constant_stride_after_symbolic_stride():
    """A constant stride added to a symbolic offset keeps it an unsigned integer."""
    i = gem.Index(extent=2)
    strides = gem.Literal(numpy.array([1, 2]), dtype=gem.uint_type)
    A = gem.Variable("A", (8,))
    node = gem.FlexiblyIndexed(A, ((0, ((1, gem.Indexed(strides, (i,))), (1, gem.Literal(2, dtype=gem.uint_type)))),))

    (offset, _), = node.dim2idxs
    assert offset.dtype == gem.uint_type
    result, = evaluate([offset])
    assert numpy.array_equal(result.arr, numpy.array([1, 2]) + 2)
