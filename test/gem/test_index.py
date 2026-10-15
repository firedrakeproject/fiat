import numpy
import pytest

import gem
from gem.gem import unique
from gem.node import traversal
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


@pytest.mark.parametrize(("build", "dtype", "expected"), [
    pytest.param(lambda n: 2 + 3 * n, gem.uint_type, [2, 5, 8], id="add_multiply"),
    pytest.param(lambda n: 2 * n - 1, numpy.int64, [-1, 1, 3], id="subtract"),
    pytest.param(lambda n: 1 - n, numpy.int64, [1, 0, -1], id="subtract_from_number"),
    pytest.param(lambda n: -n, numpy.int64, [0, -1, -2], id="negate"),
    pytest.param(lambda n: 1.5 * n, numpy.float64, [0, 1.5, 3], id="float_operand"),
])
def test_number_operand_takes_the_dtype_of_an_integer_node(build, dtype, expected):
    """A number in index arithmetic gets the smallest dtype that holds its value."""
    i = gem.Index(extent=3)
    n = gem.Indexed(gem.Literal(numpy.arange(3), dtype=gem.uint_type), (i,))
    expression = build(n)
    assert expression.dtype == dtype
    result, = evaluate([expression])
    assert numpy.array_equal(result.arr, expected)


def test_number_operand_of_an_untyped_node_is_float():
    """A number with an untyped node converts to a float literal."""
    expression = gem.Variable("x", ()) - 3
    assert expression.dtype is None
    assert all(child.dtype == numpy.float64 for child in traversal([expression]) if isinstance(child, gem.Literal))


def test_rename_indirect_componenttensor():
    """Renaming a tensor axis also renames its indirect lookups."""
    from gem.optimise import remove_componenttensors

    i, j = gem.indices(2)
    lookup = gem.VariableIndex(gem.Indexed(
        gem.Literal([2, 0, 1], dtype=gem.uint_type), (i,)))
    expression = gem.Indexed(gem.ComponentTensor(
        gem.Indexed(gem.Literal([4., 5., 6.]), (lookup,)), (i,)), (j,))
    assert expression.free_indices == (j,)
    expression, = remove_componenttensors([expression])
    result, = evaluate([expression])
    assert numpy.array_equal(result.arr, [6., 4., 5.])


def test_index_ordering_follows_indirect_lookup():
    """A lookup orders its running indices as it indexes them."""
    k, r = gem.indices(2)
    columns = gem.Literal(numpy.zeros((3, 2)), dtype=gem.uint_type)
    lookup = gem.VariableIndex(gem.Indexed(columns, (r, k)))
    j = gem.Index(extent=4)
    indexed = gem.Indexed(gem.Variable('A', (5, 4)), (lookup, j))
    assert indexed.index_ordering() == (r, k, j)
    flexibly_indexed = gem.FlexiblyIndexed(
        gem.Variable('B', (20,)), ((0, ((lookup, 4), (j, 1))),))
    assert flexibly_indexed.index_ordering() == (r, k, j)
