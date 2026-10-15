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


def test_negation_of_an_untyped_node_uses_negative_one():
    """Negating an untyped node multiplies it by negative one."""
    x = gem.Variable("x", ())
    assert -x == gem.Product(gem.Literal(-1), x)


def test_number_operand_of_an_untyped_node_is_float():
    """A number with an untyped node converts to a float literal."""
    expression = gem.Variable("x", ()) - 3
    assert expression.dtype is None
    assert all(child.dtype == numpy.float64 for child in traversal([expression]) if isinstance(child, gem.Literal))
