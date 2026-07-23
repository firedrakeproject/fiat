import pytest
import gem
import numpy

from gem.node import traversal
from gem.optimise import _distribute_sum, unflatten_returns


@pytest.fixture
def A():
    a = gem.Variable("a", ())
    b = gem.Variable("b", ())
    c = gem.Variable("c", ())
    d = gem.Variable("d", ())
    array = [[a, b], [c, d]]
    A = gem.ListTensor(array)
    return A


@pytest.fixture
def X():
    return gem.Variable("X", (2, 2))


def test_listtensor_from_indexed(X):
    k = gem.Index()
    elems = [gem.Indexed(X, (k, *i)) for i in numpy.ndindex(X.shape[1:])]
    tensor = gem.ListTensor(numpy.reshape(elems, X.shape[1:]))

    assert isinstance(tensor, gem.ComponentTensor)
    j = tensor.multiindex
    expected = gem.partial_indexed(X, (k,))
    expected = gem.ComponentTensor(gem.Indexed(expected, j), j)
    assert tensor == expected


def test_listtensor_from_fixed_indexed(A):
    elems = [gem.Indexed(A, i) for i in numpy.ndindex(A.shape)]
    tensor = gem.ListTensor(numpy.reshape(elems, A.shape))
    assert tensor == A


def test_listtensor_from_partial_indexed(A):
    elems = [gem.partial_indexed(A, i) for i in numpy.ndindex(A.shape[:1])]
    tensor = gem.ListTensor(elems)
    assert tensor == A


def test_nested_partial_indexed(A):
    i, j = gem.indices(2)
    B = gem.partial_indexed(gem.partial_indexed(A, (i,)), (j,))
    assert B == gem.Indexed(A, (i, j))


def test_componenttensor_from_indexed(A):
    i, j = gem.indices(2)
    Aij = gem.Indexed(A, (i, j))
    assert A == gem.ComponentTensor(Aij, (i, j))


def test_indexed_transpose(A):
    i, j = gem.indices(2)
    ATij = gem.Indexed(A.T, (i, j))
    Aji = gem.Indexed(A, (j, i))
    assert ATij == Aji

    i, = gem.indices(1)
    j = 1
    ATij = gem.Indexed(A.T, (i, j))
    Aji = gem.Indexed(A, (j, i))
    assert ATij == Aji

    i, j = (0, 1)
    ATij = gem.Indexed(A.T, (i, j))
    Aji = gem.Indexed(A, (j, i))
    assert ATij == Aji


def test_double_transpose(A):
    assert A.T.T == A


def test_flatten_indexsum(A):
    i, j = gem.indices(2)
    Aij = gem.Indexed(A, (i, j))

    result = gem.IndexSum(gem.IndexSum(Aij, (i,)), (j,))
    expected = gem.IndexSum(Aij, (i, j))
    assert result == expected


def test_selective_distribution():
    a = gem.Variable("a", ())
    b = gem.Variable("b", ())
    c = gem.Variable("c", ())
    i = gem.Index(extent=2)
    p = gem.Index(extent=1)
    row = gem.VariableIndex(gem.Indexed(
        gem.Literal([0], dtype=gem.uint_type), (p,)))
    delta = gem.Delta(i, row)
    common = gem.Sum(a, b)
    expression = gem.Product(common, gem.Sum(c, delta))

    terms = _distribute_sum(
        expression, predicate=lambda node: isinstance(node, gem.Delta))

    assert len(terms) == 2
    assert all(common in set(traversal((term,))) for term in terms)


def test_constant_variable_index():
    index = gem.VariableIndex(gem.Literal(1, dtype=gem.uint_type))
    assert index == 1


def test_unflatten_compatible_returns_together():
    extent = 3
    p = gem.Index(extent=extent)
    q = gem.JaggedIndex(extent=extent, parents=(p,))
    ordering = numpy.zeros((extent, extent), dtype=gem.uint_type)
    ordering[0, :3] = [0, 1, 2]
    ordering[1, :2] = [3, 4]
    ordering[2, 0] = 5
    ordering = gem.Literal(ordering, dtype=gem.uint_type)

    X = gem.Variable("X", (extent, extent))
    Y = gem.Variable("Y", (extent, extent))
    ft_x = gem.FlattenedTensor(gem.Indexed(X, (p, q)), (p, q), ordering)
    ft_y = gem.FlattenedTensor(gem.Indexed(Y, (p, q)), (p, q), ordering)

    r = gem.Index(extent=6)
    result = gem.Variable("result", (6,))
    pairs = unflatten_returns([
        (gem.Indexed(result, (r,)),
         gem.Sum(gem.Indexed(ft_x, (r,)), gem.Indexed(ft_y, (r,))))
    ])

    assert len(pairs) == 1
    variable, expression = pairs[0]
    assert variable.free_indices == expression.free_indices
    assert len(variable.free_indices) == 2
    assert not any(isinstance(node, gem.FlattenedTensor)
                   for node in traversal((expression,)))
