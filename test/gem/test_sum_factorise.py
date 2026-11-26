from itertools import combinations

import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem.optimise import contraction as optimise_contraction, sum_factorise


def contraction(nfactors, ndims, extent=2):
    """Build a product of independent contractions.

    Each factor contracts a table with a coefficient over its own indices,
    as a tensor product coefficient evaluation does, so no factor carries
    the indices of another.
    """
    numpy.random.seed(0)
    sum_indices = []
    factors = []
    expected = 1.0
    for _ in range(nfactors):
        indices = tuple(gem.Index(extent=extent) for _ in range(ndims))
        table = numpy.random.rand(*(extent,) * ndims)
        coefficient = numpy.random.rand(*(extent,) * ndims)
        factors.append(gem.Indexed(gem.Literal(table), indices))
        factors.append(gem.Indexed(gem.Literal(coefficient), indices))
        sum_indices.extend(indices)
        expected *= numpy.sum(table * coefficient)
    return tuple(sum_indices), factors, expected


@pytest.mark.parametrize("nfactors,ndims", [(1, 3), (2, 3), (3, 3), (5, 3), (3, 5)])
def test_independent_contractions(nfactors, ndims):
    # Contractions that share no factor are independent, so they are
    # factorised separately rather than by searching the orderings that
    # interleave them.  Together they exceed what one exhaustive search
    # can handle.
    sum_indices, factors, expected = contraction(nfactors, ndims)
    assert len(sum_indices) == nfactors * ndims

    expression = sum_factorise(sum_indices, factors)
    assert expression.free_indices == ()

    result, = evaluate([expression])
    assert numpy.allclose(result.arr, expected)


def test_greedy_contraction():
    # A single connected contraction too large to search exhaustively is
    # ordered greedily, cheapest contraction first.
    numpy.random.seed(0)
    extent = 2
    indices = tuple(gem.Index(extent=extent) for _ in range(7))
    table = numpy.random.rand(*(extent,) * len(indices))
    coefficient = numpy.random.rand(*(extent,) * len(indices))
    factors = [gem.Indexed(gem.Literal(table), indices),
               gem.Indexed(gem.Literal(coefficient), indices)]

    expression = sum_factorise(indices, factors)
    assert expression.free_indices == ()

    result, = evaluate([expression])
    assert numpy.allclose(result.arr, numpy.sum(table * coefficient))


def test_list_tensor_broadcast_reduction():
    i, j, k = gem.indices(3)
    a, b = gem.Literal([2, 3]), gem.Literal([7, 11])
    table = gem.ListTensor([a[j], b[k]])
    expression = gem.IndexSum(table[i], (j, k))

    result, = evaluate([optimise_contraction(expression)])
    numpy.testing.assert_array_equal(result.arr, [10, 36])


def test_list_tensor_specialisation_with_independent_selector():
    row, source, target = (gem.Index(extent=2) for _ in range(3))
    point = gem.Index()
    weights = gem.Literal([2, 3])[point]
    values = gem.Variable("values", (2,))[point]
    table = gem.ListTensor([values, 2 * values])
    expression = gem.IndexSum(table[row] * weights, (point,))
    selector = gem.Delta(source, target)

    specialised = optimise_contraction(expression)
    result = optimise_contraction(expression * selector)
    assert isinstance(result, gem.Product)
    assert set(result.children) == {specialised, selector}


def test_list_tensor_specialisation_with_selector_in_entries():
    # The gradient of a vector-valued basis repeats the component
    # selector in the entry for each direction.
    row, source, target, direction = (gem.Index(extent=2) for _ in range(4))
    point = gem.Index()
    weights = gem.Literal([[2, 3], [5, 7]])[direction, point]
    values = gem.Variable("values", (2,))[point]
    tables = [gem.ListTensor([values, 2 * values])[row],
              gem.ListTensor([3 * values, 5 * values])[row]]
    selector = gem.Delta(source, target)

    def gradient(selector):
        entries = gem.ListTensor([selector * table for table in tables])
        return gem.IndexSum(entries[direction] * weights, (direction, point))

    specialised = optimise_contraction(gradient(gem.Literal(1)))
    result = optimise_contraction(gradient(selector))
    assert isinstance(result, gem.Product)
    assert set(result.children) == {specialised, selector}


def test_delta_substitution_under_sum():
    i, j = gem.indices(2)
    table = gem.Literal([[1, 2], [8, 16]])
    a = gem.Literal([2, 3])
    expression = gem.IndexSum(gem.Delta(i, j) * (gem.IndexSum(table[i, j], (j,)) + a[i]), (i,))

    result, = evaluate([optimise_contraction(expression)])
    assert result.fids == (j,)
    numpy.testing.assert_array_equal(result.arr, [5, 27])


def test_delta_trace():
    i, j = (gem.Index(extent=2) for _ in range(2))
    expression = gem.IndexSum(gem.Delta(i, j), (i, j))

    result, = evaluate([optimise_contraction(expression)])
    assert result.arr == 2


def test_large_product():
    i, *axes = gem.indices(6)
    x = gem.Variable("x", ())
    table = gem.ListTensor([x, 2 * x])
    factors = [table[i]]
    for size in range(len(axes) + 1):
        independent = gem.Literal(numpy.full((2,) * size, 2))
        dependent = gem.Literal(numpy.full((2,) * (size + 1), 2))
        for subset in combinations(axes, size):
            factors.extend([independent[subset], dependent[i, *subset]])
    expression = gem.Product(*factors)

    expected, result = evaluate([expression, optimise_contraction(expression)],
                                bindings={x: numpy.array(2)})
    numpy.testing.assert_array_equal(result.broadcast(expected.fids), expected.arr)
