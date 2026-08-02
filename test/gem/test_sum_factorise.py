import numpy
import pytest

import gem
from gem.coffee import find_optimal_atomics
from gem.gem import one
from gem.interpreter import evaluate
from gem.optimise import estimate_cost, sum_factorise
from gem.refactorise import Monomial


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


def test_many_indices_in_one_contraction():
    indices = tuple(gem.Index(extent=2) for _ in range(7))
    table = gem.Indexed(gem.Literal(numpy.ones((2,) * 7)), indices)
    expression = sum_factorise(indices, [table])
    result, = evaluate([expression])
    assert result.arr == 2 ** len(indices)


def test_optimal_atomics_complete_bipartite():
    index = gem.Index(extent=3)
    left = tuple(
        gem.Indexed(gem.Variable(f"left{i}", (3,)), (index,))
        for i in range(5))
    right = tuple(
        gem.Indexed(gem.Variable(f"right{i}", (3,)), (index,))
        for i in range(7))
    monomials = [
        Monomial((), (a, b), one)
        for a in left for b in right
    ]

    selected = find_optimal_atomics(monomials, (index,))

    assert len(selected) == len(left)
    assert all(any(atomic in monomial.atomics for atomic in selected)
               for monomial in monomials)


def test_estimate_cost_jagged_contraction():
    p = gem.JaggedIndex(extent=4)
    q = gem.JaggedIndex(extent=4, parents=(p,))
    table = gem.Indexed(gem.Literal(numpy.ones((4, 4))), (p, q))
    expression = gem.IndexSum(table * table, (p, q))

    operations, storage, largest, _ = estimate_cost((expression,))

    assert operations == 20
    assert storage == largest == 1
