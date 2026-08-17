import numpy
import pytest

import gem
from gem.coffee import find_optimal_atomics, optimise_monomial_sum
from gem.gem import one
from gem.interpreter import evaluate
from gem.node import traversal
from gem.contraction import estimate_cost
from gem.optimise import sum_factorise
from gem.refactorise import Monomial, MonomialSum


def contraction(nfactors, ndims, extent=2):
    """Build a product of independent contractions.

    Each factor contracts a table with a coefficient over its own indices.
    This models one factor of a tensor product coefficient evaluation. No
    factor carries the indices of another.
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


def test_share_isomorphic_linear_maps():
    i = gem.Index(extent=3)
    j = gem.Index(extent=3)
    q = gem.Index(extent=2)
    table = gem.Variable("table", (3, 2))
    weight = gem.Variable("weight", (3, 2))

    def mapped(index):
        return 2 * gem.Indexed(table, (index, q)) \
            + gem.Indexed(weight, (index, q))

    monomial_sum = MonomialSum()
    monomial_sum.add((), (mapped(i), mapped(j)), one)

    expression = optimise_monomial_sum(monomial_sum, (i, j))

    tensors = [node for node in traversal((expression,))
               if isinstance(node, gem.ComponentTensor)]
    tensor, = tensors
    assert tensor.shape == (3,)
    assert tensor.free_indices == (q,)
    accesses = [node for node in traversal((expression,))
                if isinstance(node, gem.Indexed)
                and node.children[0] == tensor]
    assert {access.multiindex for access in accesses} == {(i,), (j,)}

    bindings = {
        table: numpy.arange(6).reshape(3, 2),
        weight: numpy.arange(6, 12).reshape(3, 2),
    }
    original = mapped(i) * mapped(j)
    expected, actual = evaluate([original, expression], bindings)
    assert numpy.array_equal(actual.broadcast(expected.fids), expected.arr)


def test_estimate_cost_jagged_contraction():
    p = gem.JaggedIndex(extent=4)
    q = gem.JaggedIndex(extent=4, parents=(p,))
    table = gem.Indexed(gem.Literal(numpy.ones((4, 4))), (p, q))
    expression = gem.IndexSum(table * table, (p, q))

    operations, storage, largest, _ = estimate_cost((expression,))

    assert operations == 20
    assert storage == largest == 1


def test_estimate_cost_independent_jagged_domains():
    """Multiply point counts of independent simplex lattices."""
    p = gem.JaggedIndex(extent=4)
    q = gem.JaggedIndex(extent=4, parents=(p,))
    t = gem.JaggedIndex(extent=4, parents=(p, q))
    r = gem.JaggedIndex(extent=5)
    s = gem.JaggedIndex(extent=5, parents=(r,))
    table = gem.Indexed(
        gem.Literal(numpy.ones((4, 4, 4, 5, 5))), (p, q, t, r, s))
    expression = gem.IndexSum(table * table, (p, q, t, r, s))

    operations, storage, largest, _ = estimate_cost((expression,))

    assert operations == 2 * 20 * 15
