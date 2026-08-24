from itertools import chain, combinations, islice

import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem import optimise
from gem.node import traversal
from gem.optimise import sum_factorise


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


def shared_index_contraction(nindices, nfactors, extent=2):
    """Build one connected contraction of `nfactors` factors over `nindices`.

    Every factor carries the first index, so the contraction is connected,
    and each carries a different subset of the rest, so no two factors are
    grouped together.
    """
    indices = tuple(gem.Index(extent=extent) for _ in range(nindices))
    subsets = chain.from_iterable(combinations(indices[1:], n)
                                  for n in range(1, nindices))
    factors = []
    for extra in islice(subsets, nfactors):
        shared = indices[:1] + extra
        table = numpy.random.rand(*(extent,) * len(shared))
        factors.append(gem.Indexed(gem.Literal(table), shared))
    assert len(factors) == nfactors
    return indices, factors


def test_too_many_indices_in_one_contraction():
    # A connected contraction of more factors than the planner takes falls
    # back on searching the orderings, which is still bounded.
    indices, factors = shared_index_contraction(7, 11)

    with pytest.raises(NotImplementedError):
        sum_factorise(indices, factors)


def test_more_factors_than_indices():
    # Factors and contraction indices are not tied to each other: each
    # factor carrying a different subset of the indices takes a group of
    # its own, so more factors than the planner takes can still share few
    # enough indices for the ordering search.  Planning this tree costs
    # 3**15 splits, which is what the search is here to avoid.
    numpy.random.seed(0)
    indices, factors = shared_index_contraction(5, 15)

    expression = sum_factorise(indices, factors)
    assert expression.free_indices == ()

    letters = {index: chr(ord("a") + n) for n, index in enumerate(indices)}
    spec = ",".join("".join(letters[i] for i in f.multiindex) for f in factors)
    expected = numpy.einsum(spec + "->", *[f.children[0].array for f in factors])

    result, = evaluate([expression])
    assert numpy.allclose(result.arr, expected)


def test_contraction_preserves_repeated_contractions():
    # A dual evaluation contracts the weights with an expression whose
    # coefficient evaluations are already sum factorised.  Flattening an
    # evaluation renames its indices apart, so one used several times
    # would be evaluated once per use.
    numpy.random.seed(0)
    p, q = gem.Index(extent=4), gem.Index(extent=4)
    ijk = tuple(gem.Index(extent=3) for _ in range(3))

    table = gem.Indexed(gem.Literal(numpy.random.rand(3, 3, 3, 4)), ijk + (p,))
    dofs = numpy.random.rand(3, 3, 3)
    evaluation = optimise.contraction(gem.IndexSum(gem.Product(table, gem.Indexed(gem.Literal(dofs), ijk)), ijk))
    assert isinstance(evaluation, gem.IndexSum)

    weights = numpy.random.rand(4, 4)
    cubed = gem.Product(gem.Product(gem.Indexed(gem.Literal(weights), (q, p)), evaluation),
                        gem.Product(evaluation, evaluation))
    expression = gem.IndexSum(cubed, (p,))

    optimised = optimise.contraction(expression)
    assert evaluation in set(traversal([optimised]))

    # The evaluation is contracted once and reused, so the result holds
    # only that contraction and the one over the points
    contractions = [node for node in traversal([optimised])
                    if isinstance(node, gem.IndexSum)]
    assert len(contractions) == 2

    result, = evaluate([gem.ComponentTensor(optimised, (q,))])
    expected = weights.dot(numpy.einsum("ijkp,ijk->p", numpy.asarray(table.children[0].array), dofs) ** 3)
    assert numpy.allclose(result.arr, expected)


def test_contractions_joined_by_a_shared_index():
    # A value index joins two coefficient evaluations into one connected
    # contraction of more indices than an ordering search can take, but
    # its product tree is planned by splitting the factors.
    numpy.random.seed(0)
    p = gem.Index(extent=4)
    ijk = tuple(gem.Index(extent=3) for _ in range(3))
    lmn = tuple(gem.Index(extent=3) for _ in range(3))

    tables = [numpy.random.rand(3, 3, 3, 4) for _ in range(2)]
    coefficients = [numpy.random.rand(3, 3, 3) for _ in range(2)]
    factors = []
    for indices, table, coefficient in zip((ijk, lmn), tables, coefficients):
        factors.append(gem.Indexed(gem.Literal(table), indices + (p,)))
        factors.append(gem.Indexed(gem.Literal(coefficient), indices))

    sum_indices = ijk + lmn + (p,)
    assert len(sum_indices) > 6

    expression = sum_factorise(sum_indices, factors)
    assert expression.free_indices == ()

    result, = evaluate([expression])
    evaluations = [numpy.einsum("ijkp,ijk->p", table, coefficient)
                   for table, coefficient in zip(tables, coefficients)]
    assert numpy.allclose(result.arr, evaluations[0].dot(evaluations[1]))
