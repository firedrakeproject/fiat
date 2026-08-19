from functools import partial, reduce
from itertools import chain, combinations, islice

import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem import optimise
from gem.node import traversal
from gem.optimise import sum_factorise
from gem.coffee import optimise_monomial_sum
from gem.refactorise import (ATOMIC, COMPOUND, OTHER,
                             collect_monomials)


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


def laplacian(ndofs=4, ndims=2):
    """Build a Laplacian element tensor from a mapped gradient table.

    The physical gradient of each basis function is the reference gradient
    mapped by the inverse Jacobian.  Test and trial functions apply the same
    map, over their own argument index.
    """
    numpy.random.seed(0)
    i = gem.Index(extent=ndofs)
    j = gem.Index(extent=ndofs)
    k = gem.Index(extent=ndims)
    reference = gem.Literal(numpy.random.rand(ndofs, ndims))
    jacobian = gem.Literal(numpy.random.rand(ndims, ndims))

    def gradient(argument):
        # The pullback is a contraction over the topological dimension,
        # which reaches factorisation already unrolled into a sum.
        return reduce(gem.Sum, [
            gem.Product(gem.Indexed(reference, (argument, l)),
                        gem.Indexed(jacobian, (l, k)))
            for l in range(ndims)])

    expression = gem.IndexSum(
        gem.Product(gradient(i), gradient(j)), (k,))
    expected = numpy.einsum(
        "il,lk,jm,mk->ij", reference.array, jacobian.array,
        reference.array, jacobian.array)
    return (i, j), expression, expected


def monomial_sum(linear_indices):
    """Collect the Laplacian into monomials, with or without preservation."""
    arguments, expression, expected = laplacian()
    classifier = partial(_classify, frozenset(arguments))
    result, = collect_monomials(
        [expression], classifier,
        arguments if linear_indices else ())
    return arguments, result, expected


def _classify(arguments, expression):
    shared = arguments.intersection(expression.free_indices)
    if not shared:
        return OTHER
    if len(shared) == 1 and isinstance(expression, gem.Indexed):
        return ATOMIC
    return COMPOUND


def test_linear_map_is_preserved():
    # Distributing the map expands it into the product of its entries, so
    # preserving it leaves strictly fewer monomials to factorise.
    _, preserved, _ = monomial_sum(linear_indices=True)
    _, expanded, _ = monomial_sum(linear_indices=False)
    assert len(list(preserved)) < len(list(expanded))
    assert len(list(preserved)) == 1


def test_preserved_linear_map_is_shared():
    # Test and trial apply the same map over different indices.  COFFEE
    # materialises it once, so one tensor carries both.
    arguments, preserved, expected = monomial_sum(linear_indices=True)
    expression = optimise_monomial_sum(preserved, arguments)
    tensors = [node for node in traversal((expression,))
               if isinstance(node, gem.ComponentTensor)]
    assert len(tensors) == 1

    i, j = arguments
    result, = evaluate([gem.ComponentTensor(expression, (i, j))])
    assert numpy.allclose(result.arr, expected)


def test_expanded_and_preserved_agree():
    arguments, expanded, expected = monomial_sum(linear_indices=False)
    expression = optimise_monomial_sum(expanded, arguments)
    result, = evaluate([gem.ComponentTensor(expression, arguments)])
    assert numpy.allclose(result.arr, expected)


def test_has_linear_maps_detects_preservable_maps():
    arguments, expression, _ = laplacian()
    assert optimise.has_linear_maps([expression], arguments)
    # With no argument axis declared, nothing is a linear map.
    assert not optimise.has_linear_maps([expression], ())


def test_estimate_cost_counts_the_contraction():
    _, expression, _ = laplacian(ndofs=4, ndims=2)
    flops, storage, largest, nodes = optimise.estimate_cost([expression])
    # Two mapped gradients over (argument, k, l) and their contraction
    # over k, all counted over their own domains.
    assert flops > 0
    assert storage >= largest > 0
    assert nodes > 0


def test_empty_product_contraction_counts_index_tuples() -> None:
    i, j = gem.Index(extent=2), gem.Index(extent=3)
    expression = sum_factorise((i, j), ())
    result, = evaluate([expression])
    assert result.arr == 6


def test_contraction_counts_index_absent_from_factors() -> None:
    i, j = gem.Index(extent=2), gem.Index(extent=3)
    factor = gem.Indexed(gem.Literal(numpy.arange(3.0)), (j,))

    expression = sum_factorise((i, j), (factor,))
    result, = evaluate([expression])

    assert result.arr == 2 * numpy.arange(3.0).sum()
