import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem.unconcatenate import split_contraction, unconcatenate


def contract(pairs: list) -> gem.Node:
    """Sum each product of variable and expression over the variable's indices."""
    return gem.Sum(*(gem.IndexSum(gem.Product(v, e), v.free_indices) for v, e in pairs))


def test_unconcatenate_computed_variable() -> None:
    """Split a computed variable along its own Concatenate nodes."""
    beta = gem.Index(extent=5)
    left = gem.Indexed(
        gem.Concatenate(gem.Literal([1.0, 2.0]),
                        gem.Literal([3.0, 4.0, 5.0])),
        (beta,))
    right = gem.Indexed(
        gem.Concatenate(gem.Literal([6.0, 7.0]),
                        gem.Literal([8.0, 9.0, 10.0])),
        (beta,))

    pairs = unconcatenate([(left, right)])
    assert len(pairs) == 2
    assert sorted(index.extent for v, _ in pairs for index in v.free_indices) == [2, 3]
    assert all(beta not in e.free_indices for _, e in pairs)

    original_result, split_result = evaluate([
        gem.IndexSum(gem.Product(left, right), (beta,)),
        contract(pairs),
    ])
    assert numpy.array_equal(original_result.arr, split_result.arr)


def test_unconcatenate_computed_variable_multiple_indices() -> None:
    """Split a computed variable along each index that it carries."""
    beta = gem.Index(extent=5)
    gamma = gem.Index(extent=4)
    left = gem.Indexed(
        gem.Concatenate(gem.Literal([1.0, 2.0]),
                        gem.Literal([3.0, 4.0, 5.0])),
        (beta,))
    right = gem.Indexed(
        gem.Concatenate(gem.Literal([2.0, 3.0]),
                        gem.Literal([4.0, 5.0])),
        (gamma,))
    variable = gem.Product(left, right)
    expression = gem.Literal(2.0)

    pairs = unconcatenate([(variable, expression)])
    assert len(pairs) == 4
    extents = sorted(tuple(sorted(index.extent for index in v.free_indices))
                     for v, _ in pairs)
    assert extents == [(2, 2), (2, 2), (2, 3), (2, 3)]

    original_result, split_result = evaluate([
        gem.IndexSum(gem.Product(variable, expression), (beta, gamma)),
        contract(pairs),
    ])
    assert numpy.array_equal(original_result.arr, split_result.arr)


def test_unconcatenate_computed_variable_outside_concatenate() -> None:
    """Refuse to split an index that a computed variable carries elsewhere."""
    beta = gem.Index(extent=3)
    variable = gem.Product(
        gem.Indexed(gem.Concatenate(gem.Literal([1.0]), gem.Literal([2.0, 3.0])),
                    (beta,)),
        gem.Indexed(gem.Literal([3.0, 4.0, 5.0]), (beta,)))

    with pytest.raises(ValueError):
        unconcatenate([(variable, gem.Literal(1.0))])


def contracted_blocks(beta: gem.Index) -> tuple[gem.Node, list[gem.Index]]:
    """Concatenate two blocks, each contracted over its own points."""
    blocks = []
    points = []
    for weights, values in [([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
                            ([[1.0, 0.5, 0.25]], [[2.0, 4.0, 8.0]])]:
        alpha = gem.Index(extent=len(weights))
        p = gem.Index(extent=len(weights[0]))
        summand = gem.Product(gem.Indexed(gem.Literal(weights), (alpha, p)),
                              gem.Indexed(gem.Literal(values), (alpha, p)))
        blocks.append(gem.ComponentTensor(gem.IndexSum(summand, (p,)), (alpha,)))
        points.append(p)
    return gem.Indexed(gem.Concatenate(*blocks), (beta,)), points


def contract_terms(terms: list) -> gem.Node:
    """Sum each term over its own indices."""
    return gem.Sum(*(gem.IndexSum(term, indices) for term, indices in terms))


def coefficients(beta: gem.Index) -> gem.Node:
    """Coefficients blocked like contracted_blocks."""
    return gem.Indexed(gem.Concatenate(gem.Literal([1.0, 10.0]), gem.Literal([100.0])), (beta,))


@pytest.mark.parametrize("wrap", [lambda e: e,
                                  lambda e: gem.MathFunction("conj", e),
                                  lambda e: gem.Product(gem.Literal(2.0), e)])
def test_split_contraction_lifts_block_sums(wrap) -> None:
    """Add a block's own summation indices to its term."""
    beta = gem.Index(extent=3)
    concat, _ = contracted_blocks(beta)
    expression = gem.Product(coefficients(beta), wrap(concat))
    terms = split_contraction(expression, (beta,))
    assert [sorted(i.extent for i in indices) for _, indices in terms] == [[2, 2], [1, 3]]
    assert all(set(indices) == set(term.free_indices) for term, indices in terms)

    expected, result = evaluate([gem.IndexSum(expression, (beta,)), contract_terms(terms)])
    assert numpy.allclose(expected.arr, result.arr)


def test_split_contraction_keeps_sums_that_are_not_factors() -> None:
    """Keep a block's sum when the block is not a factor of its term."""
    beta = gem.Index(extent=3)
    concat, _ = contracted_blocks(beta)
    expression = gem.Product(coefficients(beta), gem.Sum(concat, gem.Literal(1.0)))
    terms = split_contraction(expression, (beta,))
    assert [len(indices) for _, indices in terms] == [1, 1]


def test_split_contraction_renames_repeated_block_sums() -> None:
    """Give each occurrence of a block's sum its own indices."""
    beta = gem.Index(extent=3)
    concat, _ = contracted_blocks(beta)
    expression = gem.Product(concat, concat)
    terms = split_contraction(expression, (beta,))
    assert [len(set(indices)) for _, indices in terms] == [3, 3]

    expected, result = evaluate([gem.IndexSum(expression, (beta,)), contract_terms(terms)])
    assert numpy.allclose(expected.arr, result.arr)


def test_split_contraction_blocks_matching_concatenations() -> None:
    """Split a direct-sum contraction into one contraction per block."""
    beta = gem.Index(extent=5)
    left = gem.Indexed(
        gem.Concatenate(gem.Literal([1.0, 2.0]),
                        gem.Literal([3.0, 4.0, 5.0])),
        (beta,))
    right = gem.Indexed(
        gem.Concatenate(gem.Literal([6.0, 7.0]),
                        gem.Literal([8.0, 9.0, 10.0])),
        (beta,))
    expression = gem.Product(left, right)

    terms = split_contraction(expression, (beta,))
    assert len(terms) == 2
    assert sorted(index.extent for _, indices in terms for index in indices) == [2, 3]
    assert all(set(indices) == set(term.free_indices) for term, indices in terms)

    split = gem.Sum(*(gem.IndexSum(term, indices) for term, indices in terms))
    original_result, split_result = evaluate([
        gem.IndexSum(expression, (beta,)),
        split,
    ])
    assert numpy.array_equal(original_result.arr, split_result.arr)


def test_split_contraction_recurses_over_multiple_indices() -> None:
    """Split direct sums nested in a contraction over several indices."""
    beta = gem.Index(extent=5)
    gamma = gem.Index(extent=4)
    left = gem.Indexed(
        gem.Concatenate(gem.Literal([1.0, 2.0]),
                        gem.Literal([3.0, 4.0, 5.0])),
        (beta,))
    right = gem.Indexed(
        gem.Concatenate(gem.Literal([2.0, 3.0]),
                        gem.Literal([4.0, 5.0])),
        (gamma,))
    expression = gem.Product(left, right)

    terms = split_contraction(expression, (beta, gamma))
    assert len(terms) == 4
    extents = sorted(tuple(sorted(index.extent for index in indices))
                     for _, indices in terms)
    assert extents == [(2, 2), (2, 2), (2, 3), (2, 3)]

    split = gem.Sum(*(gem.IndexSum(term, indices) for term, indices in terms))
    original_result, split_result = evaluate([
        gem.IndexSum(expression, (beta, gamma)),
        split,
    ])
    assert numpy.array_equal(original_result.arr, split_result.arr)


def test_split_contraction_ignores_uncontracted_concatenations() -> None:
    """Leave a Concatenate untouched when its index is not summed."""
    beta = gem.Index(extent=3)
    point = gem.Index(extent=2)
    expression = gem.Product(
        gem.Indexed(gem.Concatenate(gem.Literal([1.0]), gem.Literal([2.0])),
                    (point,)),
        gem.Indexed(gem.Literal([3.0, 4.0, 5.0]), (beta,)))

    assert split_contraction(expression, (beta,)) == [(expression, (beta,))]


def test_split_contraction_slices_variable_views() -> None:
    """Slice a variable view summed against a direct sum into its blocks."""
    beta = gem.Index(extent=5)
    coefficients = gem.Variable("w", (5,))
    expression = gem.Product(
        gem.Indexed(coefficients, (beta,)),
        gem.Indexed(gem.Concatenate(gem.Literal([1.0, 2.0]),
                                    gem.Literal([3.0, 4.0, 5.0])),
                    (beta,)))

    terms = split_contraction(expression, (beta,))
    assert len(terms) == 2
    assert all(set(indices) == set(term.free_indices) for term, indices in terms)

    views = [next(node for node in gem.node.traversal([term])
                  if isinstance(node, gem.FlexiblyIndexed))
             for term, _ in terms]
    assert [view.children[0] for view in views] == [coefficients] * 2
    assert [view.dim2idxs[0][0] for view in views] == [0, 2]
