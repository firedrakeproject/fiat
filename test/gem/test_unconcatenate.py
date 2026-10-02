import numpy

import gem
from gem.interpreter import evaluate
from gem.unconcatenate import unconcatenate


def test_unconcatenate_contraction_blocks_matching_concatenations() -> None:
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

    terms = unconcatenate([(None, expression)], indices=(beta,))
    assert len(terms) == 2
    assert sorted(index.extent for _, term in terms for index in term.free_indices) == [2, 3]

    split = gem.Sum(*(gem.IndexSum(term, term.free_indices) for _, term in terms))
    original_result, split_result = evaluate([
        gem.IndexSum(expression, (beta,)),
        split,
    ])
    assert numpy.array_equal(original_result.arr, split_result.arr)


def test_unconcatenate_contraction_recurses_over_multiple_indices() -> None:
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

    terms = unconcatenate([(None, expression)], indices=(beta, gamma))
    assert len(terms) == 4
    extents = sorted(tuple(sorted(index.extent for index in term.free_indices))
                     for _, term in terms)
    assert extents == [(2, 2), (2, 2), (2, 3), (2, 3)]

    split = gem.Sum(*(gem.IndexSum(term, term.free_indices) for _, term in terms))
    original_result, split_result = evaluate([
        gem.IndexSum(expression, (beta, gamma)),
        split,
    ])
    assert numpy.array_equal(original_result.arr, split_result.arr)


def test_unconcatenate_contraction_ignores_uncontracted_concatenations() -> None:
    """Leave a Concatenate untouched when its index is not summed."""
    beta = gem.Index(extent=3)
    point = gem.Index(extent=2)
    expression = gem.Product(
        gem.Indexed(gem.Concatenate(gem.Literal([1.0]), gem.Literal([2.0])),
                    (point,)),
        gem.Indexed(gem.Literal([3.0, 4.0, 5.0]), (beta,)))

    assert unconcatenate([(None, expression)], indices=(beta,)) == [(None, expression)]


def test_unconcatenate_contraction_slices_variable_views() -> None:
    """Slice a variable view summed against a direct sum into its blocks."""
    beta = gem.Index(extent=5)
    coefficients = gem.Variable("w", (5,))
    expression = gem.Product(
        gem.Indexed(coefficients, (beta,)),
        gem.Indexed(gem.Concatenate(gem.Literal([1.0, 2.0]),
                                    gem.Literal([3.0, 4.0, 5.0])),
                    (beta,)))

    terms = unconcatenate([(None, expression)], indices=(beta,))
    assert len(terms) == 2
    assert all(beta not in term.free_indices for _, term in terms)

    views = [next(node for node in gem.node.traversal([term])
                  if isinstance(node, gem.FlexiblyIndexed))
             for _, term in terms]
    assert [view.children[0] for view in views] == [coefficients] * 2
    assert [view.dim2idxs[0][0] for view in views] == [0, 2]
