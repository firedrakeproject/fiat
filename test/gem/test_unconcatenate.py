import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem.unconcatenate import unconcatenate


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


def test_unconcatenate_restricted_indices() -> None:
    """Split only along the given indices, even if the variable carries more."""
    beta = gem.Index(extent=5)
    gamma = gem.Index(extent=4)
    variable = gem.Product(
        gem.Indexed(gem.Concatenate(gem.Literal([1.0, 2.0]),
                                    gem.Literal([3.0, 4.0, 5.0])),
                    (beta,)),
        gem.Indexed(gem.Concatenate(gem.Literal([2.0, 3.0]),
                                    gem.Literal([4.0, 5.0])),
                    (gamma,)))

    pairs = unconcatenate([(variable, gem.Literal(1.0))], indices=(beta,))
    assert len(pairs) == 2
    assert all(gamma in v.free_indices and beta not in v.free_indices for v, _ in pairs)
