import numpy
import pytest

import gem
from gem.interpreter import evaluate
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


def test_too_many_indices_in_one_contraction():
    # A single connected contraction is still bounded.
    indices = tuple(gem.Index(extent=2) for _ in range(7))
    table = gem.Indexed(gem.Literal(numpy.ones((2,) * 7)), indices)
    with pytest.raises(NotImplementedError):
        sum_factorise(indices, [table])
