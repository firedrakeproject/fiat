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


def test_too_many_indices_in_one_contraction():
    # A single connected contraction is still bounded.
    indices = tuple(gem.Index(extent=2) for _ in range(7))
    table = gem.Indexed(gem.Literal(numpy.ones((2,) * 7)), indices)
    with pytest.raises(NotImplementedError):
        sum_factorise(indices, [table])


def test_contraction_preserves_factorised_contractions():
    # A dual evaluation contracts the weights with an expression whose
    # coefficient evaluations are already sum factorised.  Those carry the
    # point index of the contraction, so flattening them yields a single
    # connected contraction that is too large to factorise.
    numpy.random.seed(0)
    p, q = gem.Index(extent=4), gem.Index(extent=4)
    ijk = tuple(gem.Index(extent=3) for _ in range(3))

    table = gem.Indexed(gem.Literal(numpy.random.rand(3, 3, 3, 4)), ijk + (p,))
    dofs = numpy.random.rand(3, 3, 3)
    evaluation = optimise.contraction(gem.IndexSum(gem.Product(table, gem.Indexed(gem.Literal(dofs), ijk)), ijk))
    assert optimise.is_contraction(evaluation)

    weights = numpy.random.rand(4, 4)
    cubed = gem.Product(gem.Product(gem.Indexed(gem.Literal(weights), (q, p)), evaluation),
                        gem.Product(evaluation, evaluation))
    expression = gem.IndexSum(cubed, (p,))

    with pytest.raises(NotImplementedError):
        optimise.contraction(expression)

    optimised = optimise.contraction(expression, stop_at=optimise.is_contraction)
    assert evaluation in set(traversal([optimised]))

    result, = evaluate([gem.ComponentTensor(optimised, (q,))])
    expected = weights.dot(numpy.einsum("ijkp,ijk->p", numpy.asarray(table.children[0].array), dofs) ** 3)
    assert numpy.allclose(result.arr, expected)
