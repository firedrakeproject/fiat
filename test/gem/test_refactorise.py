from functools import partial

import pytest

import gem
from gem.node import traversal
from gem.refactorise import ATOMIC, COMPOUND, OTHER, collect_monomials


def classify(atomics_set, expression):
    if expression in atomics_set:
        return ATOMIC

    for node in traversal([expression]):
        if node in atomics_set:
            return COMPOUND

    return OTHER


def test_refactorise_rebound_sum_index():
    """A contraction buried in the non-atomic factors may bind an index that
    the monomial itself already sums over."""
    f = gem.Variable('f', (3,))
    u = gem.Variable('u', (3,))
    v = gem.Variable('v', ())

    i = gem.Index()
    f_i = gem.Indexed(f, (i,))
    u_i = gem.Indexed(u, (i,))
    classifier = partial(classify, {u_i, v})

    # \sum_i u_i*(v + \sum_i f_i)
    expr = gem.IndexSum(
        gem.Product(u_i, gem.Sum(v, gem.IndexSum(f_i, (i,)))),
        (i,)
    )

    monomials = list(collect_monomials([expr], classifier)[0])
    assert [m.atomics for m in monomials] == [(u_i, v), (u_i,)]
    for m in monomials:
        assert m.sum_indices == (i,)
        # The rebound index is renamed apart from the one being summed here.
        assert i not in m.rest.free_indices


@pytest.mark.parametrize("carried", [False, True],
                         ids=["nothing-carries-i", "another-factor-carries-i"])
def test_refactorise_zero_rest(carried):
    """A Zero among the non-atomic factors absorbs the product, and with it
    the free indices of the factors it multiplies."""
    g = gem.Variable('g', (3,))
    u = gem.Variable('u', (3,))
    w = gem.Variable('w', ())

    i = gem.Index(extent=3)
    g_i = gem.Indexed(g, (i,))
    u_i = gem.Indexed(u, (i,))
    classifier = partial(classify, {u_i})

    # \sum_i [g_i*](w > 0 ? u_i : 0), whose second monomial contracts i over
    # a rest that the empty branch has zeroed.
    condition = gem.Comparison('>', w, gem.Literal(0))
    body = gem.Conditional(condition, u_i, gem.Zero())
    if carried:
        body = gem.Product(g_i, body)
    expr = gem.IndexSum(body, (i,))

    monomials = list(collect_monomials([expr], classifier)[0])
    assert [m.atomics for m in monomials] == [(u_i,), ()]
    assert monomials[1].rest == gem.Zero()
