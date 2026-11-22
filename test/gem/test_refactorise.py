from functools import partial

import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem.node import traversal
from gem.optimise import contraction
from gem.refactorise import ATOMIC, COMPOUND, OTHER, Monomial, collect_monomials


def classify(atomics_set, expression):
    if expression in atomics_set:
        return ATOMIC

    for node in traversal([expression]):
        if node in atomics_set:
            return COMPOUND

    return OTHER


@pytest.mark.parametrize("same_index", (False, True))
def test_refactorise_contracted_list_tensor(same_index):
    """A Delta that does not depend on the ListTensor index must stay
    outside the ListTensor, so that collect_monomials can find it."""
    i, q, source, target = (gem.Index(extent=2) for _ in range(4))
    component = i if same_index else gem.Index(extent=2)
    values = gem.Variable("values", (2,))
    value = values[q]
    table = gem.ListTensor([value, 2 * value])
    weights = gem.Literal([3, 5])
    tabulation = gem.IndexSum(table[i] * weights[q], (q,))
    selector = gem.IndexSum(gem.Delta(source, component) * gem.Delta(component, target), (component,))
    expression = tabulation * selector

    delta = gem.Delta(source, target)
    classifier = partial(classify, {delta})
    monomials, = collect_monomials([contraction(expression)], classifier)
    monomial, = monomials
    assert monomial.atomics == (delta,)
    assert monomial.sum_indices == ()
    result = gem.Product(*monomial.atomics, monomial.rest)
    tensor = gem.ComponentTensor(result, (i, source, target))
    actual, = evaluate([tensor], bindings={values: numpy.array([2, 7])})
    numpy.testing.assert_array_equal(actual.arr, [41 * numpy.eye(2), 82 * numpy.eye(2)])


@pytest.mark.parametrize("lookup", ("linked", "shared", "bound"))
def test_refactorise_list_tensor_selector(lookup):
    i, q, s, t = (gem.Index(extent=2) for _ in range(4))
    values = gem.Literal([2, 3])
    table = gem.ListTensor([values[q], 2 * values[q]])
    if lookup == "linked":
        expression = gem.IndexSum(table[i] * gem.Delta(q, s) * gem.Delta(q, t), (q,))
        delta = gem.Delta(s, t)
    else:
        tabulation = gem.IndexSum(table[i], (i, q) if lookup == "bound" else (q,))
        delta = gem.Delta(i, t)
        expression = tabulation * delta

    monomials, = collect_monomials([contraction(expression)], partial(classify, {delta}))
    monomial, = monomials
    assert monomial.atomics == (delta,)
    original, result = evaluate([expression, gem.Product(*monomial.atomics, monomial.rest)])
    numpy.testing.assert_array_equal(result.broadcast(original.fids), original.arr)


def test_refactorise_list_tensor_with_coupled_selector():
    row, component, value = (gem.Index(extent=2) for _ in range(3))
    point = gem.Index()
    weights = gem.Literal([2, 3])[point]
    table = gem.ListTensor([weights, 2 * weights])
    selector = gem.Delta(component, value)
    expression = gem.IndexSum(table[row] * (weights * selector + 1), (point,))

    monomials, = collect_monomials([contraction(expression)], partial(classify, {selector}))
    assert any(selector in monomial.atomics for monomial in monomials)
    result = gem.Sum(*(gem.IndexSum(gem.Product(*m.atomics, m.rest), m.sum_indices)
                       for m in monomials))
    expected, actual = evaluate([expression, result])
    numpy.testing.assert_array_equal(actual.broadcast(expected.fids), expected.arr)


@pytest.mark.parametrize("nested", (False, True))
def test_refactorise_free_sum_index(nested):
    i, = gem.indices(1)
    x = gem.Literal([2, 3])[i]
    y = gem.Literal([7, 11])[i]
    reduction = gem.IndexSum(y, (i,))
    expression = x * (reduction + 1 if nested else reduction)

    def classify_y(node):
        if isinstance(node, gem.Indexed) and node.children == y.children:
            return ATOMIC
        return COMPOUND if any(isinstance(n, gem.Indexed) and n.children == y.children
                               for n in traversal((node,))) else OTHER

    monomials, = collect_monomials([expression], classify_y)
    result = gem.Sum(*(gem.IndexSum(gem.Product(*m.atomics, m.rest), m.sum_indices)
                       for m in monomials))
    actual, = evaluate([result])
    assert actual.fids == (i,)
    numpy.testing.assert_array_equal(actual.arr, [38, 57] if nested else [36, 54])


def test_refactorise_delta_trace():
    i, j = (gem.Index(extent=2) for _ in range(2))
    k, = gem.indices(1)
    a = gem.Literal([2, 3])[k]
    expression = gem.IndexSum(a * gem.Delta(i, j), (i, j))

    monomials, = collect_monomials([expression], partial(classify, {a}))
    monomial, = monomials
    result, = evaluate([gem.Product(*monomial.atomics, monomial.rest)])
    assert monomial.sum_indices == ()
    numpy.testing.assert_array_equal(result.arr, [4, 6])


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
        # The rebound index stays bound inside its own contraction.
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


def test_refactorise_cancels_deltas():
    """A Delta cancels an index contracted by a monomial, collapsing the sum.

    The dual basis of a point evaluation element is a Delta between the node
    and the point, so a contraction against it has to collapse rather than be
    carried out.  Expansion reaches a Delta among a monomial's factors and one
    nested in a product of rests, but not one left under a Sum.
    """
    A = gem.Variable('A', (4, 3))
    B = gem.Variable('B', (3,))
    C = gem.Variable('C', (3,))

    j = gem.Index(extent=4)
    p = gem.Index(extent=3)
    q = gem.Index(extent=3)
    delta = gem.Delta(p, q)
    A_qj, A_pj = gem.Indexed(A, (j, q)), gem.Indexed(A, (j, p))
    B_q, B_p = gem.Indexed(B, (q,)), gem.Indexed(B, (p,))
    C_q, C_p = gem.Indexed(C, (q,)), gem.Indexed(C, (p,))
    classifier = partial(classify, {A_qj})

    # \sum_q (delta_pq*A_qj + delta_pq*B_q)*(A_qj + C_q)
    expr = gem.IndexSum(
        gem.Product(gem.Sum(gem.Product(delta, A_qj), gem.Product(delta, B_q)),
                    gem.Sum(A_qj, C_q)),
        (q,)
    )

    factored, under_sum, nested = collect_monomials([expr], classifier)[0]

    assert factored == Monomial((), (A_pj, A_pj), gem.Literal(1))
    assert nested == Monomial((), (), gem.Product(C_p, B_p))
    assert under_sum.sum_indices == (q,)
