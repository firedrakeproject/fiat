from functools import partial, reduce
from itertools import chain, combinations, islice

import numpy
import pytest

import gem
from gem.interpreter import evaluate
from gem import cost, optimise
from gem.node import post_traversal, traversal
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


def test_greedy_contraction():
    # A connected contraction of more factors than the planner takes falls
    # back on searching the orderings, and one of more indices than that
    # search can afford is ordered greedily rather than refused.
    numpy.random.seed(0)
    indices, factors = shared_index_contraction(7, 11)

    expression = sum_factorise(indices, factors)
    assert expression.free_indices == ()

    letters = {index: chr(ord("a") + n) for n, index in enumerate(indices)}
    spec = ",".join("".join(letters[i] for i in f.multiindex) for f in factors)
    expected = numpy.einsum(spec + "->", *[f.children[0].array for f in factors])

    result, = evaluate([expression])
    assert numpy.allclose(result.arr, expected)


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


def test_linear_map_sharing_normalises_order_and_scale():
    i, j = gem.Index(extent=3), gem.Index(extent=3)
    a_values = numpy.array([1.0, 2.0, 3.0])
    b_values = numpy.array([4.0, 5.0, 6.0])
    a = gem.Literal(a_values)
    b = gem.Literal(b_values)
    left_terms = (gem.Indexed(a, (i,)), gem.Indexed(b, (i,)))
    right_terms = (gem.Indexed(b, (j,)), gem.Indexed(a, (j,)))
    left_map = optimise.make_sum(left_terms * 2)
    right_map = optimise.make_sum(right_terms * 2)
    expression = gem.Product(left_map, right_map)

    arguments = (i, j)
    classifier = partial(_classify, frozenset(arguments))
    preserved, = collect_monomials(
        [expression], classifier, arguments)
    optimised = optimise_monomial_sum(preserved, arguments)
    tensors = [node for node in traversal((optimised,))
               if isinstance(node, gem.ComponentTensor)]
    assert len(tensors) == 1
    body, = tensors[0].children
    assert len(optimise.traverse_sum(body)) == 2

    result, = evaluate([gem.ComponentTensor(optimised, arguments)])
    expected = 4 * numpy.outer(a_values + b_values, a_values + b_values)
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
    flops, storage, largest, nodes = cost.estimate_cost([expression])
    # Two mapped gradients over (argument, k, l) and their contraction
    # over k, all counted over their own domains.
    assert flops > 0
    assert storage >= largest > 0
    assert nodes > 0


def test_common_factor_is_extracted_from_scalar_sum():
    common = gem.Variable("common", ())
    left = gem.Variable("left", ())
    right = gem.Variable("right", ())
    expression = gem.Sum(
        gem.Product(common, left), gem.Product(common, right))

    expression = optimise.factorise_scalar_sums(expression)

    _, factors = optimise.traverse_product(expression)
    assert common in factors
    result, = evaluate([expression], {
        common: numpy.asarray(2.0),
        left: numpy.asarray(3.0),
        right: numpy.asarray(4.0),
    })
    assert result.arr == 14


def test_scalar_factorisation_leaves_indexed_sum_to_planner():
    i = gem.Index(extent=3)
    common = gem.Variable("common", ())
    left = gem.Indexed(gem.Literal(numpy.arange(3.0)), (i,))
    right = gem.Indexed(gem.Literal(numpy.arange(3.0, 6.0)), (i,))
    expression = gem.Sum(
        gem.Product(common, left), gem.Product(common, right))

    assert optimise.factorise_scalar_sums(expression) is expression


def test_contraction_plan_follows_the_callers_index_order():
    """A plan depends only on the caller's ordering of the sum indices."""
    def plan(reverse_creation):
        if reverse_creation:
            r = gem.Index(extent=4)
            q = gem.Index(extent=4)
        else:
            q = gem.Index(extent=4)
            r = gem.Index(extent=4)
        # Both factors carry both indices, so the two reduce at the same
        # subset and the tie decides their order within one IndexSum.
        factors = [gem.Indexed(gem.Variable(name, (4, 4)), (q, r))
                   for name in ("A", "B")]
        expression = optimise.sum_factorise([q, r], factors)
        role = {q: "q", r: "r"}
        return [tuple(role[i] for i in node.multiindex)
                for node in post_traversal((expression,))
                if isinstance(node, gem.IndexSum)]

    assert plan(False) == plan(True) == [("q", "r")]


def test_gathers_sharing_a_table_tabulate_it_once():
    """Several gathers of one table share a single tabulation.

    A padded basis transformation reads one reference tabulation through a
    map per padded column.  Each gather alone selects as many rows as it has
    arguments, so on its own it saves nothing; together they repeat one
    contraction once per column.
    """
    nrows, npoints, nwidth = 4, 3, 3
    j = gem.Index(extent=nrows)
    q = gem.Index(extent=npoints)
    table = gem.Literal(numpy.arange(nrows * npoints, dtype=float)
                        .reshape(nrows, npoints))
    weights = gem.Literal(numpy.arange(1.0, npoints + 1))
    columns = [gem.Literal(numpy.roll(numpy.arange(nrows), k),
                           dtype=gem.uint_type)
               for k in range(nwidth)]
    terms = [gem.IndexSum(
        gem.Product(
            gem.Indexed(table,
                        (gem.VariableIndex(gem.Indexed(column, (j,))), q)),
            gem.Indexed(weights, (q,))), (q,))
        for column in columns]
    expression = optimise.make_sum(terms)

    tabulated = optimise.tabulate_indirect_contractions(expression)

    tensors = {node for node in traversal((tabulated,))
               if isinstance(node, gem.ComponentTensor)}
    assert len(tensors) == 1

    contracted = table.array @ weights.array
    expected = sum(contracted[numpy.roll(numpy.arange(nrows), k)]
                   for k in range(nwidth))
    result, = evaluate([gem.ComponentTensor(tabulated, (j,))])
    assert numpy.allclose(result.arr, expected)


@pytest.mark.parametrize("extent", [2, 3, 5])
def test_distributed_linear_map_preserves_multiplicity(extent):
    i, j = gem.Index(extent=2), gem.Index(extent=2)
    k = gem.Index(extent=extent)
    a = gem.Indexed(gem.Literal(numpy.ones((2, extent))), (i, k))
    b = gem.Indexed(gem.Literal(numpy.ones(2)), (i,))
    c = gem.Indexed(gem.Literal(numpy.ones(2)), (j,))
    expression = gem.IndexSum((a + b) * c + b * c, (k,))
    arguments = (i, j)
    monomials, = collect_monomials(
        [expression], partial(_classify, frozenset(arguments)), arguments)
    optimized = optimise_monomial_sum(monomials, arguments)
    actual, = evaluate([gem.ComponentTensor(optimized, arguments)])
    assert numpy.array_equal(actual.arr, numpy.full((2, 2), 3 * extent))


@pytest.mark.parametrize("nrows,npoints,nargs", [(6, 3, 4), (4, 5, 3)])
@pytest.mark.parametrize("distinct", ["tables", "weights", "neither"])
def test_indirect_tabulation_costs_equivalent_bodies(nrows, npoints, nargs, distinct):
    j, q = gem.Index(extent=nargs), gem.Index(extent=npoints)
    tables = [gem.Literal(numpy.arange(nrows * npoints, dtype=float)
                          .reshape(nrows, npoints) + offset)
              for offset in (0, 1)]
    weights = [gem.Literal(numpy.arange(1.0, npoints + 1) + offset)
               for offset in (0, 1)]
    terms = []
    for number in range(2):
        columns = gem.Literal(numpy.arange(nargs) + number, dtype=gem.uint_type)
        row = gem.VariableIndex(gem.Indexed(columns, (j,)))
        table = tables[number if distinct == "tables" else 0]
        weight = weights[number if distinct == "weights" else 0]
        terms.append(gem.IndexSum(
            gem.Indexed(table, (row, q)) * gem.Indexed(weight, (q,)), (q,)))
    expression = optimise.make_sum(terms)
    optimized = optimise.tabulate_indirect_contractions(expression)
    assert cost.estimate_cost((optimized,)) <= cost.estimate_cost((expression,))
    if distinct == "neither":
        assert cost.estimate_cost((optimized,))[0] < cost.estimate_cost((expression,))[0]
    expected, actual = evaluate([gem.ComponentTensor(e, (j,))
                                for e in (expression, optimized)])
    assert numpy.allclose(actual.arr, expected.arr)


def test_equal_cost_contraction_follows_axis_order():
    """Equal-work plans retain the axis order used by streaming loops."""
    i, j, k, p, q, r = gem.indices(6)
    table = gem.Literal(numpy.arange(6.).reshape(2, 3))
    coefficient = gem.Literal(numpy.arange(27.).reshape(3, 3, 3))
    expression = sum_factorise(
        (i, j, k), [table[p, i], table[q, j], table[r, k], coefficient[i, j, k]])
    reductions = [node.multiindex for node in traversal((expression,))
                  if isinstance(node, gem.IndexSum)]
    assert reductions == [(k,), (j,), (i,)]


def test_repeated_contractions_walks_shared_products_once(monkeypatch):
    """Repeated product subtrees must not expand the DAG exponentially."""
    i = gem.Index()
    evaluation = gem.IndexSum(gem.Literal([1., 2.])[i], (i,))
    expression = evaluation
    for _ in range(8):
        expression = expression * expression
    descent = optimise._product_descent
    visits = []

    def counted(node):
        visits.append(node)
        return descent(node)

    monkeypatch.setattr(optimise, '_product_descent', counted)
    assert optimise.repeated_contractions(expression) == {evaluation}
    assert len(visits) <= 2*len(tuple(traversal((expression,))))


def test_empty_product_contraction_counts_index_tuples():
    i, j = gem.Index(extent=2), gem.Index(extent=3)
    expression = sum_factorise((i, j), ())
    result, = evaluate([expression])
    assert result.arr == 6


def test_contraction_counts_index_absent_from_factors():
    i, j = gem.Index(extent=2), gem.Index(extent=3)
    factor = gem.Indexed(gem.Literal(numpy.arange(3.0)), (j,))

    expression = sum_factorise((i, j), (factor,))
    result, = evaluate([expression])

    assert result.arr == 2 * numpy.arange(3.0).sum()


def test_linear_maps_with_reused_bound_index():
    """Separate reductions may use the same bound index in their maps."""
    i, j, k = gem.indices(3)
    a = gem.Literal(numpy.arange(6.).reshape(2, 3))
    b = gem.Literal(numpy.ones((2, 3)))
    left = gem.IndexSum(a[i, k] + b[i, k], (k,))
    right = gem.IndexSum(a[j, k] + b[j, k], (k,))
    expression = left * right
    arguments = (i, j)
    monomials, = collect_monomials(
        [expression], partial(_classify, frozenset(arguments)), arguments)
    assert len(monomials) == 1
    result = optimise_monomial_sum(monomials, arguments)
    actual, expected = evaluate([gem.ComponentTensor(e, arguments)
                                for e in (result, expression)])
    assert numpy.array_equal(actual.arr, expected.arr)


def test_linear_map_contraction_remains_atomic():
    """A mapped basis is one linear operand, including its sparse row sum."""
    i, j, k, q = gem.indices(4)
    matrix = gem.Literal(numpy.arange(6.).reshape(2, 3))
    table = gem.Literal(numpy.arange(12.).reshape(3, 4))
    left = gem.IndexSum(matrix[i, k] * table[k, q], (k,))
    right = gem.IndexSum(matrix[j, k] * table[k, q], (k,))
    monomials, = collect_monomials(
        [left * right], partial(_classify, frozenset((i, j))), (i, j))
    monomial, = monomials
    assert monomial.sum_indices == ()
    assert len(monomial.atomics) == 2
    assert all(isinstance(a, gem.IndexSum) for a in monomial.atomics)


def test_estimate_cost_counts_tabulation_storage():
    """A shared shaped tabulation occupies its complete tensor domain."""
    i, q = gem.indices(2)
    table = gem.Literal(numpy.arange(6.).reshape(2, 3))
    mapped = gem.ComponentTensor(table[i, q] + gem.Literal(1.), (i,))
    _, storage, largest, _ = cost.estimate_cost([mapped])
    assert storage == largest == 6


@pytest.mark.parametrize('shared_source', [False, True])
def test_indirect_delta_pullback_domain(shared_source):
    """A sparse transpose preserves shared source axes and diagonal lookups."""
    r, a, q = gem.indices(3)
    values = gem.Literal([2., 3.])
    columns = gem.Literal([[0, 1], [0, 1]], dtype=gem.uint_type)
    column = gem.VariableIndex(columns[r, r])
    if shared_source:
        table = gem.Literal(numpy.arange(1., 9.).reshape(2, 2, 2))[a, r, q]
    else:
        table = gem.Literal(numpy.arange(1., 5.).reshape(2, 2))[a, q]
    expression = gem.IndexSum(values[r] * gem.Delta(column, a) * table, (r, a))
    result = optimise.contraction(expression)
    assert result.free_indices == expression.free_indices
    actual, expected = evaluate([result, expression])
    assert numpy.array_equal(actual.arr, expected.arr)


def test_cancel_indirect_delta_through_componenttensor():
    """A tensor view must expose its bound column to Delta cancellation."""
    r, a, b, q = gem.indices(4)
    column = gem.VariableIndex(gem.Literal([1, 0], dtype=gem.uint_type)[r])
    table = gem.Literal(numpy.arange(6.).reshape(2, 3))[a, q]
    view = gem.ComponentTensor(gem.Delta(column, a) * table, (a,))
    expression = gem.IndexSum(view[b], (b,))
    result = optimise.cancel_indirect_deltas(expression)
    assert not any(isinstance(node, gem.Delta) for node in traversal((result,)))
    actual, expected = evaluate([result, expression])
    assert numpy.array_equal(actual.arr, expected.arr)


def reduction_view():
    """A view of the reduction y_m = sum_j A_mj x_j, and its bound index m."""
    m, j = gem.indices(2)
    view = gem.ComponentTensor(gem.IndexSum(gem.Product(
        gem.Indexed(gem.Variable('A', (3, 4)), (m, j)),
        gem.Indexed(gem.Variable('x', (4,)), (j,))), (j,)), (m,))
    return view, m


def stored_tensors(assignments, prefix_ordering):
    """The tensors that compiling the assignments stores."""
    from gem.impero_utils import compile_gem

    impero_c = compile_gem(assignments, prefix_ordering)
    return [t for t in impero_c.temporaries
            if isinstance(t, gem.ComponentTensor)]


def test_reduction_view_is_shared_by_its_readers():
    """Several readers of a view read the reduction without a copy."""
    view, m = reduction_view()
    i, k = gem.indices(2)
    u = gem.Indexed(gem.Variable('u', (3, 3)), (i, k))
    value = gem.Product(gem.Indexed(view, (i,)), gem.Indexed(view, (k,)))
    assert not stored_tensors([(u, value)], (i, k))


def test_reduction_view_read_elsewhere_keeps_its_storage():
    """A loop over the bound index of a view cannot read its other entries."""
    view, m = reduction_view()
    k = gem.Index(extent=3)
    v = gem.Indexed(gem.Variable('v', (3, 3)), (m, k))
    value = gem.Product(gem.Indexed(gem.Variable('w', (3,)), (m,)),
                        gem.Indexed(view, (k,)))
    assert stored_tensors([(v, value)], (m, k))


def test_reduction_view_read_whole_keeps_its_storage():
    """A view read whole, as a FlexiblyIndexed reshape, keeps its storage."""
    view, m = reduction_view()
    i = gem.Index(extent=3)
    u = gem.Indexed(gem.Variable('u', (3,)), (i,))
    value = gem.FlexiblyIndexed(view, ((0, ((i, 1),)),))
    assert stored_tensors([(u, value)], (i,))
