import pytest
import gem
import numpy

from gem import impero
from gem.coffee import monomial_sum_to_expression
from gem.flop_count import count_flops
from gem.impero_utils import (collect_temporaries, compile_gem,
                              place_declarations)
from gem.node import traversal
from gem.interpreter import evaluate
from gem.optimise import (
    _distribute_sum,
    contraction,
    eliminate_deltas,
    preserve_linear_maps,
    unflatten_returns,
)
from gem.refactorise import (ATOMIC, COMPOUND, OTHER,
                             collect_monomials)


@pytest.fixture
def A():
    a = gem.Variable("a", ())
    b = gem.Variable("b", ())
    c = gem.Variable("c", ())
    d = gem.Variable("d", ())
    array = [[a, b], [c, d]]
    A = gem.ListTensor(array)
    return A


@pytest.fixture
def X():
    return gem.Variable("X", (2, 2))


def test_listtensor_from_indexed(X):
    k = gem.Index()
    elems = [gem.Indexed(X, (k, *i)) for i in numpy.ndindex(X.shape[1:])]
    tensor = gem.ListTensor(numpy.reshape(elems, X.shape[1:]))

    assert isinstance(tensor, gem.ComponentTensor)
    j = tensor.multiindex
    expected = gem.partial_indexed(X, (k,))
    expected = gem.ComponentTensor(gem.Indexed(expected, j), j)
    assert tensor == expected


def test_listtensor_from_fixed_indexed(A):
    elems = [gem.Indexed(A, i) for i in numpy.ndindex(A.shape)]
    tensor = gem.ListTensor(numpy.reshape(elems, A.shape))
    assert tensor == A


def test_listtensor_from_partial_indexed(A):
    elems = [gem.partial_indexed(A, i) for i in numpy.ndindex(A.shape[:1])]
    tensor = gem.ListTensor(elems)
    assert tensor == A


def test_nested_partial_indexed(A):
    i, j = gem.indices(2)
    B = gem.partial_indexed(gem.partial_indexed(A, (i,)), (j,))
    assert B == gem.Indexed(A, (i, j))


def test_componenttensor_from_indexed(A):
    i, j = gem.indices(2)
    Aij = gem.Indexed(A, (i, j))
    assert A == gem.ComponentTensor(Aij, (i, j))


def test_componenttensor_from_diagonal():
    i, = gem.indices(1)
    a = gem.Variable("a", (2,))
    b = gem.Variable("b", (2,))
    rows = gem.ListTensor([gem.Indexed(a, (i,)), gem.Indexed(b, (i,))])
    # Indexing rows on its diagonal, the index is bound by the ComponentTensor
    diagonal = gem.Indexed(rows, (i,))
    assert gem.ComponentTensor(diagonal, (i,)).free_indices == ()

    # Unrolling the trace substitutes the index in both slots
    expr, = gem.optimise.unroll_indexsum([gem.IndexSum(diagonal, (i,))],
                                         predicate=lambda index: True)
    result, = gem.optimise.remove_componenttensors([expr])
    assert result == gem.Sum(gem.Indexed(a, (0,)), gem.Indexed(b, (1,)))


def test_componenttensor_flop_count():
    i = gem.Index(extent=3)
    j = gem.Index(extent=3)
    x = gem.Variable("x", (3,))
    result = gem.Variable("result", (3,))
    tensor = gem.ComponentTensor(2 * gem.Indexed(x, (i,)), (i,))
    expression = gem.Indexed(tensor, (j,))
    impero_c = compile_gem(
        [(gem.Indexed(result, (j,)), expression)], (j,))

    assert count_flops(impero_c) == 6


def test_componenttensor_sharing_uses_scalar_temporary():
    """Keep shared work inside a component tensor's value loop."""
    i = gem.Index(extent=3)
    j = gem.Index(extent=3)
    x = gem.Variable("x", (3,))
    result = gem.Variable("result", (3,))
    shared = 2 * gem.Indexed(x, (i,))
    positive = gem.ComponentTensor(shared + 1, (i,))
    negative = gem.ComponentTensor(shared - 1, (i,))
    expression = gem.Indexed(positive, (j,)) \
        + gem.Indexed(negative, (j,))

    impero_c = compile_gem(
        [(gem.Indexed(result, (j,)), expression)], (j,))

    assert shared in impero_c.temporaries
    assert impero_c.indices[shared] == ()


def test_indexed_transpose(A):
    i, j = gem.indices(2)
    ATij = gem.Indexed(A.T, (i, j))
    Aji = gem.Indexed(A, (j, i))
    assert ATij == Aji

    i, = gem.indices(1)
    j = 1
    ATij = gem.Indexed(A.T, (i, j))
    Aji = gem.Indexed(A, (j, i))
    assert ATij == Aji

    i, j = (0, 1)
    ATij = gem.Indexed(A.T, (i, j))
    Aji = gem.Indexed(A, (j, i))
    assert ATij == Aji


def test_double_transpose(A):
    assert A.T.T == A


def test_flatten_indexsum(A):
    i, j = gem.indices(2)
    Aij = gem.Indexed(A, (i, j))

    result = gem.IndexSum(gem.IndexSum(Aij, (i,)), (j,))
    expected = gem.IndexSum(Aij, (i, j))
    assert result == expected


def test_selective_distribution():
    a = gem.Variable("a", ())
    b = gem.Variable("b", ())
    c = gem.Variable("c", ())
    i = gem.Index(extent=2)
    p = gem.Index(extent=1)
    row = gem.VariableIndex(gem.Indexed(
        gem.Literal([0], dtype=gem.uint_type), (p,)))
    delta = gem.Delta(i, row)
    common = gem.Sum(a, b)
    expression = gem.Product(common, gem.Sum(c, delta))

    terms = _distribute_sum(
        expression, predicate=lambda node: isinstance(node, gem.Delta))

    assert len(terms) == 2
    assert all(common in set(traversal((term,))) for term in terms)


def test_preserve_linear_maps_early_exit():
    """Keep a multilinear sum that contains no separate linear maps."""
    i = gem.Index(extent=2)
    j = gem.Index(extent=2)
    variables = [gem.Variable(f"a{k}", (2, 2)) for k in range(4)]
    expression = gem.Sum(*(
        gem.Indexed(variable, (i, j)) for variable in variables))

    terms, linear_maps = preserve_linear_maps(expression, (i, j))

    assert terms == (expression,)
    assert linear_maps == ()


def test_collect_monomials_preserves_linear_maps():
    """Keep finite element linear maps intact during factorization."""
    i = gem.Index(extent=2)
    j = gem.Index(extent=2)
    left = gem.Sum(
        gem.Indexed(gem.Literal([1.0, 2.0]), (i,)),
        gem.Indexed(gem.Literal([3.0, 5.0]), (i,)))
    right = gem.Sum(
        gem.Indexed(gem.Literal([7.0, 11.0]), (j,)),
        gem.Indexed(gem.Literal([13.0, 17.0]), (j,)))
    expression = left * right
    linear_indices = frozenset((i, j))

    def classifier(node: gem.Node) -> str:
        support = linear_indices.intersection(node.free_indices)
        if not support:
            return OTHER
        if isinstance(node, gem.Indexed):
            return ATOMIC
        return COMPOUND

    monomial_sum, = collect_monomials(
        (expression,), classifier, linear_indices)

    monomial, = tuple(monomial_sum)
    assert frozenset(monomial.atomics) == frozenset((left, right))
    expected, = evaluate([gem.ComponentTensor(expression, (i, j))])
    actual, = evaluate([gem.ComponentTensor(
        monomial_sum_to_expression(monomial_sum), (i, j))])
    assert numpy.array_equal(actual.arr, expected.arr)


def test_constant_variable_index():
    index = gem.VariableIndex(gem.Literal(1, dtype=gem.uint_type))
    assert index == 1


def test_place_declarations_counts_equal_impero_subtrees():
    """Equal Impero nodes are distinct occurrences in the loop tree."""
    expression = gem.Variable("a", ()) * gem.Variable("b", ())
    tree = impero.Block([
        impero.Evaluate(expression),
        impero.Evaluate(expression),
    ])
    temporaries = collect_temporaries(tree)

    declare, indices = place_declarations(
        tree, temporaries, lambda node: node.free_indices)

    assert declare[tree] == []
    assert indices[expression] == ()
    assert all(declare[statement] for statement in tree.children)


def test_delta_elimination_preserves_indirect_free_index():
    i = gem.Index(extent=4)
    k = gem.Index(extent=2)
    entries = numpy.array([1, 3], dtype=gem.uint_type)
    indirect = gem.VariableIndex(gem.Indexed(
        gem.Literal(entries, dtype=gem.uint_type), (k,)))
    values = gem.Literal([2.0, 3.0, 5.0, 7.0])
    expression = gem.IndexSum(
        gem.Delta(i, indirect) * gem.Indexed(values, (i,)), (i,))

    result = eliminate_deltas(expression)
    assert result.free_indices == (k,)
    actual, = evaluate([result])
    assert numpy.array_equal(actual.arr, values.array[entries])


def test_unflatten_compatible_returns_together():
    extent = 3
    p = gem.Index(extent=extent)
    q = gem.JaggedIndex(extent=extent, parents=(p,))
    X = gem.Variable("X", (extent, extent))
    Y = gem.Variable("Y", (extent, extent))
    ft_x = gem.FlattenedTensor(gem.Indexed(X, (p, q)), (p, q))
    ft_y = gem.FlattenedTensor(gem.Indexed(Y, (p, q)), (p, q))

    r = gem.Index(extent=6)
    result = gem.Variable("result", (6,))
    pairs = unflatten_returns([
        (gem.Indexed(result, (r,)),
         gem.Sum(gem.Indexed(ft_x, (r,)), gem.Indexed(ft_y, (r,))))
    ])

    assert len(pairs) == 1
    variable, expression = pairs[0]
    assert variable.free_indices == expression.free_indices
    assert len(variable.free_indices) == 2
    assert not any(isinstance(node, gem.FlattenedTensor)
                   for node in traversal((expression,)))


def test_unflatten_bijective_return_index():
    """Invert a compile-time permutation before exposing a return lattice."""
    extent = 3
    p = gem.Index(extent=extent)
    q = gem.JaggedIndex(extent=extent, parents=(p,))
    X = gem.Variable("X", (extent, extent))
    table = gem.FlattenedTensor(
        gem.Indexed(X, (p, q)), (p, q))

    r = gem.Index(extent=6)
    permutation = numpy.asarray([2, 0, 5, 1, 4, 3], dtype=gem.uint_type)
    mapped = gem.Indexed(table, (gem.VariableIndex(gem.Indexed(
        gem.Literal(permutation, dtype=gem.uint_type), (r,))),))
    output = gem.Variable("output", (6,))
    variable, optimized = unflatten_returns([
        (gem.Indexed(output, (r,)), mapped)
    ])[0]

    assert len(variable.free_indices) == 2
    scatter_expression = variable.multiindex[0].expression
    assert scatter_expression.children[0].shape == (extent, extent)
    assert not any(isinstance(node, gem.FlattenedTensor)
                   for node in traversal((optimized,)))

    values = numpy.arange(extent * extent, dtype=float).reshape(
        extent, extent)
    expected, = evaluate([mapped], {X: values})
    actual, scatter = evaluate(
        [optimized, variable.multiindex[0].expression], {X: values})
    points = table.lattice_points()
    coordinates = {
        index: points[:, position]
        for position, index in enumerate(variable.free_indices)
    }

    def sample(value):
        return value.arr[tuple(coordinates[index]
                               for index in value.fids)]

    dense = numpy.empty(6)
    dense[sample(scatter)] = sample(actual)
    assert numpy.array_equal(dense, expected.broadcast((r,)))


def test_unflatten_factorises_local_sum():
    extent = 3
    p = gem.JaggedIndex(extent=extent)
    q = gem.JaggedIndex(extent=extent, parents=(p,))
    ip, iq = gem.indices(2)
    A = gem.Variable("A", (extent, 2))
    B = gem.Variable("B", (extent, extent, 2))
    C = gem.Variable("C", (extent, 2))
    D = gem.Variable("D", (extent, extent, 2))
    lattice = gem.Sum(
        gem.Product(gem.Indexed(A, (p, ip)), gem.Indexed(B, (p, q, iq))),
        gem.Product(gem.Indexed(C, (p, ip)), gem.Indexed(D, (p, q, iq))),
    )
    table = gem.FlattenedTensor(lattice, (p, q))

    r = gem.Index(extent=6)
    w = gem.Variable("w", (6,))
    expression = gem.IndexSum(
        gem.Product(gem.Indexed(table, (r,)), gem.Indexed(w, (r,))), (r,))
    result = contraction(expression)

    sums = [node for node in traversal((result,))
            if isinstance(node, gem.IndexSum)]
    assert sums
    assert all(len(node.multiindex) == 1 for node in sums)
    assert not any(isinstance(node, gem.FlattenedTensor)
                   for node in traversal((result,)))


def test_unflatten_factorises_bilinear_arguments_together():
    """Two argument lattices are exposed before their local sums expand."""
    extent = 3
    p = gem.JaggedIndex(extent=extent)
    q = gem.JaggedIndex(extent=extent, parents=(p,))
    r = gem.JaggedIndex(extent=extent)
    s = gem.JaggedIndex(extent=extent, parents=(r,))
    ip, iq = gem.indices(2)

    variables = tuple(
        gem.Variable(name, shape)
        for name, shape in [
            ("A", (extent, 2)),
            ("B", (extent, extent, 2)),
            ("C", (extent, 2)),
            ("D", (extent, extent, 2)),
            ("E", (extent, 2)),
            ("F", (extent, extent, 2)),
            ("G", (extent, 2)),
            ("H", (extent, extent, 2)),
        ])
    A, B, C, D, E, F, G, H = variables
    left = gem.FlattenedTensor(gem.Sum(
        gem.Product(gem.Indexed(A, (p, ip)),
                    gem.Indexed(B, (p, q, iq))),
        gem.Product(gem.Indexed(C, (p, ip)),
                    gem.Indexed(D, (p, q, iq))),
    ), (p, q))
    right = gem.FlattenedTensor(gem.Sum(
        gem.Product(gem.Indexed(E, (r, ip)),
                    gem.Indexed(F, (r, s, iq))),
        gem.Product(gem.Indexed(G, (r, ip)),
                    gem.Indexed(H, (r, s, iq))),
    ), (r, s))

    i, j = gem.indices(2)
    output = gem.Variable("output", (6, 6))
    expression = gem.IndexSum(
        gem.Product(gem.Indexed(left, (i,)),
                    gem.Indexed(right, (j,))),
        (ip, iq))
    pairs = unflatten_returns([
        (gem.Indexed(output, (i, j)), expression)
    ])

    assert len(pairs) == 1
    variable, optimized = pairs[0]
    assert len(variable.free_indices) == 4
    assert all(isinstance(index, gem.JaggedIndex)
               for index in variable.free_indices)
    assert not any(isinstance(node, gem.FlattenedTensor)
                   for node in traversal((optimized,)))
    assert all(len(node.multiindex) == 1
               for node in traversal((optimized,))
               if isinstance(node, gem.IndexSum))

    rng = numpy.random.default_rng(2)
    bindings = {
        variable_: rng.random(variable_.shape)
        for variable_ in variables
    }
    expected, = evaluate([expression], bindings)
    actual, = evaluate([optimized], bindings)
    points = left.lattice_points()
    row_map, column_map = evaluate([
        index.expression for index in variable.multiindex
    ])
    row = row_map.arr[points[:, 0], points[:, 1]]
    column = column_map.arr[points[:, 0], points[:, 1]]
    row_indices = {
        index: points[:, position, None]
        for position, index in enumerate(row_map.fids)
    }
    column_indices = {
        index: points[None, :, position]
        for position, index in enumerate(column_map.fids)
    }
    indices = tuple((row_indices | column_indices)[index]
                    for index in actual.fids)
    values = actual.arr[indices]
    dense = numpy.empty((6, 6))
    dense[row[:, None], column[None, :]] = values
    assert numpy.allclose(dense, expected.broadcast((i, j)))


def test_distribute_sum_preserves_rectangular_multiplicity():
    """Preserve rectangular contraction multiplicity after distribution."""
    indices = tuple(gem.Index(extent=2) for _ in range(2))
    extra = gem.Index(extent=2)

    def unit(index):
        return gem.Indexed(gem.Literal(numpy.ones(2)), (index,))

    factor = gem.Sum(gem.IndexSum(unit(indices[0]) * unit(extra), (extra,)),
                     unit(indices[1]))
    expression = gem.IndexSum(factor, indices)
    terms = _distribute_sum(
        expression, predicate=lambda node: isinstance(node, gem.Sum))
    expression = gem.Sum(*terms)
    value, = evaluate([expression])
    assert value.arr == 12


def test_sum_factorise_jagged_distribution():
    """Preserve the joint jagged domain after distribution."""
    parent = gem.JaggedIndex(extent=3)
    child = gem.JaggedIndex(extent=3, parents=(parent,))

    def unit(index: gem.Index) -> gem.Node:
        """Return a unit vector carrying one free index.

        Parameters
        ----------
        index
            Free index of the vector.

        Returns
        -------
        gem.Node
            Indexed unit vector.
        """
        return gem.Indexed(gem.Literal(numpy.ones(3)), (index,))

    triangle = numpy.fromfunction(
        lambda i, j: j < 3 - i, (3, 3), dtype=int)
    factor = gem.Sum(
        unit(parent), gem.Indexed(gem.Literal(triangle), (parent, child)))
    expression = gem.IndexSum(factor, (parent, child))
    terms = _distribute_sum(
        expression, predicate=lambda node: isinstance(node, gem.Sum))
    expression = gem.Sum(*terms)
    value, = evaluate([expression])
    assert value.arr == 12


def test_literal_distinguishes_dtypes():
    """Tell an index literal apart from a value literal.

    An index table holds unsigned integers and a coefficient table holds
    floats. GEM memoizes on node identity, so the two must not compare
    equal when they happen to hold the same number.
    """
    index = gem.Literal(numpy.uint32(3), dtype=gem.uint_type)
    value = gem.Literal(3.0)

    assert index.dtype != value.dtype
    assert index != value
    assert hash(index) != hash(value)
    assert {index: "index"}.get(value) is None


def test_rename_index_under_variable_index():
    """Renaming a bound index must reach the lookup of an indirect gather."""
    values = gem.Literal(numpy.array([10.0, 20.0, 30.0]))
    lookup = gem.Literal(numpy.array([2, 0, 1], dtype=gem.uint_type),
                         dtype=gem.uint_type)

    k, kp = gem.indices(2)
    gather = gem.Indexed(
        values, (gem.VariableIndex(gem.Indexed(lookup, (k,))),))
    assert k in gather.free_indices

    # This is how make_renamer separates two sums that bind the same index.
    renamed = gem.Indexed(gem.ComponentTensor(gather, (k,)), (kp,))
    assert renamed.free_indices == (kp,)


def test_product_of_sums_over_one_index():
    """Two sums that bind the same index expand to a double sum, not one."""
    from gem.interpreter import evaluate
    from gem.optimise import make_rename_map, make_renamer

    values = gem.Literal(numpy.array([1.0, 2.0, 3.0]))
    lookup = gem.Literal(numpy.array([2, 0, 1], dtype=gem.uint_type),
                         dtype=gem.uint_type)
    k, = gem.indices(1)
    gather = gem.Indexed(
        values, (gem.VariableIndex(gem.Indexed(lookup, (k,))),))

    renamer = make_renamer(make_rename_map())
    (k0,), first = renamer((k,))
    (k1,), second = renamer((k,))
    assert k0 != k1

    square = gem.IndexSum(gem.Product(first(gather), second(gather)), (k0, k1))
    result, = evaluate([square])
    assert numpy.isclose(result.arr, (1.0 + 2.0 + 3.0) ** 2)
