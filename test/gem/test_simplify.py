import pytest
import gem
import numpy

from gem import impero
from gem.flop_count import count_flops
from gem.impero_utils import (collect_temporaries, compile_gem,
                              place_declarations)
from gem.node import traversal
from gem.interpreter import evaluate
from gem.optimise import (
    _distribute_sum,
    contraction,
    eliminate_deltas,
    factorisation_group_options,
    hoist_linear_index,
    sum_factorise,
    unflatten_returns,
)
from gem.refactorise import ATOMIC, COMPOUND, OTHER, collect_monomials


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


def test_hoist_linear_index():
    i = gem.Index(extent=3)
    j = gem.Index(extent=3)
    q = gem.Index(extent=2)
    x = gem.Variable("x", (3, 2))
    y = gem.Variable("y", (3, 2))

    def value(index):
        return 2 * gem.Indexed(x, (index, q)) \
            + gem.Indexed(y, (index, q))

    expression = value(i) * value(j)
    result = hoist_linear_index(expression, (i, j))

    tensors = [node for node in traversal((result,))
               if isinstance(node, gem.ComponentTensor)]
    tensor, = tensors
    assert tensor.shape == (3,)
    assert tensor.free_indices == (q,)
    accesses = [node for node in traversal((result,))
                if isinstance(node, gem.Indexed)
                and node.children[0] == tensor]
    assert {access.multiindex for access in accesses} == {(i,), (j,)}

    bindings = {
        x: numpy.arange(6).reshape(3, 2),
        y: numpy.arange(6, 12).reshape(3, 2),
    }
    expected, actual = evaluate([expression, result], bindings)
    assert numpy.array_equal(actual.arr, expected.arr)


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


def test_factorisation_group_options_early_exit():
    """Do not expand a multilinear sum without grouping choices."""
    i = gem.Index(extent=2)
    j = gem.Index(extent=2)
    variables = [gem.Variable(f"a{k}", (2, 2)) for k in range(4)]
    expression = gem.Sum(*(
        gem.Indexed(variable, (i, j)) for variable in variables))

    options = factorisation_group_options(expression, (i, j))

    assert options == ((expression, (), (frozenset(),)),)


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


@pytest.mark.parametrize("rows,cols,data", [
    ([0, 1, 2], [0, 1, 2], [2.0, 2.0, 2.0]),
    ([0, 1, 2], [0, 1, 2], [2.0, 2.0 + 1.0e-10, 2.0]),
    ([0, 1, 2], [0, 1, 2], [2.0, 3.0, 4.0]),
    ([0, 1, 2], [2, 0, 1], [2.0, 3.0, 4.0]),
])
def test_sparse_matrix(rows, cols, data):
    matrix = gem.sparse_matrix((3, 3), rows, cols, data)
    result, = evaluate([matrix])
    expected = numpy.zeros((3, 3))
    expected[rows, cols] = data
    assert numpy.array_equal(result.arr, expected)


def test_sparse_matrix_direct_indices():
    diagonal = gem.sparse_matrix(
        (3, 3), [0, 1, 2], [0, 1, 2], [2.0, 2.0, 2.0])
    assert not any(isinstance(node, gem.VariableIndex)
                   for node in traversal((diagonal,)))

    permutation = gem.sparse_matrix(
        (3, 3), [0, 1, 2], [2, 0, 1], [2.0, 3.0, 4.0])
    indirect = [index
                for node in traversal((permutation,))
                if isinstance(node, gem.Delta)
                for index in (node.i, node.j)
                if isinstance(index, gem.VariableIndex)]
    assert len(indirect) == 1


def test_delta_permutation_bijection():
    i = gem.Index(extent=3)
    j = gem.Index(extent=3)
    entries = numpy.array([2, 0, 1], dtype=gem.uint_type)
    pi = gem.VariableIndex(gem.Indexed(gem.Literal(entries, dtype=gem.uint_type), (i,)))
    pj = gem.VariableIndex(gem.Indexed(
        gem.Literal(entries.copy(), dtype=gem.uint_type), (j,)))

    delta = gem.Delta(pi, pj)
    assert isinstance(delta, gem.Delta)
    assert (delta.i, delta.j) == (i, j)


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


def test_sum_factorise_bounded_distribution():
    indices = tuple(gem.Index(extent=2) for _ in range(6))
    extra = gem.Index(extent=2)

    def unit(index):
        return gem.Indexed(gem.Literal(numpy.ones(2)), (index,))

    factor = gem.Sum(gem.IndexSum(unit(indices[0]) * unit(extra), (extra,)),
                     unit(indices[1]))
    expression = sum_factorise(
        indices, [factor, *(unit(index) for index in indices[2:])],
        distribute=True)
    value, = evaluate([expression])
    assert value.arr == 192


def test_sum_factorise_distribution():
    """Preserve rectangular contraction multiplicity after distribution."""
    indices = tuple(gem.Index(extent=2) for _ in range(2))
    extra = gem.Index(extent=2)

    def unit(index):
        return gem.Indexed(gem.Literal(numpy.ones(2)), (index,))

    factor = gem.Sum(gem.IndexSum(unit(indices[0]) * unit(extra), (extra,)),
                     unit(indices[1]))
    expression = sum_factorise(indices, [factor], distribute=True)
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
    expression = sum_factorise((parent, child), [factor], distribute=True)
    value, = evaluate([expression])
    assert value.arr == 12


def test_refactorise_sum_of_sparse_matvecs():
    left = gem.sparse_matrix((2, 2), [0, 1], [1, 0], [2.0, 3.0])
    right = gem.sparse_matrix((2, 2), [0, 1], [1, 0], [4.0, 5.0])
    vector = gem.Variable("vector", (2,))
    index = gem.Index(extent=2)
    expression = gem.Indexed((left + right) @ vector, (index,))

    def classify(node):
        nodes = set(traversal((node,)))
        if isinstance(node, gem.Indexed) and node.children[0] is vector:
            return ATOMIC
        return COMPOUND if vector in nodes else OTHER

    monomial_sum, = collect_monomials([expression], classify)
    monomials = list(monomial_sum)
    monomial, = monomials
    atomic, = monomial.atomics
    assert atomic.multiindex == monomial.sum_indices
