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
    eliminate_deltas,
    preserve_linear_maps,
    sum_factorise,
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
