import pytest
import gem
import numpy


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


@pytest.mark.parametrize('kind', ['abs', 'mixed'])
def test_select_facet_expression(kind):
    """Facet selection preserves node parameters and differing structures."""
    from gem.interpreter import evaluate
    from gem.optimise import select_expression

    x = gem.Variable('x', ())
    facet = gem.Index(extent=2)
    expressions = [gem.MathFunction('abs', x),
                   gem.MathFunction('abs', gem.Product(gem.Literal(4.), x))]
    if kind == 'mixed':
        expressions[1] = gem.Product(gem.Literal(2.), x)
    selected = select_expression(expressions, facet)
    actual, = evaluate([selected], {x: numpy.asarray(9.)})
    expected = evaluate(expressions, {x: numpy.asarray(9.)})
    assert numpy.array_equal(actual.arr, [value.arr for value in expected])


def test_select_shared_expression_dag(monkeypatch):
    """Facet selection visits shared products without expanding the DAG."""
    from gem.optimise import select_expression

    expressions = [gem.Variable('a', ()), gem.Variable('b', ())]
    for _ in range(8):
        expressions = [e*e for e in expressions]
    reconstruct = gem.Product.reconstruct
    visits = []

    def counted(node, *children):
        visits.append(node)
        return reconstruct(node, *children)

    monkeypatch.setattr(gem.Product, 'reconstruct', counted)
    select_expression(expressions, gem.Index(extent=2))
    assert len(visits) <= 8


def test_select_expression_is_deterministic():
    """Selecting equal expressions twice gives equal results."""
    from gem.optimise import select_expression

    x = gem.Variable('x', (3,))
    i, j = gem.indices(2)
    expressions = [
        gem.ComponentTensor(gem.IndexSum(gem.Product(
            gem.Indexed(gem.Variable(name, (2, 3)), (i, j)),
            gem.Indexed(x, (j,))), (j,)), (i,))
        for name in 'AB']
    facet = gem.Index(extent=2)
    assert (select_expression(expressions, facet)
            == select_expression(expressions, facet))


def test_select_expression_avoids_capture():
    """A bound index of one expression may be free in another."""
    from gem.interpreter import evaluate
    from gem.optimise import select_expression

    x = gem.Variable('x', (2,))
    y = gem.Variable('y', (2,))
    i, j = gem.indices(2)
    expressions = [
        gem.ComponentTensor(gem.Product(gem.Indexed(x, (i,)),
                                        gem.Indexed(x, (i,))), (i,)),
        gem.ComponentTensor(gem.Product(
            gem.Indexed(x, (j,)),
            gem.MathFunction('abs', gem.Indexed(y, (i,)))), (j,))]
    facet = gem.Index(extent=2)
    selected = gem.ComponentTensor(
        gem.Indexed(select_expression(expressions, facet), (j,)), (facet, i, j))
    expected = gem.ComponentTensor(
        gem.Indexed(gem.ListTensor(expressions), (facet, j)), (facet, i, j))
    bindings = {x: numpy.asarray([2., 3.]),
                y: numpy.asarray([-5., 7.])}
    actual, expected = evaluate([selected, expected], bindings)
    assert numpy.array_equal(actual.arr, expected.arr)
