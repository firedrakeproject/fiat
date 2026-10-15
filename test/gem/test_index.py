import gem
from gem.gem import unique


def test_unique_sorts_by_creation_order():
    """Free indices sort by creation order, whatever their addresses are."""
    indices = [gem.Index() for _ in range(64)]
    shuffled = indices[1::2] + indices[0::2]
    assert unique(shuffled) == tuple(indices)


def test_free_indices_survive_address_reuse():
    """A node orders its free indices the same way whatever preceded it."""
    def build(padding):
        scratch = [gem.Index() for _ in range(padding)]
        del scratch
        i, j = gem.indices(2)
        A = gem.Variable("A", (2, 3))
        return gem.Indexed(A, (j, i)).free_indices == (i, j)

    assert all(build(padding) for padding in range(32))


def test_rename_indirect_componenttensor():
    """Renaming a tensor axis also renames its indirect lookups."""
    from gem.interpreter import evaluate
    from gem.optimise import remove_componenttensors
    import numpy

    i, j = gem.indices(2)
    lookup = gem.VariableIndex(gem.Indexed(
        gem.Literal([2, 0, 1], dtype=gem.uint_type), (i,)))
    expression = gem.Indexed(gem.ComponentTensor(
        gem.Indexed(gem.Literal([4., 5., 6.]), (lookup,)), (i,)), (j,))
    assert expression.free_indices == (j,)
    expression, = remove_componenttensors([expression])
    result, = evaluate([expression])
    assert numpy.array_equal(result.arr, [6., 4., 5.])
