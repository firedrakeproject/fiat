import numpy

from gem import (ComponentTensor, Delta, Index, Indexed, IndexSum, Literal,
                 Node, Product, VariableIndex, as_gem, uint_type)

__all__ = ("sparse_matrix", )


def _sparse_delta(index: Index, entries: numpy.ndarray,
                  nonzero_index: Index) -> Node:
    """Construct a sparse coordinate delta without trivial indirection."""
    if len(entries) == index.extent and numpy.array_equal(
            entries, numpy.arange(index.extent)):
        return Delta(index, nonzero_index)
    entry = VariableIndex(Indexed(Literal(entries, dtype=uint_type),
                                  (nonzero_index,)))
    return Delta(index, entry)


def _coordinate_matrix(shape: tuple[int, int], rows: numpy.ndarray,
                       cols: numpy.ndarray, data: numpy.ndarray) -> Node:
    """Construct a nonempty COO matrix from coordinate deltas."""
    m, n = shape
    i = Index(extent=m)
    j = Index(extent=n)
    nnz, = rows.shape
    p = Index(extent=nnz)
    deltas = Product(_sparse_delta(i, rows, p),
                     _sparse_delta(j, cols, p))

    if data.dtype == object:
        constant = all(value == data[0] for value in data)
        unique_data = data[:1] if constant else data
        positions = numpy.zeros(nnz, dtype=int) if constant \
            else numpy.arange(nnz)
    else:
        unique_data, positions = numpy.unique(data, return_inverse=True)

    if len(unique_data) == 1:
        value = as_gem(unique_data[0])
    elif len(unique_data) == nnz:
        value = Indexed(as_gem(data), (p,))
    else:
        values = as_gem(numpy.asarray(unique_data, dtype=data.dtype))
        position = VariableIndex(Indexed(
            Literal(positions, dtype=uint_type), (p,)))
        value = Indexed(values, (position,))

    return ComponentTensor(IndexSum(Product(value, deltas), (p,)), (i, j))


def _perfect_matching(rows: numpy.ndarray, cols: numpy.ndarray,
                      size: int) -> numpy.ndarray | None:
    """Find one COO entry in every row and column, ordered by column."""
    entries = [[] for _ in range(size)]
    for position, row in enumerate(rows):
        entries[row].append(position)

    column_positions = numpy.full(size, -1, dtype=int)

    def augment(row: int, seen: numpy.ndarray) -> bool:
        for position in entries[row]:
            column = cols[position]
            if seen[column]:
                continue
            seen[column] = True
            previous = column_positions[column]
            if previous < 0 or augment(rows[previous], seen):
                column_positions[column] = position
                return True
        return False

    for row in range(size):
        if not augment(row, numpy.zeros(size, dtype=bool)):
            return None
    return column_positions


def sparse_matrix(shape: tuple[int, int], rows: numpy.ndarray,
                  cols: numpy.ndarray, data: numpy.ndarray) -> Node:
    """Construct a sparse GEM matrix from COO data.

    The returned expression represents
    ``A[i, j] = sum_p data[p] delta(i, rows[p]) delta(j, cols[p])``.

    Parameters
    ----------
    shape : tuple of int
        Matrix dimensions.
    rows, cols : array_like
        Row and column coordinates of the nonzeros.
    data : array_like
        Nonzero values.

    Returns
    -------
    Node
        Rank-two GEM expression.

    """
    rows = numpy.asarray(rows)
    cols = numpy.asarray(cols)
    data = numpy.asarray(data)
    assert rows.shape == cols.shape and rows.ndim == 1
    assert rows.size > 0
    assert data.shape == rows.shape
    m, n = shape
    assert 0 <= rows.min() and rows.max() < m
    assert 0 <= cols.min() and cols.max() < n

    if (m == n and len(rows) == m
            and numpy.array_equal(numpy.sort(rows), numpy.arange(m))
            and numpy.array_equal(numpy.sort(cols), numpy.arange(n))):
        matching = numpy.argsort(cols)
    else:
        matching = _perfect_matching(rows, cols, m) if m == n else None
    if matching is None:
        return _coordinate_matrix(shape, rows, cols, data)

    matched_rows = rows[matching]
    matched_data = data[matching]
    i = Index(extent=m)
    j = Index(extent=n)
    if numpy.array_equal(matched_rows, numpy.arange(m)):
        constant = all(value == matched_data[0] for value in matched_data)
        value = as_gem(matched_data[0]) if constant \
            else Indexed(as_gem(matched_data), (i,))
        matched_matrix = ComponentTensor(Product(value, Delta(i, j)), (i, j))
    else:
        matched_matrix = _coordinate_matrix(
            shape, matched_rows, numpy.arange(n), matched_data)

    if len(matching) == len(rows):
        return matched_matrix
    remainder = numpy.ones(len(rows), dtype=bool)
    remainder[matching] = False
    return matched_matrix + sparse_matrix(
        shape, rows[remainder], cols[remainder], data[remainder])
