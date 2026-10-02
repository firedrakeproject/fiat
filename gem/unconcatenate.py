"""Utility functions for decomposing Concatenate nodes.

The exported functions are flatten and unconcatenate.
- flatten: destroys the structure preserved within Concatenate nodes,
           essentially reducing FInAT provided tabulations to what
           FIAT could have provided, so old code can continue to work.
- unconcatenate: split up (variable, expression) pairs along
                 Concatenate nodes, thus recovering the structure
                 within them, yet eliminating the Concatenate nodes.

Let us see an example on unconcatenate.  Let us consider the form

    div(v) * dx

where v is an RTCF7 test function.  This means that the assembled
local vector has 8 * 7 + 7 * 8 = 112 entries.  So the compilation of
the form starts with a single assignment pair [(v, e)].  v is now the
indexed return variable, something equivalent to

    Indexed(Variable('A', (112,)), (j,))

where j is the basis function index of the argument.  e is just a GEM
quadrature expression with j as its only free index.  This will
contain the tabulation of the RTCF7 element, which will cause
something like

    C_j := Indexed(Concatenate(A, B), (j,))

to appear as a subexpression in e.  unconcatenate splits e along C_j
into e_1 and e_2 such that

    e_1 := e /. C_j -> A_{ja1,ja2}, and
    e_2 := e /. C_j -> B_{jb1,jb2}.

The split indices ja1, ja2, jb1, and jb2 have extents 8, 7, 7, and 8
respectively (see the RTCF7 element construction above).  So the
result of unconcatenate will be the list of pairs

     [(v_1, e_2), (v_2, e_2)]

where v_1 is the first 56 entries of v, reshaped as an 8 x 7 matrix,
indexed with (ja1, ja2), and similarly, v_2 is the second 56 entries
of v, reshaped as a 7 x 8 matrix, indexed with (jb1, jb2).

The unconcatenated form allows for sum factorisation of tensor product
elements as usual.  This pair splitting is also applicable to
coefficient evaluation: take the local basis function coefficients as
the variable, the FInAT tabulation of the element as the expression,
and apply "matrix-vector multifunction" for each pair after
unconcatenation, and then add up the results.
"""

from functools import singledispatch
from itertools import chain

import numpy

from gem.node import Memoizer, reuse_if_untouched
from gem.gem import (ComponentTensor, Concatenate, FlexiblyIndexed,
                     Index, Indexed, Literal, Node, partial_indexed,
                     Variable, reshape, view)
from gem.optimise import remove_componenttensors
from gem.interpreter import evaluate


__all__ = ['flatten', 'unconcatenate']


def find_group(expressions, splittable_indices):
    """Finds a full set of indexed Concatenate nodes with the same
    free index, if any such node exists.

    Pre-condition: ComponentTensor nodes surrounding Concatenate nodes
    must be removed.

    :arg expressions: a multi-root GEM expression DAG
    :arg splittable_indices: the indices that may be split along, that is,
        those carried by the assignment variables or summed over.  A
        Concatenate indexed by anything else has nothing to be split against,
        and is left alone.
    :returns: a list of GEM nodes, or None
    """
    # Result variables
    index = None
    nodes = []

    # Sui generis pre-order traversal so that we can avoid going
    # unnecessarily deep in the DAG.
    seen = set()
    lifo = []
    for root in expressions:
        if root not in seen:
            seen.add(root)
            lifo.append(root)

    while lifo:
        node = lifo.pop()
        if not splittable_indices.intersection(node.free_indices):
            continue

        if isinstance(node, Indexed):
            child, = node.children
            if isinstance(child, Concatenate):
                i, = node.multiindex
                if i in splittable_indices and (index or i) == i:
                    index = i
                    nodes.append(node)
                    # Skip adding children
                    continue

        for child in reversed(node.children):
            if child not in seen:
                seen.add(child)
                lifo.append(child)

    return index and nodes


def is_variable_view(node: Node) -> bool:
    """Whether a node indexes into a variable, which split_variable can slice."""
    return (isinstance(node, FlexiblyIndexed)
            or (isinstance(node, Indexed) and isinstance(node.children[0], Variable)))


def find_views(expressions: list, index: Index) -> list:
    """Finds the indexed variables that carry a concatenation index.

    Parameters
    ----------
    expressions
        A multi-root GEM expression DAG.
    index
        The :py:class:`Concatenate` index to split along.

    Returns
    -------
    list
        The variable views, as :py:func:`is_variable_view` defines them, whose
        free indices include ``index``.
    """
    views = []
    seen = set()
    lifo = []
    for root in expressions:
        if root not in seen:
            seen.add(root)
            lifo.append(root)

    while lifo:
        node = lifo.pop()
        if index not in node.free_indices:
            continue
        if is_variable_view(node):
            views.append(node)
            continue
        for child in reversed(node.children):
            if child not in seen:
                seen.add(child)
                lifo.append(child)
    return views


def split_variable(variable_ref, index, multiindices):
    """Splits an indexed variable along a concatenation index.

    :param variable_ref: indexed variable to split
    :param index: :py:class:`Concatenate` index to split along
    :param multiindices: one multiindex for each split variable

    :returns: generator of split indexed variables
    """
    assert is_variable_view(variable_ref)
    other_indices = list(variable_ref.index_ordering())
    other_indices.remove(index)
    other_indices = tuple(other_indices)
    data = ComponentTensor(variable_ref, (index,) + other_indices)
    slices = [slice(None)] * len(other_indices)
    shapes = [(other_index.extent,) for other_index in other_indices]

    offset = 0
    for multiindex in multiindices:
        shape = tuple(index.extent for index in multiindex)
        size = numpy.prod(shape, dtype=int)
        slice_ = slice(offset, offset + size)
        offset += size

        sub_ref = Indexed(reshape(view(data, slice_, *slices),
                                  shape, *shapes),
                          multiindex + other_indices)
        sub_ref, = remove_componenttensors((sub_ref,))
        yield sub_ref


def _replace_node(node, self):
    """Replace subexpressions using a given mapping.

    :param node: root of expression
    :param self: function for recursive calls
    """
    assert isinstance(node, Node)
    if self.cut(node):
        return node
    try:
        return self.mapping[node]
    except KeyError:
        return reuse_if_untouched(node, self)


def replace_node(expression, mapping, cut=None):
    """Replace subexpressions using a given mapping.

    :param expression: a GEM expression
    :param mapping: a :py:class:`dict` containing the substitutions
    :param cut: cutting predicate; if returns true, it is assumed that
                no replacements would take place in the subexpression.
    """
    mapper = Memoizer(_replace_node)
    mapper.mapping = mapping
    mapper.cut = cut or (lambda node: False)
    return mapper(expression)


def _unconcatenate(cache, terms):
    # Tail-call recursive core of unconcatenate.
    # Assumes that input has already been sanitised.
    # Each term is (variable, expression, summed indices), where the variable
    # may be None.  Only an index carried by an assignment variable or summed
    # over can be split against it.
    splittable = set(chain.from_iterable(
        chain(() if v is None else v.free_indices, s) for v, e, s in terms))
    concat_group = find_group([e for v, e, s in terms], splittable)
    if concat_group is None:
        return terms

    # Get the index split
    concat_ref = next(iter(concat_group))
    assert isinstance(concat_ref, Indexed)
    concat_expr, = concat_ref.children
    index, = concat_ref.multiindex
    assert isinstance(concat_expr, Concatenate)
    try:
        multiindices = cache[index]
    except KeyError:
        multiindices = tuple(tuple(Index(extent=d) for d in child.shape)
                             for child in concat_expr.children)
        cache[index] = multiindices

    def cut(node):
        """No need to rebuild expression of independent of the
        relevant concatenation index."""
        return index not in node.free_indices

    # Build Concatenate node replacement mappings
    mappings = [{} for i in range(len(multiindices))]
    for concat_ref in concat_group:
        concat_expr, = concat_ref.children
        for i in range(len(multiindices)):
            sub_ref = Indexed(concat_expr.children[i], multiindices[i])
            sub_ref, = remove_componenttensors((sub_ref,))
            mappings[i][concat_ref] = sub_ref

    # Slice the variable views that carry the index into the same blocks
    roots = [e for v, e, s in terms] + [v for v, e, s in terms if v is not None]
    for variable_ref in find_views(roots, index):
        for m, sub_ref in zip(mappings, split_variable(variable_ref, index, multiindices)):
            m[variable_ref] = sub_ref

    # Finally, split assignment terms
    split_terms = []
    for var, expr, summed in terms:
        if index not in summed and (var is None or index not in var.free_indices):
            split_terms.append((var, expr, summed))
            continue
        rest = tuple(i for i in summed if i != index)
        for multiindex, m in zip(multiindices, mappings):
            v = None if var is None else replace_node(var, m, cut)
            e = replace_node(expr, m, cut)
            if index in e.free_indices:
                raise ValueError(f"Cannot split {index}: it indexes a node that is "
                                 "neither a Concatenate nor a variable view.")
            split_terms.append((v, e, rest + multiindex if index in summed else rest))

    # Run again, there may be other Concatenate groups
    return _unconcatenate(cache, split_terms)


def unconcatenate(pairs: list, cache: dict | None = None, indices: tuple = ()) -> list:
    """Splits (indexed variable, expression) pairs along :py:class:`Concatenate` nodes.

    A pair splits along a Concatenate index that its variable carries, or
    that its expression is summed over.  A sum over a direct sum is the sum of
    the sums over its blocks:

        sum_j Indexed(Concatenate(A, B), (j,)) * Indexed(Concatenate(C, D), (j,))
            = sum_{ja} A_ja * C_ja + sum_{jb} B_jb * D_jb.

    All Concatenate nodes indexed by one index must contain the same blocks.
    FInAT guarantees this: an element blocks its tabulation and dual basis
    along the same summands.  A view of a variable, such as an assignment
    variable or a dat of basis coefficients, is sliced into the same blocks.

    Parameters
    ----------
    pairs
        A list of (indexed variable, expression) pairs.  The variable may be
        None if the expression is only summed.
    cache
        Index splitting cache :py:class:`dict` (optional).
    indices
        The indices that every expression is summed over (optional).

    Returns
    -------
    list
        The split (indexed variable, expression) pairs.  A split replaces an
        index of ``indices`` by the indices of each block, so the sums over
        the free indices of the split expressions add up to the sums of the
        input expressions over ``indices``.
    """
    # Set up cache
    if cache is None:
        cache = {}

    # Eliminate index renaming due to ComponentTensor nodes
    exprs = remove_componenttensors([e for v, e in pairs])
    terms = _unconcatenate(cache, [(v, e, tuple(indices)) for (v, _), e in zip(pairs, exprs)])
    return [(v, e) for v, e, s in terms]


@singledispatch
def _flatten(node, self):
    """Replace Concatenate nodes with Literal nodes.

    :arg node: root of the expression
    :arg self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


_flatten.register(Node)(reuse_if_untouched)


@_flatten.register(Concatenate)
def _flatten_concatenate(node, self):
    result, = evaluate([node])
    return partial_indexed(Literal(result.arr), result.fids)


def flatten(expressions):
    """Flatten Concatenate nodes, and destroy the structure they express.

    :arg expressions: a multi-root expression DAG
    """
    mapper = Memoizer(_flatten)
    return list(map(mapper, expressions))
