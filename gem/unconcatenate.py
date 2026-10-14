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

from gem.node import Memoizer, reuse_if_untouched, traversal
from gem.gem import (ComponentTensor, Concatenate, FlexiblyIndexed,
                     Index, IndexSum, Indexed, Literal, MathFunction, Node,
                     Product, Variable, partial_indexed, reshape, view)
from gem.optimise import (make_rename_map, make_renamer,
                          remove_componenttensors)
from gem.interpreter import evaluate


__all__ = ['flatten', 'split_contraction', 'unconcatenate']


def find_group(expressions, splittable_indices):
    """Finds a full set of indexed Concatenate nodes with the same
    free index, if any such node exists.

    Pre-condition: ComponentTensor nodes surrounding Concatenate nodes
    must be removed.

    :arg expressions: a multi-root GEM expression DAG
    :arg splittable_indices: the indices that may be split along, that is,
        those carried by the assignment variables.  A Concatenate indexed by
        anything else has nothing to be split against, and is left alone.
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


def find_views(expressions: list[Node], index: Index) -> list[Node]:
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


def split_group(cache: dict, concat_group: list[Node]) -> tuple[Index, tuple, list[dict]]:
    """Splits a group of indexed Concatenate nodes into their blocks.

    Parameters
    ----------
    cache
        Index splitting cache :py:class:`dict`.
    concat_group
        A group of indexed :py:class:`Concatenate` nodes, as
        :py:func:`find_group` returns.

    Returns
    -------
    tuple
        The index that the group shares, one multiindex for each block, and
        one substitution for each block.  A substitution replaces every node
        of the group by that block of it.
    """
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

    mappings = [{} for i in range(len(multiindices))]
    for concat_ref in concat_group:
        concat_expr, = concat_ref.children
        for i, multiindex in enumerate(multiindices):
            sub_ref = Indexed(concat_expr.children[i], multiindex)
            sub_ref, = remove_componenttensors((sub_ref,))
            mappings[i][concat_ref] = sub_ref
    return index, multiindices, mappings


def _unconcatenate(cache, pairs):
    # Tail-call recursive core of unconcatenate.
    # Assumes that input has already been sanitised.
    # Only an index carried by an assignment variable can be split against it.
    splittable = set().union(chain(*[v.free_indices for v, e in pairs]))
    concat_group = find_group([v for v, e in pairs] + [e for v, e in pairs], splittable)
    if concat_group is None:
        return pairs

    index, multiindices, mappings = split_group(cache, concat_group)

    def cut(node):
        """No need to rebuild expression of independent of the
        relevant concatenation index."""
        return index not in node.free_indices

    # Finally, split assignment pairs
    split_pairs = []
    for var, expr in pairs:
        if index not in var.free_indices:
            split_pairs.append((var, expr))
        elif isinstance(var, FlexiblyIndexed):
            for v, m in zip(split_variable(var, index, multiindices), mappings):
                split_pairs.append((v, replace_node(expr, m, cut)))
        else:
            # A computed variable is split along its own Concatenate nodes
            for m in mappings:
                v = replace_node(var, m, cut)
                if index in v.free_indices:
                    raise ValueError(f"Cannot split {index} out of {type(var).__name__}")
                split_pairs.append((v, replace_node(expr, m, cut)))

    # Run again, there may be other Concatenate groups
    return _unconcatenate(cache, split_pairs)


def unconcatenate(pairs: list[tuple[Node, Node]], cache: dict | None = None) -> list[tuple[Node, Node]]:
    """Splits (indexed variable, expression) pairs along :py:class:`Concatenate` nodes.

    Each pair is split into one pair per block of the Concatenate nodes
    indexed by an index that its variable carries.  A split pair replaces
    those nodes by one of their blocks, indexed by fresh indices, and its
    variable by the slice for that block, reshaped to the block's shape:

        v_j = f(Concatenate(A, B)_j)  ==>  v_ja = f(A_ja),  v_jb = f(B_jb).

    The split repeats until no such Concatenate is left.  A computed variable
    is split along its own Concatenate nodes rather than sliced.

    Parameters
    ----------
    pairs
        The (indexed variable, expression) pairs.
    cache
        Index splitting cache :py:class:`dict` (optional).

    Returns
    -------
    list
        The split (indexed variable, expression) pairs.
    """
    # Set up cache
    if cache is None:
        cache = {}

    # Eliminate index renaming due to ComponentTensor nodes
    exprs = remove_componenttensors([e for v, e in pairs])
    pairs = [(v, e) for (v, _), e in zip(pairs, exprs)]

    return _unconcatenate(cache, pairs)


def _split_contraction(cache: dict, expression: Node, indices: tuple[Index, ...],
                       contractions: set) -> list[tuple[Node, tuple[Index, ...]]]:
    # Recursive core of split_contraction.
    # Assumes that input has already been sanitised.
    concat_group = find_group([expression], set(indices))
    if concat_group is None:
        return [(expression, indices)]

    index, multiindices, mappings = split_group(cache, concat_group)
    for variable_ref in find_views([expression], index):
        for mapping, sub_ref in zip(mappings, split_variable(variable_ref, index, multiindices)):
            mapping[variable_ref] = sub_ref
    contractions.update(block.multiindex
                        for mapping in mappings for block in mapping.values()
                        if isinstance(block, IndexSum))

    def cut(node):
        """Do not rebuild nodes independent of the concatenation index."""
        return index not in node.free_indices

    rest = tuple(i for i in indices if i != index)
    terms = []
    for multiindex, mapping in zip(multiindices, mappings):
        term = replace_node(expression, mapping, cut)
        if index in term.free_indices:
            raise ValueError(f"Cannot split {index}: it indexes a node that is "
                             "neither a Concatenate nor a variable view.")
        terms.extend(_split_contraction(cache, term, rest + multiindex, contractions))
    return terms


def _lift_contractions(expression: Node, contractions: set, renamer) -> tuple[tuple[Index, ...], Node]:
    # Pull the block contractions out of the product tree of expression,
    # through conj.  A contraction anywhere else is not a factor of the
    # term, so it stays in place.
    if isinstance(expression, IndexSum) and expression.multiindex in contractions:
        indices, applier = renamer(expression.multiindex)
        body, = remove_componenttensors([applier(expression.children[0])])
        lifted, body = _lift_contractions(body, contractions, renamer)
        return indices + lifted, body
    elif isinstance(expression, Product):
        a, b = expression.children
        lifted_a, a = _lift_contractions(a, contractions, renamer)
        lifted_b, b = _lift_contractions(b, contractions, renamer)
        if lifted_a or lifted_b:
            expression = Product(a, b)
        return lifted_a + lifted_b, expression
    elif isinstance(expression, MathFunction) and expression.name == "conj":
        lifted, child = _lift_contractions(expression.children[0], contractions, renamer)
        if lifted:
            expression = MathFunction("conj", child)
        return lifted, expression
    else:
        return (), expression


def split_contraction(expression: Node, indices: tuple[Index, ...],
                      cache: dict | None = None) -> list[tuple[Node, tuple[Index, ...]]]:
    """Splits a contraction along the :py:class:`Concatenate` nodes it sums over.

    The sum is split into one sum per block of the Concatenate nodes indexed
    by a summed index.  A split term replaces those nodes by one of their
    blocks, indexed by fresh indices, and sums over those indices instead:

        sum_j f(Concatenate(A, B)_j)  ==>  sum_ja f(A_ja) + sum_jb f(B_jb).

    The split repeats until no such Concatenate is left.  A variable view
    carrying the split index is sliced into the same blocks.  A block that is
    itself a sum, as in the dual evaluation of a direct sum, has its summation
    indices moved to its term when the block is a factor of the term.  They
    come first in the term's indices, as points come before basis functions.

    Parameters
    ----------
    expression
        A scalar GEM expression.
    indices
        The indices that ``expression`` is summed over.
    cache
        Index splitting cache :py:class:`dict` (optional).

    Returns
    -------
    list
        (expression, indices) pairs whose sums add up to the sum of
        ``expression`` over ``indices``.
    """
    if cache is None:
        cache = {}
    expression, = remove_componenttensors([expression])
    contractions = set()
    terms = _split_contraction(cache, expression, tuple(indices), contractions)

    result = []
    for term, summed in terms:
        # A lifted index must differ from every index in the term, free or
        # bound, so that it neither aliases nor shadows one of them.
        taken = set(summed).union(term.free_indices)
        taken.update(chain.from_iterable(node.multiindex for node in traversal([term])
                                         if isinstance(node, IndexSum)))
        renamer = make_renamer(make_rename_map())
        renamer(tuple(taken))
        lifted, term = _lift_contractions(term, contractions, renamer)
        result.append((term, lifted + summed))
    return result


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
