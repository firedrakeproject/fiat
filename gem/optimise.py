"""A set of routines implementing various transformations on GEM
expressions."""

from collections import OrderedDict, defaultdict
from functools import singledispatch, partial
from itertools import combinations, permutations
from math import prod
from numbers import Integral

import numpy

from gem.utils import groupby
from gem.node import (Memoizer, MemoizerArg, reuse_if_untouched,
                      reuse_if_untouched_arg, traversal)
from gem.gem import (Node, Failure, Identity, Constant, Literal, Zero,
                     Product, Sum, Comparison, Conditional, Division,
                     Index, VariableIndex, Indexed, FlexiblyIndexed,
                     IndexSum, ComponentTensor, ListTensor, Delta,
                     partial_indexed, one)


@singledispatch
def literal_rounding(node, self):
    """Perform FFC rounding of FIAT tabulation matrices on the literals of
    a GEM expression.

    :arg node: root of the expression
    :arg self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


literal_rounding.register(Node)(reuse_if_untouched)


@literal_rounding.register(Literal)
def literal_rounding_literal(node, self):
    table = node.array
    epsilon = self.epsilon
    # Mimic the rounding applied at COFFEE formatting, which in turn
    # mimics FFC formatting.
    one_decimal = numpy.asarray(numpy.round(table, 1))
    one_decimal[numpy.logical_not(one_decimal)] = 0  # no minus zeros
    return Literal(numpy.where(abs(table - one_decimal) < epsilon, one_decimal, table))


def ffc_rounding(expression, epsilon):
    """Perform FFC rounding of FIAT tabulation matrices on the literals of
    a GEM expression.

    :arg expression: GEM expression
    :arg epsilon: tolerance limit for rounding
    """
    mapper = Memoizer(literal_rounding)
    mapper.epsilon = epsilon
    return mapper(expression)


@singledispatch
def _replace_division(node, self):
    """Replace division with multiplication

    :param node: root of expression
    :param self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


_replace_division.register(Node)(reuse_if_untouched)


@_replace_division.register(Division)
def _replace_division_division(node, self):
    a, b = node.children
    return Product(self(a), Division(one, self(b)))


def replace_division(expressions):
    """Replace divisions with multiplications in expressions"""
    mapper = Memoizer(_replace_division)
    return list(map(mapper, expressions))


@singledispatch
def replace_indices(node, self, subst):
    """Replace free indices in a GEM expression.

    :arg node: root of the expression
    :arg self: function for recursive calls
    :arg subst: tuple of pairs; each pair is a substitution
                rule with a free index to replace and an index to
                replace with.
    """
    raise AssertionError("cannot handle type %s" % type(node))


replace_indices.register(Node)(reuse_if_untouched_arg)


def _replace_indices_atomic(i, self, subst):
    if isinstance(i, VariableIndex):
        new_expr = self(i.expression, subst)
        return i if new_expr == i.expression else VariableIndex(new_expr)
    else:
        substitute = dict(subst)
        return substitute.get(i, i)


@replace_indices.register(Delta)
def replace_indices_delta(node, self, subst):
    i = _replace_indices_atomic(node.i, self, subst)
    j = _replace_indices_atomic(node.j, self, subst)
    if i == node.i and j == node.j:
        return node
    else:
        return Delta(i, j)


@replace_indices.register(Indexed)
def replace_indices_indexed(node, self, subst):
    multiindex = tuple(_replace_indices_atomic(i, self, subst) for i in node.multiindex)
    child, = node.children

    if isinstance(child, ComponentTensor):
        # Indexing into ComponentTensor
        # Inline ComponentTensor and augment the substitution rules
        substitute = dict(subst)
        substitute.update(zip(child.multiindex, multiindex))
        return self(child.children[0], tuple(sorted(substitute.items())))
    else:
        # Replace indices
        child = self(child, subst)

        # Remove fixed indices
        if isinstance(child, (Constant, ListTensor)):
            if all(isinstance(i, Integral) for i in multiindex):
                # All indices fixed
                sub = child.array[multiindex]
                child = Literal(sub, dtype=child.dtype) if isinstance(child, Constant) else sub
                multiindex = ()

            elif any(isinstance(i, Integral) for i in multiindex):
                # Some indices fixed
                slices = tuple(i if isinstance(i, Integral) else slice(None) for i in multiindex)
                sub = child.array[slices]
                child = Literal(sub, dtype=child.dtype) if isinstance(child, Constant) else ListTensor(sub)
                multiindex = tuple(i for i in multiindex if not isinstance(i, Integral))

        if multiindex == node.multiindex and child == node.children[0]:
            return node
        else:
            return Indexed(child, multiindex)


@replace_indices.register(FlexiblyIndexed)
def replace_indices_flexiblyindexed(node, self, subst):
    dim2idxs = tuple(
        (
            offset if isinstance(offset, Integral) else _replace_indices_atomic(offset, self, subst),
            tuple((_replace_indices_atomic(i, self, subst), s if isinstance(s, Integral) else self(s, subst)) for i, s in idxs)
        )
        for offset, idxs in node.dim2idxs
    )

    child, = node.children
    assert not child.free_indices
    if dim2idxs == node.dim2idxs:
        return node
    else:
        return FlexiblyIndexed(child, dim2idxs)


@replace_indices.register(IndexSum)
def replace_indices_indexsum(node, self, subst):
    targets = set()
    for _, replacement in subst:
        if isinstance(replacement, Index):
            targets.add(replacement)
        elif isinstance(replacement, VariableIndex):
            targets.update(replacement.expression.free_indices)

    # A replacement can introduce a free index with the name of a local sum.
    renames = {i: Index(extent=i.extent) for i in node.multiindex if i in targets}
    indices = tuple(renames.get(i, i) for i in node.multiindex)
    child, = node.children
    summand = self(child, tuple(sorted((*subst, *renames.items()))))
    if not renames and summand == child:
        return node

    # Specialising a ListTensor entry can make the summand invariant in an index.
    active = tuple(i for i in indices if i in summand.free_indices)
    extent = prod(i.extent for i in indices if i not in summand.free_indices)
    return Product(Literal(extent, dtype=summand.dtype), IndexSum(summand, active))


def filtered_replace_indices(node, self, subst):
    """Wrapper for :func:`replace_indices`.  At each call removes
    substitution rules that do not apply."""
    if any(isinstance(k, VariableIndex) for k, _ in subst):
        raise NotImplementedError("Can not replace VariableIndex (will need inverse)")
    filtered_subst = tuple((k, v) for k, v in subst if k in node.free_indices)
    return replace_indices(node, self, filtered_subst)


def remove_componenttensors(expressions, subst=()):
    """Removes all ComponentTensors in multi-root expression DAG."""
    mapper = MemoizerArg(filtered_replace_indices)
    return [mapper(expression, subst) for expression in expressions]


@singledispatch
def _constant_fold_zero(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_constant_fold_zero.register(Node)(reuse_if_untouched)


@_constant_fold_zero.register(Literal)
def _constant_fold_zero_literal(node, self):
    if not node.array.any():
        # A table of any shape that holds only zeros is a symbolic zero.
        return Zero(node.shape, dtype=node.dtype)
    else:
        return node


@_constant_fold_zero.register(ListTensor)
def _constant_fold_zero_listtensor(node, self):
    new_children = list(map(self, node.children))
    if all(isinstance(nc, Zero) for nc in new_children):
        return Zero(node.shape)
    elif new_children == node.children:
        return node
    else:
        return node.reconstruct(*new_children)


def constant_fold_zero(exprs):
    """Produce symbolic zeros from Literals

    :arg exprs: An iterable of gem expressions.
    :returns: A list of gem expressions where any Literal containing
        only zeros is replaced by symbolic Zero of the appropriate
        shape.

    We need a separate path for ListTensor so that its `reconstruct`
    method will not be called when the new children are `Zero()`s;
    otherwise Literal `0`s would be reintroduced.
    """
    mapper = Memoizer(_constant_fold_zero)
    return list(map(mapper, exprs))


def _select_expression(expressions, index, cache):
    """Helper function to select an expression from a list of
    expressions with an index.  This function expect sanitised input,
    one should normally call :py:func:`select_expression` instead.

    :arg expressions: a tuple of expressions
    :arg index: an index (free, fixed or variable)
    :arg cache: a dict from the tuples of expressions already selected
        with this index to their selection, so that subexpressions shared
        within the DAGs are selected once
    :returns: an expression
    """
    try:
        return cache[expressions]
    except KeyError:
        return cache.setdefault(expressions, _select_expression_uncached(expressions, index, cache))


def _select_expression_uncached(expressions, index, cache):
    expr = expressions[0]
    if all(e == expr for e in expressions):
        return expr

    types = set(map(type, expressions))
    if types <= {Indexed, Zero}:
        multiindex, = set(e.multiindex for e in expressions if isinstance(e, Indexed))
        # Shape only determined by free indices
        shape = tuple(i.extent for i in multiindex if isinstance(i, Index))

        def child(expression):
            if isinstance(expression, Indexed):
                return expression.children[0]
            elif isinstance(expression, Zero):
                return Zero(shape)
        return Indexed(_select_expression(tuple(map(child, expressions)), index, cache), multiindex)

    if types <= {Literal, Zero, Failure}:
        return partial_indexed(ListTensor(expressions), (index,))

    if types <= {ComponentTensor, Zero}:
        shape, = set(e.shape for e in expressions)
        multiindex = tuple(Index(extent=d) for d in shape)
        children = remove_componenttensors([Indexed(e, multiindex) for e in expressions])
        return ComponentTensor(_select_expression(tuple(children), index, cache), multiindex)

    if types == {Delta}:
        if all(e.i == k and e.j == expr.j for k, e in enumerate(expressions)):
            return expr.reconstruct(index, expr.j)
        elif all(e.j == k and e.i == expr.i for k, e in enumerate(expressions)):
            return expr.reconstruct(expr.i, index)

    if len(types) == 1:
        cls, = types
        if cls.__front__ or cls.__back__:
            raise NotImplementedError("How to factorise {} expressions?".format(cls.__name__))
        assert all(len(e.children) == len(expr.children) for e in expressions)
        assert len(expr.children) > 0

        return expr.reconstruct(*(_select_expression(nth_children, index, cache)
                                  for nth_children in zip(*(e.children
                                                            for e in expressions))))

    raise NotImplementedError("No rule for factorising expressions of this kind.")


def select_expression(expressions, index):
    """Select an expression from a list of expressions with an index.
    Semantically equivalent to

        partial_indexed(ListTensor(expressions), (index,))

    but has a much more optimised implementation.

    :arg expressions: a list of expressions of the same shape
    :arg index: an index (free, fixed or variable)
    :returns: an expression of the same shape as the given expressions
    """
    # Check arguments
    shape = expressions[0].shape
    assert all(e.shape == shape for e in expressions)

    # Sanitise input expressions
    alpha = tuple(Index() for s in shape)
    exprs = remove_componenttensors([Indexed(e, alpha) for e in expressions])

    # Factor the expressions recursively and convert result
    selected = _select_expression(tuple(exprs), index, {})
    return ComponentTensor(selected, alpha)


def delta_elimination(sum_indices, factors, index_replacer=None):
    """IndexSum-Delta cancellation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :kwarg index_replacer: MemoizerArg(filtered_replace_indices)

    :returns: optimised (sum_indices, factors)
    """
    if index_replacer is None:
        index_replacer = MemoizerArg(filtered_replace_indices)

    sum_indices = list(sum_indices)  # copy for modification

    def substitute(expression, from_, to_):
        if from_ not in expression.free_indices:
            return expression
        elif isinstance(expression, Delta):
            return index_replacer(expression, ((from_, to_),))
        else:
            return Indexed(ComponentTensor(expression, (from_,)), (to_,))

    delta_queue = [(f, index)
                   for f in factors if isinstance(f, Delta)
                   for index in (f.i, f.j) if index in sum_indices]
    while delta_queue:
        delta, from_ = delta_queue[0]
        to_, = list({delta.i, delta.j} - {from_})

        sum_indices.remove(from_)

        factors = [substitute(f, from_, to_) for f in factors]

        delta_queue = [(f, index)
                       for f in factors if isinstance(f, Delta)
                       for index in (f.i, f.j) if index in sum_indices]

    return sum_indices, factors


def associate(operator, operands):
    """Apply associativity rules to construct an operation-minimal expression tree.

    For best performance give factors that have different set of free indices.

    :arg operator: associative binary operator
    :arg operands: list of operands

    :returns: (reduced expression, # of floating-point operations)
    """
    def count(pair):
        """Operation count to reduce a pair of GEM expressions"""
        a, b = pair
        extents = [i.extent for i in set().union(a.free_indices, b.free_indices)]
        return numpy.prod(extents, dtype=int)

    flops = 0
    if len(operands) > 32:
        # A linear pass bounds compile time for large products and sums.
        result, *rest = operands
        for operand in rest:
            flops += count((result, operand))
            result = operator(result, operand)
        return result, flops

    while len(operands) > 1:
        # Greedy algorithm: choose a pair of operands that are the
        # cheapest to reduce.
        a, b = min(combinations(operands, 2), key=count)
        flops += count((a, b))
        # Remove chosen factors, append their product
        operands.remove(a)
        operands.remove(b)
        operands.append(operator(a, b))
    result, = operands
    return result, flops


def _independent_contractions(sum_indices, groups):
    """Split a contraction into independent subproblems.

    Two contraction indices only interact if some factor carries both of
    them, so the factors and the indices form a graph whose connected
    components can be contracted separately.

    :arg sum_indices: free indices for contractions
    :arg groups: product factors, grouped by free indices
    :returns: a pair of the list of (indices, groups) subproblems and the
              list of groups carrying no contraction index
    """
    # Union-find over the contraction indices
    parent = {index: index for index in sum_indices}

    def find(index):
        if parent[index] == index:
            return index

        parent[index] = find(parent[index])
        return parent[index]

    index_set = set(sum_indices)
    shared = [[i for i in group.free_indices if i in index_set] for group in groups]
    for indices in shared:
        for index in indices[1:]:
            root, other = find(indices[0]), find(index)
            if root != other:
                parent[other] = root

    subproblems = OrderedDict((find(index), ([], [])) for index in sum_indices)
    for index in sum_indices:
        subproblems[find(index)][0].append(index)

    rest = []
    for group, indices in zip(groups, shared):
        if indices:
            subproblems[find(indices[0])][1].append(group)
        else:
            rest.append(group)
    return list(subproblems.values()), rest


def _contract(terms, sum_index):
    """Sum one index out of a list of product terms.

    :arg terms: list of product terms
    :arg sum_index: the index to contract
    :returns: (terms after the contraction, its flop count)
    """
    # Select terms that need to be part of the contraction
    contract = [t for t in terms if sum_index in t.free_indices]
    deferred = [t for t in terms if sum_index not in t.free_indices]

    # Optimise associativity
    if contract:
        product, flops = associate(Product, contract)
        term = IndexSum(product, (sum_index,))
        flops += numpy.prod([i.extent for i in product.free_indices], dtype=int)
    else:
        term, flops = Literal(sum_index.extent), 0

    # Replace the contracted terms with the result of the contraction.
    return deferred + [term], flops


def _contract_in_order(sum_indices, groups):
    """Contract indices in the given order.

    :returns: (GEM expression, its flop count)
    """
    terms = groups[:]
    flops = 0
    for sum_index in sum_indices:
        terms, flops_ = _contract(terms, sum_index)
        flops += flops_

    # If some contraction indices were independent, then we may still
    # have several terms at this point.
    expression, flops_ = associate(Product, terms)
    return expression, flops + flops_


def _contract_greedily(sum_indices, groups):
    """Contract indices cheapest first.

    :returns: (GEM expression, its flop count)
    """
    remaining = list(sum_indices)
    ordering = []
    terms = groups[:]
    while remaining:
        sum_index = min(remaining, key=lambda index: _contract(terms, index)[1])
        remaining.remove(sum_index)
        ordering.append(sum_index)
        terms, _ = _contract(terms, sum_index)
    return _contract_in_order(ordering, groups)


def _sum_factorise_connected(sum_indices, groups):
    """Sum factorise a single connected contraction.

    Contraction orderings are searched exhaustively while that is
    affordable, and chosen greedily when it is not.

    :arg sum_indices: free indices for contractions, which must not split
                      into independent subproblems
    :arg groups: product factors, grouped by free indices
    :returns: optimised GEM expression
    """
    if len(sum_indices) > 6:
        # Exhaustive search costs a factorial in the number of indices.
        expression, _ = _contract_greedily(sum_indices, groups)
        return expression

    expression = None
    best_flops = numpy.inf

    # Consider all orderings of contraction indices
    for ordering in permutations(sum_indices):
        expr, flops = _contract_in_order(ordering, groups)
        if flops < best_flops:
            expression = expr
            best_flops = flops

    return expression


def sum_factorise(sum_indices, factors):
    """Optimise a tensor product through sum factorisation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :returns: optimised GEM expression
    """
    if len(factors) == 0 and len(sum_indices) == 0:
        # Empty product
        return one

    # Form groups by free indices
    groups = groupby(factors, key=lambda f: f.free_indices)
    groups = [Product(*terms) for _, terms in groups]

    # Contractions that share no factor are independent of each other, so
    # factorise them separately rather than searching the orderings that
    # interleave them.
    subproblems, terms = _independent_contractions(sum_indices, groups)
    terms = terms + [_sum_factorise_connected(indices, subgroups)
                     for indices, subgroups in subproblems]
    if not terms:
        return one
    expression, _ = associate(Product, terms)
    return expression


def make_sum(summands):
    """Constructs an operation-minimal sum of GEM expressions."""
    groups = groupby(summands, key=lambda f: f.free_indices)
    summands = [Sum(*terms) for _, terms in groups]
    result, flops = associate(Sum, summands)
    return result


def make_product(factors, sum_indices=()):
    """Constructs an operation-minimal (tensor) product of GEM expressions."""
    return sum_factorise(sum_indices, factors)


def make_rename_map():
    """Creates an rename map for reusing the same index renames."""
    return defaultdict(Index)


def make_renamer(rename_map):
    r"""Creates a function for renaming indices when expanding products of
    IndexSums, i.e. applying to following rule:

        (\sum_i a_i)*(\sum_i b_i) ===> \sum_{i,i'} a_i*b_{i'}

    :arg rename_map: An rename map for renaming indices the same way
                     as functions returned by other calls of this
                     function.
    :returns: A function that takes an iterable of indices to rename,
              and returns (renamed indices, applier), where applier is
              a function that remap the free indices of GEM
              expressions from the old to the new indices.
    """
    def _renamer(rename_map, current_set, incoming):
        renamed = []
        renames = []
        for i in incoming:
            j = i
            while j in current_set:
                j = rename_map[j]
            current_set.add(j)
            renamed.append(j)
            if i != j:
                renames.append((i, j))

        if renames:
            def applier(expr):
                pairs = [(i, j) for i, j in renames if i in expr.free_indices]
                if pairs:
                    current, renamed = zip(*pairs)
                    return Indexed(ComponentTensor(expr, current), renamed)
                else:
                    return expr
        else:
            applier = lambda expr: expr

        return tuple(renamed), applier
    return partial(_renamer, rename_map, set())


def traverse_product(expression, stop_at=None, renamer=None, index_replacer=None):
    """Traverses a product tree and collects factors, also descending into
    tensor contractions (IndexSum).  The numerators of divisions are
    also broken up, but not the denominators.

    :arg expression: a GEM expression
    :arg stop_at: Optional predicate on GEM expressions.  If specified
                  and returns true for some subexpression, that
                  subexpression is not broken into further factors
                  even if it is a product-like expression.
    :arg renamer: Optional renamer from :py:func:`make_renamer`.  Pass one
                  that has already seen the indices bound outside
                  ``expression``, so that the contracted indices hoisted
                  out of it stay distinct from those.
    :kwarg index_replacer: MemoizerArg(filtered_replace_indices)

    :returns: (sum_indices, terms)
              - sum_indices: list of indices to sum over
              - terms: list of product terms
    """
    if renamer is None:
        renamer = make_renamer(make_rename_map())
        renamer(expression.free_indices)
    if index_replacer is None:
        index_replacer = MemoizerArg(filtered_replace_indices)

    sum_indices = []
    terms = []

    stack = [expression]
    while stack:
        expr = stack.pop()
        if stop_at is not None and stop_at(expr):
            terms.append(expr)
        elif isinstance(expr, IndexSum):
            indices, applier = renamer(expr.multiindex)
            sum_indices.extend(indices)
            stack.extend(index_replacer(applier(c), ()) for c in expr.children)
        elif isinstance(expr, Product):
            stack.extend(reversed(expr.children))
        elif isinstance(expr, Division):
            # Break up products in the dividend, but not in divisor.
            dividend, divisor = expr.children
            if dividend == one:
                terms.append(expr)
            else:
                stack.append(Division(one, divisor))
                stack.append(dividend)
        else:
            terms.append(expr)

    return sum_indices, terms


def traverse_sum(expression, stop_at=None):
    """Traverses a summation tree and collects summands.

    :arg expression: a GEM expression
    :arg stop_at: Optional predicate on GEM expressions.  If specified
                  and returns true for some subexpression, that
                  subexpression is not broken into further summands
                  even if it is an addition.
    :returns: list of summand expressions
    """
    stack = [expression]
    result = []
    while stack:
        expr = stack.pop()
        if stop_at is not None and stop_at(expr):
            result.append(expr)
        elif isinstance(expr, Sum):
            stack.extend(reversed(expr.children))
        else:
            result.append(expr)
    return result


def contraction(expression, *, argument_indices=()):
    """Cancel Deltas and sum-factorise the tensor product at the root.

    Parameters
    ----------
    expression : Node
        Scalar GEM expression to optimise.
    argument_indices : iterable of Index, optional
        Free indices used by subsequent argument factorisation.

    Returns
    -------
    Node
        An equivalent expression with optimised contractions.
    """

    # Common memoizer to remove ComponentTensors
    index_replacer = MemoizerArg(filtered_replace_indices)

    # Eliminate annoying ComponentTensors
    expression = index_replacer(expression, ())

    def decompose(expression):
        sum_indices, factors = traverse_product(expression, index_replacer=index_replacer)
        sum_indices, factors = delta_elimination(sum_indices, factors, index_replacer=index_replacer)
        return sum_indices, [index_replacer(f, ()) for f in factors]

    # Deltas can remove the only link between a ListTensor and another factor.
    sum_indices, factors = decompose(expression)
    eligible = set(expression.free_indices)
    argument_indices = frozenset(argument_indices)

    def find_list_indices(node, recurse):
        allowed = eligible.intersection(node.free_indices)
        if not allowed:
            return ()

        # At each node, discard indices that are bound in that scope.
        children = node.children
        if isinstance(node, (Indexed, FlexiblyIndexed)):
            children += node.indirect_children
        indices = OrderedDict.fromkeys(i for child in children for i in recurse(child) if i in allowed)
        if isinstance(node, Indexed) and isinstance(node.children[0], ListTensor):
            indices.update((i, None) for i in node.multiindex if i in allowed)
        return tuple(indices)

    list_indices = Memoizer(find_list_indices)

    def rebuild(indices, group):
        axes = tuple(OrderedDict.fromkeys(i for f in group for i in list_indices(f)))
        if not axes:
            return sum_factorise(indices, group)

        # Argument factorisation needs to see coupled argument axes and Deltas.
        arguments = argument_indices.intersection(i for f in group for i in f.free_indices)
        if len(arguments) > 1 or (arguments and any(isinstance(node, Delta) for node in traversal(group))):
            return sum_factorise(indices, group)

        tensor = ComponentTensor(IndexSum(Product(*group), indices), axes)
        entries = [sum_factorise(*decompose(index_replacer(Indexed(tensor, zeta), ())))
                   for zeta in numpy.ndindex(tensor.shape)]
        return Indexed(ListTensor(numpy.array(entries).reshape(tensor.shape)), axes)

    if not any(list_indices(factor) for factor in factors):
        return sum_factorise(sum_indices, factors)

    # A shared free index permits pointwise multiplication outside the ListTensor.
    groups, terms = _independent_contractions(sum_indices, factors)
    terms = [rebuild([], [factor]) for factor in terms]
    terms.extend(rebuild(indices, group) for indices, group in groups)
    return sum_factorise((), terms)


@singledispatch
def _replace_delta(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_replace_delta.register(Node)(reuse_if_untouched)


@_replace_delta.register(Delta)
def _replace_delta_delta(node, self):
    i, j = node.i, node.j

    if isinstance(i, Index) or isinstance(j, Index):
        if isinstance(i, Index) and isinstance(j, Index):
            assert i.extent == j.extent
        if isinstance(i, Index):
            assert i.extent is not None
            size = i.extent
        if isinstance(j, Index):
            assert j.extent is not None
            size = j.extent
        return Indexed(Identity(size), (i, j))
    else:
        def expression(index):
            if isinstance(index, Integral):
                return Literal(index)
            elif isinstance(index, VariableIndex):
                return index.expression
            else:
                raise ValueError("Cannot convert running index to expression.")
        e_i = expression(i)
        e_j = expression(j)
        return Conditional(Comparison("==", e_i, e_j), one, Zero())


def replace_delta(expressions):
    """Lowers all Deltas in a multi-root expression DAG."""
    mapper = Memoizer(_replace_delta)
    return list(map(mapper, expressions))


@singledispatch
def _unroll_indexsum(node, self):
    """Unrolls IndexSums below a certain extent.

    :arg node: root of the expression
    :arg self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


_unroll_indexsum.register(Node)(reuse_if_untouched)


@_unroll_indexsum.register(IndexSum)  # noqa
def _(node, self):
    unroll = tuple(filter(self.predicate, node.multiindex))
    if unroll:
        # Unrolling
        summand = self(node.children[0])
        shape = tuple(index.extent for index in unroll)
        tensor = ComponentTensor(summand, unroll)
        unrolled = Sum(*(Indexed(tensor, alpha) for alpha in numpy.ndindex(shape)))
        return IndexSum(unrolled, tuple(index for index in node.multiindex
                                        if index not in unroll))
    else:
        return reuse_if_untouched(node, self)


def unroll_indexsum(expressions, predicate):
    """Unrolls IndexSums below a specified extent.

    :arg expressions: list of expression DAGs
    :arg predicate: a predicate function on :py:class:`Index` objects
                    that tells whether to unroll a particular index
    :returns: list of expression DAGs with some unrolled IndexSums
    """
    mapper = Memoizer(_unroll_indexsum)
    mapper.predicate = predicate
    return list(map(mapper, expressions))


def aggressive_unroll(expression):
    """Aggressively unrolls all loop structures."""
    # Unroll expression shape
    if expression.shape:
        tensor = numpy.empty(expression.shape, dtype=object)
        for alpha in numpy.ndindex(expression.shape):
            tensor[alpha] = Indexed(expression, alpha)
        expression, = remove_componenttensors((ListTensor(tensor),))

    # Unroll summation
    expression, = unroll_indexsum((expression,), predicate=lambda index: True)
    expression, = remove_componenttensors((expression,))
    return expression
