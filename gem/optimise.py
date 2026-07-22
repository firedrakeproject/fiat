"""A set of routines implementing various transformations on GEM
expressions."""

from collections import OrderedDict, defaultdict
from functools import singledispatch, partial
from itertools import combinations, permutations, zip_longest
from numbers import Integral

import numpy

from gem.utils import groupby
from gem.node import (Memoizer, MemoizerArg, reuse_if_untouched,
                      reuse_if_untouched_arg, traversal)
from gem.gem import (Node, Failure, Identity, Constant, Literal, Zero,
                     Product, Sum, Comparison, Conditional, Division,
                     Index, JaggedIndex, VariableIndex, Indexed,
                     FlexiblyIndexed, IndexSum, ComponentTensor, ListTensor,
                     Delta, FlattenedTensor, partial_indexed, one, uint_type)


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
    if numpy.array_equal(node.array, 0):
        # All zeros, make symbolic zero
        return Zero(node.shape)
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


def _select_expression(expressions, index):
    """Helper function to select an expression from a list of
    expressions with an index.  This function expect sanitised input,
    one should normally call :py:func:`select_expression` instead.

    :arg expressions: a list of expressions
    :arg index: an index (free, fixed or variable)
    :returns: an expression
    """
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
        return Indexed(_select_expression(list(map(child, expressions)), index), multiindex)

    if types <= {Literal, Zero, Failure}:
        return partial_indexed(ListTensor(expressions), (index,))

    if types <= {ComponentTensor, Zero}:
        shape, = set(e.shape for e in expressions)
        multiindex = tuple(Index(extent=d) for d in shape)
        children = remove_componenttensors([Indexed(e, multiindex) for e in expressions])
        return ComponentTensor(_select_expression(children, index), multiindex)

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

        return expr.reconstruct(*(_select_expression(nth_children, index)
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
    selected = _select_expression(exprs, index)
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
        if isinstance(to_, VariableIndex):
            # `from_` is not simply dropped: the substitution below
            # exposes `to_`'s wrapped free indices (e.g. a lattice
            # multiindex inside a Morton index expression) wherever
            # `from_` used to appear, so they must join the sum indices
            # too, or later free-index bookkeeping (e.g. the recomputed
            # ``variable.free_indices``) will no longer match.
            for index in to_.expression.free_indices:
                if index not in sum_indices:
                    sum_indices.append(index)

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
    if len(operands) > 32:
        # O(N^3) algorithm
        raise NotImplementedError("Not expected such a complicated expression!")

    def count(pair):
        """Operation count to reduce a pair of GEM expressions"""
        a, b = pair
        extents = [i.extent for i in set().union(a.free_indices, b.free_indices)]
        return numpy.prod(extents, dtype=int)

    flops = 0
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


def sum_factorise(sum_indices, factors):
    """Optimise a tensor product through sum factorisation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :returns: optimised GEM expression
    """
    if len(factors) == 0 and len(sum_indices) == 0:
        # Empty product
        return one

    if len(sum_indices) > 6:
        raise NotImplementedError("Too many indices for sum factorisation!")

    # Form groups by free indices
    groups = groupby(factors, key=lambda f: f.free_indices)
    groups = [Product(*terms) for _, terms in groups]

    # Sum factorisation
    expression = None
    best_flops = numpy.inf

    # Consider all orderings of contraction indices
    for ordering in permutations(sum_indices):
        terms = groups[:]
        flops = 0
        # Apply contraction index by index
        for sum_index in ordering:
            # Select terms that need to be part of the contraction
            contract = [t for t in terms if sum_index in t.free_indices]
            deferred = [t for t in terms if sum_index not in t.free_indices]

            # Optimise associativity
            product, flops_ = associate(Product, contract)
            term = IndexSum(product, (sum_index,))
            flops += flops_ + numpy.prod([i.extent for i in product.free_indices], dtype=int)

            # Replace the contracted terms with the result of the
            # contraction.
            terms = deferred + [term]

        # If some contraction indices were independent, then we may
        # still have several terms at this point.
        expr, flops_ = associate(Product, terms)
        flops += flops_

        if flops < best_flops:
            expression = expr
            best_flops = flops

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


def traverse_product(expression, stop_at=None, rename_map=None, index_replacer=None):
    """Traverses a product tree and collects factors, also descending into
    tensor contractions (IndexSum).  The numerators of divisions are
    also broken up, but not the denominators.

    :arg expression: a GEM expression
    :arg stop_at: Optional predicate on GEM expressions.  If specified
                  and returns true for some subexpression, that
                  subexpression is not broken into further factors
                  even if it is a product-like expression.
    :arg rename_map: an rename map for consistent index renaming
    :kwarg index_replacer: MemoizerArg(filtered_replace_indices)

    :returns: (sum_indices, terms)
              - sum_indices: list of indices to sum over
              - terms: list of product terms
    """
    if rename_map is None:
        rename_map = make_rename_map()
    renamer = make_renamer(rename_map)
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


def _clone_multiindex(multiindex):
    """Fresh copies of a (possibly jagged) multiindex, rebuilding the
    parent structure within the tuple."""
    clones = {}
    for index in multiindex:
        if isinstance(index, JaggedIndex):
            parents = tuple(clones.get(p, p) for p in index.parents)
            clones[index] = JaggedIndex(extent=index.extent, parents=parents)
        else:
            clones[index] = Index(extent=index.extent)
    return tuple(clones[index] for index in multiindex)


def _replace_indices_unflatten(node, self, subst):
    """`filtered_replace_indices` that additionally replaces one designated
    `Indexed` node with a given expression."""
    if node == self.gather:
        return self.replacement
    return filtered_replace_indices(node, self, subst)


def _find_flat_gather(expression, r):
    """The unique Indexed(FlattenedTensor, (r,)) in ``expression``, or None
    if there is none or more than one."""
    gathers = set(n for n in traversal((expression,))
                  if isinstance(n, Indexed) and n.multiindex == (r,)
                  and isinstance(n.children[0], FlattenedTensor))
    if len(gathers) != 1:
        return None
    gather, = gathers
    return gather


def _unflatten_site(gather):
    """Prepare the rewrite of one FlattenedTensor gather site: a mapper that
    replaces the gather node with the tensor's expression at a fresh copy
    ``alpha`` of the lattice multiindex (fresh because the tensor, and its
    bound multiindex, may be shared between sites), and a `VariableIndex`
    substitute for remaining uses of the flat index, gathering through the
    ordering table."""
    ft, = gather.children
    alpha = _clone_multiindex(ft.multiindex)
    index_replacer = MemoizerArg(filtered_replace_indices)
    replacement = index_replacer(ft.children[0], tuple(zip(ft.multiindex, alpha)))
    mapper = MemoizerArg(_replace_indices_unflatten)
    mapper.gather = gather
    mapper.replacement = replacement
    return mapper, alpha, VariableIndex(Indexed(ft.ordering, alpha))


def _unflatten(node, self):
    node = reuse_if_untouched(node, self)
    if not isinstance(node, IndexSum):
        return node
    summand, = node.children
    for r in node.multiindex:
        gather = _find_flat_gather(summand, r)
        if gather is None:
            continue
        mapper, alpha, r_expr = _unflatten_site(gather)
        summand = mapper(summand, ((r, r_expr),))
        rest = tuple(i for i in node.multiindex if i != r)
        return self(IndexSum(summand, rest + alpha))
    return node


def unflatten(expression):
    """Rewrite contractions over the flat index of a `FlattenedTensor` as
    loops over its jagged lattice multiindex.  The flat index disappears:
    the tensor's expression is inlined at its lattice point, and remaining
    uses of the flat index (e.g. a coefficient vector) gather through the
    tensor's ordering table.  This recovers the separable per-axis factors
    that sum factorisation needs, which the flat gather form hides."""
    mapper = Memoizer(_unflatten)
    return mapper(expression)


def unflatten_returns(pairs):
    """`unflatten` for the free (argument) indices of assignment pairs.

    For each (return variable, expression) pair whose expression scatters a
    `FlattenedTensor` along a free index of the variable, replace that flat
    index by the jagged lattice multiindex in both: the variable's index
    becomes an ordering-table gather, and the tensor's expression is inlined
    at its lattice point.  Zero-padding makes the out-of-bounds lattice
    iterations (if a consumer does not tighten the loops) accumulate zero.
    Rewritten expressions are re-factorised with `contraction`, since the
    factorised structure was built while the tensor was still atomic.
    """
    result = []
    for variable, expression in pairs:
        changed = False
        rewrite = True
        while rewrite:
            rewrite = False
            for j in variable.free_indices:
                gather = _find_flat_gather(expression, j)
                if gather is None:
                    continue
                mapper, alpha, r_expr = _unflatten_site(gather)
                subst = ((j, r_expr),)
                variable = MemoizerArg(filtered_replace_indices)(variable, subst)
                expression = mapper(expression, subst)
                changed = rewrite = True
                break
        if changed:
            expression = make_sum([contraction(s) for s in traverse_sum(expression)])
        result.append((variable, expression))
    return result


def _replace_flattened(node, self):
    node = reuse_if_untouched(node, self)
    if not isinstance(node, FlattenedTensor):
        return node
    expression, = node.children
    points = node.lattice_points()
    n, = node.shape
    r = Index(extent=n)
    subst = tuple(
        (axis, VariableIndex(Indexed(Literal(points[:, t], dtype=uint_type), (r,))))
        for t, axis in enumerate(node.multiindex))
    body = MemoizerArg(filtered_replace_indices)(expression, subst)
    return ComponentTensor(body, (r,))


def replace_flattened(expressions):
    """Lower remaining `FlattenedTensor`s to `ComponentTensor`s over a plain
    flat index, substituting each lattice axis with a gather table on the
    flat index -- jagged access instead of jagged loops."""
    mapper = Memoizer(_replace_flattened)
    return [mapper(expression) for expression in expressions]


def contraction(expression, ignore=None):
    """Optimise the contractions of the tensor product at the root of
    the expression, including:

    - IndexSum-Delta cancellation
    - Sum factorisation

    :arg ignore: Optional set of indices to ignore when applying sum
        factorisation (otherwise all summation indices will be
        considered). Use this if your expression has many contraction
        indices.

    This routine was designed with finite element coefficient
    evaluation in mind.
    """

    # Common memoizer to remove ComponentTensors
    index_replacer = MemoizerArg(filtered_replace_indices)

    # Eliminate annoying ComponentTensors
    expression = index_replacer(expression, ())

    # Rewrite flat FlattenedTensor contractions as jagged lattice loops
    expression = unflatten(expression)

    # Flatten product tree, eliminate deltas, sum factorise
    def rebuild(expression):
        sum_indices, factors = traverse_product(expression, index_replacer=index_replacer)
        sum_indices, factors = delta_elimination(sum_indices, factors, index_replacer=index_replacer)
        factors = [index_replacer(f, ()) for f in factors]
        if ignore is not None:
            # TODO: This is a really blunt instrument and one might
            # plausibly want the ignored indices to be contracted on
            # the inside rather than the outside.
            extra = tuple(i for i in sum_indices if i in ignore)
            to_factor = tuple(i for i in sum_indices if i not in ignore)
            return IndexSum(sum_factorise(to_factor, factors), extra)
        else:
            return sum_factorise(sum_indices, factors)

    # Sometimes the value shape is composed as a ListTensor, which
    # could get in the way of decomposing factors.  In particular,
    # this is the case for H(div) and H(curl) conforming tensor
    # product elements.  So if ListTensors are used, they are pulled
    # out to be outermost, so we can straightforwardly factorise each
    # of its entries.
    lt_fis = OrderedDict()  # ListTensor free indices
    for node in traversal((expression,)):
        if isinstance(node, Indexed):
            child, = node.children
            if isinstance(child, ListTensor):
                lt_fis.update(zip_longest(node.multiindex, ()))
    lt_fis = tuple(index for index in lt_fis if index in expression.free_indices)

    if lt_fis:
        # Rebuild each split component
        tensor = ComponentTensor(expression, lt_fis)
        entries = [Indexed(tensor, zeta) for zeta in numpy.ndindex(tensor.shape)]
        entries = [index_replacer(e, ()) for e in entries]
        return Indexed(ListTensor(
            numpy.array(list(map(rebuild, entries))).reshape(tensor.shape)
        ), lt_fis)
    else:
        # Rebuild whole expression at once
        return rebuild(expression)


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
