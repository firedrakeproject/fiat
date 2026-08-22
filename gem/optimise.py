"""A set of routines implementing various transformations on GEM
expressions."""

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from functools import singledispatch, partial
from itertools import combinations, permutations, zip_longest
from numbers import Integral

import numpy

from gem.cost import index_space_literal
from gem.utils import groupby
from gem.node import (Memoizer, MemoizerArg, reuse_if_untouched,
                      reuse_if_untouched_arg, traversal, traversal_children)
from gem.gem import (Node, Failure, Identity, Constant, Literal, Zero,
                     Product, Sum, Comparison, Conditional, Division,
                     Index, IndexBase, VariableIndex, Indexed, FlexiblyIndexed,
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
        if new_expr == i.expression:
            return i
        # A variable index that substitution has made constant is a fixed
        # index, and folding it lets the lookup itself be evaluated.
        if isinstance(new_expr, Literal) and not new_expr.shape:
            return int(new_expr.array)
        return VariableIndex(new_expr)
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

    if types == {IndexSum}:
        extents = {tuple(i.extent for i in e.multiindex) for e in expressions}
        if len(extents) == 1:
            multiindex = tuple(Index(extent=extent) for extent in extents.pop())
            summands = [Indexed(ComponentTensor(e.children[0], e.multiindex),
                                multiindex)
                        for e in expressions]
            return IndexSum(_select_expression(summands, index), multiindex)

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


# Planning a tree visits 3**factors subset pairs, while searching the
# orderings of the contraction indices costs indices!.  Above this many
# factors the exhaustive search over orderings is the cheaper plan.
_MAX_PLANNED_FACTORS = 10


def _plan_contraction(sum_indices, groups):
    """Choose a product tree for one connected contraction.

    Every factor is a vertex and every contraction index joins the factors
    carrying it.  A product tree is built by dynamic programming over
    subsets of factors, and each index is reduced at the smallest subtree
    holding every factor that carries it, which is the earliest its
    reduction is legal.  Unlike a search over orderings of the indices,
    this can reduce an index over part of the product and multiply the
    rest in afterwards.

    :arg sum_indices: free indices for contractions, which must not split
                      into independent subproblems
    :arg groups: product factors, grouped by free indices
    :returns: optimised GEM expression
    """
    extents = {}
    for index in sum_indices:
        extents[index] = index.extent
    for group in groups:
        for index in group.free_indices:
            extents[index] = index.extent

    def size(indices):
        return numpy.prod([extents[i] for i in indices], dtype=int)

    # The factors each contraction index occurs in, as a bit mask
    incidence = {}
    for index in sum_indices:
        incidence[index] = sum(1 << n for n, group in enumerate(groups)
                               if index in group.free_indices)

    full = (1 << len(groups)) - 1
    # Indices reducible once a subset of the factors has been multiplied
    reducible = {}
    for subset in range(1, full + 1):
        reducible[subset] = frozenset(index for index in sum_indices
                                      if not incidence[index] & ~subset)

    def order(indices):
        """Sum out the widest index first, breaking ties reproducibly."""
        return tuple(sorted(indices, key=lambda i: (-extents[i], i.count)))

    def reduce_indices(expression, free, indices):
        """Sum out indices, largest extent first, costing each reduction."""
        flops = 0
        indices = order(indices)
        for index in indices:
            flops += size(free)
            free = free - {index}
        if indices:
            expression = IndexSum(expression, indices)
        return expression, free, flops

    # plans[subset] = (flops, expression, free indices)
    plans = {}
    for n, group in enumerate(groups):
        subset = 1 << n
        free = frozenset(group.free_indices)
        expression, free, flops = reduce_indices(group, free, reducible[subset])
        plans[subset] = (flops, expression, free)

    for subset in range(1, full + 1):
        if subset in plans:
            continue
        best = None
        # Split the subset in two, taking each unordered split once
        lowest = subset & -subset
        part = (subset - 1) & subset
        while part:
            if part & lowest:
                other = subset ^ part
                left, right = plans[part], plans[other]
                free = left[2] | right[2]
                flops = left[0] + right[0] + size(free)
                expression = Product(left[1], right[1])
                indices = reducible[subset] - reducible[part] - reducible[other]
                expression, free, extra = reduce_indices(expression, free, indices)
                candidate = (flops + extra, expression, free)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            part = (part - 1) & subset
        plans[subset] = best

    return plans[full][1]


def _sum_factorise_connected(sum_indices, groups):
    """Sum factorise a single connected contraction.

    Take whichever of the two searches the contraction is small enough for;
    see `_MAX_PLANNED_FACTORS`.

    :arg sum_indices: free indices for contractions, which must not split
                      into independent subproblems
    :arg groups: product factors, grouped by free indices
    :returns: optimised GEM expression
    """
    if not groups:
        return index_space_literal(sum_indices)

    if len(groups) <= _MAX_PLANNED_FACTORS:
        return _plan_contraction(sum_indices, groups)

    if len(sum_indices) > 6:
        raise NotImplementedError("Too many indices for sum factorisation!")

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


def sum_factorise(sum_indices, factors):
    """Optimise a tensor product through sum factorisation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :returns: optimised GEM expression
    """
    if len(factors) == 0:
        return index_space_literal(sum_indices)

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


def _product_descent(expr):
    """The subexpressions a product-tree walk breaks into further factors.

    A contraction and a product are broken up, and so is the dividend of a
    division, but never its divisor.  This is the one rule shared by
    `traverse_product` and `repeated_contractions`.

    :arg expr: a GEM expression
    :returns: the subexpressions to descend into, empty when ``expr`` is
              itself a factor
    """
    if isinstance(expr, (IndexSum, Product)):
        return expr.children
    if isinstance(expr, Division):
        dividend, _ = expr.children
        # A reciprocal is a factor in its own right.
        return () if dividend == one else (dividend,)
    return ()


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
        children = () if stop_at is not None and stop_at(expr) \
            else _product_descent(expr)
        if not children:
            terms.append(expr)
        elif isinstance(expr, IndexSum):
            indices, applier = renamer(expr.multiindex)
            sum_indices.extend(indices)
            stack.extend(index_replacer(applier(c), ()) for c in children)
        elif isinstance(expr, Division):
            # The divisor is not broken up, but becomes a factor of its own.
            stack.append(Division(one, expr.children[1]))
            stack.extend(children)
        else:
            stack.extend(reversed(children))

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


def distribute_sum(expr: Node, predicate: Callable[[Node], bool]) -> list[Node]:
    """Distribute selected sums through products and contractions.

    Parameters
    ----------
    expr
        GEM expression to distribute.
    predicate
        Predicate selecting the operations to distribute.

    Returns
    -------
    list of Node
        Additive terms after distribution.

    Notes
    -----
    Memoization uses object identity.  Structurally equal GEM nodes can have
    deep expression trees, while distribution only needs to reuse actual DAG
    nodes.

    """
    results = {}
    active = {}
    stack = [(expr, False)]
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in results:
            continue
        if not expanded:
            stack.append((node, True))
            stack.extend((c, False) for c in node.children)
            continue
        active[key] = predicate(node) or any(
            active[id(child)] for child in node.children)
        if active[key] and isinstance(node, (Sum, IndexSum, Product)):
            if isinstance(node, Sum):
                results[key] = [
                    term
                    for child in node.children
                    for term in results[id(child)]]
            elif isinstance(node, IndexSum):
                body, = node.children
                results[key] = [
                    IndexSum(term, tuple(
                        index for index in node.multiindex
                        if index in term.free_indices))
                    for term in results[id(body)]]
            else:  # Product
                a, b = node.children
                ta, tb = results[id(a)], results[id(b)]
                results[key] = [node] if len(ta) == 1 and len(tb) == 1 \
                    else [Product(x, y) for x in ta for y in tb]
        else:
            results[key] = [node]
    return results[id(expr)]


def _is_linear_map(node: Node, linear_indices: frozenset) -> bool:
    """Is a node a linear map into one multilinear axis?

    Parameters
    ----------
    node
        GEM expression node.
    linear_indices
        Free indices identifying the multilinear axes.

    Returns
    -------
    bool
        Whether the node is a sum over exactly one such axis.

    """
    return (isinstance(node, Sum)
            and len(linear_indices.intersection(node.free_indices)) == 1)


def has_linear_maps(
        expressions: Iterable[Node],
        linear_indices: Iterable[Index]) -> bool:
    """Does a GEM DAG contain a finite element linear map?

    Parameters
    ----------
    expressions
        Roots of a multilinear GEM expression DAG.
    linear_indices
        Free indices identifying the multilinear axes.

    Returns
    -------
    bool
        Whether preserving one-axis sums can change the factorisation.

    Notes
    -----
    Answering this costs one traversal, where building the preserved
    factorisation to compare it costs a whole pass of monomial collection.

    """
    linear_indices = frozenset(linear_indices)
    return any(_is_linear_map(node, linear_indices)
               for node in traversal(tuple(expressions)))


def preserve_linear_maps(
        expression: Node,
        linear_indices: Iterable[Index]) -> tuple[
            tuple[Node, ...], tuple[Node, ...]]:
    """Expose multilinear terms and retain each one-axis linear map.

    A sum that depends on one linear index represents a linear map into an
    argument tabulation. A sum that depends on several linear indices
    separates multilinear form terms. This function distributes the latter
    sums and returns the former sums as factors.

    Parameters
    ----------
    expression
        Multilinear GEM expression.
    linear_indices
        Free indices identifying the linear axes.

    Returns
    -------
    tuple
        Additive terms and the linear-map factors that they contain.

    """
    linear_indices = frozenset(linear_indices)

    def multilinear_sum(node: Node) -> bool:
        return isinstance(node, Sum) and len(
            linear_indices.intersection(node.free_indices)) > 1

    if not has_linear_maps((expression,), linear_indices):
        return (expression,), ()

    terms = tuple(distribute_sum(expression, multilinear_sum))
    groups = OrderedDict()
    for term in terms:
        _, factors = traverse_product(term)
        for factor in factors:
            if _is_linear_map(factor, linear_indices):
                groups.setdefault(factor)

    if not groups:
        return (expression,), ()
    return terms, tuple(groups)


def repeated_contractions(expression):
    """Find the contractions that occur more than once in a product tree.

    Flattening a contraction renames its indices apart, so a contraction
    used more than once becomes that many independent contractions and is
    evaluated once per use.  Keeping it whole preserves the sharing.

    :arg expression: a GEM expression
    :returns: the :class:`~.IndexSum` nodes occurring more than once as a
              factor of ``expression``
    """
    counts = Counter()
    stack = [expression]
    while stack:
        expr = stack.pop()
        if isinstance(expr, IndexSum):
            counts[expr] += 1
        stack.extend(_product_descent(expr))
    return frozenset(expr for expr, count in counts.items() if count > 1)


def _delta_axes(node: Node, self: Memoizer) -> frozenset:
    """The axes compared by the Deltas below a node, including its own.

    Memoising this over the DAG keeps the search for a cancellable Delta
    linear, rather than re-walking the subtree at every enclosing
    contraction.

    :arg node: a GEM expression
    :arg self: memoizer visiting the DAG
    :returns: the indices some Delta at or below ``node`` compares
    """
    axes = frozenset().union(*map(self, traversal_children(node)))
    if isinstance(node, Delta):
        axes = axes | {node.i, node.j}
    return axes


def _constant_map(index: IndexBase) -> tuple | None:
    """The literal table behind a VariableIndex, and the indices addressing it.

    Parameters
    ----------
    index
        Index to inspect.

    Returns
    -------
    tuple or None
        ``(array, indices)`` when the index is a lookup into a Literal with a
        plain multiindex, otherwise None.

    """
    if not isinstance(index, VariableIndex):
        return None
    expression = index.expression
    if not isinstance(expression, Indexed):
        return None
    table, = expression.children
    if not isinstance(table, Literal):
        return None
    if not all(isinstance(i, Index) for i in expression.multiindex):
        return None
    return table.array, expression.multiindex


def pull_back_indirect_delta(
        sum_indices: Iterable[Index],
        factors: Iterable[Node],
        replacer: MemoizerArg) -> tuple:
    """Contract an indirect Delta's own axes before its column axis.

    ``sum_a (sum_rk v(r,k) delta(c(r,k), a)) T(a, q)`` is cancelled by
    `delta_elimination` substituting ``a := c(r,k)``, which makes ``T``
    depend on ``r`` and ``k`` and so forces that contraction inside the ``q``
    loop.  When ``r`` and ``k`` are contracted here and ``T`` carries indices
    of its own, summing them first is cheaper: it yields a dense vector
    indexed by ``a``.  Run this before `delta_elimination` to take that
    cheaper route where it exists.

    :arg sum_indices: indices contracted over the product
    :arg factors: product factors
    :arg replacer: ``MemoizerArg(filtered_replace_indices)``
    :returns: new ``(sum_indices, factors)``, unchanged when no Delta is
              worth pulling back
    """
    for delta in factors:
        if not isinstance(delta, Delta):
            continue
        column = delta.j if isinstance(delta.i, VariableIndex) else delta.i
        variable = delta.i if isinstance(delta.i, VariableIndex) else delta.j
        if not isinstance(column, Index) or not isinstance(variable, VariableIndex):
            continue
        lookup = _constant_map(variable)
        if lookup is None:
            continue
        table, source_indices = lookup
        sources = frozenset(source_indices)
        if column not in sum_indices or not sources <= set(sum_indices):
            continue

        others = [f for f in factors if f is not delta]
        spanning = [f for f in others if column in f.free_indices]
        pulled = [f for f in others if column not in f.free_indices]
        if not spanning or not pulled:
            continue
        # Cancelling couples the spanning factors to the source indices.  That
        # only costs anything when they carry indices of their own.
        if not any(set(f.free_indices) - sources - {column} for f in spanning):
            continue

        vector = numpy.empty(column.extent, dtype=object)
        contributions = defaultdict(list)
        for position in numpy.ndindex(table.shape):
            substitution = tuple(zip(source_indices, (int(p) for p in position)))
            contributions[int(table[position])].append(substitution)
        for value in range(column.extent):
            terms = [make_product([replacer(f, substitution) for f in pulled])
                     for substitution in contributions.get(value, ())]
            # A reference basis function that no row maps onto contributes
            # nothing.
            vector[value] = make_sum(terms) if terms else Zero()

        rest = tuple(i for i in sum_indices if i not in sources)
        return rest, [Indexed(ListTensor(vector), (column,)), *spanning]

    return sum_indices, factors


def cancel_nested_deltas(expression: Node) -> Node:
    """Apply `delta_elimination` at every contraction of a whole DAG.

    `delta_elimination` only inspects top-level product factors, so a Delta
    inside a preserved linear map is invisible to it.  Flattening the product
    tree first exposes it, and hoists the contractions it sits under so that
    substituting the Delta's variable index cannot capture them.

    :arg expression: root of a scalar GEM expression
    :returns: the expression with those Deltas cancelled
    """
    replacer = MemoizerArg(filtered_replace_indices)
    delta_axes = Memoizer(_delta_axes)

    def visit(node, self):
        node = reuse_if_untouched(node, self)
        if not isinstance(node, IndexSum):
            return node
        if not delta_axes(node).intersection(node.multiindex):
            return node
        sum_indices, factors = traverse_product(node, index_replacer=replacer)
        sum_indices, factors = pull_back_indirect_delta(
            sum_indices, factors, replacer)
        sum_indices, factors = delta_elimination(
            sum_indices, factors, index_replacer=replacer)
        factors = [replacer(factor, ()) for factor in factors]
        return IndexSum(make_product(factors), tuple(sum_indices))

    return Memoizer(visit)(expression)


def contraction(expression):
    """Optimise the contractions of the tensor product at the root of
    the expression, including:

    - IndexSum-Delta cancellation
    - Sum factorisation

    This routine was designed with finite element coefficient
    evaluation in mind.
    """

    # Common memoizer to remove ComponentTensors
    index_replacer = MemoizerArg(filtered_replace_indices)

    # Eliminate annoying ComponentTensors
    expression = index_replacer(expression, ())

    # Flatten product tree, eliminate deltas, sum factorise
    def rebuild(expression):
        root = expression
        # The contraction at the root is always broken up, as that is the
        # one being optimised
        keep = repeated_contractions(expression)
        sum_indices, factors = traverse_product(
            expression, index_replacer=index_replacer,
            stop_at=lambda e: e is not root and e in keep)
        sum_indices, factors = pull_back_indirect_delta(
            sum_indices, factors, index_replacer)
        sum_indices, factors = delta_elimination(
            sum_indices, factors, index_replacer=index_replacer)
        factors = [index_replacer(f, ()) for f in factors]
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


def factorise_scalar_sums(expression: Node) -> Node:
    """Factor common products from scalar sums when this lowers GEM cost.

    Parameters
    ----------
    expression
        Root of a GEM expression.

    Returns
    -------
    Node
        Expression with profitable common product factors extracted.

    Notes
    -----
    Scalar geometry and basis-transformation expressions are simplified below
    the indexed contraction structure.  Contractions are indivisible factors:
    their bound indices cannot move through an enclosing sum.  Sums carrying
    free indices are left to the contraction planner, whose cost model includes
    their iteration domains.

    """
    def choose(node):
        if node.free_indices:
            return node
        summands = traverse_sum(node)
        if len(summands) < 2:
            return node

        factorisations = []
        for summand in summands:
            _, factors = traverse_product(
                summand,
                stop_at=lambda factor: isinstance(factor, IndexSum),
            )
            factorisations.append(factors)

        common = Counter(factorisations[0])
        for factors in factorisations[1:]:
            common &= Counter(factors)
        if not common:
            return node

        common_factors = list(common.elements())
        remainders = []
        for factors in factorisations:
            remaining_common = common.copy()
            remaining = []
            for factor in factors:
                if remaining_common[factor]:
                    remaining_common[factor] -= 1
                else:
                    remaining.append(factor)
            remainders.append(make_product(remaining))

        candidate = make_product(
            (*common_factors, make_sum(remainders)))
        if candidate.free_indices != node.free_indices:
            return node
        if estimate_cost((candidate,)) < estimate_cost((node,)):
            return candidate
        return node

    cache = {}

    def visit(node):
        key = id(node)
        if key in cache:
            return cache[key]
        children = tuple(visit(child) for child in node.children)
        result = node if children == node.children else node.reconstruct(*children)
        if isinstance(result, Sum):
            result = choose(result)
        cache[key] = result
        return result

    return visit(expression)


def factorise_indirect_reductions(expression: Node) -> Node:
    """Factor a reduction through a repeated indirect table lookup.

    Parameters
    ----------
    expression
        Root of a scalar GEM expression.

    Returns
    -------
    Node
        Expression with profitable dense reductions evaluated before gathers.

    Notes
    -----
    For an indirect index c(i), linearity lets the reduction over q be
    evaluated for a new dense table-row index before gathering at c(i).
    The rewrite is selected only when GEM's cost model predicts less
    arithmetic, with storage and node count breaking ties.

    """
    def replace(node, self, substitution):
        target, replacement = substitution
        if isinstance(node, Indexed):
            child, = node.children
            multiindex = tuple(replacement if index == target else index
                               for index in node.multiindex)
            return Indexed(self(child, substitution), multiindex)
        return reuse_if_untouched_arg(node, self, substitution)

    def choose(node):
        body, = node.children
        candidates = OrderedDict()
        for indexed in traversal((body,)):
            if not isinstance(indexed, Indexed):
                continue
            aggregate, = indexed.children
            for index, extent in zip(indexed.multiindex, aggregate.shape):
                if isinstance(index, VariableIndex):
                    sources = frozenset(index.expression.free_indices)
                    if sources and sources.isdisjoint(node.multiindex):
                        candidates.setdefault(index, (extent, sources))

        if not candidates:
            return node

        best = node
        best_cost = estimate_cost((node,))
        for indirect, (extent, sources) in candidates.items():
            latent = Index(extent=extent)
            dense_body = MemoizerArg(replace)(body, (indirect, latent))
            if sources.intersection(dense_body.free_indices):
                continue
            dense = ComponentTensor(
                IndexSum(dense_body, node.multiindex), (latent,))
            candidate = Indexed(dense, (indirect,))
            if candidate.free_indices != node.free_indices:
                continue
            cost = estimate_cost((candidate,))
            if cost < best_cost:
                best = candidate
                best_cost = cost
        return best

    def visit(node, self):
        node = reuse_if_untouched(node, self)
        if isinstance(node, IndexSum):
            node = choose(node)
        return node

    return Memoizer(visit)(expression)
