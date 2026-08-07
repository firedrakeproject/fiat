"""A set of routines implementing various transformations on GEM
expressions."""

from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from functools import lru_cache, singledispatch, partial
from itertools import zip_longest
import math
from numbers import Integral

import numpy

from gem.utils import groupby
from gem.node import (Memoizer, MemoizerArg, reuse_if_untouched,
                      reuse_if_untouched_arg, traversal)
from gem.gem import (Node, Failure, Identity, Constant, Literal, Zero,
                     Product, Sum, Comparison, Conditional, Division,
                     Power, MathFunction, MinValue, MaxValue, Inverse, Solve,
                     Index, VariableIndex, Indexed, FlexiblyIndexed,
                     IndexSum, JaggedIndex, RaggedIndex, ComponentTensor, ListTensor,
                     FlattenedTensor, Delta, _jagged_lattice,
                     partial_indexed, uint_type, one)


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


def hoist_linear_index(expression: Node,
                       linear_indices: Iterable[Index]) -> Node:
    """Materialise repeated scalar expressions over one linear index.

    Expressions that differ only in which equally-sized linear axis they use
    are evaluations of the same indexed tensor.  Replacing both evaluations
    by accesses to one :class:`ComponentTensor` exposes that tensor to the
    imperative scheduler, which can compute it once in the surrounding loop
    nest.

    Parameters
    ----------
    expression
        Scalar expression to optimise.
    linear_indices
        Free indices representing multilinear argument axes.

    Returns
    -------
    Node
        Expression with profitable repeated linear expressions shared.

    """
    linear_indices = frozenset(linear_indices)
    canonical = {
        extent: Index(extent=extent)
        for extent in {index.extent for index in linear_indices}
    }
    replacer = MemoizerArg(filtered_replace_indices)
    groups = defaultdict(list)
    for node in traversal((expression,)):
        involved = linear_indices.intersection(node.free_indices)
        if node.shape or len(involved) != 1:
            continue
        index, = involved
        normal = replacer(node, ((index, canonical[index.extent]),))
        groups[normal].append((node, index))

    replacements = {}
    for normal, occurrences in groups.items():
        indices = {index for _, index in occurrences}
        if len(indices) < 2 or estimate_cost((normal,))[0] == 0:
            continue
        index = canonical[next(iter(indices)).extent]
        tensor = ComponentTensor(normal, (index,))
        replacements.update(
            (node, Indexed(tensor, (original,)))
            for node, original in occurrences)

    if not replacements:
        return expression

    def replace(node, self):
        try:
            return replacements[node]
        except KeyError:
            return reuse_if_untouched(node, self)

    return Memoizer(replace)(expression)


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


def _index_closure(indices: Iterable[Index]) -> frozenset[Index]:
    """Return indices together with all jagged-loop parents.

    Parameters
    ----------
    indices
        Indices used by an operation.

    Returns
    -------
    frozenset of Index
        The iteration indices required to execute the operation.

    """
    closure = set(indices)
    pending = list(closure)
    while pending:
        index = pending.pop()
        for parent in getattr(index, "parents", ()):
            if parent not in closure:
                closure.add(parent)
                pending.append(parent)
    return frozenset(closure)


def _index_components(
        indices: frozenset[Index]) -> tuple[frozenset[Index], ...]:
    """Find independent components of an index-parent graph.

    Parameters
    ----------
    indices
        Indices closed under the jagged parent relation.

    Returns
    -------
    tuple of frozenset of Index
        Connected components of the undirected parent graph.

    """
    neighbours = {index: set() for index in indices}
    for index in indices:
        for parent in getattr(index, "parents", ()):
            neighbours[index].add(parent)
            neighbours[parent].add(index)

    components = []
    remaining = set(indices)
    while remaining:
        pending = [min(remaining, key=lambda index: index.count)]
        component = set(pending)
        while pending:
            index = pending.pop()
            new = neighbours[index] - component
            component.update(new)
            pending.extend(new)
        remaining.difference_update(component)
        components.append(frozenset(component))
    return tuple(components)


def _component_iteration_count(indices: frozenset[Index]) -> int:
    """Count one connected rectangular or jagged index domain.

    The dynamic-programming state contains only values on the live parent
    frontier.  Values disappear as soon as no unvisited index depends on
    them, so equivalent suffixes share one count.

    Parameters
    ----------
    indices
        One connected component of an index-parent graph.

    Returns
    -------
    int
        Number of points in the component domain.

    """
    parents = {
        index: frozenset(getattr(index, "parents", ()))
        for index in indices
    }
    ordered = []
    remaining = set(indices)
    while remaining:
        available = sorted(
            (index for index in remaining
             if parents[index] <= set(ordered)),
            key=lambda index: index.count)
        if not available:
            raise ValueError("Jagged index parents contain a cycle")
        ordered.extend(available)
        remaining.difference_update(available)

    last_use = {
        index: max(
            (position for position, child in enumerate(ordered)
             if index in parents[child]),
            default=-1,
        )
        for index in ordered
    }
    frontiers = tuple(
        tuple(index for index in ordered[:position]
              if last_use[index] >= position)
        for position in range(len(ordered) + 1)
    )

    @lru_cache(maxsize=None)
    def count(position: int, state: tuple[int, ...]) -> int:
        if position == len(ordered):
            return 1
        values = dict(zip(frontiers[position], state))
        index = ordered[position]
        extent = index.iteration_extent(values)
        next_frontier = frontiers[position + 1]
        total = 0
        for value in range(extent):
            values[index] = value
            next_state = tuple(values[parent] for parent in next_frontier)
            total += count(position + 1, next_state)
        return total

    return count(0, ())


@lru_cache(maxsize=1024)
def _iteration_count(indices: frozenset[Index]) -> int:
    """Count points in a rectangular or jagged iteration space.

    Independent parent-graph components form a Cartesian product, so their
    point counts multiply.  Each jagged component is counted by dynamic
    programming over its live parent frontier.

    Parameters
    ----------
    indices
        Indices on which an operation depends.

    Returns
    -------
    int
        Number of executions of the operation.

    """
    indices = _index_closure(indices)
    if not indices:
        return 1
    if not any(getattr(index, "parents", ()) for index in indices):
        return int(numpy.prod(
            [index.extent for index in indices], dtype=int))
    return math.prod(map(
        _component_iteration_count, _index_components(indices)))


def _storage_count(indices: Iterable[Index]) -> int:
    """Return the rectangular allocation size for a set of indices.

    Parameters
    ----------
    indices
        Indices retained by an intermediate.

    Returns
    -------
    int
        Number of scalar entries in the intermediate.

    """
    indices = _index_closure(indices)
    return int(numpy.prod(
        [index.extent for index in indices], dtype=int))


def _operation_count(node: Node) -> int:
    """Estimate scalar operations performed by one GEM node.

    Parameters
    ----------
    node
        Scalar expression node in a contraction DAG.

    Returns
    -------
    int
        Operations over the node's complete iteration domain.

    """
    domain = _iteration_count(frozenset(node.free_indices))
    if isinstance(node, Product):
        if any(isinstance(child, Literal) and not child.shape
               and child.value == -1 for child in node.children):
            return 0
        return domain
    if isinstance(node, (Sum, Division, MathFunction, MinValue, MaxValue)):
        return domain
    if isinstance(node, Power):
        _, exponent = node.children
        if isinstance(exponent, Literal) and not exponent.shape:
            value = exponent.value
            if value > 0 and value == math.floor(value):
                return math.ceil(math.log2(value)) * domain
        return 5 * domain
    if isinstance(node, IndexSum):
        body, = node.children
        return _iteration_count(frozenset(body.free_indices))
    if isinstance(node, Inverse):
        n, _ = node.shape
        return 2 * n ** 3
    if isinstance(node, Solve):
        n, m = node.shape
        return 2 * n * m + 2 * n ** 3
    return 0


def estimate_cost(expressions: Iterable[Node]) -> tuple[int, int, int, int]:
    """Estimate arithmetic work and contraction storage for a GEM DAG.

    Each structurally shared operation is counted once over the exact
    rectangular or jagged domain induced by its free indices.  An
    :class:`IndexSum` contributes one accumulation per point of its body
    domain.  Storage counts the result domains of contractions, which are the
    mathematical intermediates exposed to scheduling.

    Parameters
    ----------
    expressions
        Roots of a scalar GEM expression DAG.

    Returns
    -------
    tuple of int
        Operation count, total contraction storage, largest contraction, and
        expression-node count.

    """
    nodes = tuple(traversal(tuple(expressions)))
    sizes = [
        _storage_count(node.free_indices)
        for node in nodes if isinstance(node, IndexSum)
    ]
    return (
        sum(map(_operation_count, nodes)),
        sum(sizes),
        max(sizes, default=0),
        len(nodes),
    )


def associate(operator, operands: Iterable[Node]) -> tuple[Node, int]:
    """Construct a minimum-operation associative expression tree.

    Dynamic programming examines every bipartition of each operand subset.
    For unusually large expressions a deterministic greedy search bounds
    compile time.

    Parameters
    ----------
    operator
        Associative binary GEM operator.
    operands
        Expressions to combine.

    Returns
    -------
    Node
        Associated GEM expression.
    int
        Estimated number of floating-point operations.

    """
    operands = tuple(operands)
    if not operands:
        return operator(), 0
    if len(operands) == 1:
        return operands[0], 0

    def combine(left: Node, right: Node) -> tuple[Node, int]:
        result = operator(left, right)
        folded = result is left or result is right
        indices = frozenset(left.free_indices) | frozenset(right.free_indices)
        return result, 0 if folded else _iteration_count(indices)

    if len(operands) > 8:
        terms = list(operands)
        flops = 0
        while len(terms) > 1:
            candidates = (
                (combine(terms[i], terms[j])[1], i, j)
                for i in range(len(terms))
                for j in range(i + 1, len(terms)))
            _, i, j = min(candidates)
            result, cost = combine(terms[i], terms[j])
            flops += cost
            terms = [term for k, term in enumerate(terms)
                     if k not in (i, j)]
            terms.append(result)
        return terms[0], flops

    plans = {
        1 << position: ((0, 0), operand)
        for position, operand in enumerate(operands)
    }
    for size in range(2, len(operands) + 1):
        for mask in range(1, 1 << len(operands)):
            if mask.bit_count() != size:
                continue
            anchor = mask & -mask
            best = None
            left = (mask - 1) & mask
            while left:
                if left & anchor:
                    right = mask ^ left
                    if right:
                        left_score, left_expr = plans[left]
                        right_score, right_expr = plans[right]
                        result, cost = combine(left_expr, right_expr)
                        score = (
                            left_score[0] + right_score[0] + cost,
                            max(left_score[1], right_score[1]) + 1,
                        )
                        candidate = (score, result)
                        if best is None or score < best[0]:
                            best = candidate
                left = (left - 1) & mask
            plans[mask] = best
    score, result = plans[(1 << len(operands)) - 1]
    return result, score[0]


def _ordered_contraction_indices(
        indices: Iterable[Index],
        ordering: tuple[Index, ...]) -> tuple[Index, ...]:
    """Order contractions with jagged parents outside their children.

    Parameters
    ----------
    indices
        Indices to order.
    ordering
        Preferred deterministic order.

    Returns
    -------
    tuple of Index
        Legal loop order for an :class:`IndexSum`.

    """
    indices = frozenset(indices)
    result = []
    pending = [index for index in ordering if index in indices]
    while pending:
        for position, index in enumerate(pending):
            parents = set(getattr(index, "parents", ())) & indices
            if parents <= set(result):
                result.append(index)
                pending.pop(position)
                break
        else:
            raise ValueError("Jagged index parents contain a cycle")
    return tuple(result)


def _contraction_component(
        sum_indices: tuple[Index, ...],
        factors: tuple[Node, ...]) -> tuple[Node, tuple[int, int, int]]:
    """Optimize one connected tensor contraction by subset DP.

    Parameters
    ----------
    sum_indices
        Contracted indices in deterministic order.
    factors
        Factors connected through at least one contracted index.

    Returns
    -------
    Node
        Optimized contraction.
    tuple of int
        Estimated FLOPs, peak storage, and total storage.

    """
    if len(factors) > 10:
        terms = list(factors)
        flops = 0
        storage = 0
        for index in reversed(sum_indices):
            contract = [term for term in terms
                        if index in term.free_indices]
            if not contract:
                continue
            deferred = [term for term in terms
                        if index not in term.free_indices]
            product, product_flops = associate(Product, contract)
            extent = _iteration_count(frozenset(product.free_indices))
            result = IndexSum(product, (index,))
            flops += product_flops + extent
            result_storage = _storage_count(result.free_indices)
            storage += result_storage
            terms = deferred + [result]
        result, product_flops = associate(Product, terms)
        return result, (flops + product_flops, storage, storage)

    full_mask = (1 << len(factors)) - 1
    factor_indices = tuple(
        _index_closure(factor.free_indices) for factor in factors)
    supports = {
        index: sum(
            1 << position
            for position, indices in enumerate(factor_indices)
            if index in indices)
        for index in sum_indices
    }

    @lru_cache(maxsize=None)
    def closed(mask: int) -> frozenset[Index]:
        return frozenset(
            index for index, support in supports.items()
            if support and not support & ~mask)

    @lru_cache(maxsize=None)
    def live(mask: int) -> frozenset[Index]:
        indices = set().union(*(
            factor_indices[position]
            for position in range(len(factors))
            if mask & (1 << position)))
        indices.difference_update(closed(mask))
        return frozenset(indices)

    def reduce(
            expression: Node, mask: int,
            child_closed: frozenset[Index]) -> tuple[Node, int, int]:
        newly_closed = closed(mask) - child_closed
        direct = _ordered_contraction_indices(
            (index for index in newly_closed
             if index in expression.free_indices),
            sum_indices)
        if not direct:
            return expression, 0, 0
        extent = _iteration_count(
            frozenset(expression.free_indices))
        result = IndexSum(expression, direct)
        return result, extent, _storage_count(live(mask))

    plans = {}
    for position, factor in enumerate(factors):
        mask = 1 << position
        expression, cost, result_storage = reduce(
            factor, mask, frozenset())
        score = (cost, result_storage, result_storage)
        plans[mask] = (
            score, expression, result_storage, closed(mask))

    for size in range(2, len(factors) + 1):
        for mask in range(1, full_mask + 1):
            if mask.bit_count() != size:
                continue
            anchor = mask & -mask
            best = None
            left = (mask - 1) & mask
            while left:
                if left & anchor:
                    right = mask ^ left
                    if right:
                        left_plan = plans[left]
                        right_plan = plans[right]
                        left_score, left_expr, left_storage, left_closed = left_plan
                        right_score, right_expr, right_storage, right_closed = right_plan
                        product = Product(left_expr, right_expr)
                        if product in (left_expr, right_expr):
                            product_cost = 0
                        else:
                            product_cost = _iteration_count(
                                live(left) | live(right))
                        expression, reduction_cost, result_storage = reduce(
                            product, mask, left_closed | right_closed)
                        flops = (left_score[0] + right_score[0]
                                 + product_cost + reduction_cost)
                        live_storage = (left_storage + right_storage
                                        + result_storage)
                        peak = max(
                            left_score[1], right_score[1], live_storage)
                        total = (left_score[2] + right_score[2]
                                 + result_storage)
                        score = (flops, peak, total)
                        candidate = (
                            score, expression, result_storage, closed(mask))
                        if best is None or score < best[0]:
                            best = candidate
                left = (left - 1) & mask
            plans[mask] = best
    score, expression, _, _ = plans[full_mask]
    return expression, score


def sum_factorise(
        sum_indices: Iterable[Index],
        factors: Iterable[Node],
        distribute: bool = False) -> Node:
    """Optimize a tensor contraction using sum factorization.

    The factors form a tensor network whose hyperedges are contraction
    indices.  Independent connected components are reduced separately.  A
    subset dynamic program then chooses the contraction tree minimizing
    arithmetic work, with peak and total intermediate storage as tie-breakers.

    Parameters
    ----------
    sum_indices
        Free indices to contract.
    factors
        Scalar tensor factors.
    distribute
        Split selected sums when doing so exposes contractions over fewer
        indices.

    Returns
    -------
    Node
        Optimized GEM expression.

    """
    sum_indices = tuple(sum_indices)
    factors = tuple(factors)
    if len(factors) == 0 and len(sum_indices) == 0:
        # Empty product
        return one

    factor_indices = set().union(*(factor.free_indices for factor in factors))
    jagged_domain = set().union(*(
        _index_closure((index,))
        for index in sum_indices if isinstance(index, JaggedIndex)))
    if jagged_domain - factor_indices:
        domain_indices = tuple(index for index in sum_indices
                               if index in jagged_domain)
        active = _index_closure(factor_indices & jagged_domain) \
            & jagged_domain
        active_indices = tuple(index for index in domain_indices
                               if index in active)
        points = _jagged_lattice(domain_indices)
        if active_indices:
            positions = [domain_indices.index(index)
                         for index in active_indices]
            multiplicity = numpy.zeros(
                tuple(index.extent for index in active_indices))
            numpy.add.at(
                multiplicity,
                tuple(points[:, position] for position in positions), 1)
            domain_factor = Indexed(
                Literal(multiplicity), active_indices)
        else:
            domain_factor = Literal(len(points))
        factors += (domain_factor,)
        factor_indices.update(active_indices)
        marginalised = jagged_domain - active
        sum_indices = tuple(index for index in sum_indices
                            if index not in marginalised)

    missing = set(sum_indices) - factor_indices
    if missing:
        # A rectangular sum of an index-independent expression is its extent
        # times that expression.
        factors += tuple(Literal(index.extent)
                         for index in sum_indices if index in missing)
        sum_indices = tuple(index for index in sum_indices
                            if index not in missing)

    if distribute:
        contraction_indices = frozenset(sum_indices)
        for position, factor in enumerate(factors):
            summands = traverse_sum(factor)
            involved = contraction_indices.intersection(factor.free_indices)
            if (len(summands) > 1 and involved
                    and any(any(
                        contraction_indices.intersection(term.free_indices)
                        < involved
                        for term in traverse_product(summand)[1])
                        for summand in summands)):
                expressions = []
                for summand in summands:
                    extra, summand_factors = traverse_product(summand)
                    indices = tuple(OrderedDict.fromkeys((*sum_indices, *extra)))
                    if len(indices) > 6:
                        break
                    expressions.append(sum_factorise(
                        indices,
                        factors[:position] + tuple(summand_factors)
                        + factors[position + 1:]))
                else:
                    return make_sum(expressions)

    contraction_set = frozenset(sum_indices)
    factor_indices = [
        _index_closure(factor.free_indices) & contraction_set
        for factor in factors]
    remaining = set(range(len(factors)))
    components = []
    while remaining:
        component = {remaining.pop()}
        active_indices = set().union(*(
            factor_indices[position] for position in component))
        changed = True
        while changed:
            connected = {
                position for position in remaining
                if active_indices & factor_indices[position]}
            changed = bool(connected)
            component.update(connected)
            remaining.difference_update(connected)
            active_indices.update(*(
                factor_indices[position] for position in connected))
        components.append(tuple(sorted(component)))

    expressions = []
    for component in components:
        component_factors = tuple(factors[position] for position in component)
        involved = set().union(*(
            factor_indices[position] for position in component))
        component_indices = tuple(
            index for index in sum_indices if index in involved)
        expression, _ = _contraction_component(
            component_indices, component_factors)
        expressions.append(expression)
    expression, _ = associate(Product, expressions)
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
            def applier(expr):
                return expr

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


def _distribute_sum(expr: Node, predicate=None) -> list[Node]:
    """Distribute selected sums through products and contractions.

    Parameters
    ----------
    expr
        GEM expression to distribute.
    predicate
        Optional predicate selecting operations to distribute.

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
    if predicate is None:
        def predicate(node):
            return True

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

    Notes
    -----
    Polynomial factorization can recover any partial grouping from a fully
    expanded expression. The map-preserving representation remains useful
    because it bounds expansion and exposes basis transformation as a
    separate contraction.

    """
    linear_indices = frozenset(linear_indices)

    def multilinear_sum(node: Node) -> bool:
        return isinstance(node, Sum) and len(
            linear_indices.intersection(node.free_indices)) > 1

    if not any(
            isinstance(node, Sum)
            and len(linear_indices.intersection(node.free_indices)) == 1
            for node in traversal((expression,))):
        return (expression,), ()

    terms = tuple(_distribute_sum(
        expression, predicate=multilinear_sum))
    groups = OrderedDict()
    for term in terms:
        _, factors = traverse_product(term)
        for factor in factors:
            if (isinstance(factor, Sum)
                    and len(linear_indices.intersection(
                        factor.free_indices)) == 1):
                groups.setdefault(factor)

    if not groups:
        return (expression,), ()
    return terms, tuple(groups)


def eliminate_deltas(expression):
    """Cancel contracted deltas without changing other contractions."""
    replacer = MemoizerArg(filtered_replace_indices)
    expression = replacer(expression, ())
    nodes = tuple(traversal((expression,)))
    contracted = frozenset(
        index
        for node in nodes if isinstance(node, IndexSum)
        for index in node.multiindex)

    def cancellable(node):
        return isinstance(node, Delta) \
            and bool({node.i, node.j} & contracted)

    if not any(isinstance(node, Delta) and cancellable(node)
               for node in nodes):
        return expression

    terms = []
    for term in _distribute_sum(expression, predicate=cancellable):
        indices, factors = traverse_product(term, index_replacer=replacer)
        indices, factors = delta_elimination(
            indices, factors, index_replacer=replacer)
        factors = [replacer(factor, ()) for factor in factors]
        terms.append(IndexSum(Product(*factors), indices))
    return make_sum(terms)


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

    distribute = any(isinstance(node, FlattenedTensor)
                     for node in traversal((expression,)))

    # Common memoizer to remove ComponentTensors
    index_replacer = MemoizerArg(filtered_replace_indices)

    # Eliminate annoying ComponentTensors
    expression = index_replacer(expression, ())

    # Flatten product tree, eliminate deltas, sum factorise
    def rebuild(expression):
        expression = eliminate_deltas(expression)
        sum_indices, factors = traverse_product(expression, index_replacer=index_replacer)
        sum_indices, factors = delta_elimination(sum_indices, factors, index_replacer=index_replacer)
        factors = [index_replacer(f, ()) for f in factors]

        expression = IndexSum(Product(*factors), sum_indices)
        expression = unflatten(expression)
        sum_indices, factors = traverse_product(expression, index_replacer=index_replacer)
        sum_indices, factors = delta_elimination(sum_indices, factors, index_replacer=index_replacer)
        factors = [index_replacer(f, ()) for f in factors]
        if ignore is not None:
            # TODO: This is a really blunt instrument and one might
            # plausibly want the ignored indices to be contracted on
            # the inside rather than the outside.
            extra = tuple(i for i in sum_indices if i in ignore)
            to_factor = tuple(i for i in sum_indices if i not in ignore)
            return IndexSum(sum_factorise(to_factor, factors,
                                          distribute=distribute), extra)
        else:
            return sum_factorise(sum_indices, factors, distribute=distribute)

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
    expression, = unroll_indexsum(
        (expression,), predicate=lambda index: not isinstance(index, RaggedIndex))
    expression, = remove_componenttensors((expression,))
    return expression


def _clone_multiindex(multiindex: Iterable[Index]) -> tuple[Index, ...]:
    """Clone an index tuple while preserving its internal jagged parents."""
    clones = {}
    for index in multiindex:
        if isinstance(index, JaggedIndex):
            parents = tuple(clones[parent] for parent in index.parents)
            clones[index] = JaggedIndex(extent=index.extent, parents=parents)
        else:
            clones[index] = Index(extent=index.extent)
    return tuple(clones[index] for index in multiindex)


def _replace_gathers(node: Node, self, subst: tuple) -> Node:
    """Replace selected flat gathers, then apply ordinary index substitution."""
    try:
        return self.replacements[node]
    except KeyError:
        return filtered_replace_indices(node, self, subst)


def _flattened_layout(gather: Indexed) -> tuple:
    """Return a structural key for a flattened tensor's iteration lattice."""
    tensor, = gather.children
    positions = {index: position
                 for position, index in enumerate(tensor.multiindex)}
    return tuple((type(index), index.extent,
                  tuple(positions[parent]
                        for parent in getattr(index, "parents", ())))
                 for index in tensor.multiindex)


def _find_unflattenable_index(
        nodes: Iterable[Node],
        indices: Iterable[Index]) -> tuple | None:
    """Find compatible flat gathers at one unconstrained index.

    An index occurring in a :class:`Delta` is not a candidate.  Cancelling
    that delta is cheaper than replacing the flat index with a full lattice
    loop and an indirect comparison.
    """
    indices = tuple(indices)
    index_set = frozenset(indices)
    constrained = set()
    gathers = defaultdict(OrderedDict)
    for node in nodes:
        if isinstance(node, Delta):
            constrained.update(node.free_indices)
        elif (isinstance(node, Indexed)
              and len(node.multiindex) == 1
              and node.multiindex[0] in index_set
              and isinstance(node.children[0], FlattenedTensor)):
            gathers[node.multiindex[0]].setdefault(node)

    for index in indices:
        candidates = tuple(gathers[index])
        layouts = {_flattened_layout(gather) for gather in candidates}
        if candidates and len(layouts) == 1 and index not in constrained:
            layout, = layouts
            return index, layout, candidates
    return None


def _prepare_unflattening(
        gathers: tuple[Indexed, ...]) -> tuple[MemoizerArg, tuple, VariableIndex]:
    """Prepare one joint rewrite of compatible flat gathers.

    Each flattened tensor is inlined on the same fresh lattice multiindex.
    ``flat_index`` maps that lattice point back to the original flat ordering
    wherever the old index is still needed, notably in the return variable.
    """
    gather = gathers[0]
    tensor, = gather.children
    assert all(_flattened_layout(other) == _flattened_layout(gather)
               for other in gathers)
    multiindex = _clone_multiindex(tensor.multiindex)
    replacer = MemoizerArg(filtered_replace_indices)
    mapper = MemoizerArg(_replace_gathers)
    mapper.replacements = {}
    for other in gathers:
        tensor, = other.children
        mapper.replacements[other] = replacer(
            tensor.children[0], tuple(zip(tensor.multiindex, multiindex)))

    shape = tuple(index.extent for index in multiindex)
    points = gather.children[0].lattice_points()
    ordering = numpy.zeros(shape, dtype=uint_type)
    ordering[tuple(points.T)] = numpy.arange(len(points))
    flat_index = VariableIndex(Indexed(
        Literal(ordering, dtype=uint_type), multiindex))
    return mapper, multiindex, flat_index


def _separable_sum(node: Node, indices: frozenset[Index]) -> bool:
    """Whether distributing a sum exposes smaller index dependencies."""
    if not isinstance(node, Sum):
        return False
    involved = indices.intersection(node.free_indices)
    return bool(involved) and any(
        any(indices.intersection(factor.free_indices) < involved
            for factor in traverse_product(summand)[1])
        for summand in traverse_sum(node))


def _unflatten_contracted_terms(
        summand: Node, index: Index) -> tuple[list, list[Node]]:
    """Unflatten additive terms containing a gather at a contracted index."""
    rewritten = []
    leftover = []
    for term in traverse_sum(summand):
        candidate = _find_unflattenable_index(
            traversal((term,)), (index,))
        if candidate is None:
            leftover.append(term)
            continue
        _, _, gathers = candidate
        mapper, multiindex, flat_index = _prepare_unflattening(gathers)
        term = mapper(term, ((index, flat_index),))
        own = frozenset(multiindex)
        predicate = partial(_separable_sum, indices=own)
        rewritten.extend(
            (multiindex, piece)
            for piece in _distribute_sum(term, predicate=predicate))
    return rewritten, leftover


def _unflatten_contractions(node: Node, self) -> Node:
    """Memoizer callback for flat indices bound by an :class:`IndexSum`."""
    node = reuse_if_untouched(node, self)
    if not isinstance(node, IndexSum):
        return node
    summand, = node.children
    for index in node.multiindex:
        rewritten, leftover = _unflatten_contracted_terms(summand, index)
        if not rewritten:
            continue
        rest = tuple(other for other in node.multiindex if other != index)
        pieces = []
        for own, term in rewritten:
            term = self(IndexSum(
                term, own + tuple(i for i in rest if i in term.free_indices)))
            indices, factors = traverse_product(term)
            indices, factors = delta_elimination(indices, factors)
            pieces.append(sum_factorise(indices, factors))
        if leftover:
            residual = make_sum(leftover)
            indices = tuple(i for i in (index,) + rest
                            if i in residual.free_indices)
            pieces.append(self(IndexSum(residual, indices)))
        return make_sum(pieces)
    return node


def unflatten(expression: Node) -> Node:
    """Replace flat contractions by loops over their jagged lattices.

    A contraction over the flat axis of a :class:`FlattenedTensor` hides its
    tensor-product factors.  This rewrite substitutes the tensor's own
    (possibly jagged) multiindex for that flat axis, enabling the ordinary
    contraction optimizer to recover sum factorisation.
    """
    if not any(isinstance(node, FlattenedTensor)
               for node in traversal((expression,))):
        return expression
    return Memoizer(_unflatten_contractions)(expression)


def _has_flat_gather(nodes: Iterable[Node],
                     indices: Iterable[Index]) -> bool:
    """Whether ``nodes`` gather a flattened tensor at one of ``indices``."""
    indices = frozenset(indices)
    return any(isinstance(node, Indexed)
               and len(node.multiindex) == 1
               and node.multiindex[0] in indices
               and isinstance(node.children[0], FlattenedTensor)
               for node in nodes)


def _unflatten_free_indices(
        variable: Node, expression: Node, *,
        split_separable_sums: bool) -> tuple[list[tuple[Node, Node]], bool]:
    """Replace flattened gathers at free indices of one assignment.

    Compatible gathers are rewritten together.  Distribution is restricted
    to deltas until a lattice has been exposed; the optional legacy path then
    splits separable sums immediately.
    """
    pending = [(variable, expression)]
    outputs = []
    changed = False
    while pending:
        current_variable, current_expression = pending.pop()
        nodes = tuple(traversal((current_expression,)))
        candidate = _find_unflattenable_index(
            nodes, current_variable.free_indices)
        if candidate is not None:
            index, layout, gathers = candidate
            groups = OrderedDict([
                ((index, layout), ([current_expression], list(gathers)))])
        elif not _has_flat_gather(nodes, current_variable.free_indices):
            outputs.append((current_variable, current_expression))
            continue
        else:
            predicate = (lambda node: isinstance(node, Delta)) \
                if any(isinstance(node, Delta) for node in nodes) else None
            groups = OrderedDict()
            for term in _distribute_sum(
                    current_expression, predicate=predicate):
                candidate = _find_unflattenable_index(
                    traversal((term,)), current_variable.free_indices)
                key = candidate[:2] if candidate is not None else None
                terms, gathers = groups.setdefault(key, ([], []))
                terms.append(term)
                if candidate is not None:
                    gathers.extend(candidate[2])

        for key, (terms, gathers) in groups.items():
            term = make_sum(terms)
            if key is None:
                outputs.append((current_variable, term))
                continue

            changed = True
            index, _ = key
            gathers = tuple(OrderedDict.fromkeys(gathers))
            mapper, multiindex, flat_index = _prepare_unflattening(gathers)
            substitution = ((index, flat_index),)
            new_variable = MemoizerArg(filtered_replace_indices)(
                current_variable, substitution)
            new_expression = mapper(term, substitution)
            if split_separable_sums:
                lattice_indices = frozenset(multiindex)
                predicate = partial(_separable_sum, indices=lattice_indices)
                pending.extend(
                    (new_variable, piece)
                    for piece in _distribute_sum(
                        new_expression, predicate=predicate))
            else:
                pending.append((new_variable, new_expression))
    return outputs, changed


def _refactor_unflattened_outputs(
        outputs: list[tuple[Node, Node]]) -> list[tuple[Node, Node]] | None:
    r"""Recover sum factorisation after exposing every argument lattice.

    Consider a bilinear form whose argument tabulations are transformed by
    sparse matrices,

    .. math::

        A_{ij} = \sum_q (S B(q))_i\,G(q)\,(T C(q))_j.

    Sparse-delta cancellation moves rows of ``S`` and ``T`` into the return
    scatter.  Both flat argument indices must then be replaced by their
    jagged lattice multiindices *before* expanding derivative sums.  Expanding
    after only one replacement duplicates the second lattice once per
    summand, producing many equivalent loop nests.

    The refactorisation is valid when the non-tabulation part of every
    monomial is independent of the argument indices.  This condition says
    exactly that the sparse transforms have been absorbed by the scatter.
    Dense or otherwise residual transforms use the conservative legacy path.
    """
    # Local imports avoid a module cycle: refactorise and coffee both use
    # primitive transformations defined in this module.
    from gem.coffee import optimise_monomial_sum
    from gem.refactorise import (
        ATOMIC, COMPOUND, OTHER, FactorisationError, collect_monomials,
    )

    candidates = []
    has_nontrivial_sum = False
    for variable, expression in outputs:
        argument_indices = frozenset(variable.free_indices)
        contraction_indices = frozenset(
            index
            for node in traversal((expression,))
            if isinstance(node, IndexSum)
            for index in node.multiindex)

        def classify(node):
            involved = argument_indices.intersection(node.free_indices)
            if not involved:
                return OTHER
            if isinstance(node, Indexed):
                return ATOMIC if contraction_indices.intersection(
                    node.free_indices) else OTHER
            return COMPOUND

        try:
            monomial_sum, = collect_monomials([expression], classify)
        except FactorisationError:
            return None

        monomials = tuple(monomial_sum)
        if any(argument_indices.intersection(monomial.rest.free_indices)
               for monomial in monomials):
            return None
        has_nontrivial_sum |= len(monomials) > 1
        sum_indices = tuple(OrderedDict.fromkeys(
            index
            for monomial in monomials
            for index in monomial.sum_indices))
        candidates.append(
            (variable, monomial_sum, sum_indices))

    if not has_nontrivial_sum:
        return None
    return [
        (variable, optimise_monomial_sum(
            monomial_sum, variable.index_ordering(), sum_indices))
        for variable, monomial_sum, sum_indices in candidates
    ]


def unflatten_returns(
        pairs: Iterable[tuple[Node, Node]]) -> list[tuple[Node, Node]]:
    """Unflatten free argument indices in assignment pairs.

    Every compatible argument lattice is exposed jointly.  Bilinear
    expressions are then refactorised as monomial sums so that sparse
    argument transforms remain in the output scatter while quadrature
    contractions recover their one-dimensional factors.  Expressions with
    residual argument-dependent transforms retain the established local
    distribution strategy.
    """
    pairs = list(pairs)
    if not any(isinstance(node, FlattenedTensor)
               for _, expression in pairs
               for node in traversal((expression,))):
        return pairs

    result = []
    for variable, expression in pairs:
        expression = eliminate_deltas(expression)
        joint_outputs, changed = _unflatten_free_indices(
            variable, expression, split_separable_sums=False)
        refactored = _refactor_unflattened_outputs(
            joint_outputs) if changed else None
        if refactored is not None:
            result.extend(refactored)
            continue

        outputs, changed = _unflatten_free_indices(
            variable, expression, split_separable_sums=True)
        if changed or any(isinstance(node, Delta)
                          for _, output in outputs
                          for node in traversal((output,))):
            outputs = [(output_variable, contraction(output))
                       for output_variable, output in outputs]
        result.extend(outputs)
    return result


def _replace_flattened(node, self):
    node = reuse_if_untouched(node, self)
    if not isinstance(node, FlattenedTensor):
        return node
    expression, = node.children
    points = node.lattice_points()
    index = Index(extent=node.shape[0])
    subst = tuple(
        (axis, VariableIndex(Indexed(
            Literal(points[:, position], dtype=uint_type), (index,))))
        for position, axis in enumerate(node.multiindex))
    body = MemoizerArg(filtered_replace_indices)(expression, subst)
    return ComponentTensor(body, (index,))


def replace_flattened(expressions):
    """Lower remaining flattened tensors to indirect flat-index gathers."""
    mapper = Memoizer(_replace_flattened)
    return [mapper(expression) for expression in expressions]
