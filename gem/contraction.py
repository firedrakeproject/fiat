"""Choose arithmetic- and storage-efficient GEM contraction trees.

This module treats scalar association and index contraction as one
optimization problem on a tensor-network hypergraph.  Its implementation
counts rectangular and jagged iteration domains exactly, partitions
independent networks, and uses subset dynamic programming to place each
reduction at the earliest legal product subtree.  Plans minimize arithmetic
work first, then peak live and total materialized storage.

Finite-element rewrites remain in :mod:`gem.optimise`.  The seam here knows
nothing about element families, tabulations, or quadrature construction; it
receives contraction indices and scalar factors and returns a GEM expression.
"""

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from functools import lru_cache
import math
from typing import TypeVar

import numpy

from gem.gem import (Division, Index, IndexSum, Inverse, Literal,
                     MathFunction, MaxValue, MinValue, Node, Power, Product,
                     Solve, Sum)
from gem.node import traversal


T = TypeVar("T")


def partition_connected(
        items: Iterable[T],
        support: Callable[[T], Iterable[Hashable]]) -> tuple[tuple[T, ...], ...]:
    """Partition objects connected by transitive support overlap.

    Parameters
    ----------
    items
        Objects to partition.
    support
        Function returning the hypergraph vertices incident on an object.

    Returns
    -------
    tuple of tuple
        Connected components in deterministic input order.

    Notes
    -----
    Factors joined by contraction indices and monomials joined by common
    atomics are both incidence hypergraphs.  Traversing their bipartite
    incidence relation avoids constructing a quadratic pairwise-overlap
    graph and gives both optimizers the same definition of independence.

    """
    items = tuple(items)
    supports = tuple(frozenset(support(item)) for item in items)
    incidence = defaultdict(list)
    for position, keys in enumerate(supports):
        for key in keys:
            incidence[key].append(position)

    unseen = set(range(len(items)))
    components = []
    for seed in range(len(items)):
        if seed not in unseen:
            continue
        component = []
        pending = [seed]
        while pending:
            position = pending.pop()
            if position not in unseen:
                continue
            unseen.remove(position)
            component.append(position)
            for key in supports[position]:
                pending.extend(reversed(incidence[key]))
        components.append(tuple(items[position]
                                for position in sorted(component)))
    return tuple(components)


def index_closure(indices: Iterable[Index]) -> frozenset[Index]:
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
    indices = index_closure(indices)
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
    indices = index_closure(indices)
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


def has_arithmetic(expressions: Iterable[Node]) -> bool:
    """Does a GEM DAG perform any scalar arithmetic?

    Parameters
    ----------
    expressions
        Roots of a scalar GEM expression DAG.

    Returns
    -------
    bool
        Whether the operation count of :func:`estimate_cost` is nonzero.

    Notes
    -----
    A tabulation reference costs nothing to evaluate, so materializing it
    buys no arithmetic.  Deciding that needs no storage model.

    """
    return any(map(_operation_count, traversal(tuple(expressions))))


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


@lru_cache(maxsize=4096)
def _contraction_component(
        sum_indices: tuple[Index, ...],
        factors: tuple[Node, ...]) -> Node:
    """Optimize one connected tensor contraction by subset DP.

    A state is a subset of factors.  A contraction index becomes *closed*
    when the state contains every factor incident on that index; reducing it
    at that point is precisely the earliest legal code motion.  The state
    retains only indices needed by factors outside the subset.  Consequently,
    its score compares plans lexicographically by FLOPs, peak live
    intermediate storage, and total materialized storage.

    This dynamic program subsumes scalar reassociation for a connected
    contraction: each bipartition chooses both a product association and the
    reductions that become legal there.  :func:`associate` handles expressions
    for which no contraction-placement decision remains.  Beyond ten factors
    a deterministic index-at-a-time plan bounds the exponential search.

    Plans are memoized because a loop-ordering search re-plans the same
    components once per candidate ordering.

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

    """
    if len(factors) > 10:
        terms = list(factors)
        for index in reversed(sum_indices):
            contract = [term for term in terms
                        if index in term.free_indices]
            if not contract:
                continue
            deferred = [term for term in terms
                        if index not in term.free_indices]
            product, _ = associate(Product, contract)
            result = IndexSum(product, (index,))
            terms = deferred + [result]
        result, _ = associate(Product, terms)
        return result

    full_mask = (1 << len(factors)) - 1
    factor_indices = tuple(
        index_closure(factor.free_indices) for factor in factors)
    supports = {
        index: sum(
            1 << position
            for position, indices in enumerate(factor_indices)
            if index in indices)
        for index in sum_indices
    }

    closed_cache = {}
    live_cache = {}

    def closed(mask: int) -> frozenset[Index]:
        try:
            return closed_cache[mask]
        except KeyError:
            pass
        result = frozenset(
            index for index, support in supports.items()
            if support and not support & ~mask)
        closed_cache[mask] = result
        return result

    def live(mask: int) -> frozenset[Index]:
        try:
            return live_cache[mask]
        except KeyError:
            pass
        indices = set().union(*(
            factor_indices[position]
            for position in range(len(factors))
            if mask & (1 << position)))
        indices.difference_update(closed(mask))
        result = frozenset(indices)
        live_cache[mask] = result
        return result

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
    _, expression, _, _ = plans[full_mask]
    return expression


def factor_contraction(
        sum_indices: Iterable[Index],
        factors: Iterable[Node]) -> Node:
    """Plan a normalized tensor contraction.

    Parameters
    ----------
    sum_indices
        Indices to contract in deterministic loop order.
    factors
        Scalar factors whose product is contracted.

    Returns
    -------
    Node
        A contraction tree with independent tensor networks reduced
        separately.

    Notes
    -----
    The incidence hypergraph has one vertex for each contracted index and
    one hyperedge for each factor.  Connected components have no shared
    contraction and can therefore be planned independently.  Within each
    component, subset dynamic programming chooses both product association
    and the earliest legal reductions.

    """
    sum_indices = tuple(sum_indices)
    factors = tuple(factors)
    contraction_set = frozenset(sum_indices)
    factor_indices = tuple(
        index_closure(factor.free_indices) & contraction_set
        for factor in factors)
    components = partition_connected(
        range(len(factors)), lambda position: factor_indices[position])

    expressions = []
    for component in components:
        component_factors = tuple(factors[position] for position in component)
        involved = set().union(*(
            factor_indices[position] for position in component))
        component_indices = tuple(
            index for index in sum_indices if index in involved)
        expressions.append(_contraction_component(
            component_indices, component_factors))
    expression, _ = associate(Product, expressions)
    return expression
