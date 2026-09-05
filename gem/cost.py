"""Cost models for GEM expressions.

`node_cost` is the single home for the per-node cost constants.  Two
counters weigh them differently: `gem.flop_count` walks a scheduled Impero
tree and multiplies by the enclosing loop extents, while `estimate_cost`
scores a GEM DAG over each node's own free indices, before any schedule
exists.  Only the weighting differs, so the constants and the special cases
live here.
"""

import math
from collections.abc import Iterable
from functools import singledispatch

import numpy

from gem.gem import (Comparison, Division, Index, IndexSum, Inverse, Literal,
                     MathFunction, MaxValue, MinValue, Node, Power, Product,
                     Solve, Sum)
from gem.node import traversal


@singledispatch
def node_cost(node: Node) -> int:
    """Scalar operations performed by one evaluation of a GEM node.

    The children are not included; a caller adds them, and weighs this by
    however many times the node is evaluated.

    :arg node: a GEM expression node
    :returns: operations performed at the node itself
    """
    return 0


@node_cost.register(Product)
def _(node):
    # Negation folds into the operation that consumes it, so a product with
    # -1 costs nothing.
    if any(isinstance(child, Literal) and not child.shape and child.value == -1
           for child in node.children):
        return 0
    return 1


@node_cost.register(Sum)
@node_cost.register(Division)
@node_cost.register(Comparison)
@node_cost.register(MathFunction)
@node_cost.register(MinValue)
@node_cost.register(MaxValue)
def _(node):
    return 1


@node_cost.register(Power)
def _(node):
    _, exponent = node.children
    if isinstance(exponent, Literal) and not exponent.shape:
        value = exponent.value
        if value > 0 and value == math.floor(value):
            # Repeated squaring.
            return int(math.ceil(math.log2(value)))
    return 5  # heuristic


@node_cost.register(Inverse)
def _(node):
    n, _ = node.shape
    return 2 * n ** 3


@node_cost.register(Solve)
def _(node):
    n, m = node.shape
    # A right-hand side each, on top of inverting the matrix.
    return 2 * n * m + 2 * n ** 3


def iteration_count(indices: Iterable[Index]) -> int:
    """Count the points of a rectangular iteration space.

    :arg indices: indices an operation depends on
    :returns: the number of executions of the operation
    """
    return int(numpy.prod([index.extent for index in indices], dtype=int))


def operation_count(node: Node) -> int:
    """Estimate the scalar operations performed by one GEM node.

    :arg node: scalar expression node
    :returns: operations over the node's complete iteration domain
    """
    if isinstance(node, IndexSum):
        # One accumulation per point of the body's domain.
        body, = node.children
        return iteration_count(body.free_indices)
    if isinstance(node, (Inverse, Solve)):
        # A shaped node carries its own domain in its shape.
        return node_cost(node)
    return node_cost(node) * iteration_count(node.free_indices)


def has_arithmetic(expressions: Iterable[Node]) -> bool:
    """Does a GEM DAG perform any scalar arithmetic?

    Materialising a tabulation reference buys no arithmetic, so sharing one
    only adds storage.

    :arg expressions: roots of a scalar GEM expression DAG
    :returns: whether any node costs arithmetic to evaluate
    """
    return any(map(operation_count, traversal(tuple(expressions))))


def estimate_cost(expressions: Iterable[Node]) -> tuple[int, int, int, int]:
    """Estimate arithmetic work and contraction storage for a GEM DAG.

    Each structurally shared operation counts once over the domain its free
    indices induce.  An :class:`~gem.gem.IndexSum` contributes one
    accumulation per point of its body domain.  Storage counts the result
    domains of contractions, the intermediates that scheduling exposes.

    :arg expressions: roots of a scalar GEM expression DAG
    :returns: operation count, total contraction storage, largest
              contraction, and expression-node count.  Comparing the tuples
              lexicographically ranks arithmetic first and breaks ties by
              storage.
    """
    nodes = tuple(traversal(tuple(expressions)))
    sizes = [iteration_count(node.free_indices)
             for node in nodes if isinstance(node, IndexSum)]
    return (sum(map(operation_count, nodes)),
            sum(sizes),
            max(sizes, default=0),
            len(nodes))
