"""Lower and optimise :class:`gem.gem.FlattenedTensor` expressions."""

from collections import OrderedDict, defaultdict
from collections.abc import Iterable

import numpy

from gem.gem import (ComponentTensor, Delta, FlattenedTensor, Index, Node,
                     Indexed, IndexSum, JaggedIndex, Literal, Sum,
                     VariableIndex, uint_type)
from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, traversal
from gem.optimise import (_distribute_sum, delta_elimination,
                          eliminate_deltas, filtered_replace_indices, make_sum,
                          sum_factorise, traverse_product, traverse_sum)


def _clone_multiindex(multiindex: Iterable[Index]) -> tuple[Index, ...]:
    clones = {}
    for index in multiindex:
        if isinstance(index, JaggedIndex):
            parents = tuple(clones[parent] for parent in index.parents)
            clones[index] = JaggedIndex(extent=index.extent, parents=parents)
        else:
            clones[index] = Index(extent=index.extent)
    return tuple(clones[index] for index in multiindex)


def _replace_indices(node: Node, self, subst: tuple) -> Node:
    try:
        return self.replacements[node]
    except KeyError:
        return filtered_replace_indices(node, self, subst)


def _layout(gather: Indexed) -> tuple:
    tensor, = gather.children
    positions = {index: position
                 for position, index in enumerate(tensor.multiindex)}
    return tuple((type(index), index.extent,
                  tuple(positions[parent]
                        for parent in getattr(index, "parents", ())))
                 for index in tensor.multiindex)


def _candidate(nodes: Iterable[Node], indices: Iterable[Index]):
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
        layouts = {_layout(gather) for gather in candidates}
        if candidates and len(layouts) == 1 and index not in constrained:
            layout, = layouts
            return index, layout, candidates
    return None


def _sites(gathers: tuple[Indexed, ...]) -> tuple:
    gather = gathers[0]
    tensor, = gather.children
    assert all(_layout(other) == _layout(gather) for other in gathers)
    multiindex = _clone_multiindex(tensor.multiindex)
    replacer = MemoizerArg(filtered_replace_indices)
    mapper = MemoizerArg(_replace_indices)
    mapper.replacements = {}
    for other in gathers:
        tensor, = other.children
        mapper.replacements[other] = replacer(
            tensor.children[0], tuple(zip(tensor.multiindex, multiindex)))

    shape = tuple(index.extent for index in multiindex)
    points = gather.children[0].lattice_points()
    ordering = numpy.zeros(shape, dtype=uint_type)
    ordering[tuple(points.T)] = numpy.arange(len(points))
    flat = VariableIndex(Indexed(Literal(ordering, dtype=uint_type),
                                 multiindex))
    return mapper, multiindex, flat


def _separable_sum(node: Node, indices: frozenset[Index]) -> bool:
    """Whether distributing a sum exposes smaller index dependencies."""
    if not isinstance(node, Sum):
        return False
    involved = indices.intersection(node.free_indices)
    return bool(involved) and any(
        any(indices.intersection(factor.free_indices) < involved
            for factor in traverse_product(summand)[1])
        for summand in traverse_sum(node))


def _terms(summand: Node, index: Index) -> tuple[list, list[Node]]:
    rewritten = []
    leftover = []
    for term in traverse_sum(summand):
        candidate = _candidate(traversal((term,)), (index,))
        if candidate is None:
            leftover.append(term)
            continue
        _, _, gathers = candidate
        mapper, multiindex, flat = _sites(gathers)
        term = mapper(term, ((index, flat),))
        own = frozenset(multiindex)
        predicate = lambda node: _separable_sum(node, own)
        rewritten.extend(
            (multiindex, piece)
            for piece in _distribute_sum(term, predicate=predicate))
    return rewritten, leftover


def _unflatten(node: Node, self) -> Node:
    node = reuse_if_untouched(node, self)
    if not isinstance(node, IndexSum):
        return node
    summand, = node.children
    for index in node.multiindex:
        rewritten, leftover = _terms(summand, index)
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
    """Replace flat contractions by loops over their jagged lattices."""
    if not any(isinstance(node, FlattenedTensor)
               for node in traversal((expression,))):
        return expression
    return Memoizer(_unflatten)(expression)


def unflatten_returns(
        pairs: Iterable[tuple[Node, Node]]) -> list[tuple[Node, Node]]:
    """Unflatten free argument indices in assignment pairs."""
    pairs = list(pairs)
    if not any(isinstance(node, FlattenedTensor)
               for _, expression in pairs
               for node in traversal((expression,))):
        return pairs

    from gem.optimise import contraction

    result = []
    for variable, expression in pairs:
        pending = [(variable, eliminate_deltas(expression))]
        outputs = []
        changed = False
        while pending:
            var, expr = pending.pop()
            nodes = tuple(traversal((expr,)))
            candidate = _candidate(nodes, var.free_indices)
            if candidate is not None:
                index, layout, gathers = candidate
                groups = OrderedDict([
                    ((index, layout), ([expr], list(gathers)))])
            elif not any(isinstance(node, Indexed)
                         and len(node.multiindex) == 1
                         and node.multiindex[0] in var.free_indices
                         and isinstance(node.children[0], FlattenedTensor)
                         for node in nodes):
                outputs.append((var, expr))
                continue
            else:
                predicate = (lambda node: isinstance(node, Delta)) \
                    if any(isinstance(node, Delta) for node in nodes) else None
                groups = OrderedDict()
                for term in _distribute_sum(expr, predicate=predicate):
                    candidate = _candidate(traversal((term,)),
                                           var.free_indices)
                    key = candidate[:2] if candidate is not None else None
                    terms, gathers = groups.setdefault(key, ([], []))
                    terms.append(term)
                    if candidate is not None:
                        gathers.extend(candidate[2])

            for key, (terms, gathers) in groups.items():
                term = make_sum(terms)
                if key is None:
                    outputs.append((var, term))
                    continue
                changed = True
                index, _ = key
                gathers = tuple(OrderedDict.fromkeys(gathers))
                mapper, multiindex, flat = _sites(gathers)
                subst = ((index, flat),)
                new_var = MemoizerArg(filtered_replace_indices)(var, subst)
                new_term = mapper(term, subst)
                own = frozenset(multiindex)
                predicate = lambda node: _separable_sum(node, own)
                pending.extend((new_var, piece)
                               for piece in _distribute_sum(
                                   new_term, predicate=predicate))
        if changed or any(isinstance(node, Delta)
                          for _, expr in outputs
                          for node in traversal((expr,))):
            outputs = [(var, contraction(expr)) for var, expr in outputs]
        result.extend(outputs)
    return result


def _replace(node: Node, self) -> Node:
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


def replace_flattened(expressions: Iterable[Node]) -> list[Node]:
    """Lower remaining flattened tensors to indirect flat-index gathers."""
    mapper = Memoizer(_replace)
    return [mapper(expression) for expression in expressions]
