"""Jagged tensors: the geometry of a `JaggedIndex` lattice, and the
rewrites that trade a flat axis for the lattice it enumerates.

`gem.gem` defines what a jagged domain *is* and enumerates its points.  This
module builds on that: it compacts a product of simplex lattices into
storage, and it rewrites contractions and assignments that gather a
`FlattenedTensor` so that the ordinary contraction optimiser can see the
tensor-product factors again.
"""

from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from functools import lru_cache, partial

import numpy

from gem.gem import (ComponentTensor, Delta, FlattenedTensor, Index, IndexSum,
                     Indexed, JaggedIndex, Literal, Node, Sum, VariableIndex,
                     jagged_lattice, jagged_layout, lattice_points, uint_type)
from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, traversal
from gem.optimise import (delta_elimination, distribute_sum,
                          filtered_replace_indices, make_sum, sum_factorise,
                          traverse_product, traverse_sum)


def _index_components(indices: tuple[Index, ...]) -> tuple[tuple, ...]:
    """Partition indices into connected parent domains."""
    index_set = frozenset(indices)
    neighbours = {index: set() for index in indices}
    for index in indices:
        for parent in index.parents:
            if parent in index_set:
                neighbours[index].add(parent)
                neighbours[parent].add(index)

    components = []
    unseen = set(indices)
    for seed in indices:
        if seed not in unseen:
            continue
        component = {seed}
        pending = [seed]
        unseen.remove(seed)
        while pending:
            for index in neighbours[pending.pop()] & unseen:
                unseen.remove(index)
                component.add(index)
                pending.append(index)
        components.append(tuple(index for index in indices
                                if index in component))
    return tuple(components)


def _is_simplex_lattice(component: tuple[Index, ...]) -> bool:
    """Check whether indices describe a nested simplex lattice."""
    return all(
        isinstance(index, JaggedIndex)
        and index.extent == component[0].extent
        and index.parents == component[:position]
        for position, index in enumerate(component))


def compact_index_layout(
        indices: tuple[Index, ...]) -> tuple[tuple[int, ...], tuple]:
    """Compact a product of independent simplex lattices.

    Parameters
    ----------
    indices
        Indices in loop order.

    Returns
    -------
    shape
        Compact storage shape.
    layout
        Scalar indices and compact simplex components.

    Notes
    -----
    A simplex lattice is stored along one compact dimension holding just
    its lattice points, in the same lexicographic order that
    `FlattenedTensor` flattens a jagged tensor. Rectangular padding is
    exponential in the lattice dimension, so it is avoided whenever the
    lattice is smaller than the box enclosing it.

    """
    shape = []
    layout = []
    for component in _index_components(indices):
        points = _compact_extent(component)
        if points is None:
            shape.extend(index.extent for index in component)
            layout.extend(component)
        else:
            shape.append(points)
            layout.append(component)
    return tuple(shape), tuple(layout)


def _compact_extent(component: tuple[Index, ...]) -> int | None:
    """Return the compact extent of a simplex lattice, or None to pad.

    A lattice that already fills its box gains nothing from compaction:
    the rank lookup would just be the identity.

    """
    if not _is_simplex_lattice(component):
        return None
    points = len(jagged_lattice(component))
    if points >= numpy.prod([index.extent for index in component]):
        return None
    return points


@lru_cache(maxsize=128)
def _lattice_ranks(layout: tuple) -> numpy.ndarray:
    """Rank of every point of one structural jagged iteration domain."""
    points = lattice_points(layout)
    ranks = numpy.zeros(tuple(extent for extent, _ in layout), dtype=uint_type)
    ranks[tuple(points.T)] = numpy.arange(len(points), dtype=uint_type)
    ranks.flags.writeable = False
    return ranks


def simplex_lattice_ranks(component: tuple[Index, ...]) -> numpy.ndarray:
    """Tabulate the compact rank of every point of a simplex lattice.

    Parameters
    ----------
    component
        Nested simplex lattice indices, in loop order.

    Returns
    -------
    numpy.ndarray
        Rectangular table of the lexicographic rank of each lattice
        point, indexed by the lattice indices themselves. Entries
        outside the jagged bounds are never read and are set to zero.

    """
    return _lattice_ranks(jagged_layout(component))


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
                  tuple(positions[parent] for parent in index.parents))
                 for index in tensor.multiindex)


def _flat_index_bijection(
        index, extent: int) -> tuple[Index, tuple[int, ...] | None] | None:
    """Identify a direct or compile-time bijective flat index.

    Parameters
    ----------
    index
        Index of a flattened tensor.
    extent
        Length of the flattened tensor.

    Returns
    -------
    tuple or None
        Source index and its forward permutation. A direct index has no
        permutation.

    """
    if isinstance(index, Index):
        return (index, None) if index.extent == extent else None
    if not isinstance(index, VariableIndex):
        return None

    expression = index.expression
    if not (isinstance(expression, Indexed)
            and len(expression.multiindex) == 1
            and isinstance(expression.multiindex[0], Index)
            and isinstance(expression.children[0], Literal)):
        return None
    source, = expression.multiindex
    table, = expression.children
    if table.shape != (source.extent,) or source.extent != extent:
        return None

    permutation = tuple(map(int, table.array))
    if tuple(sorted(permutation)) != tuple(range(extent)):
        return None
    return source, permutation


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
    gathers = defaultdict(lambda: defaultdict(OrderedDict))
    for node in nodes:
        if isinstance(node, Delta):
            constrained.update(node.free_indices)
        elif (isinstance(node, Indexed)
              and len(node.multiindex) == 1
              and isinstance(node.children[0], FlattenedTensor)):
            tensor, = node.children
            bijection = _flat_index_bijection(
                node.multiindex[0], tensor.shape[0])
            if bijection is None:
                continue
            source, permutation = bijection
            if source in index_set:
                key = _flattened_layout(node), permutation
                gathers[source][key].setdefault(node)

    for index in indices:
        groups = gathers[index]
        if len(groups) == 1 and index not in constrained:
            layout, candidates = next(iter(groups.items()))
            return index, layout, tuple(candidates)
    return None


def _prepare_unflattening(
        gathers: tuple[Indexed, ...],
        source: Index) -> tuple[MemoizerArg, tuple, VariableIndex]:
    """Prepare one joint rewrite of compatible flat gathers.

    Each flattened tensor is inlined on the same fresh lattice multiindex.
    The returned index maps each lattice point to the original source index.
    A compile-time permutation is inverted before the index is used in the
    return variable.
    """
    gather = gathers[0]
    tensor, = gather.children
    assert all(_flattened_layout(other) == _flattened_layout(gather)
               for other in gathers)
    bijection = _flat_index_bijection(
        gather.multiindex[0], tensor.shape[0])
    assert bijection is not None and bijection[0] == source
    permutation = bijection[1]
    assert all(_flat_index_bijection(
        other.multiindex[0], other.children[0].shape[0]) == bijection
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
    if permutation is not None:
        inverse = numpy.empty(len(permutation), dtype=uint_type)
        inverse[numpy.asarray(permutation)] = numpy.arange(
            len(permutation), dtype=uint_type)
        ordering = inverse[ordering]
    source_index = VariableIndex(Indexed(
        Literal(ordering, dtype=uint_type), multiindex))
    return mapper, multiindex, source_index


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
        mapper, multiindex, source_index = _prepare_unflattening(
            gathers, index)
        term = mapper(term, ((index, source_index),))
        own = frozenset(multiindex)
        predicate = partial(_separable_sum, indices=own)
        rewritten.extend(
            (multiindex, piece)
            for piece in distribute_sum(term, predicate=predicate))
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


def unflatten_free_indices(
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
            for term in distribute_sum(
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
            mapper, multiindex, source_index = _prepare_unflattening(
                gathers, index)
            substitution = ((index, source_index),)
            new_variable = MemoizerArg(filtered_replace_indices)(
                current_variable, substitution)
            new_expression = mapper(term, substitution)
            if split_separable_sums:
                lattice_indices = frozenset(multiindex)
                predicate = partial(_separable_sum, indices=lattice_indices)
                pending.extend(
                    (new_variable, piece)
                    for piece in distribute_sum(
                        new_expression, predicate=predicate))
            else:
                pending.append((new_variable, new_expression))
    return outputs, changed


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
