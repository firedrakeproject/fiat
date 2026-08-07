"""This module contains an implementation of the COFFEE optimisation
algorithm operating on a GEM representation.

This file is NOT for code generation as a COFFEE AST.
"""

from collections import OrderedDict, defaultdict
from itertools import chain
import logging

import numpy

from gem.gem import IndexSum, Node, one
from gem.optimise import make_sum, make_product
from gem.refactorise import Monomial, MonomialSum
from gem.utils import groupby


__all__ = ['optimise_monomial_sum']


def monomial_sum_to_expression(monomial_sum):
    """Convert a monomial sum to a GEM expression.

    :arg monomial_sum: an iterable of :class:`Monomial`s

    :returns: GEM expression
    """
    indexsums = []  # The result is summation of indexsums
    # Group monomials according to their sum indices
    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    # Create IndexSum's from each monomial group
    for _, monomials in groups:
        sum_indices = monomials[0].sum_indices
        products = [make_product(monomial.atomics + (monomial.rest,)) for monomial in monomials]
        indexsums.append(IndexSum(make_sum(products), sum_indices))
    return make_sum(indexsums)


def index_extent(factor, linear_indices):
    """Compute the product of the extents of linear indices of a GEM expression

    :arg factor: GEM expression
    :arg linear_indices: set of linear indices

    :returns: product of extents of linear indices
    """
    return numpy.prod([i.extent for i in factor.free_indices if i in linear_indices])


def sort_monomials(monomials):
    """Sort monomials to produce a better initial guess for :func:`find_optimal_atomics`.

    :arg monomials: A list of :class:`Monomial`s

    :returns: the reordered list of monomials.
    """
    if len(monomials) <= 2:
        return monomials
    # Construct a monomial subset with non-intersecting atomics
    head = []
    rest = []
    atomics = set()
    for m in monomials:
        if atomics.intersection(m.atomics):
            rest.append(m)
        else:
            atomics.update(m.atomics)
            head.append(m)
    # Put non-intersecting subset first and recurse on the rest
    monomials = head + sort_monomials(rest)
    return monomials


def find_optimal_atomics(
        monomials: list[Monomial],
        linear_indices: tuple) -> tuple[Node, ...]:
    """Find a minimum-cost set of atomics intersecting every monomial.

    Parameters
    ----------
    monomials
        Monomials with equal contraction indices.
    linear_indices
        Free indices belonging to form arguments.

    Returns
    -------
    tuple of Node
        Atomics selected for common-subexpression factorization.

    """
    monomials = sort_monomials(list(monomials))
    atomics = tuple(dict.fromkeys(chain.from_iterable(
        monomial.atomics for monomial in monomials)))
    if not atomics:
        return ()

    positions = {atomic: position
                 for position, atomic in enumerate(atomics)}
    constraints = tuple(dict.fromkeys(
        sum({1 << positions[atomic] for atomic in monomial.atomics})
        for monomial in monomials))
    constraints = tuple(
        constraint for constraint in constraints
        if not any(
            other != constraint and other & constraint == other
            for other in constraints))
    costs = tuple(int(index_extent(atomic, linear_indices))
                  for atomic in atomics)
    covers = tuple(sum(
        1 << position
        for position, constraint in enumerate(constraints)
        if constraint & (1 << atomic))
        for atomic in range(len(atomics)))
    full_cover = (1 << len(constraints)) - 1

    covered = 0
    best_solution = 0
    while covered != full_cover:
        atomic = max(
            range(len(atomics)),
            key=lambda candidate: (
                ((covers[candidate] & ~covered).bit_count(),
                 costs[candidate], -candidate)))
        best_solution |= 1 << atomic
        covered |= covers[atomic]

    def solution_cost(solution):
        selected = [position for position in range(len(atomics))
                    if solution & (1 << position)]
        return len(selected), -sum(costs[position] for position in selected)

    best_cost = solution_cost(best_solution)
    seen = {}
    states = 0
    max_states = 1 << 16

    def lower_bound(uncovered):
        disjoint = 0
        count = 0
        for position in sorted(
                (position for position in range(len(constraints))
                 if uncovered & (1 << position)),
                key=lambda position: constraints[position].bit_count()):
            constraint = constraints[position]
            if not constraint & disjoint:
                disjoint |= constraint
                count += 1
        return count

    def solve(covered, solution, cardinality, extent):
        nonlocal best_solution, best_cost, states
        states += 1
        if states > max_states:
            raise StopIteration

        cost = cardinality, -extent
        if seen.get(covered, (numpy.inf, numpy.inf)) <= cost:
            return
        seen[covered] = cost
        if covered == full_cover:
            best_solution, best_cost = solution, cost
            return

        uncovered = full_cover & ~covered
        if cardinality + lower_bound(uncovered) > best_cost[0]:
            return
        constraint = min(
            (position for position in range(len(constraints))
             if uncovered & (1 << position)),
            key=lambda position: constraints[position].bit_count())
        choices = [atomic for atomic in range(len(atomics))
                   if constraints[constraint] & (1 << atomic)]
        choices.sort(
            key=lambda atomic: (
                (covers[atomic] & uncovered).bit_count(),
                costs[atomic], -atomic),
            reverse=True)
        for atomic in choices:
            candidate_cost = cardinality + 1, -(extent + costs[atomic])
            if candidate_cost < best_cost:
                solve(
                    covered | covers[atomic],
                    solution | (1 << atomic),
                    cardinality + 1,
                    extent + costs[atomic])

    try:
        solve(0, 0, 0, 0)
    except StopIteration:
        logging.getLogger("tsfc").warning(
            "Solution to hitting-set problem may not be optimal: search "
            "interrupted after examining %d states.", max_states)

    return tuple(
        atomic for position, atomic in enumerate(atomics)
        if best_solution & (1 << position))


def factorise_atomics(monomials, optimal_atomics, linear_indices):
    """Group and factorise monomials using a list of atomics as common
    subexpressions. Create new monomials for each group and optimise them recursively.

    :arg monomials: an iterable of :class:`Monomial`s, all of which should have
                    the same sum indices
    :arg optimal_atomics: list of tuples of atomics to be used as common subexpression
    :arg linear_indices: tuple of linear indices

    :returns: an iterable of :class:`Monomials`s after factorisation
    """
    if not optimal_atomics or len(monomials) <= 1:
        return monomials

    # Group monomials with respect to each optimal atomic
    def group_key(monomial):
        for oa in optimal_atomics:
            if oa in monomial.atomics:
                return oa
        assert False, "Expect at least one optimal atomic per monomial."
    factor_groups = OrderedDict()
    for monomial in monomials:
        factor_groups.setdefault(group_key(monomial), []).append(monomial)

    sum_indices = next(iter(monomials)).sum_indices
    new_monomials = []
    for oa, monomials in factor_groups.items():
        # Create new MonomialSum for the factorised out terms
        sub_monomials = []
        for monomial in monomials:
            atomics = list(monomial.atomics)
            atomics.remove(oa)  # remove common factor
            sub_monomials.append(Monomial((), tuple(atomics), monomial.rest))
        # Continue to factorise the remaining expression
        sub_monomials = optimise_monomials(sub_monomials, linear_indices)
        if len(sub_monomials) == 1:
            # Factorised part is a product, we add back the common atomics then
            # add to new MonomialSum directly rather than forming a product node
            # Retaining the monomial structure enables applying associativity
            # when forming GEM nodes later.
            sub_monomial, = sub_monomials
            new_monomials.append(
                Monomial(sum_indices, (oa,) + sub_monomial.atomics, sub_monomial.rest))
        else:
            # Factorised part is a summation, we need to create a new GEM node
            # and multiply with the common factor
            node = monomial_sum_to_expression(sub_monomials)
            # If the free indices of the new node intersect with linear indices,
            # add to the new monomial as `atomic`, otherwise add as `rest`.
            # Note: we might want to continue to factorise with the new atomics
            # by running optimise_monoials twice.
            if set(linear_indices) & set(node.free_indices):
                new_monomials.append(Monomial(sum_indices, (oa, node), one))
            else:
                new_monomials.append(Monomial(sum_indices, (oa, ), node))
    return new_monomials


def collect_common_rests(monomials):
    """Group monomials with a common scalar coefficient.

    This applies ``r*a + r*b = r*(a + b)`` after argument-factor
    extraction, allowing the scalar part of a finite element contraction to
    participate in COFFEE factorisation.

    Parameters
    ----------
    monomials : iterable of Monomial
        Monomials with equal contraction indices.

    Returns
    -------
    list of Monomial
        Monomials after common coefficients have been collected.

    """
    def group_key(monomial):
        linear = frozenset(chain.from_iterable(
            atomic.free_indices for atomic in monomial.atomics))
        return frozenset(monomial.sum_indices), linear, monomial.rest

    result = []
    for (_, _, rest), group in groupby(monomials, key=group_key):
        if len(group) > 1 and rest != one and all(
                monomial.atomics for monomial in group):
            sum_indices = group[0].sum_indices
            node = make_sum(tuple(
                make_product(monomial.atomics) for monomial in group))
            result.append(Monomial(sum_indices, (node,), rest))
        else:
            result.extend(group)
    return result


def optimise_monomial_sum(
        monomial_sum: MonomialSum,
        linear_indices: tuple,
        contraction_order: tuple = ()) -> Node:
    """Factor monomial algebra and place ordered contractions.

    Parameters
    ----------
    monomial_sum
        Sum-of-products representation to optimize.
    linear_indices
        Free indices identifying argument tabulations.
    contraction_order
        Contraction indices, from outermost to innermost stage.

    Returns
    -------
    Node
        Factorized GEM expression.

    """
    if contraction_order:
        grouped = defaultdict(MonomialSum)
        order = OrderedDict()
        remaining = frozenset(contraction_order)
        for monomial in monomial_sum:
            inner_indices = tuple(index for index in monomial.sum_indices
                                  if index in remaining)
            involved = frozenset(inner_indices)
            inner_atomics = tuple(
                atomic for atomic in monomial.atomics
                if involved.intersection(atomic.free_indices))
            outer_indices = tuple(index for index in monomial.sum_indices
                                  if index not in remaining)
            outer_atomics = tuple(
                atomic for atomic in monomial.atomics
                if atomic not in inner_atomics)
            key = outer_indices, outer_atomics
            order.setdefault(key)
            grouped[key].add(
                inner_indices, inner_atomics, monomial.rest)

        outer_sum = MonomialSum()
        for outer_indices, outer_atomics in order:
            inner = optimise_monomial_sum(
                grouped[(outer_indices, outer_atomics)],
                linear_indices, contraction_order[1:])
            outer_sum.add(outer_indices, outer_atomics, inner)
        monomial_sum = outer_sum

    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    optimized = []
    for _, monomials in groups:
        old_size = len(monomials) + 1
        while len(monomials) < old_size:
            old_size = len(monomials)
            monomials = optimise_monomials(monomials, linear_indices)
        optimized.extend(monomials)
    return monomial_sum_to_expression(optimized)


def optimise_monomials(monomials, linear_indices):
    """Choose optimal common atomic subexpressions and factorise an iterable
    of monomials.

    :arg monomials: a list of :class:`Monomial`s, all of which should have
                    the same sum indices
    :arg linear_indices: tuple of linear indices

    :returns: an iterable of factorised :class:`Monomials`s
    """
    assert len(set(frozenset(m.sum_indices) for m in monomials)) <= 1, \
        "All monomials required to have same sum indices for factorisation"

    result = [m for m in monomials if not m.atomics]  # skipped monomials
    active_monomials = [m for m in monomials if m.atomics]

    while len(active_monomials) > 0:
        # Extract a connected component: maximal subset of monomials with intersecting atomics
        old_size = 0
        subset = {active_monomials[0]}
        while len(subset) > old_size:
            old_size = len(subset)
            for candidate in active_monomials:
                if candidate not in subset:
                    candidate_atomics = frozenset(candidate.atomics)
                    if any(candidate_atomics.intersection(m.atomics) for m in subset):
                        subset.add(candidate)
        connected_monomials = [m for m in active_monomials if m in subset]

        # Optimise the connected component and append to the result
        optimal_atomics = find_optimal_atomics(connected_monomials, linear_indices)
        result += factorise_atomics(connected_monomials, optimal_atomics, linear_indices)

        # Discard the connected component
        active_monomials = [m for m in active_monomials if m not in subset]

    return collect_common_rests(result)
