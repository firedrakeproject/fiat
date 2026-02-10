"""This module contains an implementation of the COFFEE optimisation
algorithm operating on a GEM representation.

This file is NOT for code generation as a COFFEE AST.
"""

from itertools import chain, repeat
import logging

import numpy

from gem.gem import IndexSum, one
from gem.optimise import make_sum, make_product
from gem.refactorise import Monomial
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


def find_optimal_atomics(monomials, linear_indices):
    """Find optimal atomic common subexpressions, which produce least number of
    terms in the resultant IndexSum when factorised.

    :arg monomials: A list of :class:`Monomial`s, all of which should have
                    the same sum indices
    :arg linear_indices: tuple of linear indices

    :returns: list of atomic GEM expressions
    """
    monomials = sort_monomials(monomials)

    atomics = tuple(dict.fromkeys(chain.from_iterable(monomial.atomics for monomial in monomials)))

    # Create a list of sets of indices to avoid any hashing during the search
    monomial_atomics = [set(map(atomics.index, m.atomics)) for m in monomials]

    # Precompute the cost of each atomic
    atomic_costs = list(map(index_extent, atomics, repeat(linear_indices)))

    def cost(solution):
        extent = sum(atomic_costs[i] for i in solution)
        # Prefer shorter solutions, but larger extents
        return (len(solution), -extent)

    optimal_solution = set(range(len(atomics)))  # pessimal but feasible solution
    optimal_cost = cost(optimal_solution)
    solution = set()
    solution_cost = (0, 0)

    max_it = 1 << 12
    it = iter(range(max_it))

    def solve(idx):
        nonlocal solution_cost, optimal_cost

        while idx < len(monomials) and solution.intersection(monomial_atomics[idx]):
            idx += 1

        if idx < len(monomials):
            if len(solution) < len(optimal_solution):
                for atomic in monomial_atomics[idx]:
                    atomic_cost = atomic_costs[atomic]
                    old_solution_cost = solution_cost
                    solution_cost = (solution_cost[0]+1, solution_cost[1]-atomic_cost)
                    if solution_cost < optimal_cost:
                        solution.add(atomic)
                        solve(idx + 1)
                        solution.remove(atomic)
                    solution_cost = old_solution_cost
        else:
            if solution_cost < optimal_cost:
                optimal_solution.clear()
                optimal_solution.update(solution)
                optimal_cost = solution_cost
            next(it)

    try:
        solve(0)
    except StopIteration:
        logger = logging.getLogger('tsfc')
        logger.warning("Solution to ILP problem may not be optimal: search "
                       "interrupted after examining %d solutions.", max_it)

    return tuple(atomics[i] for i in optimal_solution)


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
    factor_group = groupby(monomials, key=group_key)

    # We should not drop monomials
    assert sum(len(ms) for _, ms in factor_group) == len(monomials)

    sum_indices = next(iter(monomials)).sum_indices
    new_monomials = []
    for oa, monomials in factor_group:
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


def optimise_monomial_sum(monomial_sum, linear_indices):
    """Choose optimal common atomic subexpressions and factorise a
    :class:`MonomialSum` object to create a GEM expression.

    :arg monomial_sum: a :class:`MonomialSum` object
    :arg linear_indices: tuple of linear indices

    :returns: factorised GEM expression
    """
    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    new_monomials = []
    for _, monomials in groups:
        new_monomials.extend(optimise_monomials(monomials, linear_indices))
    return monomial_sum_to_expression(new_monomials)


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
        # Extract a connected component: subset of active monomials with intersecting atomics
        subset = {0}
        for candidate in range(1, len(active_monomials)):
            candidate_atomics = set(active_monomials[candidate].atomics)
            if any(candidate_atomics.intersection(active_monomials[j].atomics) for j in subset):
                subset.add(candidate)
        connected_monomials = [active_monomials[j] for j in subset]

        # Optimise the connected component and append to the result
        optimal_atomics = find_optimal_atomics(connected_monomials, linear_indices)
        result += factorise_atomics(connected_monomials, optimal_atomics, linear_indices)

        # Discard the connected component
        rest = set(range(len(active_monomials))) - subset
        active_monomials = [active_monomials[j] for j in rest]

    return result
