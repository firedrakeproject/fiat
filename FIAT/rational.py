# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
r"""Rational functions in barycentric coordinates.

The rational monomials

.. math::

    R^\alpha_\beta = \prod_j \lambda_j^{\alpha_j} (1 - \lambda_j)^{-\beta_j}

span the rational finite element spaces of Guzman and Neilan and the
blow-up Whitney forms, see

Diening, Storn, Tscherpel, Exact integration for singular Zienkiewicz and
Guzman--Neilan finite elements with implementation, CAMWA 191 (2025).

A rational monomial is encoded by its exponents ``(alpha, beta)``, stored as a
row of an integer array of shape ``(num_monomials, 2*(sd+1))``.
"""

import itertools
import math
from functools import lru_cache

import numpy

from FIAT import expansions
from FIAT.polynomial_set import mis
from FIAT.reference_element import make_affine_mapping


def barycentric_gradients(ref_el):
    """Return the gradients of the barycentric coordinates of a simplex.

    :arg ref_el: a simplex.
    :returns: an array whose entry ``[j, k]`` is d lambda_j / d x_k.
    """
    sd = ref_el.get_spatial_dimension()
    A, _ = make_affine_mapping(ref_el.get_vertices(), numpy.eye(sd+1))
    return A


def rational_union(*exponents):
    """Express several rational monomial sets in a common set.

    :arg exponents: arrays of exponents of rational monomials.
    :returns: a tuple ``(union, indices)`` where ``union`` contains the
        unique exponents, and ``union[indices[i]] == exponents[i]``.
    """
    union, inverse = numpy.unique(numpy.concatenate(exponents), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    offsets = numpy.cumsum([0] + [len(E) for E in exponents])
    indices = [inverse[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    return union, indices


def rational_derivative(exponents, coeffs, direction):
    r"""Differentiate a linear combination of rational monomials.

    Applies the quotient rule, which corrects the sign of the second term in
    (Diening, Storn, and Tscherpel, 2025, eq. (3.6)),

    .. math::

        \partial_{\lambda_j} R^\alpha_\beta
        = \alpha_j R^{\alpha-e_j}_\beta + \beta_j R^\alpha_{\beta+e_j}.

    :arg exponents: an array of exponents of rational monomials.
    :arg coeffs: an array whose last axis indexes the monomials.
    :arg direction: the derivative direction in barycentric coordinates.
    :returns: a tuple ``(exponents, coeffs)`` of the derivative.
    """
    exponents = numpy.asarray(exponents)
    num_monomials, num_cols = exponents.shape
    nb = num_cols // 2
    shifted = []
    weights = []
    for j in range(nb):
        for col, step in ((j, -1), (nb + j, 1)):
            E = exponents.copy()
            E[:, col] += step
            shifted.append(E)
            weights.append(exponents[:, col] * direction[j])
    shifted = numpy.concatenate(shifted)
    weights = numpy.concatenate(weights)
    source = numpy.tile(numpy.arange(num_monomials), 2*nb)

    keep = weights != 0
    if not keep.any():
        return exponents[:0], numpy.zeros(coeffs.shape[:-1] + (0,))
    new_exponents, (target,) = rational_union(shifted[keep])
    D = numpy.zeros((num_monomials, len(new_exponents)))
    numpy.add.at(D, (source[keep], target), weights[keep])
    return new_exponents, numpy.dot(coeffs, D)


def rational_multiply(exponents1, coeffs1, exponents2, coeffs2):
    """Multiply two sets of linear combinations of rational monomials.

    :arg exponents1: an array of exponents of rational monomials.
    :arg coeffs1: an array of shape ``(n1, len(exponents1))``.
    :arg exponents2: an array of exponents of rational monomials.
    :arg coeffs2: an array of shape ``(n2, len(exponents2))``.
    :returns: a tuple ``(exponents, coeffs)`` where ``coeffs`` has shape
        ``(n1, n2, len(exponents))`` and holds all pairwise products.
    """
    exponents1 = numpy.asarray(exponents1)
    exponents2 = numpy.asarray(exponents2)
    products = (exponents1[:, None, :] + exponents2[None, :, :]).reshape(-1, exponents1.shape[1])
    exponents, (index,) = rational_union(products)
    coeffs = numpy.zeros((len(coeffs1), len(coeffs2), len(exponents)))
    terms = numpy.einsum("ia,jb->ijab", coeffs1, coeffs2).reshape(len(coeffs1), len(coeffs2), -1)
    for k, col in enumerate(index):
        coeffs[:, :, col] += terms[:, :, k]
    return exponents, coeffs


def rational_evaluate(exponents, bary):
    """Evaluate rational monomials at points given in barycentric coordinates.

    At a vertex, the monomials are evaluated by their limit along the median,
    which coincides with the value of any linear combination of monomials
    that is continuous at that vertex.

    :arg exponents: an array of exponents of rational monomials.
    :arg bary: an array of shape ``(num_points, sd+1)`` of barycentric coordinates.
    :returns: an array of shape ``(num_monomials, num_points)``.
    """
    exponents = numpy.asarray(exponents)
    nb = exponents.shape[1] // 2
    alpha = exponents[:, None, :nb]
    beta = exponents[:, None, nb:]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        values = numpy.prod(bary ** alpha / (1 - bary) ** beta, axis=-1)

    for j in range(nb):
        at_vertex = numpy.isclose(bary[:, j], 1)
        if at_vertex.any():
            # Approach vertex j along the median: lambda_k = s / (nb - 1) for k != j
            order = exponents[:, :nb].sum(axis=1) - exponents[:, j]
            gap = order - exponents[:, nb + j]
            limit = numpy.where(gap > 0, 0.0, numpy.where(gap == 0, (nb-1.0) ** -order, numpy.inf))
            values[:, at_vertex] = limit[:, None]
    return values


def _mean_two_poles(a1, a2, b1, b2):
    """Mean of lambda_1^a1 lambda_2^a2 / ((1-lambda_1)^b1 (1-lambda_2)^b2) as (q, p), meaning q + p pi^2."""
    if b2 < b1:
        a1, a2, b1, b2 = a2, a1, b2, b1
    if b1 == 0:
        return (2 / (a1+1) * math.factorial(a2) * math.factorial(a1-b2+1) / math.factorial(a1+a2-b2+2), 0.0)
    if b1 == 1 and b2 == 1:
        q = -2 * sum(1 / i**2 for i in range(1, a2+1))
        q -= 2 * sum(math.factorial(a2) * math.factorial(j-1) / (j * math.factorial(a2+j)) for j in range(1, a1+1))
        return (q, 1/3)
    if b1 == 1:
        c = (b2-a2-2) / (b2-1)
        rest = 2 / (b2-1) * math.factorial(a1-b2+1) * math.factorial(a2) / math.factorial(a1-b2+a2+2)
        q, p = _mean_two_poles(a1, a2, 1, b2-1) if c != 0 else (0.0, 0.0)
        return (c*q + rest, c*p)
    c = (b1-a1-2) / (b1-1)
    rest = 2 / (b1-1) * math.factorial(a2-b1+1) * math.factorial(a1-b2+1) / math.factorial(a2-b1+a1-b2+3)
    q, p = _mean_two_poles(a1, a2, b1-1, b2) if c != 0 else (0.0, 0.0)
    return (c*q + rest, c*p)


@lru_cache(maxsize=None)
def _mean(alpha, beta):
    """Mean of R^alpha_beta over a triangle as (q, p), meaning q + p pi^2.

    Implements the recursion of (Diening, Storn, and Tscherpel, 2025, Algorithm 1).
    """
    order = sorted(range(3), key=lambda j: beta[j])
    a0, a1, a2 = (alpha[j] for j in order)
    b0, b1, b2 = (beta[j] for j in order)
    e = numpy.eye(3, dtype=int)
    alpha = numpy.array((a0, a1, a2))
    beta = numpy.array((b0, b1, b2))
    mean = lambda a, b: _mean(tuple(a), tuple(b))
    combine = lambda *terms: tuple(sum(c * v[i] for c, v in terms) for i in range(2))

    if b0 == 0 and b1 == 0:
        A = a0 + a1 + a2
        return (2 * math.factorial(a0) * math.factorial(a1) * math.factorial(a2) / math.factorial(A-b2+2)
                * math.factorial(a0+a1+1-b2) / math.factorial(a0+a1+1), 0.0)
    if b0 >= 1:
        return combine(*((0.5, mean(alpha, beta - e[j])) for j in range(3)))
    if a0 == 0:
        return _mean_two_poles(a1, a2, b1, b2)
    A = a0 + a1 + a2
    if a1 + b1 < A + 1:
        return combine((1, mean(alpha - e[0], beta - e[2])), (-1, mean(alpha - e[0] + e[1], beta)))
    if a2 + b2 < A + 1:
        return combine((1, mean(alpha - e[0], beta - e[1])), (-1, mean(alpha - e[0] + e[2], beta)))
    terms = [(-1, mean(alpha - e[0] + e[1] + e[2], beta))]
    for j in (1, 2):
        terms.append((0.5, mean(alpha, beta - e[j])))
        terms.append((0.5, mean(alpha - e[0] + e[j], beta - e[j])))
    return combine(*terms)


def integrable_rational_monomials(numerator_degree, denominator_degree):
    """Return the exponents of the integrable rational monomials on a triangle.

    :arg numerator_degree: the maximum degree |alpha| of the numerator.
    :arg denominator_degree: the maximum degree |beta| of the denominator.
    :returns: an integer array of exponents.
    """
    exponents = [alpha + beta
                 for alpha in itertools.product(range(numerator_degree+1), repeat=3)
                 if sum(alpha) <= numerator_degree
                 for beta in itertools.product(range(denominator_degree+1), repeat=3)
                 if sum(beta) <= denominator_degree
                 if max(numpy.add(alpha, beta)) <= sum(alpha) + 1]
    return numpy.array(exponents, dtype=int)


def rational_integral(ref_el, exponents):
    """Integrate rational monomials over a triangle exactly.

    Uses the recursion of (Diening, Storn, and Tscherpel, 2025, Sec. 3).

    :arg ref_el: a triangle.
    :arg exponents: an array of exponents of rational monomials.
    :returns: an array with the integrals, which are infinite for
        non-integrable monomials.
    """
    if ref_el.get_spatial_dimension() != 2:
        raise ValueError("Exact rational integration is only implemented on triangles")
    result = []
    for row in numpy.asarray(exponents):
        alpha, beta = tuple(row[:3]), tuple(row[3:])
        if max(numpy.add(alpha, beta)) > sum(alpha) + 1:
            result.append(numpy.inf)
        else:
            q, p = _mean(alpha, beta)
            result.append(ref_el.volume() * (q + p * math.pi**2))
    return numpy.array(result)


class RationalExpansionSet(expansions.ExpansionSet):
    """A set of rational monomials on a simplex.

    The members are independent of the degree passed to ``tabulate``.

    :arg ref_el: a simplex.
    :arg exponents: an integer array of shape ``(num_members, 2*(sd+1))``
        whose rows hold the exponents ``(alpha, beta)`` of each member.
    """
    def __init__(self, ref_el, exponents):
        super().__init__(ref_el)
        self.exponents = numpy.asarray(exponents, dtype=int)
        sd = ref_el.get_spatial_dimension()
        if self.exponents.ndim != 2 or self.exponents.shape[1] != 2*(sd+1):
            raise ValueError(f"Expecting exponents of shape (num_members, {2*(sd+1)})")
        self.grad_lambda = barycentric_gradients(ref_el)

    def reconstruct(self, ref_el=None, **kwargs):
        """Reconstruct this RationalExpansionSet on another reference element."""
        return RationalExpansionSet(ref_el or self.ref_el, self.exponents)

    def get_num_members(self, n):
        return len(self.exponents)

    def _tabulate(self, n, pts, order=0):
        """Returns a dict of tabulations such that
        tabulations[alpha][i, j] = D^alpha phi_i(pts[j])."""
        pts = numpy.asarray(pts)
        sd = self.ref_el.get_spatial_dimension()
        bary = self.ref_el.compute_barycentric_coordinates(pts.reshape(-1, sd))
        terms = {(0,) * sd: (self.exponents, numpy.eye(len(self.exponents)))}
        for r in range(1, order+1):
            for alpha in mis(sd, r):
                k = next(i for i, a in enumerate(alpha) if a > 0)
                base = tuple(a - (i == k) for i, a in enumerate(alpha))
                terms[alpha] = rational_derivative(*terms[base], self.grad_lambda[:, k])

        shape = (len(self.exponents), *pts.shape[:-1])
        return {alpha: numpy.dot(C, rational_evaluate(E, bary)).reshape(shape)
                for alpha, (E, C) in terms.items()}

    def get_dmats(self, degree, cell=0):
        raise NotImplementedError("Rational monomials are not closed under differentiation")

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.ref_el == other.ref_el and
                numpy.array_equal(self.exponents, other.exponents))
