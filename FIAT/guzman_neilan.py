# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

# The extended element includes dim-1 tangential constraints on each facet.
# These are numbered last and used in the basis transformation.

from FIAT import finite_element, polynomial_set, expansions
from FIAT.bernardi_raugel import BernardiRaugelSpace, BernardiRaugelDualSet
from FIAT.alfeld_sorokina import AlfeldSorokina
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.macro import AlfeldSplit
from FIAT.quadrature_schemes import create_quadrature
from FIAT.restricted import RestrictedElement
from FIAT.nodal_enriched import NodalEnrichedElement

import numpy
import math


def GuzmanNeilanSpace(ref_el, order, kind=1, reduced=False):
    r"""Return a basis for the (extended) Guzman-Neilan H1 space.

    Project the extended Bernardi-Raugel space (Pk + FacetBubble)^d into
    C0 Pd(Alfeld)^d, correcting the facet bubbles with interior bubbles so that
    they have constant divergence, while preserving their trace.
    For kind = 1 and k > 1, the union with the interior bubbles that invert the
    divergence from P_{k-1} / P0 gives the P_{k-1} divergence.
    For kind = 2, C0 Pk(Alfeld) already contains interior bubbles.

    Parameters
    ----------
    ref_el : FIAT.reference_element.Cell
        A simplex.
    order : int
        The polynomial degree k.
    kind : int, optional
        Use Pk for kind 1, or C0 Pk(Alfeld) for kind 2.
    reduced : bool, optional
        Omit tangential facet bubbles if True.

    Returns
    -------
    FIAT.polynomial_set.PolynomialSet
        A basis for the Guzman-Neilan H1 space.
    """
    sd = ref_el.get_spatial_dimension()
    ref_complex = AlfeldSplit(ref_el)
    C0 = polynomial_set.ONPolynomialSet(ref_complex, sd, shape=(sd,), scale=1, variant="bubble")
    B = take_interior_bubbles(C0)
    if sd > 2:
        B = modified_bubble_subspace(B)

    K = ref_complex if kind == 2 else ref_el
    top = K.get_topology()
    facets = sorted(set(top[sd-1]) - set(K.get_interior_facets(sd-1) or ()))

    # The facet bubbles are numbered after the Pk^d basis
    dimPk = sd * sum(math.comb(order-1, dim) * len(top[dim]) for dim in range(order))
    num_bubbles = sd * len(facets)
    BR = BernardiRaugelSpace(K, order, interior=False)
    if reduced:
        # Keep only the normal component of the facet bubbles
        coeffs = BR.get_coeffs()
        bubbles = coeffs[dimPk:].reshape(len(facets), sd, *coeffs.shape[1:])
        normal_bubbles = [numpy.tensordot(K.compute_normal(f), bubble, axes=(0, 0))
                          for f, bubble in zip(facets, bubbles)]
        coeffs = numpy.concatenate([coeffs[:dimPk], normal_bubbles])
        num_bubbles = len(facets)
        BR = polynomial_set.PolynomialSet(BR.get_reference_element(), BR.degree, BR.embedded_degree,
                                          BR.get_expansion_set(), coeffs)

    GN = constant_div_projection(BR, C0, B, num_bubbles)
    if kind == 1 and order > 1:
        # Add the interior bubbles with divergence in P_{k-1} / P0
        Q = polynomial_set.ONPolynomialSet(ref_el, order-1)
        Q = Q.take(list(range(1, Q.get_num_members())))
        interior = invert_divergence(B, lambda pts: Q.tabulate(pts)[(0,)*sd])
        GN = polynomial_set.polynomial_set_union_normalized(GN, interior)
    return GN


class GuzmanNeilanH1(finite_element.CiarletElement):
    """The Guzman-Neilan H1-conforming (extended) macroelement."""
    def __init__(self, ref_el, order=1, kind=1, quad_scheme=None):
        sd = ref_el.get_spatial_dimension()
        if order >= sd:
            raise ValueError(f"{type(self).__name__} is only defined for order < dim")
        degree = sd
        poly_set = GuzmanNeilanSpace(ref_el, order, kind=kind)
        ref_complex = poly_set.get_reference_element()
        K = ref_complex if kind == 2 else ref_el
        dual = BernardiRaugelDualSet(K, order, degree=degree, ref_complex=ref_complex, quad_scheme=quad_scheme)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")


class GuzmanNeilanFirstKindH1(GuzmanNeilanH1):
    """The Guzman-Neilan H1-conforming (extended) macroelement of the first kind.

    Reference element: a simplex of any dimension.
    Function space: Pk^d + facet bubbles with div in P0
                    + interior bubbles with div in P_{k-1} / P0, 1 <= k < dim.
    Degrees of freedom: evaluation at Pk lattice points, normal moments on faces,
    and cell moments for k >= 2.

    This element belongs to a Stokes complex, and is paired with unsplit DG_{k-1}.
    """
    def __init__(self, ref_el, order=1, quad_scheme=None):
        super().__init__(ref_el, order=order, kind=1, quad_scheme=quad_scheme)


class GuzmanNeilanSecondKindH1(GuzmanNeilanH1):
    """The Guzman-Neilan H1-conforming (extended) macroelement of the second kind.

    Reference element: a simplex of any dimension.
    Function space: C0 Pk^d(Alfeld) + facet bubbles with div in P0, 1 <= k < dim.
    Degrees of freedom: evaluation at Pk(Alfeld) lattice points, and normal moments on faces.

    This element belongs to a Stokes complex, and is paired with DG_{k-1}(Alfeld).
    """
    def __init__(self, ref_el, order=1, quad_scheme=None):
        super().__init__(ref_el, order=order, kind=2, quad_scheme=quad_scheme)


def GuzmanNeilanH1div(ref_el, degree=2, reduced=False, quad_scheme=None):
    """The Guzman-Neilan H1(div)-conforming (extended) macroelement.

    Reference element: a simplex of any dimension.
    Function space: C0 P2^d(Alfeld) with C0 P1 divergence + normal facet bubbles with div in P0.
    Degrees of freedom: evaluation at P2(Alfeld) lattice points, divergence at P1 lattice points,
                        and normal moments on faces.

    This element belongs to a Stokes complex, and is paired with CG1(Alfeld).
    """
    order = 0
    AS = AlfeldSorokina(ref_el, 2)
    if reduced or ref_el.get_spatial_dimension() <= 2:
        order = 1
        # Only extract the div bubbles
        div_nodes = [i for i, node in enumerate(AS.dual_basis())
                     if len(node.deriv_dict) > 0]
        AS = RestrictedElement(AS, indices=div_nodes)
    GN = GuzmanNeilanH1(ref_el, order=order, quad_scheme=quad_scheme)
    return NodalEnrichedElement(AS, GN)


def inner(v, u, qwts):
    """Compute the L2 inner product from tabulation arrays and quadrature weights"""
    return numpy.tensordot(numpy.multiply(v, qwts), u,
                           axes=(range(1, v.ndim), range(1, u.ndim)))


def div(U):
    """Compute the divergence from tabulation dict."""
    return sum(U[k][:, k.index(1), :] for k in U if sum(k) == 1)


def take_interior_bubbles(P):
    """Extract the interior bubbles from a complete PolynomialSet."""
    ref_complex = P.get_reference_element()
    ncomp = numpy.prod(P.get_shape())
    dimPk = P.expansion_set.get_num_members(P.degree)
    assert ncomp * dimPk == P.get_num_members()
    continuity = P.expansion_set.continuity
    entity_ids = expansions.polynomial_entity_ids(ref_complex, P.degree,
                                                  continuity=continuity)
    ids = [i + j * dimPk
           for dim in entity_ids
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f]
           for j in range(ncomp)]
    return P.take(ids)


def modified_bubble_subspace(B):
    """Construct the interior bubble space M_k(K^r) from (Guzman and Neilan, 2019)."""
    ref_complex = B.get_reference_element()
    sd = ref_complex.get_spatial_dimension()
    degree = B.degree
    rule = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = rule.get_points(), rule.get_weights()

    # tabulate the linear hat function associated with the barycenter
    hat = B.take([0])
    hat_at_qpts = hat.tabulate(qpts)[(0,)*sd][0, 0]

    # tabulate the bubbles = hat ** (degree - k) * BDMk_facet
    ref_el = ref_complex.get_parent()
    bubbles = [numpy.eye(sd)[:, :, None] * hat_at_qpts[None, None, :] ** degree]
    for k in range(1, degree):
        # tabulate the BDM facet functions
        BDM = BrezziDouglasMarini(ref_el, k)
        BDM_facet = BDM.get_nodal_basis().take(BDM.dual.get_indices("facet"))
        phis = BDM_facet.tabulate(qpts)[(0,)*sd]

        bubbles.append(numpy.multiply(phis, hat_at_qpts ** (degree-k)))

    bubbles = numpy.concatenate(bubbles, axis=0)

    # store the bubbles into a PolynomialSet via L2 projection
    v = B.tabulate(qpts)[(0,) * sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, bubbles, qwts))
    coeffs = numpy.tensordot(coeffs, B.get_coeffs(), axes=(0, 0))
    M = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                     B.get_expansion_set(), coeffs)
    return M


def invert_divergence(M, f):
    """Return the interior bubbles whose divergence is given modulo constants.

    Parameters
    ----------
    M : FIAT.polynomial_set.PolynomialSet
        Interior bubbles whose divergences span mean-zero piecewise P_{d-1}.
    f : callable
        Tabulates the target divergences at an array of points, with shape
        (number of targets, number of points).

    Returns
    -------
    FIAT.polynomial_set.PolynomialSet
        A subset of the span of M, with one member per target.
    """
    ref_complex = M.get_reference_element()
    sd = ref_complex.get_spatial_dimension()
    rule = create_quadrature(ref_complex, 2*M.degree)
    qpts, qwts = rule.get_points(), rule.get_weights()

    # Take the test space for the divergence in L2 \ R
    Q = polynomial_set.ONPolynomialSet(ref_complex, M.degree-1)
    Q = Q.take(list(range(1, Q.get_num_members())))
    P = Q.tabulate(qpts)[(0,)*sd]
    P -= numpy.dot(P, qwts)[:, None] / sum(qwts)

    B = inner(P, div(M.tabulate(qpts, 1)), qwts)
    g = inner(P, f(qpts), qwts)
    w = numpy.linalg.solve(B, g)
    coeffs = numpy.tensordot(w, M.get_coeffs(), axes=(0, 0))
    return polynomial_set.PolynomialSet(ref_complex, M.degree, M.degree,
                                        M.get_expansion_set(), coeffs)


def constant_div_projection(BR, C0, M, num_bubbles):
    """Project the BR space into C0 Pk(Alfeld)^d with constant divergence on the bubbles.

    Parameters
    ----------
    BR : FIAT.polynomial_set.PolynomialSet
        The extended Bernardi-Raugel space, with bubbles numbered last.
    C0 : FIAT.polynomial_set.PolynomialSet
        The continuous piecewise polynomial space containing the result.
    M : FIAT.polynomial_set.PolynomialSet
        Interior bubbles whose divergences span mean-zero piecewise P_{d-1}.
    num_bubbles : int
        Number of BR bubbles to correct.

    Returns
    -------
    FIAT.polynomial_set.PolynomialSet
        The corrected space, with the same traces as BR.
    """
    ref_complex = C0.get_reference_element()
    sd = ref_complex.get_spatial_dimension()
    degree = C0.degree
    rule = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = rule.get_points(), rule.get_weights()

    # Add correction to BR bubbles
    corrections = invert_divergence(M, lambda pts: div(BR.tabulate(pts, 1))[-num_bubbles:])
    v = C0.tabulate(qpts)[(0,)*sd]
    x = BR.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, x, qwts))
    coeffs = coeffs.T.reshape(BR.get_num_members(), sd, -1)
    coeffs[-num_bubbles:] -= corrections.get_coeffs()
    return polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                        C0.get_expansion_set(), coeffs)
