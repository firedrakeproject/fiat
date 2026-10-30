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

    Project the extended Bernardi-Raugel space (Pk + FacetBubble + InteriorBubble P_{k-2})^d
    into C0 Pk(Alfeld)^d with P_{k-1} divergence, preserving its trace.
    For kind = 1, the interior bubble is the hat function associated with the
    barycenter of the Alfeld split, so that the space does not increase its degree.
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
    BR = BernardiRaugelSpace(K, order)
    interior = None
    if kind == 1 and order > 1:
        # Use the Alfeld hat in place of the unsplit cell bubble.
        interior = take_interior_bubbles(C0, degree=order-1)
        BR = BR.take(list(range(len(BR) - len(interior))))

    # The bubbles are numbered after the Pk^d basis
    top = K.get_topology()
    dimPk = sd * sum(math.comb(order-1, dim) * len(top[dim]) for dim in range(order))
    if reduced:
        # Keep only the normal component of the facet bubbles
        facets = sorted(set(top[sd-1]) - set(K.get_interior_facets(sd-1) or ()))
        coeffs = BR.get_coeffs()
        ids = slice(dimPk, dimPk + sd * len(facets))
        facet_bubbles = []
        for i, f in enumerate(facets):
            normal = K.compute_normal(f)
            bubbles = coeffs[dimPk + sd*i:dimPk + sd*(i+1)]
            facet_bubbles.append(sum(n * bubble for n, bubble in zip(normal, bubbles)))
        coeffs = numpy.concatenate([coeffs[:ids.start], facet_bubbles, coeffs[ids.stop:]])
        BR = polynomial_set.PolynomialSet(BR.get_reference_element(), BR.degree, BR.embedded_degree,
                                          BR.get_expansion_set(), coeffs)
    num_bubbles = len(BR) - dimPk + (len(interior) if interior is not None else 0)

    # The divergence of the bubbles is projected onto P_{k-1}(ref_el),
    # this includes the constants as the facet bubbles have nonzero flux
    pressure = polynomial_set.ONPolynomialSet(ref_el, max(order-1, 0), scale="orthonormal")
    return divergence_projection(BR, C0, B, num_bubbles, pressure, interior=interior)


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
    Function space: Pk^d + facet and interior bubbles with div in P_{k-1}, 1 <= k < dim.
    Degrees of freedom: evaluation at Pk lattice points, normal moments on faces,
    and cell moments for k >= 2.

    This element belongs to a Stokes complex, and is paired with unsplit DG_{k-1}.
    """
    def __init__(self, ref_el, order=1, quad_scheme=None):
        super().__init__(ref_el, order=order, kind=1, quad_scheme=quad_scheme)


class GuzmanNeilanSecondKindH1(GuzmanNeilanH1):
    """The Guzman-Neilan H1-conforming (extended) macroelement of the second kind.

    Reference element: a simplex of any dimension.
    Function space: C0 Pk^d(Alfeld) + facet bubbles with div in P_{k-1}, 1 <= k < dim.
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


def take_interior_bubbles(P, degree=None):
    """Extract the interior bubbles up to the given degree from a complete PolynomialSet."""
    ref_complex = P.get_reference_element()
    ncomp = numpy.prod(P.get_shape())
    dimPk = P.expansion_set.get_num_members(P.degree)
    assert ncomp * dimPk == P.get_num_members()
    continuity = P.expansion_set.continuity
    entity_ids = expansions.polynomial_entity_ids(ref_complex, P.degree,
                                                  continuity=continuity)
    if degree is None or degree >= P.degree:
        slices = {dim: slice(None) for dim in entity_ids}
    else:
        slices = {dim: slice(math.comb(degree-1, dim)) for dim in entity_ids}

    ids = [i + j * dimPk
           for dim in slices
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f][slices[dim]]
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


def divergence_projection(BR, C0, M, num_bubbles, pressure, interior=None):
    """Project the BR space into C0 Pk(Alfeld)^d with divergence in the pressure space.

    The divergence of the BR bubbles is corrected with the modified bubbles M,
    so that it becomes its L2 projection onto the pressure space.

    Parameters
    ----------
    BR : FIAT.polynomial_set.PolynomialSet
        The extended Bernardi-Raugel space, with bubbles numbered last.
    C0 : FIAT.polynomial_set.PolynomialSet
        The continuous piecewise polynomial space containing the result.
    M : FIAT.polynomial_set.PolynomialSet
        Interior bubbles whose divergences span mean-zero piecewise P_{d-1}.
    num_bubbles : int
        Number of basis functions to correct, including interior.
    pressure : FIAT.polynomial_set.PolynomialSet
        An L2-orthonormal basis for the target divergence space on the unsplit simplex.
    interior : FIAT.polynomial_set.PolynomialSet, optional
        Interior bubbles to append to BR before correcting the divergence.

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

    # Take the test space for the divergence in L2 \ R
    Q = polynomial_set.ONPolynomialSet(ref_complex, degree-1)
    Q = Q.take(list(range(1, Q.get_num_members())))
    P = Q.tabulate(qpts)[(0,)*sd]
    P -= numpy.dot(P, qwts)[:, None] / sum(qwts)

    U = M.tabulate(qpts, 1)
    X = BR.tabulate(qpts, 1)
    if interior is not None:
        Y = interior.tabulate(qpts, 1)
        X = {alpha: numpy.concatenate([X[alpha], Y[alpha]]) for alpha in X}
    # Remove the L2 projection onto the pressure space from the divergence of the bubbles
    R = pressure.tabulate(qpts)[(0,)*sd]
    divX = div(X)[-num_bubbles:]
    divX -= inner(divX, R, qwts) @ R
    # Invert the divergence
    B = inner(P, div(U), qwts)
    g = inner(P, divX, qwts)
    w = numpy.linalg.solve(B, g)

    # Add correction to BR bubbles
    v = C0.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, X[(0,)*sd], qwts))
    coeffs = coeffs.T.reshape(-1, sd, C0.get_coeffs().shape[-1])
    coeffs[-num_bubbles:] -= numpy.tensordot(w, M.get_coeffs(), axes=(0, 0))
    GN = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                      C0.get_expansion_set(), coeffs)
    return GN
