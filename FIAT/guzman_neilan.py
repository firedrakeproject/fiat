# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

# This is not quite Guzman-Neilan, but it has 2*dim*(dim+1) dofs and includes
# dim**2-1 extra constraint functionals.  The first (dim+1)**2 basis functions
# are the reference element bfs, but the extra dim**2-1 are used in the
# transformation theory.

from FIAT import finite_element, polynomial_set, expansions
from FIAT.bernardi_raugel import BernardiRaugelSpace, BernardiRaugelDualSet, BernardiRaugel
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

    Project the extended Bernardi-Raugel space (Pk + FacetBubble)^d
    into C0 Pk(Alfeld)^d with P_{k-1} divergence, preserving its trace.

    :arg ref_el: a simplex
    :arg order: the maximal polynomial degree
    :kwarg kind: kind = 1 gives Pk^d + GN bubbles,
                 kind = 2 gives C0 Pk(Alfeld)^d + GN bubbles.
    :kwarg reduced: Include tangential bubbles if reduced = False.

    :returns: a PolynomialSet basis for the Guzman-Neilan H1 space.
    """
    sd = ref_el.get_spatial_dimension()
    ref_complex = AlfeldSplit(ref_el)
    C0 = polynomial_set.ONPolynomialSet(ref_complex, sd, shape=(sd,), scale=1, variant="bubble")
    B = take_interior_bubbles(C0)
    if sd > 2:
        B = modified_bubble_subspace(B)

    K = ref_complex if kind == 2 else ref_el
    num_bubbles = sd + 1
    if reduced:
        BR = BernardiRaugel(K, order).get_nodal_basis()
        reduced_dim = BR.get_num_members() - (sd-1) * (sd+1)
        BR = BR.take(list(range(reduced_dim)))
    else:
        num_bubbles *= sd
        BR = BernardiRaugelSpace(K, order)

    GN = constant_div_projection(BR, C0, B, num_bubbles)
    return GN


class GuzmanNeilanH1(finite_element.CiarletElement):
    """The Guzman-Neilan H1-conforming (extended) macroelement."""
    def __init__(self, ref_el, order=1, kind=1):
        sd = ref_el.get_spatial_dimension()
        if order >= sd:
            raise ValueError(f"{type(self).__name__} is only defined for order < dim")
        degree = sd
        poly_set = GuzmanNeilanSpace(ref_el, order, kind=kind)
        ref_complex = poly_set.get_reference_element() if kind == 2 else ref_el
        dual = BernardiRaugelDualSet(ref_complex, order, degree=degree)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")


class GuzmanNeilanFirstKindH1(GuzmanNeilanH1):
    """The Guzman-Neilan H1-conforming (extended) macroelement of the first kind.

    Reference element: a simplex of any dimension.
    Function space: Pk^d + normal facet bubbles with div in P0, with 1 <= k < dim.
    Degrees of freedom: evaluation at Pk lattice points, and normal moments on faces.

    This element belongs to a Stokes complex, and is paired with unsplit DG_{k-1}.
    """
    def __init__(self, ref_el, order=1):
        super().__init__(ref_el, order=order, kind=1)


class GuzmanNeilanSecondKindH1(GuzmanNeilanH1):
    """The Guzman-Neilan H1-conforming (extended) macroelement of the second kind.

    Reference element: a simplex of any dimension.
    Function space: C0 Pk^d(Alfeld) + normal facet bubbles with div in P0, with 1 <= k < dim.
    Degrees of freedom: evaluation at Pk(Alfeld) lattice points, and normal moments on faces.

    This element belongs to a Stokes complex, and is paired with DG_{k-1}(Alfeld).
    """
    def __init__(self, ref_el, order=1):
        super().__init__(ref_el, order=order, kind=2)


def GuzmanNeilanH1div(ref_el, degree=2, reduced=False):
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
    GN = GuzmanNeilanH1(ref_el, order=order)
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


def constant_div_projection(BR, C0, M, num_bubbles):
    """Project the BR space into C0 Pk(Alfeld)^d with P_{k-1} divergence."""
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
    # Invert the divergence
    B = inner(P, div(U), qwts)
    g = inner(P, div(X)[-num_bubbles:], qwts)
    w = numpy.linalg.solve(B, g)

    # Add correction to BR bubbles
    v = C0.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, X[(0,)*sd], qwts))
    coeffs = coeffs.T.reshape(BR.get_num_members(), sd, -1)
    coeffs[-num_bubbles:] -= numpy.tensordot(w, M.get_coeffs(), axes=(0, 0))
    GN = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                      C0.get_expansion_set(), coeffs)
    return GN
