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
from FIAT.bernardi_raugel import ExtendedBernardiRaugelSpace, BernardiRaugelDualSet, BernardiRaugel
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.macro import AlfeldSplit
from FIAT.quadrature_schemes import create_quadrature
from itertools import chain

import numpy
import math


def ExtendedGuzmanNeilanSpace(ref_el, subdegree, reduced=False):
    """Return a basis for the extended Guzman-Neilan space."""
    sd = ref_el.get_spatial_dimension()
    ref_complex = AlfeldSplit(ref_el)
    C0 = polynomial_set.ONPolynomialSet(ref_complex, sd, shape=(sd,), scale=1, variant="bubble")
    B = take_interior_bubbles(C0)
    if sd > 2:
        B = modified_bubble_subspace(B)

    if reduced:
        BR = BernardiRaugel(ref_el, sd, subdegree=subdegree).get_nodal_basis()
        reduced_dim = BR.get_num_members() - (sd-1) * (sd+1)
        BR = BR.take(list(range(reduced_dim)))
    else:
        BR = ExtendedBernardiRaugelSpace(ref_el, subdegree)
    GN = constant_div_projection(BR, C0, B)
    return GN


class GuzmanNeilan(finite_element.CiarletElement):
    """The Guzman-Neilan extended element."""
    def __init__(self, ref_el, degree=None, subdegree=1):
        sd = ref_el.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError("Guzman-Neilan only defined for degree = dim")
        poly_set = ExtendedGuzmanNeilanSpace(ref_el, subdegree)
        dual = BernardiRaugelDualSet(ref_el, degree, subdegree=subdegree)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")


def inner(v, u, qwts):
    """Compute the L2 inner product from tabulation arrays and quadrature weights"""
    return numpy.tensordot(numpy.multiply(v, qwts), u,
                           axes=(range(1, v.ndim), range(1, u.ndim)))


def div(U):
    """Compute the divergence from tabulation dict."""
    return sum(U[k][:, k.index(1), :] for k in U if sum(k) == 1)


def take_interior_bubbles(C0):
    """Take the interior bubbles from a vector-valued C0 PolynomialSet."""
    ref_complex = C0.get_reference_element()
    sd = ref_complex.get_spatial_dimension()
    dimC0 = C0.get_num_members() // sd
    entity_ids = expansions.polynomial_entity_ids(ref_complex, C0.degree, continuity="C0")
    ids = [i + j * dimC0
           for dim in range(sd+1)
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f]
           for j in range(sd)]
    return C0.take(ids)


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

    # tabulate the BDM facet functions
    ref_el = ref_complex.get_parent()
    BDM = BrezziDouglasMarini(ref_el, degree-1)
    entity_dofs = BDM.entity_dofs()
    facet_dofs = list(range(BDM.space_dimension() - len(entity_dofs[sd][0])))
    BDM_facet = BDM.get_nodal_basis().take(facet_dofs)
    phis = BDM_facet.tabulate(qpts)[(0,)*sd]

    # tabulate the bubbles = hat ** (degree - k) * BDMk_facet
    bubbles = [numpy.eye(sd)[:, :, None] * hat_at_qpts[None, None, :] ** degree]
    for k in range(1, degree):
        dimPk = math.comb(k + sd-1, sd-1)
        idsPk = list(chain.from_iterable(entity_dofs[sd-1][f][:dimPk]
                     for f in entity_dofs[sd-1]))
        bubbles.append(numpy.multiply(phis[idsPk], hat_at_qpts ** (degree-k)))
    bubbles = numpy.concatenate(bubbles, axis=0)

    # store the bubbles into a PolynomialSet via L2 projection
    v = B.tabulate(qpts)[(0,) * sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, bubbles, qwts))
    coeffs = numpy.tensordot(coeffs, B.get_coeffs(), axes=(0, 0))
    M = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                     B.get_expansion_set(), coeffs)
    return M


def constant_div_projection(BR, C0, M):
    """Project the BR space into the space of C0 polynomials with constant divergence."""
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
    g = inner(P, div(X), qwts)
    w = numpy.linalg.solve(B, g)

    # Add correction to BR bubbles
    v = C0.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, X[(0,)*sd], qwts))
    coeffs = coeffs.T.reshape(BR.get_num_members(), sd, -1)
    coeffs -= numpy.tensordot(w, M.get_coeffs(), axes=(0, 0))
    GN = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                      C0.get_expansion_set(), coeffs)
    return GN
