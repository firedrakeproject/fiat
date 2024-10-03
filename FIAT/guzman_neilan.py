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
from FIAT.bernardi_raugel import BernardiRaugel, BernardiRaugelDualSet
from FIAT.macro import AlfeldSplit
from FIAT.quadrature_schemes import create_quadrature
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
import numpy


def inner(v, u, qwts):
    """compute the L2 inner product from tabulation arrays and quadrature weights"""
    return numpy.tensordot(numpy.multiply(v, qwts), u,
                           axes=(range(1, v.ndim), range(1, u.ndim)))


def div(U):
    """compute the divergence from tabulation dict"""
    return sum(U[k][:, k.index(1), :] for k in U if sum(k) == 1)


def constant_div_projection(BR, C0, V):
    ref_complex = V.get_reference_element()
    sd = ref_complex.get_spatial_dimension()
    degree = V.degree

    rule = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = rule.get_points(), rule.get_weights()

    # Pressure space
    Q = polynomial_set.ONPolynomialSet(ref_complex, degree-1)
    Q = Q.take(list(range(1, Q.get_num_members())))
    # Take pressure test functions in L2 \ R
    P = Q.tabulate(qpts)[(0,)*sd]
    P -= numpy.dot(P, qwts)[:, None] / sum(qwts)

    U = V.tabulate(qpts, 1)
    X = BR.tabulate(qpts, 1)
    B = inner(P, div(U), qwts)
    g = inner(P, div(X), qwts)
    u = numpy.linalg.solve(B, g)

    phi = C0.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.linalg.solve(inner(phi, phi, qwts), inner(phi, X[(0,)*sd], qwts))
    coeffs = coeffs.T.reshape(BR.get_num_members(), sd, -1)
    coeffs -= numpy.tensordot(u, V.get_coeffs(), axes=(0, 0))

    GN = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                      C0.get_expansion_set(), coeffs)
    return GN


def modified_bubble_subspace(C0):
    degree = C0.degree
    ref_complex = C0.get_reference_element()
    ref_el = ref_complex.get_parent()
    sd = ref_complex.get_spatial_dimension()
    dimC0 = C0.get_expansion_set().get_num_members(degree)

    # Get interior bubbles
    entity_ids = expansions.polynomial_entity_ids(ref_complex, degree, continuity="C0")
    ids = [i + j * dimC0
           for dim in range(sd+1)
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f]
           for j in range(sd)]
    V = C0.take(ids)
    if sd > 2:
        # Trim the bubble space
        rule = create_quadrature(ref_complex, 2*degree)
        qpts, qwts = rule.get_points(), rule.get_weights()

        hat = C0.take([sd+1])
        hat_at_qpts = hat.tabulate(qpts)[(0,)*sd][:, 0, :]
        bubbles = []
        for k in range(sd):
            if k < 2:
                Pk = polynomial_set.ONPolynomialSet(ref_el, k, shape=(sd,))
            else:
                BDMk = BrezziDouglasMarini(ref_el, k)
                entity_ids = BDMk.entity_dofs()
                num_facet_dofs = len(entity_ids[sd-1]) * len(entity_ids[sd-1][0])
                Pk = BDMk.get_nodal_basis().take(list(range(num_facet_dofs)))
            phis = Pk.tabulate(qpts)[(0,)*sd]
            bubbles.append(numpy.multiply(phis, hat_at_qpts ** (sd-k)))

        bubbles = numpy.concatenate(bubbles)
        _, sig, _ = numpy.linalg.svd(bubbles.reshape(bubbles.shape[0], -1), full_matrices=True)
        phi = V.tabulate(qpts)[(0,)*sd]
        coeffs = numpy.linalg.solve(inner(phi, phi, qwts), inner(phi, bubbles, qwts))
        coeffs = numpy.tensordot(coeffs, V.get_coeffs(), axes=(0, 0))
        V = polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                         C0.get_expansion_set(), coeffs)
    return V


def GuzmanNeilanSpace(ref_el, degree):
    r"""Return a basis for the extended Guzman-Neilan space."""
    sd = ref_el.get_spatial_dimension()
    if degree != sd:
        raise ValueError("Guzman-Neilan only defined for degree = dim")

    BR = BernardiRaugel(ref_el, degree).get_nodal_basis()
    ref_complex = AlfeldSplit(ref_el)
    C0 = polynomial_set.ONPolynomialSet(ref_complex, degree, shape=(sd,), scale=1, variant="bubble")

    V = modified_bubble_subspace(C0)
    GN = constant_div_projection(BR, C0, V)
    return GN


class GuzmanNeilan(finite_element.CiarletElement):
    """The Guzman-Neilan extended element."""
    def __init__(self, ref_el, degree=None):
        sd = ref_el.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError("Guzman-Neilan only defined for degree = dim")
        poly_set = GuzmanNeilanSpace(ref_el, degree)
        dual = BernardiRaugelDualSet(ref_el, degree)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
