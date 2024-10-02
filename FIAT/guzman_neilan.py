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
import numpy


def GuzmanNeilanSpace(ref_el, degree):
    r"""Return a basis for the extended Guzman-Neilan space."""
    sd = ref_el.get_spatial_dimension()
    if degree != sd:
        raise ValueError("Guzman-Neilan only defined for degree = dim")

    BR = BernardiRaugel(ref_el, degree).get_nodal_basis()

    ref_complex = AlfeldSplit(ref_el)
    C0 = polynomial_set.ONPolynomialSet(ref_complex, degree, shape=(sd,), scale=1, variant="bubble")
    expansion_set = C0.get_expansion_set()
    dimC0 = expansion_set.get_num_members(degree)

    entity_ids = expansions.polynomial_entity_ids(ref_complex, degree, continuity="C0")
    ids = [i + j * dimC0
           for dim in range(sd+1)
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f]
           for j in range(sd)]
    V = C0.take(ids)
    Q = polynomial_set.ONPolynomialSet(ref_complex, degree-1)
    Q = Q.take(list(range(1, Q.get_num_members())))

    rule = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = rule.get_points(), rule.get_weights()

    # Compute the divergence from tabulation dict
    div = lambda tab: sum(tab[alpha][:, alpha.index(1), :] for alpha in tab if sum(alpha) == 1)
    # compute the L2 inner product from tabulation arrays and quadrature weights
    inner = lambda v, u, qwts: numpy.tensordot(numpy.multiply(v, qwts), u,
                                               axes=(range(1, v.ndim), range(1, u.ndim)))
    # Stokes bilinear forms
    a = lambda v, u: sum(inner(v[alpha], u[alpha], qwts) for alpha in u if sum(alpha) == 1)
    b = lambda q, u: inner(q, div(u), qwts)

    X = BR.tabulate(qpts, 1)
    U = V.tabulate(qpts, 1)
    # Take pressure test functions in L2 \ R
    P = Q.tabulate(qpts)[(0,)*sd]
    P -= numpy.dot(P, qwts)[:, None] / sum(qwts)

    # Stokes LHS
    A = a(U, U)
    B = b(P, U)
    # Stokes RHS
    f = a(U, X)
    g = b(P, X)

    # Solve using the Schur complement
    AinvBT = numpy.linalg.solve(A, B.T)
    S = B @ AinvBT
    u = numpy.linalg.solve(A, f)
    p = numpy.linalg.solve(S, B @ u - g)
    u -= AinvBT @ p

    phi = C0.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.linalg.solve(inner(phi, phi, qwts), inner(phi, X[(0,)*sd], qwts))
    coeffs = coeffs.T.reshape(BR.get_num_members(), sd, dimC0)
    coeffs -= numpy.tensordot(u, V.get_coeffs(), axes=(0, 0))

    GN = polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, coeffs)
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
        ref_complex = poly_set.get_reference_element()
        dual = BernardiRaugelDualSet(ref_complex, degree)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
