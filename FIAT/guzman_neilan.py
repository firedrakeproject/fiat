# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

# This is not quite Guzman-Neilan, but it has 2*dim*(dim+1) dofs and includes
# dim**2-1 extra constraint functionals.  The first (dim+1)**2 basis functions
# are the reference element bfs, but the extra dim**2-1 are used in the
# transformation theory.

from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.macro import AlfeldSplit, CkPolynomialSet
from FIAT.bernardi_raugel import ExtendedBernardiRaugelSpace, BernardiRaugelDualSet
from FIAT.quadrature_schemes import create_quadrature
import numpy


def GuzmanNeilanSpace(ref_el, degree):
    r"""Return a basis for the extended Guzman-Neilan space."""
    BR = ExtendedBernardiRaugelSpace(ref_el, degree)

    top = ref_el.get_topology()
    ref_complex = AlfeldSplit(ref_el)
    sd = ref_complex.get_spatial_dimension()
    Pk = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), scale=1, variant="bubble")
    expansion_set = Pk.get_expansion_set()
    dimPk = expansion_set.get_num_members(degree)
    entity_ids = expansions.polynomial_entity_ids(ref_complex, degree, continuity="C0")

    ids = [i + j*dimPk for j in range(sd)
           for dim in range(sd+1)
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f]]
    V = Pk.take(ids)
    Q = polynomial_set.ONPolynomialSet(ref_complex, degree-1)
    Q = Q.take(list(range(1, Q.get_num_members())))

    div = lambda tab: sum(tab[alpha][:, alpha.index(1), :] for alpha in tab if sum(alpha) == 1)
    inner = lambda u, v, qwts: numpy.tensordot(numpy.multiply(u, qwts), v,
                                               axes=(range(1, u.ndim), range(1, v.ndim)))

    rule = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = rule.get_points(), rule.get_weights()
    P = Q.tabulate(qpts)[(0,)*sd]
    U = V.tabulate(qpts, 1)
    X = BR.tabulate(qpts, 1)
    stiff = lambda u, v: sum(inner(u[alpha], v[alpha], qwts) for alpha in u if sum(alpha) == 1)

    divb = div(X)
    divb -= numpy.dot(divb, qwts)[:, None]/sum(qwts)
    P -= numpy.dot(P, qwts)[:, None]/sum(qwts)
    A = stiff(U, U)
    B = inner(P, div(U), qwts)
    f = stiff(U, X)
    g = inner(P, divb, qwts)

    S = B @ numpy.linalg.solve(A, B.T)
    u = numpy.linalg.solve(A, f)
    p = numpy.linalg.solve(S, B @ u - g)
    u -= numpy.linalg.solve(A, B.T @ p)

    p1, pk = BR.tabulate(qpts)[(0,)*sd], Pk.tabulate(qpts)[(0,)*sd]
    coeffs = numpy.tensordot(inner(p1, pk, qwts), numpy.linalg.inv(inner(pk, pk, qwts)), axes=(1, 1))
    coeffs = coeffs.reshape(-1, sd, dimPk)
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
