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
from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.quadrature_schemes import create_quadrature
import numpy


def GuzmanNeilanSpace(ref_el, degree):
    r"""Return a basis for the extended Guzman-Neilan space."""
    top = ref_el.get_topology()
    ref_complex = AlfeldSplit(ref_el)
    sd = ref_complex.get_spatial_dimension()
    Pk = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), scale=1, variant="bubble")
    expansion_set = Pk.get_expansion_set()
    dimPk = expansion_set.get_num_members(degree)
    entity_ids = expansions.polynomial_entity_ids(ref_complex, degree, continuity="C0")

    bcoeffs = numpy.zeros((sd+1, sd, dimPk))
    for f in sorted(top[sd-1]):
        i = entity_ids[sd-1][f][0]
        bcoeffs[f, :, i] = ref_el.compute_normal(f)
    bubbles = polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, bcoeffs)

    ids = [i + j*dimPk for j in range(sd)
           for dim in range(sd+1)
           for f in sorted(ref_complex.get_interior_facets(dim))
           for i in entity_ids[dim][f]]
    V = Pk.take(ids)
    Q = polynomial_set.ONPolynomialSet(ref_complex, degree-1)

    div = lambda tab: sum(tab[alpha][:, alpha.index(1), :] for alpha in tab if sum(alpha) == 1)
    inner = lambda u, v, qwts: numpy.tensordot(numpy.multiply(u, qwts), v,
                                               axes=(range(1, u.ndim), range(1, v.ndim)))

    rule = create_quadrature(ref_complex, 2*degree)
    qpts, qwts = rule.get_points(), rule.get_weights()
    Qtab = Q.tabulate(qpts)
    Vtab = V.tabulate(qpts, 1)
    Btab = bubbles.tabulate(qpts, 1)
    U, P, b = Vtab[(0,)*sd], Qtab[(0,)*sd], Btab[(0,)*sd]
    divU, divb = div(Vtab), div(Btab)

    c = numpy.dot(divb, qwts)[:, None]/sum(qwts)
    A = inner(U, U, qwts)
    B = inner(P, divU, qwts)
    f = inner(U, -b, qwts)
    g = inner(P, c - divb, qwts)

    S = B @ numpy.linalg.solve(A, B.T)
    u = numpy.linalg.solve(A, f)
    p = numpy.linalg.solve(S, B @ u - g)
    u -= numpy.linalg.solve(A, B.T @ p)

    bcoeffs += numpy.tensordot(u, V.get_coeffs(), axes=(0, 0))

    tcoeffs = numpy.zeros(((sd-1)*(sd+1), sd, dimPk))
    cur = 0
    for f in sorted(top[sd-1]):
        n = ref_el.compute_normal(f)
        ncoeff = numpy.dot(n, bcoeffs[f])
        for t in ref_el.compute_normalized_tangents(sd-1, f):
            tcoeffs[cur] = t[:, None] * ncoeff
            cur += 1

    P1 = CkPolynomialSet(ref_el, 1, shape=(sd,), scale=1, variant="bubble")
    pk, p1 = P1.tabulate(qpts)[(0,)*sd], Pk.tabulate(qpts)[(0,)*sd]
    P1coeffs = numpy.tensordot(numpy.linalg.inv(inner(pk, pk, qwts)), inner(pk, p1, qwts), axes=(1, 0))

    coeffs = numpy.concatenate((P1coeffs.reshape(-1, sd, dimPk), bcoeffs, tcoeffs))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree, expansion_set, coeffs)


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
