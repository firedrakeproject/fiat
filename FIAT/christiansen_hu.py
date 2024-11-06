# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

# This is not quite Christiansen-Hu, but it has 2*dim*(dim+1) dofs and includes
# dim**2-1 extra constraint functionals.  The first (dim+1)**2 basis functions
# are the reference element bfs, but the extra dim**2-1 are used in the
# transformation theory.

from FIAT import finite_element, polynomial_set
from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import CkPolynomialSet, WorseyFarinSplit

import numpy


def ChristiansenHuSpace(ref_el, degree, reduced=False):
    """Return a basis for the Christianse-Hu space
    set(v in C0 P1(WF)^d : div(v) = 0) + P_0 x if reduced = True, and
    this space is agumented with rotated facet bubbles if reduced = False."""
    sd = ref_el.get_spatial_dimension()
    ref_complex = WorseyFarinSplit(ref_el)
    C0 = CkPolynomialSet(ref_complex, degree, order=0, shape=(sd,), scale=1, variant="bubble")
    Q = create_quadrature(ref_complex, degree-1)
    tab = C0.tabulate(Q.get_points(), 1)
    divC0 = sum(tab[alpha][:, alpha.index(1), :] for alpha in tab if sum(alpha) == 1)

    nsp = polynomial_set.spanning_basis(divC0.T, nullspace=True)
    coeffs = numpy.tensordot(nsp, C0.get_coeffs(), axes=(-1, 0))

    verts = numpy.array(ref_complex.get_vertices())
    WT = verts[-1]
    P0x_coeffs = numpy.transpose(verts - WT[None, :])
    coeffs = numpy.concatenate((coeffs, P0x_coeffs[None, ...]), axis=0)

    if not reduced:
        # Compute the primal basis via Vandermonde and extract the facet bubbles
        dual = BernardiRaugelDualSet(ref_el, degree, degree=degree, ref_complex=ref_complex, reduced=True)
        dualmat = dual.to_riesz(C0)
        V = numpy.tensordot(dualmat, coeffs, axes=((1, 2), (1, 2)))
        coeffs = numpy.tensordot(numpy.linalg.inv(V.T), coeffs, axes=(-1, 0))
        facet_bubbles = coeffs[-(sd+1):]

        # Rotate the facet bubbles onto the tangent space of the facet
        # NOTE they are not aligned with the normal, but they point in the direction
        # that connects the split point on the facet with the split point of the cell
        WF = verts[sd+1:-1]
        top = ref_el.get_topology()
        ext = []
        for f in top[sd-1]:
            ehat = WF[f] - WT
            FB = numpy.dot(ehat, facet_bubbles[f])
            thats = ref_el.compute_tangents(sd-1, f)
            for that in thats:
                ext.append(that[:, None] * FB[None, :])
        ext_coeffs = numpy.array(ext)
        coeffs = numpy.concatenate((coeffs, ext_coeffs), axis=0)

    return polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                        C0.get_expansion_set(), coeffs)


class ChristiansenHu(finite_element.CiarletElement):
    """The Christiansen-Hu C^0(Worsey-Farin) linear macroelement with divergence in P0.
    This element belongs to a Stokes complex, and is paired with unsplit DG0."""
    def __init__(self, ref_el, degree=1):
        if degree != 1:
            raise ValueError("Christiansen-Hu only defined for degree = 1")
        poly_set = ChristiansenHuSpace(ref_el, degree)
        ref_complex = poly_set.get_reference_element()
        dual = BernardiRaugelDualSet(ref_el, degree, degree=degree, ref_complex=ref_complex)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
