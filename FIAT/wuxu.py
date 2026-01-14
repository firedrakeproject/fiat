# Copyright (C) 2022 Robert C. Kirby (Baylor University)
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.

from FIAT import expansions, polynomial_set, dual_set, finite_element
from FIAT.functional import (IntegralMomentOfDerivative,
                             IntegralMomentOfBidirectionalDerivative,
                             PointDerivative, PointEvaluation)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.polynomial_set import mis
from FIAT.bubble import Bubble
from FIAT.lagrange import Lagrange
import numpy

polydim = expansions.polynomial_dimension


def WuXuRobustH3NCSpace(ref_el):
    """Constructs a basis for the the Wu Xu H^3 nonconforming space
    P^{(3,2)}(T) = P_3(T) + b_T P_1(T) + b_T^2 P_1(T),
    where b_T is the standard cubic bubble."""

    sd = ref_el.get_spatial_dimension()
    assert sd == 2

    em_deg = 7

    # Unfortunately,  b_T^2 P_1 has degree 7 (cubic squared times a linear)
    # so we need a high embedded degree!
    p7 = polynomial_set.ONPolynomialSet(ref_el, 7)

    dimp1 = polydim(ref_el, 1)
    dimp3 = polydim(ref_el, 3)
    dimp7 = polydim(ref_el, 7)

    # Here's the first bit we'll work with.  It's already expressed in terms
    # of the ON basis for P7, so we're golden.
    p3fromp7 = p7.take(list(range(dimp3)))

    # Rather than creating the barycentric coordinates ourself, let's
    # reuse the existing bubble functionality
    bT = Bubble(ref_el, 3)
    p1 = Lagrange(ref_el, 1)

    # next, we'll have to project b_T P1 and b_T^2 P1 onto P^7
    Q = create_quadrature(ref_el, 14)
    Qpts = numpy.array(Q.get_points())
    Qwts = numpy.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    # it's just one bubble function: let's get a 1d array!
    bT_at_qpts = bT.tabulate(0, Qpts)[zero_index][0, :]
    p1_at_qpts = p1.tabulate(0, Qpts)[zero_index]

    # Note: difference in signature because bT, p1 are FE and p7 is a
    # polynomial set
    p7_at_qpts = p7.tabulate(Qpts)[zero_index]

    bubble_coeffs = numpy.zeros((6, dimp7), "d")

    # first three: bT P1, last three will be bT^2 P1
    foo = bT_at_qpts * p1_at_qpts * Qwts
    bubble_coeffs[:dimp1, :] = numpy.dot(foo, p7_at_qpts.T)

    foo = bT_at_qpts * foo
    bubble_coeffs[dimp1:2*dimp1, :] = numpy.dot(foo, p7_at_qpts.T)

    bubbles = polynomial_set.PolynomialSet(ref_el, 3, em_deg,
                                           p7.get_expansion_set(),
                                           bubble_coeffs)

    return polynomial_set.polynomial_set_union_normalized(p3fromp7, bubbles)


def WuXuH3NCSpace(ref_el):
    """Constructs a basis for the the Wu Xu H^3 nonconforming space
    P(T) = P_3(T) + b_T P_1(T),
    where b_T is the standard cubic bubble."""

    assert ref_el.get_spatial_dimension() == 2

    em_deg = 4
    p4 = polynomial_set.ONPolynomialSet(ref_el, em_deg)

    # Here's the first bit we'll work with.  It's already expressed in terms
    # of the ON basis for P4, so we're golden.
    dimp3 = polydim(ref_el, 3)
    p3fromp4 = p4.take(list(range(dimp3)))

    # Rather than creating the barycentric coordinates ourself, let's
    # reuse the existing bubble functionality
    bT = Bubble(ref_el, 4)

    return polynomial_set.polynomial_set_union_normalized(p3fromp4, bT.get_nodal_basis())


class WuXuRobustH3NCDualSet(dual_set.DualSet):
    """Dual basis for WuXu H3 nonconforming element consisting of
    vertex values and gradients and first and second normals at edge midpoints."""

    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        assert sd == 2
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # jet at each vertex
        verts = ref_el.get_vertices()
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, verts[v]))
            nodes.extend(PointDerivative(ref_el, verts[v], alpha) for alpha in mis(sd, 1))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # average of first and second normal derivative along each edge
        Q_ref = create_quadrature(ref_el.construct_subelement(1), degree-1)
        for e in sorted(top[1]):
            n = ref_el.compute_normal(e)
            Q = FacetQuadratureRule(ref_el, 1, e, Q_ref)
            f = numpy.full(Q.get_weights().shape, 1/Q.jacobian_determinant())
            cur = len(nodes)
            nodes.append(IntegralMomentOfDerivative(ref_el, n, Q, f))
            nodes.append(IntegralMomentOfBidirectionalDerivative(ref_el, Q, f, n, n))
            entity_ids[1][e].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class WuXuH3NCDualSet(dual_set.DualSet):
    """Dual basis for WuXu H3 nonconforming element consisting of
    vertex values and gradients and second normals at edge midpoints."""

    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        assert sd == 2
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # jet at each vertex
        verts = ref_el.get_vertices()
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, verts[v]))
            nodes.extend(PointDerivative(ref_el, verts[v], alpha) for alpha in mis(sd, 1))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # average of second normal derivative along each edge
        Q_ref = create_quadrature(ref_el.construct_subelement(1), degree-2)
        for e in sorted(top[1]):
            n = ref_el.compute_normal(e)
            Q = FacetQuadratureRule(ref_el, 1, e, Q_ref)
            f = numpy.full(Q.get_weights().shape, 1/Q.jacobian_determinant())
            cur = len(nodes)
            nodes.append(IntegralMomentOfBidirectionalDerivative(ref_el, Q, f, n, n))
            entity_ids[1][e].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class WuXuRobustH3NC(finite_element.CiarletElement):
    """The Wu-Xu robust H3 nonconforming finite element"""
    def __init__(self, ref_el, degree=7):
        poly_set = WuXuRobustH3NCSpace(ref_el)
        assert degree == poly_set.degree
        dual = WuXuRobustH3NCDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)


class WuXuH3NC(finite_element.CiarletElement):
    """The Wu-Xu H3 nonconforming finite element"""
    def __init__(self, ref_el, degree=4):
        poly_set = WuXuH3NCSpace(ref_el)
        assert degree == poly_set.degree
        dual = WuXuH3NCDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)
