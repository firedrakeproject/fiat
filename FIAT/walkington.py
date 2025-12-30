# Copyright (C) 2025 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This is not quite Walkington, but is 65-dofs and includes 20 extra constraint
# functionals.  The first 45 basis functions are the reference element
# bfs, but the extra 20 are used in the transformation theory.

from FIAT import finite_element, polynomial_set, dual_set, macro
from FIAT.functional import (
    PointEvaluation, PointDerivative,
    IntegralMomentOfNormalDerivative,
    IntegralMomentOfBidirectionalDerivative,
)
from FIAT.reference_element import TETRAHEDRON
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.jacobi import eval_jacobi
from FIAT.hierarchical import make_dual_bubbles
import numpy


def inner(u, v, wts):
    return numpy.dot(numpy.multiply(u, wts), v.T)


class WalkingtonDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # Vertex dofs: second order jet
        for v in sorted(top[0]):
            cur = len(nodes)
            x, = ref_el.make_points(0, v, degree)
            nodes.append(PointEvaluation(ref_el, x))

            # first and second derivatives
            nodes.extend(PointDerivative(ref_el, x, alpha)
                         for i in (1, 2) for alpha in polynomial_set.mis(sd, i))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # Face dofs: moments or normal derivative
        ref_face = ref_el.construct_subelement(2)
        Q_face = create_quadrature(ref_face, degree-1)
        f_at_qpts = numpy.ones(Q_face.get_weights().shape)
        for face in sorted(top[2]):
            cur = len(nodes)
            nodes.append(IntegralMomentOfNormalDerivative(ref_el, face, Q_face, f_at_qpts))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        # Interior dof: point evaluation at barycenter
        for entity in top[sd]:
            cur = len(nodes)
            x, = ref_el.make_points(sd, entity, sd+1)
            nodes.append(PointEvaluation(ref_el, x))
            entity_ids[sd][entity].extend(range(cur, len(nodes)))

        # Constraint dofs
        # Face-edge constraint: normal derivative along edge is cubic
        edges = ref_el.get_connectivity()[(2, 1)]
        ref_edge = ref_el.construct_subelement(1)
        Q_edge = create_quadrature(ref_edge, 2*(degree-1))
        x = ref_edge.compute_barycentric_coordinates(Q_edge.get_points())
        leg4_at_qpts = eval_jacobi(0, 0, 4, x[:, 1] - x[:, 0])
        # Face constraint: normal derivative is cubic
        Q_face, phis = make_dual_bubbles(ref_face, degree-2, scale=1)

        for face in sorted(top[2]):
            cur = len(nodes)
            thats = ref_el.compute_tangents(sd-1, face)
            nface = numpy.cross(*thats)
            nface *= -1/numpy.dot(nface, nface)

            for i, e in enumerate(edges[face]):
                Q = FacetQuadratureRule(ref_face, 1, i, Q_edge)

                te = ref_el.compute_edge_tangent(e)
                nfe = numpy.cross(te, nface)
                nfe /= numpy.linalg.norm(nfe)
                nfe /= Q.jacobian_determinant()

                nodes.append(IntegralMomentOfNormalDerivative(ref_el, face, Q, leg4_at_qpts, n=nfe))

            Q = FacetQuadratureRule(ref_el, 2, face, Q_face)
            nodes.extend(IntegralMomentOfBidirectionalDerivative(ref_el, Q, phis[0], nface, t) for t in thats)
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Walkington(finite_element.CiarletElement):
    """The Walkington C1 macroelement."""

    def __init__(self, ref_el, degree=5):
        if ref_el.get_shape() != TETRAHEDRON:
            raise ValueError(f"{type(self).__name__} only defined on tetrahedron")
        if degree != 5:
            raise ValueError(f"{type(self).__name__} only defined for degree=5.")

        dual = WalkingtonDualSet(ref_el, degree)
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1, vorder=4, variant="bubble")
        super().__init__(poly_set, dual, degree)
