# Copyright (C) 2025 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This is not quite Walkington, but is 65-dofs and includes 20 extra constraint
# functionals.  The first 45 basis functions are the reference element
# bfs, but the extra 20 are used in the transformation theory.

from FIAT import finite_element, polynomial_set, macro
from FIAT.dual_set import DualSet
from FIAT.functional import (
    PointEvaluation, PointDerivative,
    IntegralMomentOfDerivative,
)
from FIAT.reference_element import TETRAHEDRON
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.hierarchical import make_projected_bubble_moments
from FIAT.jacobi import eval_jacobi
import numpy


class WalkingtonDualSet(DualSet):
    def __init__(self, ref_el, degree, reduced=False):
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
        if reduced:
            Q_face, phis = make_projected_bubble_moments(
                ref_face, degree-2, interpolant_deg=degree-1)
            f_at_qpts = phis[-1]
        for face in sorted(top[2]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, 2, face, Q_face, avg=True)
            n = ref_el.compute_normal(face)
            nodes.append(IntegralMomentOfDerivative(ref_el, Q, f_at_qpts, n))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        # Interior dof: point evaluation at barycenter
        for entity in top[sd]:
            cur = len(nodes)
            nodes.extend(PointEvaluation(ref_el, x)
                         for x in ref_el.make_points(sd, entity, sd + 1 - reduced))
            entity_ids[sd][entity].extend(range(cur, len(nodes)))

        # Constraint dofs
        # Face-edge constraint: normal derivative along edge is cubic
        edges = ref_el.get_connectivity()[(2, 1)]
        ref_edge = ref_el.construct_subelement(1)
        Q_edge = create_quadrature(ref_edge, 2*(degree-1))
        x = ref_edge.compute_barycentric_coordinates(Q_edge.get_points())
        leg4_at_qpts = eval_jacobi(0, 0, 4, x[:, 1] - x[:, 0])
        # Face constraint: normal derivative drops to degree-2
        Q_face, phis = make_projected_bubble_moments(ref_face, degree-2)
        phi = phis[-1]

        for face in sorted(top[2]):
            cur = len(nodes)
            thats = ref_el.compute_tangents(sd-1, face)
            nface = -numpy.cross(*thats)
            nface /= numpy.linalg.norm(nface)

            for e in sorted(edges[face]):
                Q = FacetQuadratureRule(ref_el, 1, e, Q_edge, avg=True)
                te = ref_el.compute_edge_tangent(e)
                nfe = numpy.cross(te, nface)
                nfe /= numpy.linalg.norm(nfe)
                nodes.append(IntegralMomentOfDerivative(ref_el, Q, leg4_at_qpts, nfe))

            Q = FacetQuadratureRule(ref_el, 2, face, Q_face, avg=True)
            nodes.extend(IntegralMomentOfDerivative(ref_el, Q, phi, nface, t) for t in thats)
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


def _reduced_polynomial_set(ref_el):
    """Construct the reduced Walkington polynomial space."""
    ref_complex = macro.AlfeldSplit(ref_el)
    full_poly_set = macro.CkPolynomialSet(ref_complex, 5, order=1, vorder=4, variant="bubble")
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
    vertices = numpy.asarray(ref_el.get_vertices())
    center = numpy.mean(vertices, axis=0)
    interior_nodes = [PointEvaluation(ref_el, center)]
    interior_weights = [1.0]
    for vertex in vertices:
        offset = center - vertex
        point = tuple(vertex)
        interior_nodes.append(PointEvaluation(ref_el, point))
        interior_weights.append(-0.25)
        for i in range(sd):
            alpha = tuple(int(i == j) for j in range(sd))
            interior_nodes.append(PointDerivative(ref_el, point, alpha))
            interior_weights.append(-offset[i] / 6)
        for i in range(sd):
            for j in range(i, sd):
                alpha = tuple(int(k == i) + int(k == j) for k in range(sd))
                factor = 1 if i == j else 2
                interior_nodes.append(PointDerivative(ref_el, point, alpha))
                interior_weights.append(-factor * offset[i] * offset[j] / 24)

    entity_ids[sd][0] = list(range(len(interior_nodes)))
    riesz = DualSet(interior_nodes, ref_el, entity_ids).to_riesz(full_poly_set)
    constraint = numpy.dot(interior_weights, riesz)
    constraint = numpy.dot(constraint, full_poly_set.get_coeffs().T)
    constraint /= numpy.linalg.norm(constraint)
    basis = polynomial_set.spanning_basis(constraint[numpy.newaxis, :], nullspace=True)
    if basis.shape[0] != 64:
        raise numpy.linalg.LinAlgError(
            f"Expected a 64-dimensional reduced Walkington space, got {basis.shape[0]}.")

    coeffs = numpy.dot(basis, full_poly_set.get_coeffs())
    ref_complex = full_poly_set.get_reference_element()
    return polynomial_set.PolynomialSet(
        ref_complex, 5, 5, full_poly_set.get_expansion_set(), coeffs)


class Walkington(finite_element.CiarletElement):
    """The Walkington C1 macroelement."""

    def __init__(self, ref_el, degree=5, reduced=False):
        if ref_el.get_shape() != TETRAHEDRON:
            raise ValueError(f"{type(self).__name__} only defined on tetrahedron")
        if degree != 5:
            raise ValueError(f"{type(self).__name__} only defined for degree=5.")

        if reduced:
            poly_set = _reduced_polynomial_set(ref_el)
        else:
            ref_complex = macro.AlfeldSplit(ref_el)
            poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1, vorder=4, variant="bubble")
        dual = WalkingtonDualSet(ref_el, degree, reduced=reduced)
        super().__init__(poly_set, dual, degree)
