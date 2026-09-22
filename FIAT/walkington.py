# Copyright (C) 2025 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This is not quite Walkington, but is 65-dofs and includes 20 extra constraint
# functionals.  The first 45 basis functions are the reference element
# bfs, but the extra 20 are used in the transformation theory.

from collections import defaultdict

from FIAT import finite_element, polynomial_set, macro
from FIAT.dual_set import DualSet
from FIAT.functional import (
    Functional, PointEvaluation, PointDerivative,
    PointDirectionalDerivative, IntegralMomentOfDerivative,
)
from FIAT.reference_element import TETRAHEDRON
from FIAT.quadrature import QuadratureRule, FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.hierarchical import make_projected_bubble_moments
from FIAT.jacobi import eval_jacobi
import numpy


class WalkingtonDualSet(DualSet):
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
            Q = FacetQuadratureRule(ref_el, 2, face, Q_face, avg=True)
            n = ref_el.compute_normal(face)
            nodes.append(IntegralMomentOfDerivative(ref_el, Q, f_at_qpts, n))
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
        # Face constraint: normal derivative drops to degree-2
        Q_face, phis = make_projected_bubble_moments(ref_face, degree-2)
        phi = phis[-1]

        extra_entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        extra_nodes = []

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

            cur = len(extra_nodes)
            extra_nodes.extend(IntegralMomentOfDerivative(ref_el, Q, phi, thats[i], thats[j])
                               for i in range(2) for j in range(i, 2))
            extra_entity_ids[2][face].extend(range(cur, len(extra_nodes)))

        self.nodal_completion = DualSet(extra_nodes, ref_el, extra_entity_ids)
        super().__init__(nodes, ref_el, entity_ids)


def combine(ref_el, terms, nm):
    """Sum FIAT functionals, weighted, into a single functional.

    Weights sharing a point and multi-index are added rather than listed
    separately, as dual set evaluation keeps one weight per pair.
    """
    pt_dict = defaultdict(lambda: defaultdict(float))
    deriv_dict = defaultdict(lambda: defaultdict(float))
    for scale, node in terms:
        for pt, wc in node.get_point_dict().items():
            for w, c in wc:
                pt_dict[pt][tuple(c)] += scale * w
        for pt, wac in node.deriv_dict.items():
            for w, alpha, c in wac:
                deriv_dict[pt][(tuple(alpha), tuple(c))] += scale * w
    return Functional(ref_el, tuple(),
                      {pt: [(w, c) for c, w in wc.items()]
                       for pt, wc in pt_dict.items()},
                      {pt: [(w, alpha, c) for (alpha, c), w in wac.items()]
                       for pt, wac in deriv_dict.items()},
                      nm)


def derivative_sum(ref_el, pts, wts, *directions):
    """A weighted sum of directional derivatives sampled at several points."""
    Q = QuadratureRule(ref_el, numpy.asarray(pts, dtype=float), numpy.asarray(wts, dtype=float))
    return IntegralMomentOfDerivative(ref_el, Q, numpy.ones(len(wts)), *directions)


def edge_constraint(ref_el, edge, n):
    """Walkington's edge condition, eq. (2.1), applied to p = du/dn on an edge.

    A quartic p on e satisfies p(c_e) = (p(v_a) + p(v_b))/2
    + (|e|/8)(p'(v_a) - p'(v_b)) if and only if p is cubic.  Writing the
    arclength derivative against the unnormalized tangent absorbs |e|.
    """
    verts = numpy.asarray(ref_el.get_vertices())
    a, b = ref_el.get_topology()[1][edge]
    va, vb = verts[a], verts[b]
    cm, = ref_el.make_points(1, edge, 2)
    return combine(ref_el, [
        (1.0, derivative_sum(ref_el, [cm, va, vb], [1.0, -0.5, -0.5], n)),
        (1.0, derivative_sum(ref_el, [va, vb], [-0.125, 0.125], vb - va, n)),
    ], "WalkingtonEdgeConstraint")


def face_residual(ref_el, face, n, i):
    """The defect in the i-th identity of Walkington's Lemma 2.2 for p = du/dn.

    On a face whose normal derivative is already cubic along each edge,
    the three defects are equal precisely when p is cubic on the face,
    and they vanish when in addition p(c_f) = 0.  The two gradient groups
    carry the sign opposite to the one printed in the paper, which makes
    the defects equal 81 p(c_f) / 125 on the cubics, as its first identity
    requires.
    """
    verts = numpy.asarray(ref_el.get_vertices())
    vi, vj, vk = (verts[v] for v in numpy.roll(ref_el.get_topology()[2][face], -i))
    x = 0.6*vi + 0.2*vj + 0.2*vk
    return combine(ref_el, [
        (1.0, derivative_sum(ref_el, [x, vi, vj, vk],
                             [1.0, -12/25, 8/125, 8/125], n)),
        (1.0, derivative_sum(ref_el, [vi], [1.0], n, (6/125)*(2*vi - vj - vk))),
        (1.0, derivative_sum(ref_el, [vj], [1.0], n, (-2/125)*(vj - vk))),
        (1.0, derivative_sum(ref_el, [vk], [1.0], n, (2/125)*(vj - vk))),
    ], "WalkingtonFaceResidual")


class WalkingtonPointDualSet(DualSet):
    """Walkington's own point-based nodes, from [Walkington 2014, section 2].

    The degrees of freedom are those of his Lemma 2.3: the second order jet
    at each vertex, the normal derivative at each face centroid, and the
    value at the barycenter.  The twenty constraints are his pointwise
    characterizations of the reduction rather than moments: eq. (2.1) on
    each edge of each face, and, on each face, two differences of the three
    defects of Lemma 2.2, which annihilate the cubics without also pinning
    the face degree of freedom.
    """
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
            nodes.extend(PointDerivative(ref_el, x, alpha)
                         for i in (1, 2) for alpha in polynomial_set.mis(sd, i))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # Face dofs: normal derivative at the centroid
        for face in sorted(top[2]):
            cur = len(nodes)
            x, = ref_el.make_points(2, face, 3)
            n = ref_el.compute_normal(face)
            nodes.append(PointDirectionalDerivative(ref_el, n, x, nm="PointNormalDeriv"))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        # Interior dof: point evaluation at barycenter
        for entity in top[sd]:
            cur = len(nodes)
            x, = ref_el.make_points(sd, entity, sd+1)
            nodes.append(PointEvaluation(ref_el, x))
            entity_ids[sd][entity].extend(range(cur, len(nodes)))

        # Constraint dofs, owned by the face whose normal derivative they reduce
        edges = ref_el.get_connectivity()[(2, 1)]
        for face in sorted(top[2]):
            cur = len(nodes)
            n = ref_el.compute_normal(face)
            nodes.extend(edge_constraint(ref_el, e, n) for e in sorted(edges[face]))
            defects = [face_residual(ref_el, face, n, i) for i in range(3)]
            nodes.extend(combine(ref_el, [(1.0, defects[i]), (-1.0, defects[i+1])],
                                 "WalkingtonFaceConstraint") for i in range(2))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Walkington(finite_element.CiarletElement):
    """The Walkington C1 macroelement.

    :arg variant: ``"moment"`` for moment-based nodes and constraints, or
        ``"point"`` for Walkington's own point-based ones.
    """

    def __init__(self, ref_el, degree=5, variant="moment"):
        if ref_el.get_shape() != TETRAHEDRON:
            raise ValueError(f"{type(self).__name__} only defined on tetrahedron")
        if degree != 5:
            raise ValueError(f"{type(self).__name__} only defined for degree=5.")
        try:
            make_dual = {"moment": WalkingtonDualSet, "point": WalkingtonPointDualSet}[variant]
        except KeyError:
            raise ValueError(f"Unsupported {type(self).__name__} variant {variant!r}") from None

        dual = make_dual(ref_el, degree)
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1, vorder=4, variant="bubble")
        super().__init__(poly_set, dual, degree)
