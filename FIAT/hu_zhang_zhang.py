# Copyright (C) 2026 Pablo D. Brubeck
# SPDX-License-Identifier: LGPL-3.0-or-later

# The extended element includes 8 constraints on the tangential components of
# the curl.  The constraints are numbered last and used in the transformation
# theory.

r"""The Hu-Zhang-Zhang H(grad curl) element on the Alfeld tetrahedron.

This follows the notation of Hu, Zhang, and Zhang, "A family of finite element
Stokes complexes in three dimensions", arXiv:2008.03793, with the local complex

.. math::

    P_r \xrightarrow{\nabla} V^{r-1,k+1} \xrightarrow{\nabla\times}
    \Sigma^{k,+} \xrightarrow{\nabla\cdot} P_{k-1}.

Only the case :math:`r=k` is implemented.
"""

import numpy

from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.dual_set import DualSet
from FIAT.expansions import polynomial_entity_ids
from FIAT.finite_element import CiarletElement
from FIAT.functional import FrobeniusIntegralMoment, IntegralMomentOfCurl, PointCurlComponent
from FIAT.fu_guzman_neilan import PoincareSpace, grad, poincare_curl
from FIAT.guzman_neilan import GuzmanNeilanSpace, inner
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TETRAHEDRON


def corrected_poincare_curl(V, points):
    r"""Tabulate the Poincare operator with a conforming tangential trace.

    The operator is :math:`\tilde\kappa g = \kappa g - \nabla\varphi`, where
    :math:`\kappa` is the Poincare operator about the barycenter, and
    :math:`\varphi` is a sum of bubbles of one degree above :math:`\kappa g`,
    with

    .. math::

        \int_S \tilde\kappa g \cdot M_S \nabla b \,\mathrm{d}S = 0,
        \qquad M_S = \sum_{z \in S} (z - x_S) (z - x_S)^T,

    for all bubbles :math:`b` of the entity :math:`S` of positive dimension,
    where :math:`z` are the vertices of :math:`S` and :math:`x_S` is its barycenter.

    The correction is curl-free.  The form is positive definite on the
    gradients of bubbles, it is preserved by the covariant Piola map, and it
    vanishes on the lowest order Nedelec space.  The tangential trace on
    :math:`S` therefore only depends on the lowest order moments on its edges
    and on the normal component of the curl on :math:`S`.
    """
    ref_complex = V.get_reference_element()
    ref_el = ref_complex.get_parent()
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    vertices = numpy.array(ref_el.get_vertices())

    degree = V.degree + 2
    P = ONPolynomialSet(ref_el, degree, variant="bubble")
    entity_ids = polynomial_entity_ids(ref_el, degree, continuity="C0")
    bubbles = [i for dim in range(1, sd + 1) for entity in sorted(top[dim]) for i in entity_ids[dim][entity]]

    rules = []
    metrics = []
    dofs = []
    for dim in range(1, sd + 1):
        ref_entity = ref_complex if dim == sd else ref_el.construct_subelement(dim)
        rule = create_quadrature(ref_entity, 2 * degree - 2)
        for entity in sorted(top[dim]):
            z = vertices[list(top[dim][entity])]
            z = z - numpy.mean(z, axis=0)
            rules.append(rule if dim == sd else FacetQuadratureRule(ref_el, dim, entity, rule))
            metrics.append(z.T @ z)
            dofs.append(entity_ids[dim][entity])

    qpts = numpy.concatenate([Q.get_points() for Q in rules])
    qwts = numpy.concatenate([Q.get_weights() for Q in rules])
    dP = grad(P.tabulate(qpts, 1))

    # Each bubble only tests on the points of its entity
    tests = []
    start = 0
    for Q, metric, ids in zip(rules, metrics, dofs):
        stop = start + len(Q.get_weights())
        test = numpy.zeros_like(dP[ids])
        test[..., start:stop] = metric @ dP[ids][..., start:stop]
        tests.append(test)
        start = stop
    test = numpy.concatenate(tests)

    A = inner(test, dP[bubbles], qwts)
    b = inner(test, poincare_curl(V, qpts), qwts)
    coeffs = numpy.linalg.solve(A, b)
    dP = grad(P.tabulate(points, 1))
    return poincare_curl(V, points) - numpy.tensordot(coeffs.T, dP[bubbles], axes=1)


def HuZhangZhangSpace(ref_el, k=1):
    r"""Build the extended H(grad curl) space with :math:`r=k`.

    The space is

    .. math::

        V^{k-1,k+1} = \nabla P_k \oplus \tilde\kappa (GN \cap \ker \operatorname{div}),

    where :math:`GN` is the extended Guzman-Neilan space of order :math:`k`, and
    :math:`\tilde\kappa` is given by :func:`corrected_poincare_curl`.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    k : int, optional
        The degree of the complete polynomials in the curl, which must be 1 or 2.

    Returns
    -------
    PolynomialSet
        The extended space on the Alfeld split, with dimension 26 for
        :math:`k=1` and 50 for :math:`k=2`.
    """
    if ref_el.get_shape() != TETRAHEDRON:
        raise ValueError("H(grad curl) is only defined on tetrahedra")
    P = ONPolynomialSet(ref_el, k)
    GN = GuzmanNeilanSpace(ref_el, k)
    pressure = ONPolynomialSet(ref_el, k - 1)
    return PoincareSpace(P, GN, pressure, GN.degree + 1, poincare=corrected_poincare_curl)


class HuZhangZhangDualSet(DualSet):
    """The curl at vertices and edge dofs, augmented by constraints.

    The edge dofs are moments of the tangential component, and the curl at
    interior lattice points.  The constraints are the tangential Guzman-Neilan
    constraints pulled back through the curl.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    k : int, optional
        The degree of the complete polynomials in the curl, which must be 1 or 2.
    """

    def __init__(self, ref_el, k=1):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # Vertex dofs: the curl
        for v in sorted(top[0]):
            cur = len(nodes)
            x, = ref_el.make_points(0, v, 4)
            nodes.extend(PointCurlComponent(ref_el, c, x) for c in range(sd))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # Edge dofs: tangential moments and the curl at interior lattice points
        ref_edge = ref_el.construct_subelement(1)
        Q_edge = create_quadrature(ref_edge, 4 + k - 1)
        moments = ONPolynomialSet(ref_edge, k - 1).tabulate(Q_edge.get_points())[(0,)]
        for e in sorted(top[1]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, 1, e, Q_edge, avg=True)
            t = ref_el.compute_edge_tangent(e)
            nodes.extend(FrobeniusIntegralMoment(ref_el, Q, numpy.outer(t, phi)) for phi in moments)
            nodes.extend(PointCurlComponent(ref_el, c, x)
                         for x in ref_el.make_points(1, e, k) for c in range(sd))
            entity_ids[1][e].extend(range(cur, len(nodes)))

        # Constraint dofs: tangential GN constraints on the curl
        gn = BernardiRaugelDualSet(ref_el, k, degree=3)
        for face in sorted(top[2]):
            cur = len(nodes)
            # Skip the normal moment, which comes first
            for i in gn.entity_ids[2][face][1:]:
                node = gn.nodes[i]
                nodes.append(IntegralMomentOfCurl(ref_el, node.Q, node.f_at_qpts))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HuZhangZhang(CiarletElement):
    r"""The extended Hu-Zhang-Zhang H(grad curl) tetrahedron.

    This is the space :math:`V^{r-1,k+1}` with :math:`r=k`, which contains
    complete polynomials of degree :math:`\min(r-1, k+1)`.

    The first basis functions form the element, with 18 for :math:`k=1` and 42
    for :math:`k=2`: the curl at each vertex, and on each edge the tangential
    moments and the curl at interior lattice points.  The remaining 8 nodes
    constrain the tangential components of the curl on faces, and support
    physical basis transformations.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    k : int, optional
        The degree of the complete polynomials in the curl, which must be 1 or 2.
    """

    def __init__(self, ref_el, k=1):
        if k not in {1, 2}:
            raise ValueError(f"{type(self).__name__} is only implemented for k = 1 or 2")
        poly_set = HuZhangZhangSpace(ref_el, k)
        dual = HuZhangZhangDualSet(ref_el, k)
        super().__init__(poly_set, dual, poly_set.degree, formdegree=1, mapping="covariant piola")
