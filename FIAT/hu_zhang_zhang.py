# Copyright (C) 2026 Pablo D. Brubeck
# SPDX-License-Identifier: LGPL-3.0-or-later

# The extended element has 26 dofs, of which the last 8 are constraints on the
# tangential components of the curl.  The constraints are used in the
# transformation theory.

r"""The Hu-Zhang-Zhang H(grad curl) element on the Alfeld tetrahedron.

This follows the notation of Hu, Zhang, and Zhang, "A family of finite element
Stokes complexes in three dimensions", arXiv:2008.03793, with the local complex

.. math::

    P_r \xrightarrow{\nabla} V^{r-1,k+1} \xrightarrow{\nabla\times}
    \Sigma^{k,+} \xrightarrow{\nabla\cdot} P_{k-1}.

Only the lowest order case :math:`r=k=1` is implemented.
"""

import numpy

from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.dual_set import DualSet
from FIAT.expansions import polynomial_entity_ids
from FIAT.finite_element import CiarletElement
from FIAT.functional import FrobeniusIntegralMoment, IntegralMomentOfCurl, PointCurlComponent
from FIAT.fu_guzman_neilan import PoincareSpace, grad
from FIAT.guzman_neilan import GuzmanNeilanSpace, inner
from FIAT.polynomial_set import ONPolynomialSet, spanning_basis
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TETRAHEDRON


def HuZhangZhangSpace(ref_el):
    r"""Build the extended lowest order H(grad curl) space.

    The space is

    .. math::

        \{v \in \nabla P_5 \oplus \kappa (GN \cap \ker \operatorname{div}) :
        \int_S v \cdot M_S \nabla b \,\mathrm{d}S = 0
        \text{ for all bubbles } b \in P_5(S)\},

    for each entity :math:`S` of positive dimension, where :math:`GN` is the
    extended Guzman-Neilan space of order 1, :math:`\kappa` is the Poincare
    operator about the barycenter, and

    .. math::

        M_S = \sum_{z \in S} (z - x_S) (z - x_S)^T

    sums over the vertices :math:`z` of :math:`S`, with barycenter :math:`x_S`.

    The constraints fix the gradient part, so that the curl is unchanged.  The
    form is positive definite on the gradients of bubbles, it is preserved by
    the covariant Piola map, and it vanishes on the lowest order Nedelec space.
    The tangential trace on :math:`S` therefore only depends on the lowest order
    moments on its edges and on the normal component of the curl on :math:`S`.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.

    Returns
    -------
    PolynomialSet
        The 26-dimensional extended space on the Alfeld split.
    """
    if ref_el.get_shape() != TETRAHEDRON:
        raise ValueError("H(grad curl) is only defined on tetrahedra")
    sd = ref_el.get_spatial_dimension()
    degree = 5
    GN = GuzmanNeilanSpace(ref_el, 1)
    P = ONPolynomialSet(ref_el, degree, variant="bubble")
    V = PoincareSpace(P, GN, ONPolynomialSet(ref_el, 0), degree - 1)

    entity_ids = polynomial_entity_ids(ref_el, degree, continuity="C0")
    top = ref_el.get_topology()
    vertices = numpy.array(ref_el.get_vertices())
    constraints = []
    for dim in range(1, sd + 1):
        ref_entity = V.get_reference_element() if dim == sd else ref_el.construct_subelement(dim)
        rule = create_quadrature(ref_entity, 2 * degree - 2)
        for entity in sorted(top[dim]):
            Q = rule if dim == sd else FacetQuadratureRule(ref_el, dim, entity, rule)
            qpts, qwts = Q.get_points(), Q.get_weights()
            z = vertices[list(top[dim][entity])]
            z = z - numpy.mean(z, axis=0)
            bubbles = grad(P.tabulate(qpts, 1))[entity_ids[dim][entity]]
            constraints.append(inner((z.T @ z) @ bubbles, V.tabulate(qpts)[(0,) * sd], qwts))

    return V.recombine(spanning_basis(numpy.concatenate(constraints), nullspace=True))


class HuZhangZhangDualSet(DualSet):
    """The curl at vertices and edge circulations, augmented by constraints.

    The constraints are the tangential Guzman-Neilan constraints pulled back
    through the curl.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    """

    def __init__(self, ref_el):
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

        # Edge dofs: tangential moment
        ref_edge = ref_el.construct_subelement(1)
        Q_edge = create_quadrature(ref_edge, 4)
        for e in sorted(top[1]):
            Q = FacetQuadratureRule(ref_el, 1, e, Q_edge, avg=True)
            t = ref_el.compute_edge_tangent(e)
            f_at_qpts = numpy.outer(t, numpy.ones(Q.get_weights().shape))
            entity_ids[1][e].append(len(nodes))
            nodes.append(FrobeniusIntegralMoment(ref_el, Q, f_at_qpts))

        # Constraint dofs: tangential GN constraints on the curl
        gn = BernardiRaugelDualSet(ref_el, 1, degree=3)
        for face in sorted(top[2]):
            cur = len(nodes)
            # Skip the normal moment, which comes first
            for i in gn.entity_ids[2][face][1:]:
                nodes.append(IntegralMomentOfCurl(ref_el, gn.nodes[i].Q, gn.nodes[i].f_at_qpts))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HuZhangZhang(CiarletElement):
    r"""The extended lowest order Hu-Zhang-Zhang H(grad curl) tetrahedron.

    This is the space :math:`V^{r-1,k+1}` with :math:`r=k`, which contains
    complete polynomials of degree :math:`\min(r-1, k+1)`.

    The first 18 basis functions form the element: the curl at each vertex and
    a circulation on each edge.  The remaining 8 nodes constrain the tangential
    components of the curl on faces, and support physical basis transformations.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    k : int, optional
        The degree of the complete polynomials in the curl, which must be 1.
    """

    def __init__(self, ref_el, k=1):
        if k != 1:
            raise ValueError(f"{type(self).__name__} is only implemented for k = 1")
        poly_set = HuZhangZhangSpace(ref_el)
        dual = HuZhangZhangDualSet(ref_el)
        super().__init__(poly_set, dual, 4, formdegree=1, mapping="covariant piola")
