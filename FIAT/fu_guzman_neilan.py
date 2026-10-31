# Copyright (C) 2026 Pablo D. Brubeck
# SPDX-License-Identifier: LGPL-3.0-or-later

# This is not quite the reduced Fu-Guzman-Neilan element, but is 86-dofs and
# includes 32 extra constraint functionals.  The first 54 basis functions are
# the reference element bfs, but the extra 32 are used in the transformation
# theory.

"""H1(curl) elements on the Alfeld tetrahedron."""

import numpy

from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.bernstein import BernsteinPolynomialSet
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
from FIAT.functional import (ComponentPointEvaluation, FrobeniusIntegralMoment,
                             IntegralMomentOfDerivative, PointDirectionalDerivative)
from FIAT.guzman_neilan import GuzmanNeilanSpace, div, inner
from FIAT.macro import AlfeldSplit
from FIAT.polynomial_set import (ONPolynomialSet, TaylorConstrainedPolynomialSet, mis,
                                 spanning_basis)
from FIAT.quadrature import FacetQuadratureRule, GaussLegendreQuadratureLineRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TETRAHEDRON, UFCInterval
from FIAT.walkington import WalkingtonDualSet


def ReducedFuGuzmanNeilanH1curlSpace(ref_el):
    r"""Build the extended quartic H1(curl) space.

    The space is

    .. math::

        \nabla W \oplus \kappa (GN \cap \ker \operatorname{div}),

    where :math:`W` is the extended reduced Walkington space, :math:`GN` is
    the extended Guzman-Neilan space, and :math:`\kappa` is the Poincaré
    operator about the barycenter :math:`x_T`

    .. math::

        \kappa g(x) = \int_0^1 t\, g(x_T + t (x - x_T)) \times (x - x_T) \,\mathrm{d}t.

    Since :math:`(x - x_T) \cdot \kappa g = 0`, the image of :math:`\kappa`
    has no gradient part, and satisfies the barycenter supersmoothness and
    Taylor constraints that :math:`\nabla W` inherits from :math:`W`.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.

    Returns
    -------
    PolynomialSet
        The 86-dimensional extended space on the Alfeld split.
    """
    if ref_el.get_shape() != TETRAHEDRON:
        raise ValueError("H1(curl) is only defined on tetrahedra")
    sd = ref_el.get_spatial_dimension()
    ref_complex = AlfeldSplit(ref_el)
    rule = create_quadrature(ref_complex, 8)
    qpts, qwts = rule.get_points(), rule.get_weights()

    # Gradients of the extended reduced Walkington space
    W = BernsteinPolynomialSet(ref_complex, 5, order=1, vorder=4)
    W = TaylorConstrainedPolynomialSet(W, (1, 2/3, 1/3))
    dW = W.tabulate(qpts, 1)
    grads = numpy.stack([dW[alpha] for alpha in mis(sd, 1)], axis=1)

    # The divergence of GN is constant, so its kernel has codimension one
    GN = GuzmanNeilanSpace(ref_el, 1)
    mean_div = numpy.dot(div(GN.tabulate(qpts, 1)), qwts)
    GN = GN.recombine(spanning_basis(mean_div[None, :], nullspace=True))

    # The Poincaré integrand is a quartic in t, so three Gauss points are exact
    barycenter = numpy.mean(ref_el.get_vertices(), axis=0)
    x = qpts - barycenter
    line = GaussLegendreQuadratureLineRule(UFCInterval(), 3)
    kappa = sum(w * t * numpy.cross(GN.tabulate(barycenter + t * x)[(0,) * sd], x.T[None], axis=1)
                for (t,), w in zip(line.get_points(), line.get_weights()))

    # The L2 projection is exact, and the gradient of the constant vanishes
    P = ONPolynomialSet(ref_complex, 4, shape=(sd,), variant="bubble")
    v = P.tabulate(qpts)[(0,) * sd]
    f = numpy.concatenate([grads, kappa])
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, f, qwts))
    return P.recombine(spanning_basis(coeffs.T))


class ReducedFuGuzmanNeilanH1curlDualSet(DualSet):
    """Vertex jets and edge circulations, augmented by constraints.

    The constraints are the reduced Walkington constraints pulled back
    through the gradient, and the tangential Guzman-Neilan constraints
    pulled back through the curl.

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

        # Vertex dofs: first order jet
        for v in sorted(top[0]):
            cur = len(nodes)
            x, = ref_el.make_points(0, v, 4)
            nodes.extend(ComponentPointEvaluation(ref_el, c, (sd,), x) for c in range(sd))
            nodes.extend(PointDirectionalDerivative(ref_el, d, x, comp=(c,), shp=(sd,))
                         for d in numpy.eye(sd) for c in range(sd))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # Edge dofs: tangential moment
        ref_edge = ref_el.construct_subelement(1)
        Q_edge = create_quadrature(ref_edge, 4)
        for e in sorted(top[1]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, 1, e, Q_edge, avg=True)
            t = ref_el.compute_edge_tangent(e)
            f_at_qpts = numpy.outer(t, numpy.ones(Q.get_weights().shape))
            nodes.append(FrobeniusIntegralMoment(ref_el, Q, f_at_qpts))
            entity_ids[1][e].extend(range(cur, len(nodes)))

        # Constraint dofs: Walkington constraints on the potential, pairing
        # second derivatives with the symmetric gradient
        walkington = WalkingtonDualSet(ref_el, 5, reduced=True)
        for dim in range(1, sd+1):
            for entity in sorted(top[dim]):
                cur = len(nodes)
                for i in walkington.entity_ids[dim][entity]:
                    node = walkington.nodes[i]
                    f_at_qpts = node.f_at_qpts
                    if f_at_qpts.ndim == 2:
                        nodes.append(FrobeniusIntegralMoment(ref_el, node.Q, f_at_qpts))
                    else:
                        f_at_qpts = (f_at_qpts + f_at_qpts.swapaxes(0, 1)) / 2
                        nodes.append(IntegralMomentOfDerivative(ref_el, node.Q, f_at_qpts, shp=(sd,)))
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        # Constraint dofs: tangential GN constraints on the curl, where
        # f . curl v = sum_d (f x e_d) . D_d v
        gn = BernardiRaugelDualSet(ref_el, 1, degree=3)
        for face in sorted(top[2]):
            cur = len(nodes)
            # Skip the normal moment, which comes first
            for i in gn.entity_ids[2][face][1:]:
                node = gn.nodes[i]
                f_at_qpts = numpy.cross(node.f_at_qpts[:, None], numpy.eye(sd)[..., None], axis=0)
                nodes.append(IntegralMomentOfDerivative(ref_el, node.Q, f_at_qpts, shp=(sd,)))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class ReducedFuGuzmanNeilanH1curl(CiarletElement):
    """Extended H1(curl) tetrahedron.

    The first 54 basis functions form the reduced element: twelve first-jet
    degrees of freedom at each vertex and one circulation on each edge.
    The remaining 32 nodes constrain the boundary traces and curl, and
    support physical basis transformations.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    degree : int, optional
        Polynomial degree, which must be four.
    """

    def __init__(self, ref_el, degree=4):
        if degree != 4:
            raise ValueError("H1(curl) is only implemented for degree four")
        poly_set = ReducedFuGuzmanNeilanH1curlSpace(ref_el)
        dual = ReducedFuGuzmanNeilanH1curlDualSet(ref_el)
        super().__init__(poly_set, dual, degree, formdegree=1, mapping="covariant piola")
