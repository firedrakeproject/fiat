# Copyright (C) 2026 Pablo D. Brubeck
# SPDX-License-Identifier: LGPL-3.0-or-later

# This is not quite the reduced Fu-Guzman-Neilan element.  The extended element
# of order 2 has 86 dofs, of which the last 32 are extra constraint functionals.
# The extended element of order 3 has 105 dofs, of which the last 28 are extra
# constraint functionals.  The constraints are used in the transformation theory.

"""H1(curl) elements on the Alfeld tetrahedron."""

import numpy

from FIAT.bernardi_raugel import BernardiRaugelDualSet
from FIAT.bernstein import BernsteinPolynomialSet
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
from FIAT.functional import (ComponentPointEvaluation, FrobeniusIntegralMoment,
                             IntegralMomentOfCurl, IntegralMomentOfDerivative,
                             PointCurlComponent, PointDirectionalDerivative)
from FIAT.guzman_neilan import GuzmanNeilanSpace, div, inner
from FIAT.macro import AlfeldSplit
from FIAT.polynomial_set import (ONPolynomialSet, TaylorConstrainedPolynomialSet, mis,
                                 spanning_basis)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import TETRAHEDRON, UFCInterval
from FIAT.walkington import WalkingtonDualSet


def grad(U):
    """Compute the gradient from tabulation dict."""
    sd = len(next(iter(U)))
    return numpy.stack([U[alpha] for alpha in mis(sd, 1)], axis=-2)


def poincare_curl(V, points):
    r"""Tabulate the Poincare operator that inverts the curl.

    The operator about the barycenter :math:`x_T` is

    .. math::

        \kappa g(x) = \int_0^1 t\, g(x_T + t (x - x_T)) \times (x - x_T) \,\mathrm{d}t.

    Each subcell of an Alfeld split is a cone from :math:`x_T`, so the integrand
    is a polynomial in :math:`t`, and :math:`\kappa` raises the degree by one.
    """
    ref_el = V.get_reference_element()
    cell = ref_el.get_parent() or ref_el
    sd = cell.get_spatial_dimension()
    barycenter = numpy.mean(cell.get_vertices(), axis=0)
    x = points - barycenter
    line = create_quadrature(UFCInterval(), V.degree + 1)
    return sum(w * t * numpy.cross(V.tabulate(barycenter + t * x)[(0,) * sd], x.T[None], axis=1)
               for (t,), w in zip(line.get_points(), line.get_weights()))


def PoincareSpace(V0, V2, V3, degree, d0=grad, d2=div, poincare=poincare_curl):
    r"""Fill in the space between two spaces of a complex.

    Given the spaces :math:`V_0, V_2, V_3` of the complex

    .. math::

        V_0 \xrightarrow{d_0} V_1 \xrightarrow{d_1} V_2 \xrightarrow{d_2} V_3,

    return

    .. math::

        V_1 = d_0 V_0 \oplus \kappa (V_2 \cap \ker d_2),

    where :math:`\kappa` is a Poincare operator with :math:`d_1 \kappa = I` on
    :math:`\ker d_2` and with an image that intersects :math:`\ker d_1` trivially.

    Parameters
    ----------
    V0 : PolynomialSet
        The space before :math:`V_1`.
    V2 : PolynomialSet
        The space after :math:`V_1`, on the cell or macrocell of :math:`V_1`.
    V3 : PolynomialSet
        A space containing :math:`d_2 V_2`.
    degree : int
        The polynomial degree of :math:`V_1`.
    d0 : callable, optional
        A first order operator that maps a tabulation dict to :math:`d_0`.
    d2 : callable, optional
        A first order operator that maps a tabulation dict to :math:`d_2`.
    poincare : callable, optional
        Maps a polynomial set and points to the tabulation of :math:`\kappa`,
        which encodes :math:`d_1`.

    Returns
    -------
    PolynomialSet
        The space :math:`V_1` in C0 piecewise polynomials of the given degree.
    """
    ref_complex = V2.get_reference_element()
    sd = ref_complex.get_spatial_dimension()
    rule = create_quadrature(ref_complex, 2 * degree)
    qpts, qwts = rule.get_points(), rule.get_weights()

    q = V3.tabulate(qpts)[(0,) * sd]
    dV2 = d2(V2.tabulate(qpts, 1))
    kernel = V2.recombine(spanning_basis(inner(q, dV2, qwts), nullspace=True))
    dV0 = d0(V0.tabulate(qpts, 1))
    f = numpy.concatenate([dV0, poincare(kernel, qpts)])

    # The L2 projection is exact, and any zero image of d0 drops out of the span
    P = ONPolynomialSet(ref_complex, degree, shape=f.shape[1:-1], variant="bubble")
    v = P.tabulate(qpts)[(0,) * sd]
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, f, qwts))
    return P.recombine(spanning_basis(coeffs.T))


def ReducedFuGuzmanNeilanH1curlSpace(ref_el, order=2):
    r"""Build the extended quartic H1(curl) space.

    The space is

    .. math::

        \nabla W \oplus \kappa (GN \cap \ker \operatorname{div}),

    where :math:`GN` is the extended Guzman-Neilan space of order ``order - 1``,
    :math:`W` is the extended Walkington space, which is reduced for order 2,
    and :math:`\kappa` is the Poincare operator about the barycenter.

    Since :math:`(x - x_T) \cdot \kappa g = 0`, the image of :math:`\kappa`
    has no gradient part, and satisfies the barycenter supersmoothness and
    Taylor constraints that :math:`\nabla W` inherits from :math:`W`.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    order : int, optional
        The embedded subdegree, which must be 2 or 3.

    Returns
    -------
    PolynomialSet
        The extended space on the Alfeld split, with dimension 86 for order 2
        and 105 for order 3.
    """
    if ref_el.get_shape() != TETRAHEDRON:
        raise ValueError("H1(curl) is only defined on tetrahedra")
    GN = GuzmanNeilanSpace(ref_el, order - 1)
    W = BernsteinPolynomialSet(GN.get_reference_element(), 5, order=1, vorder=4)
    if order == 2:
        W = TaylorConstrainedPolynomialSet(W, (1, 2/3, 1/3))
    pressure = ONPolynomialSet(ref_el, order - 2)
    return PoincareSpace(W, GN, pressure, 4)


class ReducedFuGuzmanNeilanH1curlDualSet(DualSet):
    """Vertex jets and edge circulations, augmented by constraints.

    For order 3, the dofs also include the curl at edge midpoints, normal
    moments on faces, and a cell moment against the position vector.  The
    constraints are the Walkington constraints pulled back through the
    gradient, and the tangential Guzman-Neilan constraints pulled back through
    the curl.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    order : int, optional
        The embedded subdegree, which must be 2 or 3.
    """

    def __init__(self, ref_el, order=2):
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

        # Edge dofs: curl at the interior GN lattice points
        for e in sorted(top[1]):
            cur = len(nodes)
            nodes.extend(PointCurlComponent(ref_el, c, x)
                         for x in ref_el.make_points(1, e, order - 1) for c in range(sd))
            entity_ids[1][e].extend(range(cur, len(nodes)))

        # Walkington dofs and constraints on the potential, pairing second
        # derivatives with the symmetric gradient.  The interior dof of W, its
        # barycenter value, becomes a cell average against the position vector,
        # which annihilates the image of the Poincare operator and is preserved
        # by the covariant Piola map.
        walkington = WalkingtonDualSet(ref_el, 5, reduced=(order == 2))
        entities = {i: (dim, entity)
                    for dim in walkington.entity_ids
                    for entity in walkington.entity_ids[dim]
                    for i in walkington.entity_ids[dim][entity]}
        Q_cell = create_quadrature(AlfeldSplit(ref_el), 5)
        barycenter = numpy.mean(ref_el.get_vertices(), axis=0)
        for i, node in enumerate(walkington.nodes):
            dim, entity = entities[i]
            if dim == 0:
                continue
            if dim == sd:
                f_at_qpts = (Q_cell.get_points() - barycenter).T / ref_el.volume()
                functional = FrobeniusIntegralMoment(ref_el, Q_cell, f_at_qpts)
            elif node.f_at_qpts.ndim == 2:
                functional = FrobeniusIntegralMoment(ref_el, node.Q, node.f_at_qpts)
            else:
                f_at_qpts = (node.f_at_qpts + node.f_at_qpts.swapaxes(0, 1)) / 2
                functional = IntegralMomentOfDerivative(ref_el, node.Q, f_at_qpts, shp=(sd,))
            entity_ids[dim][entity].append(len(nodes))
            nodes.append(functional)

        # Constraint dofs: tangential GN constraints on the curl
        gn = BernardiRaugelDualSet(ref_el, order - 1, degree=3)
        for face in sorted(top[2]):
            cur = len(nodes)
            # Skip the normal moment, which comes first
            for i in gn.entity_ids[2][face][1:]:
                node = gn.nodes[i]
                nodes.append(IntegralMomentOfCurl(ref_el, node.Q, node.f_at_qpts))
            entity_ids[2][face].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class ReducedFuGuzmanNeilanH1curl(CiarletElement):
    """Extended quartic H1(curl) tetrahedron.

    The first basis functions form the reduced element, with 54 for order 2
    and 77 for order 3.  The remaining nodes constrain the boundary traces and
    curl, and support physical basis transformations.

    Parameters
    ----------
    ref_el : Cell
        A tetrahedron.
    order : int, optional
        The embedded subdegree, which must be 2 or 3.
    """

    def __init__(self, ref_el, order=2):
        if order not in {2, 3}:
            raise ValueError(f"{type(self).__name__} is only implemented for order 2 or 3")
        poly_set = ReducedFuGuzmanNeilanH1curlSpace(ref_el, order)
        dual = ReducedFuGuzmanNeilanH1curlDualSet(ref_el, order)
        super().__init__(poly_set, dual, 4, formdegree=1, mapping="covariant piola")
