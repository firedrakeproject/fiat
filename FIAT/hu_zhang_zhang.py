# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This is the lowest-order H(grad curl) element of Hu, Zhang and Zhang in the
# same extended form as Guzman-Neilan: it has 26 dofs, of which the last 8 are
# tangential face moments of the curl.  The first 18 basis functions are the
# reference element bfs, and the extra 8 are used in the transformation theory.

from FIAT import finite_element, dual_set, polynomial_set
from FIAT.functional import Functional
from FIAT.guzman_neilan import GuzmanNeilanFirstKindH1, GuzmanNeilanSpace
from FIAT.macro import AlfeldSplit, CkPolynomialSet
from FIAT.nedelec import Nedelec
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import ufc_simplex

import numpy

# The Levi-Civita symbol
EPS = numpy.zeros((3, 3, 3))
for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[i, j, k] = 1
    EPS[i, k, j] = -1


def curl(U):
    """Compute the curl from a tabulation dict of a 3D vector field."""
    alphas = [tuple(row) for row in numpy.eye(3, dtype=int)]
    grad = numpy.stack([U[alpha] for alpha in alphas], axis=1)
    return numpy.einsum("ijk,njkq->niq", EPS, grad)


def scale_rows(A):
    return A / numpy.maximum(abs(A).max(axis=1, keepdims=True), 1E-300)


def HuZhangZhangSpace(ref_el, degree=4):
    r"""Return a basis for the extended Hu-Zhang-Zhang H(grad curl) space.

    This is the space of C0 piecewise polynomial fields u on the Alfeld split
    such that

    * curl u lies in the extended Guzman-Neilan space,
    * u . t is constant on each edge,
    * \int_f u . q dA = 0 for q = (x - x_f) b - mean(q) with b in B_f P_2(f),
      where B_f is the cubic bubble on each face f,
    * \int_K u . q dx = 0 for q = (x - x_K) b - mean(q) with b a C1 bubble on
      the Alfeld split.

    The first two conditions determine u up to gradients of bubbles, and the
    last two remove those bubbles: the pairing against (x - x_c) b is a Koszul
    duality, nondegenerate since \int grad b' . (x - x_c) b = -d/2 \int b b'
    when b = b'. Every condition is invariant under affine maps with the
    covariant Piola transform, so this space is too.  The space differs from the
    construction of Hu, Zhang and Zhang only by curl-free fields.

    :arg ref_el: a tetrahedron.
    :kwarg degree: the polynomial degree of the ambient C0 space.

    :returns: a PolynomialSet basis for the extended Hu-Zhang-Zhang space.
    """
    sd = ref_el.get_spatial_dimension()
    if sd != 3:
        raise ValueError("Hu-Zhang-Zhang is only defined on tetrahedra")
    ref_complex = AlfeldSplit(ref_el)
    top = ref_el.get_topology()
    verts = numpy.asarray(ref_el.get_vertices())

    U = ONPolynomialSet(ref_complex, degree, shape=(sd,), scale=1, variant="bubble")
    GN = GuzmanNeilanSpace(ref_el, 1, kind=1, reduced=False)
    num_u = U.get_num_members()
    num_gn = GN.get_num_members()
    z = (0,) * sd

    def pad(rows):
        return numpy.concatenate([rows, numpy.zeros((rows.shape[0], num_gn))], axis=1)

    # curl u = w with w in GN
    Q = create_quadrature(ref_complex, 2*degree + 2)
    qpts, qwts = numpy.asarray(Q.get_points()), numpy.asarray(Q.get_weights())
    U_tab = U.tabulate(qpts, 1)
    U_at_qpts = U_tab[z]
    GN_at_qpts = GN.tabulate(qpts)[z]
    rows = [numpy.concatenate([curl(U_tab).reshape(num_u, -1),
                               -GN_at_qpts.reshape(num_gn, -1)]).T]

    # u . t is constant on each edge
    line = ufc_simplex(1)
    Q_ref = create_quadrature(line, 2*degree)
    legendre = ONPolynomialSet(line, degree).tabulate(Q_ref.get_points())[(0,)][1:]
    for e in sorted(top[1]):
        Qe = FacetQuadratureRule(ref_el, 1, e, Q_ref)
        t = ref_el.compute_edge_tangent(e)
        ut = numpy.tensordot(U.tabulate(Qe.get_points())[z], t, axes=(1, 0))
        rows.append(pad(numpy.dot(ut * Qe.get_weights(), legendre.T).T))

    # Koszul gauge on each face against the cubic bubble times P2
    tri = ufc_simplex(2)
    Q_ref = create_quadrature(tri, 2*degree + 4)
    for f in sorted(top[sd-1]):
        Qf = FacetQuadratureRule(ref_el, sd-1, f, Q_ref)
        pts, wts = numpy.asarray(Qf.get_points()), numpy.asarray(Qf.get_weights())
        lam = ref_el.compute_barycentric_coordinates(pts)
        fverts = list(top[sd-1][f])
        bubble = numpy.prod(lam[:, fverts], axis=1)
        quadratics = [lam[:, a] * lam[:, b] for i, a in enumerate(fverts) for b in fverts[i:]]
        xf = numpy.mean(verts[fverts], axis=0)
        U_at_pts = U.tabulate(pts)[z]
        for p in quadratics:
            q = (pts - xf) * (bubble * p)[:, None]
            q -= numpy.dot(wts, q) / sum(wts)
            rows.append(pad(numpy.einsum("nkq,qk,q->n", U_at_pts, q, wts)[None, :]))

    # Koszul gauge in the interior against the C1 bubbles on the Alfeld split
    C1 = CkPolynomialSet(ref_complex, degree+1, order=1, variant="bubble")
    bdry_pts = numpy.concatenate([FacetQuadratureRule(ref_el, sd-1, f, Q_ref).get_points()
                                  for f in sorted(top[sd-1])])
    bubbles = polynomial_set.spanning_basis(C1.tabulate(bdry_pts)[z].T, nullspace=True)
    xK = numpy.mean(verts, axis=0)
    for b in numpy.dot(bubbles, C1.tabulate(qpts)[z]):
        q = (qpts - xK) * b[:, None]
        q -= numpy.dot(qwts, q) / sum(qwts)
        rows.append(pad(numpy.einsum("nkq,qk,q->n", U_at_qpts, q, qwts)[None, :]))

    A = scale_rows(numpy.concatenate(rows))
    kernel = polynomial_set.spanning_basis(A, nullspace=True)
    expected_dim = (num_gn - 1) + sd
    if len(kernel) != expected_dim:
        raise RuntimeError(f"Expected a space of dimension {expected_dim}, got {len(kernel)}")

    coeffs = numpy.tensordot(kernel[:, :num_u], U.get_coeffs(), axes=(1, 0))
    return polynomial_set.PolynomialSet(ref_complex, degree, degree,
                                        U.get_expansion_set(), coeffs)


def curl_of(ell):
    """Return the functional u -> ell(curl u) from a functional on vector fields."""
    sd = ell.ref_el.get_spatial_dimension()
    if len(ell.deriv_dict) > 0:
        raise ValueError("Expecting a functional without derivatives")
    alphas = [tuple(row) for row in numpy.eye(sd, dtype=int)]
    deriv_dict = {pt: [(wt * EPS[i, j, k], alphas[j], (k,))
                       for wt, (i,) in entries
                       for j in range(sd) for k in range(sd) if EPS[i, j, k]]
                  for pt, entries in ell.pt_dict.items()}
    return Functional(ell.ref_el, (sd,), {}, deriv_dict, f"CurlOf{ell.functional_type}")


class HuZhangZhangDualSet(dual_set.DualSet):
    """The extended Hu-Zhang-Zhang dual set.

    The vertex values and tangential face constraints are the Guzman-Neilan
    functionals composed with the curl, so this dual set inherits the
    Guzman-Neilan transformation theory.  The normal face moments of the curl
    are omitted, since by Stokes' theorem they are sums of the edge moments.
    """
    def __init__(self, ref_el, quad_scheme=None):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        GN = GuzmanNeilanFirstKindH1(ref_el, 1, quad_scheme=quad_scheme)
        gn_nodes = GN.dual_basis()
        gn_ids = GN.entity_dofs()
        NED = Nedelec(ref_el, 1, variant="integral")
        ned_nodes = NED.dual_basis()
        ned_ids = NED.entity_dofs()

        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []
        # Vertex values of the curl
        for v in sorted(top[0]):
            cur = len(nodes)
            nodes.extend(curl_of(gn_nodes[i]) for i in gn_ids[0][v])
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # Tangential edge moments
        for e in sorted(top[1]):
            cur = len(nodes)
            nodes.extend(ned_nodes[i] for i in ned_ids[1][e])
            entity_ids[1][e].extend(range(cur, len(nodes)))

        # Tangential face moments of the curl (constraints)
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            nodes.extend(curl_of(gn_nodes[i]) for i in gn_ids[sd-1][f][1:])
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HuZhangZhang(finite_element.CiarletElement):
    """The Hu-Zhang-Zhang H(grad curl)-conforming (extended) macroelement.

    Reference element: a tetrahedron.
    Function space: C0 piecewise quartics on the Alfeld split whose curl is in
                    Guzman-Neilan, with the curl-free part fixed by a Koszul gauge.
    Degrees of freedom: the curl at the vertices, and tangential moments on edges.

    This element belongs to the Stokes complex CG1 -> HZZ -> GN -> DG0.
    """
    def __init__(self, ref_el, degree=1, quad_scheme=None):
        if degree != 1:
            raise NotImplementedError("Only the lowest-order Hu-Zhang-Zhang element is implemented")
        poly_set = HuZhangZhangSpace(ref_el)
        dual = HuZhangZhangDualSet(ref_el, quad_scheme=quad_scheme)
        formdegree = 1
        super().__init__(poly_set, dual, poly_set.get_degree(), formdegree,
                         mapping="covariant piola")
