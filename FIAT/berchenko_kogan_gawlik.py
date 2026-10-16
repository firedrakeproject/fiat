# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Lowest-order blow-up Whitney forms on a triangle.

Berchenko-Kogan and Gawlik, Blow-up Whitney forms, shadow forms, and
Poisson processes, arXiv:2402.03198.
"""

import itertools

import numpy

from FIAT import dual_set, finite_element, polynomial_set
from FIAT.functional import FrobeniusIntegralMoment, IntegralMoment
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature, rational_quadrature
from FIAT.rational import (RationalExpansionSet, barycentric_gradients, rational_derivative, rational_evaluate,
                           rational_multiply, rational_union)


# Rotation from the covariant to the contravariant proxy of a 1-form,
# u = (u_x, u_y) -> (u_y, -u_x), so that div of the rotated proxy is rot u
ROTATION = numpy.array([[0, 1], [-1, 0]])


def monomial(alpha=(0, 0, 0), beta=(0, 0, 0)):
    """Return the exponents of the rational monomial lambda^alpha / (1-lambda)^beta."""
    return numpy.concatenate((alpha, beta))


def BerchenkoKoganGawlikSpace(ref_el, k, degree=1, rotated=False):
    """Return a basis for the lowest-order blow-up Whitney k-forms on a triangle.

    The basis is ordered as the dual basis of :class:`BerchenkoKoganGawlikDualSet`:

    * k=0: psi_ijk = lambda_i lambda_j / (1 - lambda_i), for each edge (i, j)
      and each vertex i of that edge.
    * k=1: the Whitney forms phi_ij = lambda_i dlambda_j - lambda_j dlambda_i
      for each edge (i, j), followed by the arc forms
      psi_i{jk} = lambda_i phi_jk (1/(1 - lambda_i) + 1/(1 - lambda_i)^2), j < k,
      for each vertex i.
    * k=2: the constant form.

    :arg ref_el: a triangle.
    :arg k: the form degree.
    :kwarg degree: the polynomial degree, only degree 1 is implemented.
    :kwarg rotated: whether to rotate the proxy of the 1-forms into H(div).

    :returns: a PolynomialSet basis over a :class:`RationalExpansionSet`.
    """
    if degree != 1:
        raise ValueError("Blow-up Whitney forms are only implemented for degree 1")
    if rotated and k != 1:
        raise ValueError("Only the blow-up Whitney 1-forms can be rotated")
    sd = ref_el.get_spatial_dimension()
    if sd != 2:
        raise ValueError("Blow-up Whitney forms are only implemented on triangles")
    top = ref_el.get_topology()
    e = numpy.eye(sd+1, dtype=int)

    if k == 0:
        exponents = [monomial(e[i] + e[j], e[i])
                     for edge in sorted(top[1])
                     for i, j in itertools.permutations(top[1][edge])]
        coeffs = numpy.eye(len(exponents))

    elif k == 1:
        linear = [monomial(e[i]) for i in range(sd+1)]
        arcs = [monomial(e[i] + e[j], p * e[i])
                for i in range(sd+1) for j in range(sd+1) if j != i for p in (1, 2)]
        exponents, (ilinear, iarcs) = rational_union(linear, arcs)
        iarcs = iarcs.reshape(sd+1, sd, 2)

        grad_lambda = barycentric_gradients(ref_el)
        coeffs = numpy.zeros((len(top[1]) + len(top[0]), sd, len(exponents)))
        for edge in sorted(top[1]):
            i, j = top[1][edge]
            coeffs[edge, :, ilinear[i]] += grad_lambda[j]
            coeffs[edge, :, ilinear[j]] -= grad_lambda[i]
        for i in sorted(top[0]):
            j, l = (m for m in range(sd+1) if m != i)
            row = len(top[1]) + i
            jj, ll = (m - (m > i) for m in (j, l))
            for p in range(2):
                coeffs[row, :, iarcs[i, jj, p]] += grad_lambda[l]
                coeffs[row, :, iarcs[i, ll, p]] -= grad_lambda[j]

    elif k == 2:
        exponents = [monomial()]
        coeffs = numpy.eye(1)

    else:
        raise ValueError(f"Invalid form degree {k}")

    if rotated:
        coeffs = numpy.einsum("ij,njm->nim", ROTATION, coeffs)
    expansion_set = RationalExpansionSet(ref_el, exponents)
    return polynomial_set.PolynomialSet(ref_el, 1, 1, expansion_set, coeffs)


class BerchenkoKoganGawlikDualSet(dual_set.DualSet):
    """The degrees of freedom of the lowest-order blow-up Whitney k-forms.

    * k=0: the limit at vertex i of the trace on edge (i, j). The trace on
      each edge is linear, so this is a moment against the dual P1 basis on
      the edge.
    * k=1: the tangential moment on each edge, followed by the integral over
      the infinitesimal arc at each vertex i, oriented from j to k with j < k.
      The cubic cutoff chi_i = lambda_i (3 lambda_i - 2) + 10 lambda_0 lambda_1 lambda_2
      equals one at vertex i, vanishes at the other vertices, and has zero mean
      on the triangle and on each edge. On this space, du is constant and the
      tangential traces are constant on each edge, so Stokes' theorem on the
      blown-up triangle gives the arc integral as the moment of u against the
      L2 Riesz representer d chi_i.
    * k=2: the integral over the triangle.

    The edge degrees of freedom of the 0- and 1-forms determine the trace, and
    the arc degrees of freedom are interior to the cell.

    For the rotated 1-forms, the weights are rotated as the proxy, so that the
    tangential moments become normal moments.

    :arg ref_el: a triangle.
    :arg k: the form degree.
    :kwarg degree: the polynomial degree, only degree 1 is implemented.
    :kwarg rotated: whether to rotate the proxy of the 1-forms into H(div).
    """
    def __init__(self, ref_el, k, degree=1, rotated=False):
        if degree != 1:
            raise ValueError("Blow-up Whitney forms are only implemented for degree 1")
        if rotated and k != 1:
            raise ValueError("Only the blow-up Whitney 1-forms can be rotated")
        rotation = ROTATION if rotated else numpy.eye(2)
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        facet = ref_el.construct_subelement(1)
        Q_facet = create_quadrature(facet, 2)

        nodes = []
        if k == 0:
            # Dual basis of P1 on the reference edge
            x = Q_facet.get_points()[:, 0]
            phis = numpy.array([1 - x, x])
            gram = numpy.dot(phis * Q_facet.get_weights(), phis.T)
            duals = numpy.linalg.solve(gram, phis)
            for edge in sorted(top[1]):
                Q = FacetQuadratureRule(ref_el, 1, edge, Q_facet, avg=True)
                cur = len(nodes)
                nodes.extend(IntegralMoment(ref_el, Q, phi) for phi in duals)
                entity_ids[1][edge].extend(range(cur, len(nodes)))

        elif k == 1:
            for edge in sorted(top[1]):
                Q = FacetQuadratureRule(ref_el, 1, edge, Q_facet, avg=True)
                t = ref_el.compute_edge_tangent(edge)
                phi = numpy.outer(t, numpy.ones(Q.get_weights().shape))
                nodes.append(FrobeniusIntegralMoment(ref_el, Q, numpy.dot(rotation, phi)))
                entity_ids[1][edge].append(len(nodes) - 1)

            # The L2 Riesz representers q_i = sign_i curl chi_i of the arc integrals
            e = numpy.eye(sd+1, dtype=int)
            verts = numpy.asarray(ref_el.get_vertices())
            chi_exponents = [monomial(2*e[i]) for i in sorted(top[0])]
            chi_exponents.extend(monomial(e[i]) for i in sorted(top[0]))
            chi_exponents.append(monomial(numpy.ones(sd+1, dtype=int)))
            chi_coeffs = numpy.zeros((len(top[0]), len(chi_exponents)))
            for i in sorted(top[0]):
                chi_coeffs[i, [i, len(top[0]) + i, -1]] = (3, -2, 10)
            grad_lambda = barycentric_gradients(ref_el)
            dx, dy = (rational_derivative(chi_exponents, chi_coeffs, grad_lambda[:, c]) for c in range(sd))
            q_exponents, (idx, idy) = rational_union(dx[0], dy[0])
            q_coeffs = numpy.zeros((len(top[0]), sd, len(q_exponents)))
            q_coeffs[:, 0, idy] = dy[1]
            q_coeffs[:, 1, idx] = -dx[1]
            for i in sorted(top[0]):
                j, l = (m for m in range(sd+1) if m != i)
                q_coeffs[i] *= numpy.sign(numpy.linalg.det([verts[j] - verts[i], verts[l] - verts[i]]))

            # Quadrature that integrates q_i . u exactly for u in the space
            space = BerchenkoKoganGawlikSpace(ref_el, k, degree=degree)
            products = [rational_multiply(q_exponents, q_coeffs[:, c], space.get_expansion_set().exponents,
                                          space.get_coeffs()[:, c]) for c in range(sd)]
            exponents, indices = rational_union(*(E for E, _ in products))
            coeffs = numpy.zeros((len(top[0]), len(space), len(exponents)))
            for (_, C), index in zip(products, indices):
                coeffs[..., index] += C
            Q = rational_quadrature(ref_el, exponents, coeffs)

            bary = ref_el.compute_barycentric_coordinates(Q.get_points())
            q_at_qpts = numpy.dot(q_coeffs, rational_evaluate(q_exponents, bary))
            for i in sorted(top[0]):
                nodes.append(FrobeniusIntegralMoment(ref_el, Q, numpy.dot(rotation, q_at_qpts[i])))
                entity_ids[sd][0].append(len(nodes) - 1)

        elif k == 2:
            Q = create_quadrature(ref_el, 0)
            nodes.append(IntegralMoment(ref_el, Q, numpy.ones(Q.get_weights().shape)))
            entity_ids[sd][0].append(0)

        else:
            raise ValueError(f"Invalid form degree {k}")
        super().__init__(nodes, ref_el, entity_ids)


class BerchenkoKoganGawlik(finite_element.CiarletElement):
    """The lowest-order blow-up Whitney k-forms on a triangle.

    The 0- and 1-forms are rational, contain the Lagrange P1 and Nedelec
    first kind spaces, and are singular at the vertices. Their nodal basis
    is the basis of (Berchenko-Kogan and Gawlik, 2024, Table 1).

    With rotated=True, the 1-forms are represented by their H(div) proxy.

    :arg ref_el: a triangle.
    :arg k: the form degree.
    :kwarg degree: the polynomial degree, only degree 1 is implemented.
    :kwarg rotated: whether to rotate the proxy of the 1-forms into H(div).
    """
    def __init__(self, ref_el, k, degree=1, rotated=False):
        poly_set = BerchenkoKoganGawlikSpace(ref_el, k, degree=degree, rotated=rotated)
        dual = BerchenkoKoganGawlikDualSet(ref_el, k, degree=degree, rotated=rotated)
        if k == 1:
            mapping = "contravariant piola" if rotated else "covariant piola"
        else:
            mapping = "affine"
        super().__init__(poly_set, dual, degree, formdegree=k, mapping=mapping)
