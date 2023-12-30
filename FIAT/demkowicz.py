# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (polynomial_set, dual_set,
                  finite_element, functional)
import numpy
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.nedelec import Nedelec
from FIAT.raviart_thomas import RaviartThomas


def as_table(P):
    PT = numpy.transpose(P, (1, 0, 2))
    P_table = dict(zip(polynomial_set.mis(len(PT), 1), PT))
    return P_table


def grad(table_u):
    return table_u


def curl(table_u):
    grad_u = [None for alpha in table_u if sum(alpha) == 1]
    for alpha in table_u:
        if sum(alpha) == 1:
            grad_u[alpha.index(1)] = table_u[alpha]

    nbfs, *_, npts = grad_u[0].shape
    d = len(grad_u)
    ncomp = (d * (d - 1)) // 2
    indices = ((i, j) for i in range(d) for j in range(i+1, d))
    curl_u = numpy.empty((nbfs, ncomp, npts), "d")
    for k, (i, j) in enumerate(indices):
        s = (-1)**k
        curl_u[:, k, :] = s*grad_u[i][:, j, :] - s*grad_u[j][:, i, :]
    return as_table(curl_u)


def div(table_u):
    grad_u = [None for alpha in table_u if sum(alpha) == 1]
    for alpha in table_u:
        if sum(alpha) == 1:
            grad_u[alpha.index(1)] = table_u[alpha]
    nbfs, *_, npts = grad_u[0].shape
    div_u = numpy.empty((nbfs, 1, npts), "d")
    div_u[:, 0, :] = sum(grad_u[i][:, i, :] for i in range(len(grad_u)))
    return as_table(div_u)


class DemkowiczDual(dual_set.DualSet):

    def __init__(self, ref_el, degree, sobolev_space):
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        self.formdegree = 1 if sobolev_space == "HCurl" else sd - 1

        for dim in top:
            entity_ids[dim] = {}
            if dim < self.formdegree or degree < dim:
                for entity in top[dim]:
                    entity_ids[dim][entity] = []
            else:
                Q_ref, Phis = self._reference_duals(ref_el, dim, degree, sobolev_space)
                if dim == sd:
                    trace = None
                else:
                    trace = "tangential" if sobolev_space == "HCurl" else "normal"
                if trace == "tangential":
                    Phis = numpy.transpose(Phis, (0, 2, 1))

                for entity in top[dim]:
                    cur = len(nodes)
                    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                    if trace == "normal":
                        Jdet = Q.jacobian_determinant()
                        n = ref_el.compute_scaled_normal(entity) / Jdet
                        phis = n[None, :, None] * Phis
                    elif trace == "tangential":
                        J = Q.jacobian()
                        piola_map = numpy.linalg.pinv(J.T)
                        phis = numpy.dot(Phis, piola_map.T)
                        phis = numpy.transpose(phis, (0, 2, 1))
                    else:
                        phis = Phis
                    nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q, phi)
                                 for phi in phis)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super(DemkowiczDual, self).__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, ref_el, dim, degree, sobolev_space):
        facet = ref_el.construct_subelement(dim)
        Q = create_quadrature(facet, 2 * degree)
        if dim == self.formdegree:
            P = polynomial_set.ONPolynomialSet(facet, degree, (1,))
            duals = P.tabulate(Q.get_points())[(0,) * dim]
            return Q, duals

        Qpts, Qwts = Q.get_points(), Q.get_weights()
        inner = lambda v, u: numpy.dot(numpy.multiply(v, Qwts), u.T)
        galerkin = lambda order, V, U: sum(inner(V[k], U[k]) for k in V if sum(k) == order)
        P = polynomial_set.ONPolynomialSet(facet, degree, (dim,))
        P_table = P.tabulate(Qpts, 1)

        if self.formdegree == 1:
            B = polynomial_set.make_bubbles(facet, degree+1)
            grad_basis = B.tabulate(Qpts, 1)

            if sobolev_space == "HCurl":
                curl_coeffs, curl_basis = self.exterior_derivative_bubbles(facet, degree, 1, Qpts, galerkin)
                K1 = numpy.dot(curl_coeffs, galerkin(1, curl_basis, curl(P_table)))

            elif sobolev_space == "HDiv":
                # Swap grad -> rot, and curl -> div
                grad_basis[(1, 0)], grad_basis[(0, 1)] = grad_basis[(0, 1)], -grad_basis[(1, 0)]
                div_coeffs, div_basis = self.exterior_derivative_bubbles(facet, degree, 2, Qpts, galerkin)
                K1 = numpy.dot(div_coeffs, galerkin(1, div_basis, div(P_table)))
            else:
                raise ValueError("Invalid Sobolev space")

            K0 = galerkin(1, grad_basis, as_table(P_table[(0,) * dim]))
            K = numpy.vstack((K0, K1))

        elif self.formdegree == 2:
            curl_coeffs, curl_basis = self.exterior_derivative_bubbles(facet, degree+1, 1, Qpts, galerkin)
            K1 = numpy.dot(curl_coeffs, galerkin(1, curl_basis, as_table(P_table[(0,) * dim])))

            div_coeffs, div_basis = self.exterior_derivative_bubbles(facet, degree, 2, Qpts, galerkin)
            K2 = numpy.dot(div_coeffs, galerkin(1, div_basis, div(P_table)))
            K = numpy.vstack((K1, K2))
        else:
            raise ValueError("Invalid form degree")

        duals = P_table[(0,) * dim]
        shp = (-1, ) + duals.shape[1:]
        duals = numpy.dot(K, duals.reshape((K.shape[1], -1))).reshape(shp)
        return Q, duals

    def exterior_derivative_bubbles(self, facet, degree, formdegree, Qpts, galerkin):
        dim = facet.get_spatial_dimension()
        comp = (dim, dim*(dim-1)//2, 1)[formdegree]
        Pkm1 = polynomial_set.ONPolynomialSet(facet, degree-1, (comp,))
        basis = as_table(Pkm1.tabulate(Qpts)[(0,) * dim])

        family = (None, Nedelec, RaviartThomas)[formdegree]
        d = (grad, curl, div)[formdegree]

        N1 = family(facet, degree)
        B = N1.get_nodal_basis().take(N1.entity_dofs()[dim][0])
        dB = d(B.tabulate(Qpts, 1))

        new_coeffs = galerkin(1, dB, basis)
        U, S, VT = numpy.linalg.svd(new_coeffs)
        num_sv = len([s for s in S if s > 1E-10])
        coeffs = VT[:num_sv]
        return coeffs, basis


class N2Curl(finite_element.CiarletElement):

    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        dual = DemkowiczDual(ref_el, degree, "HCurl")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, (sd,))
        super(N2Curl, self).__init__(poly_set, dual, degree, dual.formdegree,
                                     mapping="covariant piola")


class N2Div(finite_element.CiarletElement):

    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        dual = DemkowiczDual(ref_el, degree, "HDiv")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, (sd,))
        super(N2Div, self).__init__(poly_set, dual, degree, dual.formdegree,
                                    mapping="contravariant piola")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from FIAT.reference_element import symmetric_simplex
    dim = 3
    degree = 7
    ref_el = symmetric_simplex(dim)
    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()
    inner = lambda v, u: numpy.dot(numpy.multiply(v, Qwts), u.T)
    galerkin = lambda order, V, U: sum(inner(V[k], U[k]) for k in V if sum(k) == order)

    d, fe = curl, N2Curl(ref_el, degree)
    # d, fe = div, N2Div(ref_el, degree)

    phi = fe.tabulate(1, Qpts)
    dphi = d(phi)
    A = galerkin(1, dphi, dphi)
    A[abs(A) < 1E-10] = 0.0

    phi_table = as_table(phi[(0,) * dim])
    B = galerkin(1, phi_table, phi_table)
    B[abs(B) < 1E-10] = 0.0

    # A = B
    plt.spy(A, markersize=0)
    plt.pcolor(numpy.log(abs(A)))
    plt.show()
