# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (polynomial_set, dual_set,
                  finite_element, functional)
import numpy
import scipy
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.nedelec import Nedelec
from FIAT.raviart_thomas import RaviartThomas
from FIAT.reference_element import symmetric_simplex


def as_table(P):
    if len(P.shape) == 3:
        PT = numpy.transpose(P, (1, 0, 2))
    else:
        PT = P[None, :, :]
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
    indices = ((i, j) for i in reversed(range(d)) for j in reversed(range(i+1, d)))
    curl_u = numpy.empty((nbfs, ncomp, npts), grad_u[0].dtype)
    for k, (i, j) in enumerate(indices):
        curl_u[:, k, :] = ((-1)**k)*(grad_u[j][:, i, :] - grad_u[i][:, j, :])
    return as_table(curl_u)


def div(table_u):
    grad_u = [None for alpha in table_u if sum(alpha) == 1]
    for alpha in table_u:
        if sum(alpha) == 1:
            grad_u[alpha.index(1)] = table_u[alpha]

    div_u = sum(grad_u[i][:, i, :] for i in range(len(grad_u)))
    return as_table(div_u)


class DemkowiczDual(dual_set.DualSet):

    def __init__(self, ref_el, degree, sobolev_space):
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        self.formdegree = {"H1": 0, "HCurl": 1, "HDiv": sd-1, "L2": sd}[sobolev_space]

        for dim in sorted(top):
            entity_ids[dim] = {}
            if dim < self.formdegree or degree < dim:
                for entity in top[dim]:
                    entity_ids[dim][entity] = []
            elif dim == 0 and self.formdegree == 0:
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(functional.PointEvaluation(ref_el, pt) for pt in pts)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
            else:
                Q_ref, Phis = self._reference_duals(dim, degree, sobolev_space)
                if dim == sd or self.formdegree == 0:
                    trace = None
                else:
                    trace = "tangential" if sobolev_space == "HCurl" else "normal"
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                    if trace == "normal":
                        Jdet = Q.jacobian_determinant()
                        n = ref_el.compute_scaled_normal(entity) / Jdet
                        phis = n[None, :, None] * Phis
                    elif trace == "tangential":
                        J = Q.jacobian()
                        piola_map = numpy.linalg.pinv(J.T)
                        phis = numpy.dot(piola_map, Phis).transpose((1, 0, 2))
                    else:
                        Jdet = Q.jacobian_determinant()
                        phis = Phis / Jdet
                    nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q, phi)
                                 for phi in phis)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super(DemkowiczDual, self).__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, dim, degree, sobolev_space):
        facet = symmetric_simplex(dim)
        Q = create_quadrature(facet, 2 * degree)
        if dim == self.formdegree:
            P = polynomial_set.ONPolynomialSet(facet, degree, (1,))
            duals = P.tabulate(Q.get_points())[(0,) * dim]
            return Q, duals

        Qpts, Qwts = Q.get_points(), Q.get_weights()
        fd = self.formdegree
        shp = () if fd == 0 else (dim,)
        P = polynomial_set.ONPolynomialSet(facet, degree, shp)
        P_table = P.tabulate(Qpts, 1)
        if sobolev_space == "H1":
            dtrial = grad(P_table)
            K = self._bubble_moments(facet, degree, fd, Qpts, Qwts, dtrial)
        else:
            rot = fd == 1 and sobolev_space == "HDiv"
            trial = as_table(P_table[(0,) * dim])
            dtrial = div(P_table) if sobolev_space == "HDiv" else curl(P_table)
            K0 = self._bubble_moments(facet, degree+1, fd-1, Qpts, Qwts, trial, rot=rot)
            K1 = self._bubble_moments(facet, degree, fd, Qpts, Qwts, dtrial)
            K = numpy.vstack((K0, K1))

        duals = P_table[(0,) * dim]
        shp = (-1, ) + duals.shape[1:]
        duals = numpy.dot(K, duals.reshape((K.shape[1], -1))).reshape(shp)
        return Q, duals

    def _bubble_moments(self, facet, degree, formdegree, Qpts, Qwts, trial, rot=False):
        inner = lambda v, u: numpy.dot(numpy.multiply(v, Qwts), u.T)
        galerkin = lambda V, U: sum(inner(V[k], U[k]) for k in V if sum(k) == 1)

        dim = facet.get_spatial_dimension()
        if formdegree >= dim - 1:
            Pkm1 = polynomial_set.ONPolynomialSet(facet, degree-1, (1,))
            P0 = Pkm1.take(list(range(1, Pkm1.get_num_members())))
            dtest = as_table(P0.tabulate(Qpts)[(0,) * dim])
            return galerkin(dtest, trial)

        element = (None, Nedelec, RaviartThomas)[formdegree]
        if element is None:
            B = polynomial_set.make_bubbles(facet, degree)
        else:
            fe = element(facet, degree)
            B = fe.get_nodal_basis().take(fe.entity_dofs()[dim][0])

        d = (grad, curl, div)[formdegree]
        dtest = d(B.tabulate(Qpts, 1))
        if rot:
            assert dim == 2
            dtest[(1, 0)], dtest[(0, 1)] = dtest[(0, 1)], -dtest[(1, 0)]

        A = galerkin(dtest, dtest)
        sig, S = scipy.linalg.eigh(A)
        nullspace_dim = len([s for s in sig if abs(s) <= 1.e-10])
        S = S[:, nullspace_dim:]
        S *= numpy.sqrt(1 / sig[None, nullspace_dim:])
        return numpy.dot(S.T, galerkin(dtest, trial))


class FDMDual(dual_set.DualSet):

    def __init__(self, ref_el, degree, sobolev_space):
        nodes = []
        entity_ids = {}
        element, d = {"H1": (CG, grad),
                      "HCurl": (N2Curl, curl),
                      "HDiv": (N2Div, div),
                      }[sobolev_space]

        sd = ref_el.get_spatial_dimension()
        Ref_el = symmetric_simplex(sd)
        fe = element(Ref_el, degree)
        ells = fe.dual_basis()
        entity_dofs = fe.entity_dofs()
        self.formdegree = fe.formdegree

        Q = create_quadrature(Ref_el, 2 * degree)
        X, W = Q.get_points(), Q.get_weights()
        inner = lambda v: numpy.dot(numpy.multiply(v, W), v.T)
        galerkin = lambda V, dofs: sum(inner(V[k][dofs]) for k in V if sum(k) == 1)

        V = fe.tabulate(1, X)
        V0 = as_table(V[(0,) * sd])
        V1 = d(V)

        for dim in sorted(entity_dofs):
            entity_ids[dim] = {}
            if dim == 0 and self.formdegree == 0:
                for entity in sorted(entity_dofs[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(functional.PointEvaluation(ref_el, pt) for pt in pts)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                continue
            dofs = entity_dofs[dim][0]
            if len(dofs) == 0:
                for entity in sorted(entity_dofs[dim]):
                    entity_ids[dim][entity] = []
                continue

            A = galerkin(V1, dofs)
            B = galerkin(V0, dofs)
            nullspace = [i for i, a in enumerate(A.diagonal()) if a < 1E-10]
            if len(nullspace) == 0:
                A, B = B, A
            _, S = scipy.linalg.eigh(A, B)
            Sinv = numpy.dot(S.T, B)

            Q_dof = ells[dofs[0]].Q
            Q_ref = Q_dof.reference_rule()
            if dim == sd or self.formdegree == 0:
                trace = None
            else:
                trace = "tangential" if sobolev_space == "HCurl" else "normal"

            # apply pushforward
            Phis = numpy.array([ells[i].f_at_qpts for i in dofs])
            Jdet = Q_dof.jacobian_determinant()
            if trace == "normal":
                n = Ref_el.compute_scaled_normal(0) / Jdet
                n *= 1 / numpy.dot(n, n)
                Phis, = numpy.dot(n[None, :], Phis)
            elif trace == "tangential":
                J = Q_dof.jacobian()
                Phis = numpy.dot(J.T, Phis).transpose((1, 0, 2))
            else:
                Phis *= Jdet

            Phis = numpy.dot(Sinv, Phis.reshape((Sinv.shape[0], -1))).reshape(Phis.shape)

            for entity in sorted(entity_dofs[dim]):
                cur = len(nodes)
                Q_facet = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                # apply pullback
                Jdet = Q_facet.jacobian_determinant()
                if trace == "normal":
                    n = ref_el.compute_scaled_normal(entity) / Jdet
                    phis = n[None, :, None] * Phis[:, None, :]
                elif trace == "tangential":
                    J = Q_facet.jacobian()
                    piola_map = numpy.linalg.pinv(J.T)
                    phis = numpy.dot(piola_map, Phis).transpose((1, 0, 2))
                else:
                    phis = (1 / Jdet) * Phis

                nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q_facet, phi)
                             for phi in reversed(phis))
                entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super(FDMDual, self).__init__(nodes, ref_el, entity_ids)


class CG(finite_element.CiarletElement):

    def __init__(self, ref_el, degree, variant=None):
        make_dual = FDMDual if variant == "fdm" else DemkowiczDual
        dual = make_dual(ref_el, degree, "H1")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        super(CG, self).__init__(poly_set, dual, degree, dual.formdegree,
                                 mapping="affine")


class N2Curl(finite_element.CiarletElement):

    def __init__(self, ref_el, degree, variant=None):
        sd = ref_el.get_spatial_dimension()
        make_dual = FDMDual if variant == "fdm" else DemkowiczDual
        dual = make_dual(ref_el, degree, "HCurl")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, (sd,))
        super(N2Curl, self).__init__(poly_set, dual, degree, dual.formdegree,
                                     mapping="covariant piola")


class N2Div(finite_element.CiarletElement):

    def __init__(self, ref_el, degree, variant=None):
        sd = ref_el.get_spatial_dimension()
        make_dual = FDMDual if variant == "fdm" else DemkowiczDual
        dual = make_dual(ref_el, degree, "HDiv")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree, (sd,))
        super(N2Div, self).__init__(poly_set, dual, degree, dual.formdegree,
                                    mapping="contravariant piola")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dim = 3
    degree = 7
    ref_el = symmetric_simplex(dim)
    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()
    inner = lambda v, u: numpy.dot(numpy.multiply(v, Qwts), u.T)
    galerkin = lambda order, V, U: sum(inner(V[k], U[k]) for k in V if sum(k) == order)

    variant = "fdm"
    # variant = None
    spaces = ("H1", "HCurl", "HDiv")

    space_dict = {"H1": (CG, grad),
                  "HCurl": (N2Curl, curl),
                  "HDiv": (N2Div, div),
                  }

    fig, axes = plt.subplots(ncols=len(spaces), nrows=2, figsize=(6*len(spaces), 12))
    axes = axes.T.flat
    for space in spaces:
        element, d = space_dict[space]
        fe = element(ref_el, degree, variant)
        phi = fe.tabulate(1, Qpts)
        dphi = d(phi)
        stiff = galerkin(1, dphi, dphi)

        phi_table = as_table(phi[(0,) * dim])
        mass = galerkin(1, phi_table, phi_table)

        mats = (stiff, mass)
        title = f"{type(fe).__name__}({degree})"
        names = (f"{title} stiff", f"{title} mass")
        for name, A in zip(names, mats):
            A[abs(A) < 1E-10] = 0.0
            nnz = numpy.count_nonzero(A)
            ax = next(axes)
            ax.spy(A, markersize=0)
            ax.pcolor(numpy.log(abs(A)))
            ax.set_title(f"{name} nnz {nnz}")
    plt.show()
