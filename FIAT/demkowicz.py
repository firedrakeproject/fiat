# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import scipy

from FIAT.dual_set import DualSet
from FIAT.functional import PointEvaluation, FrobeniusIntegralMoment
from FIAT.polynomial_set import ONPolynomialSet, make_bubbles
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import symmetric_simplex
from FIAT.nedelec import Nedelec
from FIAT.raviart_thomas import RaviartThomas


def grad(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    grad_u = [grad_u[k] for k in sorted(grad_u)]
    return numpy.transpose(grad_u, (1, 0, 2))


def curl(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    d = len(grad_u)
    indices = ((i, j) for i in reversed(range(d)) for j in reversed(range(i+1, d)))
    curl_u = [((-1)**k) * (grad_u[j][:, i, :] - grad_u[i][:, j, :]) for k, (i, j) in enumerate(indices)]
    return numpy.transpose(curl_u, (1, 0, 2))


def div(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    div_u = sum(grad_u[i][:, i, :] for i in grad_u)
    return div_u


def inner(v, u, Qwts):
    return numpy.tensordot(numpy.multiply(v, Qwts), u, axes=(range(1, v.ndim), range(1, u.ndim)))


def map_duals(ref_el, dim, entity, mapping, Q_ref, Phis):
    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
    if mapping == "normal":
        n = ref_el.compute_normal(entity)
        phis = n[None, :, None] * Phis[:, None, :]
    elif mapping == "covariant":
        piola_map = numpy.linalg.pinv(Q.jacobian().T)
        phis = numpy.dot(piola_map, Phis).transpose((1, 0, 2))
    elif mapping == "contravariant":
        piola_map = Q.jacobian() / Q.jacobian_determinant()
        phis = numpy.dot(piola_map, Phis).transpose((1, 0, 2))
    else:
        Jdet = Q.jacobian_determinant()
        phis = (1 / Jdet) * Phis
    return Q, phis


class DemkowiczDual(DualSet):

    def __init__(self, ref_el, degree, sobolev_space):
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        formdegree = {"H1": 0, "HCurl": 1, "HDiv": sd-1, "L2": sd}[sobolev_space]
        trace = {"HCurl": "contravariant", "HDiv": "normal"}.get(sobolev_space, None)
        dual_mapping = {"HCurl": "contravariant", "HDiv": "covariant"}.get(sobolev_space, None)

        for dim in sorted(top):
            entity_ids[dim] = {}
            if dim < formdegree or degree <= dim - formdegree:
                for entity in top[dim]:
                    entity_ids[dim][entity] = []
            elif dim == 0 and formdegree == 0:
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(PointEvaluation(ref_el, pt) for pt in pts)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
            else:
                Q_ref, Phis = self._reference_duals(dim, degree, formdegree, sobolev_space)
                mapping = dual_mapping if dim == sd else trace
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super(DemkowiczDual, self).__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, dim, degree, formdegree, sobolev_space):
        facet = symmetric_simplex(dim)
        Q = create_quadrature(facet, 2 * degree)
        if formdegree == dim:
            shp = (dim,) if sobolev_space == "HCurl" else ()
            P = ONPolynomialSet(facet, degree, shp)
            duals = P.tabulate(Q.get_points())[(0,) * dim]
            return Q, duals

        exterior_derivative = {"H1": grad, "HCurl": curl, "HDiv": div}[sobolev_space]
        Qpts, Qwts = Q.get_points(), Q.get_weights()
        shp = () if formdegree == 0 else (dim,)
        P = ONPolynomialSet(facet, degree, shp)
        P_at_qpts = P.tabulate(Qpts, 1)
        dtrial = exterior_derivative(P_at_qpts)
        K = self._bubble_derivative_moments(facet, degree, formdegree, Qpts, Qwts, dtrial)
        if formdegree > 0:
            trial = P_at_qpts[(0,) * dim]
            if formdegree == 1 and sobolev_space == "HDiv":
                rot = numpy.array([[0.0, 1.0], [-1.0, 0.0]], "d")
                trial = numpy.dot(rot, trial).transpose((1, 0, 2))
            M = self._bubble_derivative_moments(facet, degree+1, formdegree-1, Qpts, Qwts, trial)
            K = numpy.vstack((K, M))

        duals = numpy.tensordot(K, P_at_qpts[(0,) * dim], axes=(1, 0))
        return Q, duals

    def _bubble_derivative_moments(self, facet, degree, formdegree, Qpts, Qwts, trial):
        """Integrate trial expressions against an orthonormal basis for
           the exterior derivative of bubbles.
        """
        dim = facet.get_spatial_dimension()
        if formdegree >= dim - 1:
            # We are at the end of the complex
            # derivative of bubbles is P_k-1 minus constants
            Pkm1 = ONPolynomialSet(facet, degree-1, trial.shape[1:-1])
            P0 = Pkm1.take(list(range(1, Pkm1.get_num_members())))
            dtest = P0.tabulate(Qpts)[(0,) * dim]
            return inner(dtest, trial, Qwts)

        # Get bubbles
        element = (None, Nedelec, RaviartThomas)[formdegree]
        if element is None:
            B = make_bubbles(facet, degree)
        else:
            fe = element(facet, degree)
            B = fe.get_nodal_basis().take(fe.entity_dofs()[dim][0])
        # Tabulate the exterior derivate
        d = (grad, curl, div)[formdegree]
        dtest = d(B.tabulate(Qpts, 1))
        # Build an orthonormal basis, remove nullspace
        A = inner(dtest, dtest, Qwts)
        sig, S = scipy.linalg.eigh(A)
        nullspace_dim = len([s for s in sig if abs(s) <= 1.e-10])
        S = S[:, nullspace_dim:]
        S *= numpy.sqrt(1 / sig[None, nullspace_dim:])
        # Apply change of basis
        dtest = numpy.tensordot(S.T, dtest, axes=(1, 0))
        return inner(dtest, trial, Qwts)


class FDMDual(DualSet):

    def __init__(self, ref_el, degree, sobolev_space, element):
        nodes = []
        entity_ids = {}
        sd = ref_el.get_spatial_dimension()

        Ref_el = symmetric_simplex(sd)
        fe = element(Ref_el, degree, variant="demkowicz")
        ells = fe.dual_basis()
        entity_dofs = fe.entity_dofs()
        formdegree = fe.formdegree

        Q = create_quadrature(Ref_el, 2 * degree)
        X, W = Q.get_points(), Q.get_weights()
        exterior_derivative = {"H1": grad, "HCurl": curl, "HDiv": div}[sobolev_space]
        trace = {"HCurl": "contravariant", "HDiv": "normal"}.get(sobolev_space, None)
        dual_mapping = {"HCurl": "contravariant", "HDiv": "covariant"}.get(sobolev_space, None)

        phi_at_qpts = fe.tabulate(1, X)
        V0 = phi_at_qpts[(0,) * sd]
        V1 = exterior_derivative(phi_at_qpts)

        for dim in sorted(entity_dofs):
            entity_ids[dim] = {}
            if dim == 0 and formdegree == 0:
                for entity in sorted(entity_dofs[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(PointEvaluation(ref_el, pt) for pt in pts)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                continue
            dofs = entity_dofs[dim][0]
            if len(dofs) == 0:
                for entity in sorted(entity_dofs[dim]):
                    entity_ids[dim][entity] = []
                continue

            B = inner(V0[dofs], V0[dofs], W)
            if dim == sd:
                _, S = scipy.linalg.eigh(B)
            else:
                A = inner(V1[dofs], V1[dofs], W)
                if formdegree > 0:
                    A += B
                _, S = scipy.linalg.eigh(B, A)
                S = numpy.dot(A, S)

            phis = numpy.array([ells[i].f_at_qpts for i in dofs])
            phis = numpy.tensordot(S.T, phis, axes=(1, 0))

            Q_dof = ells[dofs[0]].Q
            Q_ref = Q_dof.reference_rule()
            mapping = dual_mapping if dim == sd else trace
            # map physical phis to reference values Phis
            if mapping == "normal":
                n = Ref_el.compute_normal(0)
                Phis, = numpy.dot(n[None, :], phis)
            elif mapping == "covariant":
                piola_map = Q_dof.jacobian().T
                Phis = numpy.dot(piola_map, phis).transpose((1, 0, 2))
            elif mapping == "contravariant":
                piola_map = numpy.linalg.pinv(Q_dof.jacobian()) * Q_dof.jacobian_determinant()
                Phis = numpy.dot(piola_map, phis).transpose((1, 0, 2))
            else:
                Jdet = Q_dof.jacobian_determinant()
                Phis = Jdet * phis

            for entity in sorted(entity_dofs[dim]):
                cur = len(nodes)
                Q_facet, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                nodes.extend(FrobeniusIntegralMoment(ref_el, Q_facet, phi) for phi in phis)
                entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super(FDMDual, self).__init__(nodes, ref_el, entity_ids)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from FIAT import IntegratedLegendre as CG
    from FIAT import NedelecSecondKind as N2Curl
    from FIAT import BrezziDouglasMarini as N2Div

    dim = 3
    degree = 7
    ref_el = symmetric_simplex(dim)
    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    variant = "fdm"
    # variant = "demkowicz"
    # variant = None
    space_dict = {"H1": (CG, grad),
                  "HCurl": (N2Curl, curl),
                  "HDiv": (N2Div, div),
                  }
    spaces = list(space_dict.keys())

    fig, axes = plt.subplots(ncols=len(spaces), nrows=2, figsize=(6*len(spaces), 12))
    axes = axes.T.flat
    for space in spaces:
        element, d = space_dict[space]
        fe = element(ref_el, degree, variant)
        phi_at_qpts = fe.tabulate(1, Qpts)
        V0 = phi_at_qpts[(0,) * dim]
        V1 = d(phi_at_qpts)
        mass = inner(V0, V0, Qwts)
        stiff = inner(V1, V1, Qwts)

        mats = (stiff, mass)
        title = f"{type(fe).__name__}({degree})"
        names = (f"{title} stiff", f"{title} mass")
        for name, A in zip(names, mats):
            A[abs(A) < 1E-10] = 0.0
            # print(A.diagonal())
            nnz = numpy.count_nonzero(A)
            ax = next(axes)
            ax.spy(A, markersize=0)
            ax.pcolor(numpy.log(abs(A)))
            ax.set_title(f"{name} nnz {nnz}")
    plt.show()
