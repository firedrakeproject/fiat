# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import scipy

from FIAT.dual_set import DualSet
from FIAT.functional import PointEvaluation, FrobeniusIntegralMoment
from FIAT.polynomial_set import make_bubbles, ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import symmetric_simplex


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


def perp(u):
    u_perp = numpy.empty_like(u)
    u_perp[:, 0, :] = u[:, 1, :]
    u_perp[:, 1, :] = -u[:, 0, :]
    return u_perp


def map_duals(ref_el, dim, entity, mapping, Q_ref, Phis):
    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
    if mapping == "normal":
        n = ref_el.compute_normal(entity)
        phis = n[None, :, None] * Phis
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

    def __init__(self, ref_el, degree, sobolev_space, kind=None):
        nodes = []
        entity_ids = {}
        reduced_dofs = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        formdegree = {"H1": 0, "HCurl": 1, "HDiv": sd-1, "L2": sd}[sobolev_space]
        trace = {"HCurl": "contravariant", "HDiv": "normal"}.get(sobolev_space, None)
        dual_mapping = {"HCurl": "contravariant", "HDiv": "covariant"}.get(sobolev_space, None)
        if kind is None:
            kind = 1 if formdegree == 0 else 2

        for dim in sorted(top):
            entity_ids[dim] = {}
            if dim < formdegree or degree <= dim - formdegree:
                for entity in top[dim]:
                    entity_ids[dim][entity] = []
                reduced_dofs[dim] = 0
            elif dim == 0 and formdegree == 0:
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(PointEvaluation(ref_el, pt) for pt in pts)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                reduced_dofs[dim] = len(nodes)
            else:
                Q_ref, Phis, rdofs = self._reference_duals(dim, degree, formdegree, sobolev_space, kind)
                reduced_dofs[dim] = rdofs
                mapping = dual_mapping if dim == sd else trace
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        self._reduced_dofs = reduced_dofs
        super(DemkowiczDual, self).__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, dim, degree, formdegree, sobolev_space, kind):
        facet = symmetric_simplex(dim)
        Q = create_quadrature(facet, 2 * degree)
        Qpts, Qwts = Q.get_points(), Q.get_weights()
        exterior_derivative = {"H1": grad, "HCurl": curl, "HDiv": div, "L2": None}[sobolev_space]

        shp = () if formdegree == 0 else (dim,)
        if sobolev_space == "L2" and dim > 2:
            shp = ()
        elif formdegree == dim:
            shp = (1,)

        P = ONPolynomialSet(facet, degree, shp, scale="orthonormal")
        P_at_qpts = P.tabulate(Qpts, 1)
        trial = P_at_qpts[(0,) * dim]
        # Evaluate type-I degrees of freedom on P
        if formdegree >= dim:
            K = inner(trial[:1], trial, Qwts)
        else:
            dtrial = exterior_derivative(P_at_qpts)
            if dim == 2 and formdegree == 1 and sobolev_space == "HDiv":
                dtrial = dtrial[:, None, :]
            K = self._bubble_derivative_moments(facet, degree, formdegree, kind, Qpts, Qwts, dtrial)
        reduced_dofs = K.shape[0]

        # Evaluate type-II degrees of freedom on P
        if formdegree > 0:
            q = degree + 1 if kind == 2 else degree
            if q > degree:
                Q2 = create_quadrature(facet, 2 * q)
                Qpts, Qwts = Q2.get_points(), Q2.get_weights()
                trial = P.tabulate(Qpts, 0)[(0,) * dim]

            if dim == 2 and formdegree == 1 and sobolev_space == "HDiv":
                trial = perp(trial)
            M = self._bubble_derivative_moments(facet, q, formdegree-1, kind, Qpts, Qwts, trial)
            K = numpy.vstack((K, M))

        duals = numpy.tensordot(K, P_at_qpts[(0,) * dim], axes=(1, 0))
        return Q, duals, reduced_dofs

    def _bubble_derivative_moments(self, facet, degree, formdegree, kind, Qpts, Qwts, trial, deriv=True):
        """Integrate trial expressions against an orthonormal basis for
           the exterior derivative of bubbles.
        """
        dim = facet.get_spatial_dimension()
        # Get bubbles
        if formdegree == 0:
            B = make_bubbles(facet, degree)
        elif kind == 1:
            from FIAT.nedelec import Nedelec as N1curl
            from FIAT.raviart_thomas import RaviartThomas as N1div
            fe = (N1curl, N1div)[formdegree-1](facet, degree)
            B = fe.get_nodal_basis().take(fe.entity_dofs()[dim][0])
        else:
            from FIAT.nedelec_second_kind import NedelecSecondKind as N2curl
            from FIAT.brezzi_douglas_marini import BrezziDouglasMarini as N2div
            fe = (N2curl, N2div)[formdegree-1](facet, degree)
            B = fe.get_nodal_basis().take(fe.entity_dofs()[dim][0])

        # Tabulate the exterior derivate
        B_at_qpts = B.tabulate(Qpts, 1)
        d = (grad, curl, div)[formdegree]
        dtest = d(B_at_qpts)
        test = B_at_qpts[(0,) * dim]
        expr = dtest if deriv else test
        if len(dtest) > 0:
            # Build an orthonormal basis, remove nullspace
            B = inner(test, test, Qwts)
            A = inner(dtest, dtest, Qwts)
            sig, S = scipy.linalg.eigh(A, B)
            tol = sig[-1] * 1E-12
            nullspace_dim = len([s for s in sig if abs(s) <= tol])
            S = S[:, nullspace_dim:]
            if deriv:
                S *= numpy.sqrt(1 / sig[None, nullspace_dim:])
            # Apply change of basis
            expr = numpy.tensordot(S, expr, axes=(0, 0))
        return inner(expr, trial, Qwts)

    def get_indices(self, restriction_domain, take_closure=True):
        """Return the list of dofs with support on the given restriction domain.
        Allows for reduced Demkowicz elements, excluding the exterior
        derivative of the previous space in the de Rham complex.

        :arg restriction_domain: can be 'reduced', 'interior', 'vertex',
                                 'edge', 'face' or 'facet'
        :kwarg take_closure: Are we taking the closure of the restriction domain?
        """
        if restriction_domain == "reduced":
            indices = []
            entity_ids = self.get_entity_ids()
            for dim in entity_ids:
                reduced_dofs = self._reduced_dofs[dim]
                for entity, ids in entity_ids[dim].items():
                    indices.extend(ids[:reduced_dofs])
            return indices
        else:
            return DualSet.get_indices(self, restriction_domain, take_closure=take_closure)


class FDMDual(DemkowiczDual):

    def __init__(self, ref_el, degree, sobolev_space, element):
        exterior_derivative = {"H1": grad, "HCurl": curl, "HDiv": div}[sobolev_space]
        self.trace = {"HCurl": "contravariant", "HDiv": "normal"}.get(sobolev_space, None)
        self.dual_mapping = {"HCurl": "contravariant", "HDiv": "covariant"}.get(sobolev_space, None)

        sd = ref_el.get_spatial_dimension()
        base_ref_el = symmetric_simplex(sd)
        self.fe = element(base_ref_el, degree, variant="demkowicz")

        Q = create_quadrature(base_ref_el, 2 * degree)
        phis = self.fe.tabulate(1, Q.get_points())
        self.Q = Q
        self.V0 = phis[(0,) * sd]
        self.V1 = exterior_derivative(phis)
        super(FDMDual, self).__init__(ref_el, degree, sobolev_space, kind=None)

    def _reference_duals(self, dim, degree, formdegree, sobolev_space, kind):
        entity_dofs = self.fe.entity_dofs()
        ells = self.fe.dual_basis()
        Ref_el = self.fe.get_reference_element()
        sd = Ref_el.get_spatial_dimension()

        dofs = entity_dofs[dim][0]
        V0 = self.V0[dofs]
        W = self.Q.get_weights()
        B = inner(V0, V0, W)
        if dim == sd:
            _, S = scipy.linalg.eigh(B)
        else:
            V1 = self.V1[dofs]
            A = inner(V1, V1, W)
            if formdegree > 0:
                A += B
            _, S = scipy.linalg.eigh(B, A)
            S = numpy.dot(A, S)

        phis = numpy.array([ells[i].f_at_qpts for i in dofs])
        phis = numpy.tensordot(S.T, phis, axes=(1, 0))

        Q_dof = ells[dofs[0]].Q
        Q_ref = Q_dof.reference_rule()
        mapping = self.dual_mapping if dim == sd else self.trace
        # map physical phis to reference values Phis
        if mapping == "normal":
            n = Ref_el.compute_normal(0)
            Phis = numpy.dot(n[None, :], phis).transpose((1, 0, 2))
        elif mapping == "covariant":
            piola_map = Q_dof.jacobian().T
            Phis = numpy.dot(piola_map, phis).transpose((1, 0, 2))
        elif mapping == "contravariant":
            piola_map = numpy.linalg.pinv(Q_dof.jacobian()) * Q_dof.jacobian_determinant()
            Phis = numpy.dot(piola_map, phis).transpose((1, 0, 2))
        else:
            Jdet = Q_dof.jacobian_determinant()
            Phis = Jdet * phis

        reduced_dofs = self.fe.dual._reduced_dofs[dim]
        return Q_ref, Phis, reduced_dofs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import savemat, loadmat
    from os.path import exists

    from FIAT import IntegratedLegendre as CG
    from FIAT import Nedelec as N1Curl
    from FIAT import RaviartThomas as N1Div
    from FIAT import NedelecSecondKind as N2Curl
    from FIAT import BrezziDouglasMarini as N2Div

    dim = 3
    degree = 7
    ref_el = symmetric_simplex(dim)
    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    kind = 1
    variant = "fdm"
    variant = "demkowicz"
    # variant = None
    space_dict = {"H1": (CG, grad),
                  "HCurl": (N1Curl if kind == 1 else N2Curl, curl),
                  "HDiv": (N1Div if kind == 1 else N2Div, div),
                  }
    spaces = list(space_dict.keys())

    fig, axes = plt.subplots(ncols=len(spaces), nrows=2, figsize=(6*len(spaces), 12))
    axes = axes.T.flat
    fname = "fiat.mat"
    mdict = dict()
    if exists(fname):
        loadmat(fname, mdict=mdict)

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
        names = ("A", "B")
        for name, A in zip(names, mats):
            A[abs(A) < 1E-10] = 0.0
            scipy_mat = scipy.sparse.csr_matrix(A)
            nnz = scipy_mat.count_nonzero()
            ms = 0
            ax = next(axes)
            ax.spy(A, markersize=ms)
            if ms == 0:
                ax.pcolor(numpy.log(abs(A)))
            ax.set_title(f"{title} {name} nnz {nnz}")

            if False:
                family = {"H1": "Lagrange", "HCurl": "N1curl", "HDiv": "N1div"}[space]
                if kind == 2:
                    family = family.replace("N1", "N2")
                mat_name = "%s%dd_%s%d_%s" % (name, dim, family, degree, variant or "integral")
                old_mat = mdict.get(mat_name, None)
                old_mat = None
                if old_mat is None or scipy_mat.shape[0] == old_mat.shape[0]:
                    mdict[mat_name] = scipy_mat

    savemat(fname, mdict)
    plt.show()
