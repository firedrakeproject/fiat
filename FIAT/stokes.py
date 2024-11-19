# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import scipy

from FIAT import finite_element, dual_set
from FIAT.functional import ComponentPointEvaluation, FrobeniusIntegralMoment
from FIAT.polynomial_set import make_bubbles, ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import symmetric_simplex


def eps(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    d = len(grad_u)
    indices = ((i, j) for i in range(d) for j in range(d))
    eps_u = [0.5*(grad_u[j][:, i, :] + grad_u[i][:, j, :]) for k, (i, j) in enumerate(indices)]

    num_members = table_u[(0,)*d].shape[0]
    return numpy.transpose(eps_u, (1, 0, 2)).reshape(num_members, d, d, -1)


def grad(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    grad_u = [grad_u[k] for k in sorted(grad_u)]
    return numpy.transpose(grad_u, (1, 0, 2, 3))


def inner(v, u, Qwts):
    return numpy.tensordot(numpy.multiply(v, Qwts), u, axes=(range(1, v.ndim), range(1, u.ndim)))



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


class StokesDual(dual_set.DualSet):

    def __init__(self, ref_el, degree):
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        self.shp = (sd,)

        mapping = None
        for dim in sorted(top):
            entity_ids[dim] = {}
            if dim == 0:
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(ComponentPointEvaluation(ref_el, (i,), self.shp, pt)
                                 for pt in pts for i in range(sd))
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
            else:
                Q_ref, Phis = self._reference_duals(dim, degree)
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, dim, degree):
        facet = symmetric_simplex(dim)
        Q = create_quadrature(facet, 2 * degree)
        Qpts, Qwts = Q.get_points(), Q.get_weights()

        P = ONPolynomialSet(facet, degree, self.shp, scale="orthonormal")
        P_at_qpts = P.tabulate(Qpts, 1)
        trial = P_at_qpts[(0,) * dim]

        if dim == self.shp[0]:
            dtrial = eps(P_at_qpts)
            moments = self._interior_moments
        else:
            dtrial = grad(P_at_qpts)
            moments = self._facet_moments

        K = moments(facet, degree, Qpts, Qwts, dtrial)
        duals = numpy.tensordot(K, P_at_qpts[(0,) * dim], axes=(1, 0))
        return Q, duals

    def _facet_moments(self, facet, degree, Qpts, Qwts, trial):
        """Integrate trial expressions against an orthonormal basis for
           the exterior derivative of bubbles.
        """
        dim = facet.get_spatial_dimension()
        # Get bubbles
        B = make_bubbles(facet, degree, shape=self.shp)

        # Tabulate the exterior derivate
        B_at_qpts = B.tabulate(Qpts, 1)
        dtest = grad(B_at_qpts)
        test = B_at_qpts[(0,) * dim]
        expr = dtest
        if len(dtest) > 0:
            # Build an orthonormal basis, remove nullspace
            B = inner(test, test, Qwts)
            A = inner(dtest, dtest, Qwts)
            sig, S = scipy.linalg.eigh(A, B)
            tol = sig[-1] * 1E-12
            nullspace_dim = len([s for s in sig if abs(s) <= tol])
            S = S[:, nullspace_dim:]
            S *= numpy.sqrt(1 / sig[None, nullspace_dim:])
            # Apply change of basis
            expr = numpy.tensordot(S, expr, axes=(0, 0))
        return inner(expr, trial, Qwts)


    def _interior_moments(self, facet, degree, Qpts, Qwts, eps_trial):
        """Integrate trial expressions against an orthonormal basis for
           the exterior derivative of bubbles.
        """
        dim = facet.get_spatial_dimension()
        # Get bubbles
        B = make_bubbles(facet, degree, shape=self.shp)

        # Tabulate the exterior derivate
        B_at_qpts = B.tabulate(Qpts, 1)
        eps_test = eps(B_at_qpts)
        div_test = numpy.trace(eps_test, axis1=1, axis2=2)
        div_trial = numpy.trace(eps_trial, axis1=1, axis2=2)

        # Build an orthonormal basis, remove nullspace
        B = inner(eps_test, eps_test, Qwts)
        A = inner(div_test, div_test, Qwts)

        sig, S = scipy.linalg.eigh(A, B)
        tol = sig[-1] * 1E-12
        nullspace_dim = len([s for s in sig if abs(s) <= tol])

        S2 = S[:, :nullspace_dim]
        S1 = S[:, nullspace_dim:]
        S1 *= numpy.sqrt(1 / sig[None, nullspace_dim:])

        # Apply change of basis
        expr = [inner(numpy.tensordot(S1, div_test, axes=(0, 0)), div_trial, Qwts),
                inner(numpy.tensordot(S2, eps_test, axes=(0, 0)), eps_trial, Qwts)]
        expr = numpy.concatenate(expr, axis=0)
        return expr


class Stokes(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        if degree < 2*sd:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= {2*sd}")
        poly_set = ONPolynomialSet(ref_el, degree, shape=(sd,), variant="bubble")
        dual = StokesDual(ref_el, degree)

        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from FIAT import ufc_simplex

    dim = 2
    degree = 10
    ref_el = symmetric_simplex(dim)
    # ref_el = ufc_simplex(dim)
    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    space_dict = {"H1": (Stokes, eps)}
    spaces = list(space_dict.keys())

    fig, axes = plt.subplots(ncols=2, nrows=len(spaces), figsize=(12, 6*len(spaces)))
    axes = axes.flat

    for space in spaces:
        element, d = space_dict[space]
        fe = element(ref_el, degree)
        phi_at_qpts = fe.tabulate(1, Qpts)

        Veps = d(phi_at_qpts)
        Vdiv = div(phi_at_qpts)
        Aeps = inner(Veps, Veps, Qwts)
        Adiv = inner(Vdiv, Vdiv, Qwts)

        title = f"{type(fe).__name__}({degree})"
        names = ("eps", "div")
        mats = (Aeps, Adiv)
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

    plt.show()
