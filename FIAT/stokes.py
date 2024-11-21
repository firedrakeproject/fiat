# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import scipy

from FIAT import finite_element, dual_set
from FIAT.functional import (ComponentPointEvaluation,
                             PointTangentialDerivative,
                             PointTangentialSecondDerivative,
                             PointSecondDerivative,
                             IntegralMomentOfDerivative,
                             IntegralMoment,
                             FrobeniusIntegralMoment)
from FIAT.polynomial_set import make_bubbles, ONPolynomialSet
from FIAT.expansions import polynomial_dimension
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
        self.sd = sd
        shp = (sd,)

        mapping = None
        for dim in sorted(top):
            entity_ids[dim] = {}
            if dim == 0:
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(ComponentPointEvaluation(ref_el, (k,), shp, pt)
                                 for pt in pts for k in range(sd))
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
            else:
                # degree of bubbles
                moment_degree = degree
                if dim == 1:
                    moment_degree -= 2*sd
                elif dim == 2 and sd == 3:
                    moment_degree -= 6
                Q_ref, Phis = self._reference_duals(dim, degree, moment_degree)

                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    if dim == 1:
                        verts = ref_el.get_vertices_of_subcomplex(top[dim][entity])
                        ells = [PointTangentialDerivative]
                        if sd == 3:
                            ells.append(PointTangentialSecondDerivative)
                        nodes.extend(ell(ref_el, entity, pt, comp=(i,), shp=shp)
                                     for pt in verts for ell in ells for i in range(sd))

                    elif dim == 2 and sd == 3:
                        # Face vertex dofs
                        verts = numpy.array(ref_el.get_vertices_of_subcomplex(top[dim][entity]))
                        for i in range(len(verts)):
                            tangents = [verts[j] - verts[i] for j in range(len(verts)) if j != i]
                            nodes.extend(PointSecondDerivative(ref_el, *tangents, verts[i], comp=(k,), shp=shp)
                                         for k in range(sd))

                        # Face edge dofs
                        face = set(top[dim][entity])
                        edges = [e for e in top[1] if set(top[1][e]) < face]

                        phi_degree = degree - 5
                        ref_edge = ref_el.construct_subelement(1)
                        Q_edge = create_quadrature(ref_edge, degree + phi_degree)
                        Phis_edge = ONPolynomialSet(ref_edge, phi_degree).tabulate(Q_edge.get_points())[(0,)]
                        n = ref_el.compute_scaled_normal(entity)
                        for e in edges:
                            t = ref_el.compute_edge_tangent(e)
                            s = numpy.cross(n, t)
                            Q, phis = map_duals(ref_el, 1, e, mapping, Q_edge, Phis_edge)
                            nodes.extend(IntegralMomentOfDerivative(ref_el, s, Q, phi, comp=(k,), shp=shp)
                                         for phi in phis for k in range(sd))

                    # Rest of the facet moments
                    Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                    if dim == sd:
                        nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    else:
                        nodes.extend(IntegralMoment(ref_el, Q, phi, comp=(k,), shp=shp)
                                     for phi in phis for k in range(sd))

                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, dim, degree, moment_degree):
        facet = symmetric_simplex(dim)
        Q = create_quadrature(facet, 2 * degree)
        Qpts, Qwts = Q.get_points(), Q.get_weights()

        shp = (dim,) if dim == self.sd else tuple()
        V = ONPolynomialSet(facet, degree, shp, scale="orthonormal")

        moments = self._interior_moments if dim == self.sd else self._facet_moments
        duals = moments(facet, moment_degree, Qpts, Qwts, V)
        return Q, duals

    def _facet_moments(self, facet, moment_degree, Qpts, Qwts, V):
        """Integrate trial expressions against an orthonormal basis for
           the exterior derivative of bubbles.
        """
        dim = facet.get_spatial_dimension()

        V_at_qpts = V.tabulate(Qpts)
        trial = V_at_qpts[(0,) * dim]
        test = trial[:polynomial_dimension(facet, moment_degree)]
        K = inner(test, trial, Qwts)
        return numpy.tensordot(K, trial, axes=(1, 0))

    def _interior_moments(self, facet, moment_degree, Qpts, Qwts, V):
        """Integrate trial expressions against an orthonormal basis for
           the exterior derivative of bubbles.
        """
        dim = facet.get_spatial_dimension()

        V_at_qpts = V.tabulate(Qpts, 1)
        trial = V_at_qpts[(0,) * dim]
        eps_trial = eps(V_at_qpts)

        # Get bubbles
        B = make_bubbles(facet, V.degree, shape=(dim,))

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
        K = numpy.concatenate(expr, axis=0)
        return numpy.tensordot(K, trial, axes=(1, 0))


class Stokes(finite_element.CiarletElement):
    """Simplicial continuous element with integrated Legendre polynomials."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        if degree < 2*sd:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= {2*sd}")
        poly_set = ONPolynomialSet(ref_el, degree, shape=(sd,), variant="bubble")
        dual = StokesDual(ref_el, degree)
        formdegree = sd-1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree,
                         mapping="contravariant piola")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dim = 2
    degree = 10
    ref_el = symmetric_simplex(dim)
    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    axes = axes.flat

    fe = Stokes(ref_el, degree)
    phi_at_qpts = fe.tabulate(1, Qpts)

    Veps = eps(phi_at_qpts)
    Vdiv = numpy.trace(Veps, axis1=1, axis2=2)
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
