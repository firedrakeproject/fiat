# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import scipy

from FIAT import finite_element, dual_set, expansions, jacobi, macro
from FIAT.functional import (ComponentPointEvaluation,
                             PointTangentialDerivative,
                             PointTangentialSecondDerivative,
                             PointSecondDerivative,
                             IntegralMomentOfDerivative,
                             IntegralMoment,
                             FrobeniusIntegralMoment)
from FIAT.polynomial_set import make_bubbles, ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


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
    assert mapping is None
    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
    Jdet = Q.jacobian_determinant()
    phis = (1 / Jdet) * Phis
    return Q, phis


def jacobi_duals(ref_el, weight, moment_degree, degree):
    facet = ref_el.construct_subelement(1)
    Q = create_quadrature(facet, degree + moment_degree)
    x = facet.compute_barycentric_coordinates(Q.get_points())
    xhat = x[:, 1:] - x[:, :1]
    a = weight
    phis = jacobi.eval_jacobi_batch(a, a, moment_degree, xhat)
    return Q, phis


class StokesDual(dual_set.DualSet):

    def __init__(self, ref_el, degree):
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        moment_degree = degree - 2*sd
        edge_weight = sd-1
        Q_edge, phis_edge = jacobi_duals(ref_el, edge_weight, degree-4, degree)

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
                if dim == 1:
                    Q_ref, Phis = Q_edge, phis_edge[:moment_degree+1]
                elif dim < sd:
                    Q_ref, Phis = self._reference_duals(ref_el, dim, degree, moment_degree)

                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    if dim == sd:
                        # Interior dofs
                        Q, phis = self._interior_duals(ref_el, degree)
                        nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                        entity_ids[dim][entity] = list(range(cur, len(nodes)))
                        continue

                    if dim == 1:
                        # Vertex-edge dofs
                        verts = ref_el.get_vertices_of_subcomplex(top[dim][entity])
                        ells = [PointTangentialDerivative]
                        if sd == 3:
                            ells.append(PointTangentialSecondDerivative)
                        nodes.extend(ell(ref_el, entity, pt, comp=(i,), shp=shp)
                                     for pt in verts for ell in ells for i in range(sd))

                    elif dim == 2:
                        # Face-vertex dofs
                        verts = numpy.array(ref_el.get_vertices_of_subcomplex(top[dim][entity]))
                        for i in range(len(verts)):
                            tangents = [verts[j] - verts[i] for j in range(len(verts)) if j != i]
                            nodes.extend(PointSecondDerivative(ref_el, *tangents, verts[i], comp=(k,), shp=shp)
                                         for k in range(sd))

                        # Face-edge dofs
                        face = set(top[dim][entity])
                        edges = [e for e in top[1] if set(top[1][e]) < face]
                        n = ref_el.compute_scaled_normal(entity)
                        for e in edges:
                            t = ref_el.compute_edge_tangent(e)
                            s = numpy.cross(n, t)
                            Q, phis = map_duals(ref_el, 1, e, mapping, Q_edge, phis_edge[:degree-5+1])
                            nodes.extend(IntegralMomentOfDerivative(ref_el, s, Q, phi, comp=(k,), shp=shp)
                                         for phi in phis for k in range(sd))

                    # Rest of the facet moments
                    Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                    nodes.extend(IntegralMoment(ref_el, Q, phi, comp=(k,), shp=shp)
                                 for phi in phis for k in range(sd))

                    entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)

    def _reference_duals(self, ref_el, dim, degree, moment_degree):
        """Compute moments of trial space V against a subset of itself.
        """
        sd = ref_el.get_spatial_dimension()
        assert dim != sd
        facet = ref_el.construct_subelement(dim)

        V = ONPolynomialSet(facet, moment_degree, scale="orthonormal")
        Q = create_quadrature(facet, degree + V.degree)
        phis = V.tabulate(Q.get_points())[(0,)*dim]
        return Q, phis

    def _interior_duals(self, ref_el, degree):
        """Compute div-div and eps-eps moments of the trial space against an
           orthonormal bases for div(V_0) and eps(V_0).
        """
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        Q = create_quadrature(ref_el, degree * 2)
        Qpts, Qwts = Q.get_points(), Q.get_weights()

        # Test space: bubbles
        if ref_el.is_macrocell():
            B = ONPolynomialSet(ref_el, degree, shp, variant="bubble")
            es = B.get_expansion_set()
            ids = expansions.polynomial_entity_ids(ref_el, degree, continuity=es.continuity)
            indices = []
            for dim in sorted(ids):
                for entity in ref_el.get_interior_facets(dim):
                    indices.extend(ids[dim][entity])

            dimPk = es.get_num_members(degree)
            indices = [dimPk*k + i for i in indices for k in range(sd)]
            V0 = B.take(indices)
        else:
            V0 = make_bubbles(ref_el, degree, shape=shp)

        V0_at_qpts = V0.tabulate(Qpts, 1)
        eps_test = eps(V0_at_qpts)
        div_test = numpy.trace(eps_test, axis1=1, axis2=2)

        # Build an orthonormal basis that splits in the div kernel
        B = inner(eps_test, eps_test, Qwts)
        A = inner(div_test, div_test, Qwts)

        sig, S = scipy.linalg.eigh(A, B)
        tol = sig[-1] * 1E-12
        nullspace_dim = len([s for s in sig if abs(s) <= tol])

        S2 = S[:, :nullspace_dim]
        S1 = S[:, nullspace_dim:]
        S1 *= numpy.sqrt(1 / sig[None, nullspace_dim:])

        # Apply change of basis
        div_test = numpy.tensordot(S1, div_test, axes=(0, 0))
        eps_test = numpy.tensordot(S2, eps_test, axes=(0, 0))

        # Trial space
        V = ONPolynomialSet(ref_el, degree, shp, scale="orthonormal")
        V_at_qpts = V.tabulate(Qpts, 1)
        trial = V_at_qpts[(0,) * sd]
        eps_trial = eps(V_at_qpts)
        div_trial = numpy.trace(eps_trial, axis1=1, axis2=2)

        K = numpy.concatenate((inner(div_test, div_trial, Qwts),
                               inner(eps_test, eps_trial, Qwts)), axis=0)
        phis = numpy.tensordot(K, trial, axes=(1, 0))
        return Q, phis


class Stokes(finite_element.CiarletElement):
    """Simplicial continuous element that decouples div-free modes and
    simultaneously diagonalizes the div-div and eps-eps inner-products on
    the reference element."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        if degree < 2*sd:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= {2*sd}")

        poly_set = ONPolynomialSet(ref_el, degree, shape=(sd,), variant="bubble")
        dual = StokesDual(ref_el, degree)
        formdegree = sd-1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree)


class MacroStokesDual(StokesDual):

    def __init__(self, ref_complex, degree):
        nodes = []
        entity_ids = {}
        ref_el = ref_complex.get_parent()
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
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
                if dim < sd:
                    moment_degree = degree - (1+dim)
                    Q_ref, Phis = self._reference_duals(ref_complex, dim, degree, moment_degree)

                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    if dim == sd:
                        # Interior dofs
                        Q, phis = self._interior_duals(ref_complex, degree)
                        nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    else:
                        # Facet dofs
                        Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                        nodes.extend(IntegralMoment(ref_el, Q, phi, comp=(k,), shp=shp)
                                     for phi in phis for k in range(sd))
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
        super(StokesDual, self).__init__(nodes, ref_el, entity_ids)


class MacroStokes(finite_element.CiarletElement):
    """Simplicial continuous element that decouples div-free modes and
    simultaneously diagonalizes the div-div and eps-eps inner-products on
    the reference element."""
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        if degree < sd:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= {sd}")

        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = ONPolynomialSet(ref_complex, degree, shape=(sd,), variant="bubble")
        dual = MacroStokesDual(ref_complex, degree)
        formdegree = sd-1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import FIAT

    dim = 3
    degree = 3
    ref_el = FIAT.reference_element.symmetric_simplex(dim)
    # fe = Stokes(ref_el, degree)
    fe = MacroStokes(ref_el, degree)

    Q = create_quadrature(fe.ref_complex, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()
    phi_at_qpts = fe.tabulate(1, Qpts)

    Veps = eps(phi_at_qpts)
    Vdiv = numpy.trace(Veps, axis1=1, axis2=2)
    Aeps = inner(Veps, Veps, Qwts)
    Adiv = inner(Vdiv, Vdiv, Qwts)

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    axes = axes.flat
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
