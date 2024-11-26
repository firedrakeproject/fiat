# Copyright (C) 2023 Pablo D. Brubeck (University of Oxford)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import scipy

from FIAT import finite_element, dual_set, expansions, jacobi, macro
from FIAT.functional import (PointEvaluation,
                             ComponentPointEvaluation,
                             PointTangentialDerivative,
                             PointTangentialSecondDerivative,
                             PointSecondDerivative,
                             IntegralMomentOfDerivative,
                             IntegralMoment,
                             FrobeniusIntegralMoment)
from FIAT.polynomial_set import make_bubbles, ONPolynomialSet
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.bernstein import Bernstein


def eps(table_u):
    grad_u = {alpha.index(1): table_u[alpha] for alpha in table_u if sum(alpha) == 1}
    d = len(grad_u)
    indices = ((i, j) for i in range(d) for j in range(d))
    eps_u = [0.5*(grad_u[j][:, i, :] + grad_u[i][:, j, :]) for k, (i, j) in enumerate(indices)]
    num_members = table_u[(0,)*d].shape[0]
    return numpy.transpose(eps_u, (1, 0, 2)).reshape(num_members, d, d, -1)


def inner(v, u, Qwts):
    return numpy.tensordot(numpy.multiply(v, Qwts), u,
                           axes=(range(1, v.ndim), range(1, u.ndim)))


def map_duals(ref_el, dim, entity, mapping, Q_ref, Phis):
    assert mapping is None
    Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
    Jdet = Q.jacobian_determinant()
    phis = (1 / Jdet) * Phis
    return Q, phis


def jacobi_duals(ref_el, weight, trial_degree, test_degree):
    facet = ref_el.construct_subelement(1)
    Q = create_quadrature(facet, trial_degree + test_degree)
    x = facet.compute_barycentric_coordinates(Q.get_points())
    xhat = x[:, 1:] - x[:, :1]
    a = weight
    phis = jacobi.eval_jacobi_batch(a, a, test_degree, xhat)
    return Q, phis


def dubiner_duals(ref_el, dim, trial_degree, test_degree):
    if dim == 0:
        return None, []

    facet = ref_el.construct_subelement(dim)
    V = ONPolynomialSet(facet, test_degree, scale="orthonormal")
    Q = create_quadrature(facet, trial_degree + V.degree)
    phis = V.tabulate(Q.get_points())[(0,)*dim]
    return Q, phis


class StokesDual(dual_set.DualSet):

    def __init__(self, ref_el, degree):
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)
        comps = list(numpy.ndindex(shp))

        moment_degree = degree - 2*sd
        edge_weight = sd-1
        Q_edge, phis_edge = jacobi_duals(ref_el, edge_weight, degree, degree-4)

        mapping = None
        self._reduced_dofs = {}
        for dim in sorted(top):
            entity_ids[dim] = {}
            self._reduced_dofs[dim] = None

            if dim == 1:
                Q_ref, Phis = Q_edge, phis_edge[:moment_degree+1]
            elif dim < sd:
                Q_ref, Phis = dubiner_duals(ref_el, dim, degree, moment_degree)

            for entity in sorted(top[dim]):
                cur = len(nodes)
                if dim == sd:
                    # Interior dofs
                    Q, phis, nullspace_dim = self._interior_duals(ref_el, degree)
                    self._reduced_dofs[dim] = nullspace_dim
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                    continue

                elif dim == 0:
                    # Vertex dofs
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(ComponentPointEvaluation(ref_el, comp, shp, pt)
                                 for pt in pts for comp in comps)
                    entity_ids[dim][entity] = list(range(cur, len(nodes)))
                    continue

                elif dim == 1:
                    # Vertex-edge dofs
                    verts = ref_el.get_vertices_of_subcomplex(top[dim][entity])
                    ells = [PointTangentialDerivative]
                    if sd == 3:
                        ells.append(PointTangentialSecondDerivative)
                    nodes.extend(ell(ref_el, entity, pt, comp=comp, shp=shp)
                                 for pt in verts for ell in ells for comp in comps)

                elif dim == 2:
                    # Face-vertex dofs
                    verts = numpy.array(ref_el.get_vertices_of_subcomplex(top[dim][entity]))
                    for i in range(len(verts)):
                        tangents = [verts[j] - verts[i] for j in range(len(verts)) if j != i]
                        nodes.extend(PointSecondDerivative(ref_el, *tangents, verts[i], comp=comp, shp=shp)
                                     for comp in comps)

                    # Face-edge dofs
                    mid_face, = numpy.asarray(ref_el.make_points(dim, entity, dim+1))
                    edges = ref_el.connectivity[(dim, dim-1)][entity]
                    for e in edges:
                        mid_edge, = numpy.asarray(ref_el.make_points(dim-1, e, dim))
                        s = mid_face - mid_edge
                        Q, phis = map_duals(ref_el, dim-1, e, mapping, Q_edge, phis_edge[:degree-5+1])
                        nodes.extend(IntegralMomentOfDerivative(ref_el, s, Q, phi, comp=comp, shp=shp)
                                     for phi in phis for comp in comps)

                # Rest of the facet moments
                Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                nodes.extend(IntegralMoment(ref_el, Q, phi, comp=comp, shp=shp)
                             for phi in phis for comp in comps)

                entity_ids[dim][entity] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)

    def _interior_duals(self, ref_el, degree):
        """Compute div-div and eps-eps moments of the trial space against an
           orthonormal bases for div(V_0) and eps(V_0).
        """
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        Q = create_quadrature(ref_el, 2*degree)
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

        S1 = S[:, :nullspace_dim]
        S2 = S[:, nullspace_dim:]
        S2 *= numpy.sqrt(1 / sig[None, nullspace_dim:])

        # Apply change of basis
        eps_test = numpy.tensordot(S1, eps_test, axes=(0, 0))
        div_test = numpy.tensordot(S2, div_test, axes=(0, 0))

        # Trial space
        V = ONPolynomialSet(ref_el, degree, shp, scale="orthonormal")
        V_at_qpts = V.tabulate(Qpts, 1)
        trial = V_at_qpts[(0,) * sd]
        eps_trial = eps(V_at_qpts)
        div_trial = numpy.trace(eps_trial, axis1=1, axis2=2)

        K = numpy.concatenate((inner(eps_test, eps_trial, Qwts),
                               inner(div_test, div_trial, Qwts),
                               ), axis=0)
        phis = numpy.tensordot(K, trial, axes=(1, 0))
        return Q, phis, nullspace_dim

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
            return dual_set.DualSet.get_indices(self, restriction_domain, take_closure=take_closure)


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
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")


class MacroStokesDual(StokesDual):

    def __init__(self, ref_complex, degree):
        nodes = []
        entity_ids = {}
        ref_el = ref_complex.get_parent()
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)
        comps = list(numpy.ndindex(shp))

        mapping = None
        self._reduced_dofs = {}
        for dim in sorted(top):
            entity_ids[dim] = {}

            moment_degree = degree - (dim+1)
            if dim == 1:
                Q_ref, Phis = jacobi_duals(ref_complex, 0, degree, moment_degree)
            elif dim > 0 and dim < sd:
                Q_ref, Phis = dubiner_duals(ref_complex, dim, degree, moment_degree)

            self._reduced_dofs[dim] = None
            for entity in sorted(top[dim]):
                cur = len(nodes)
                if dim == 0:
                    # Vertex dofs
                    pts = ref_el.make_points(dim, entity, degree)
                    nodes.extend(ComponentPointEvaluation(ref_el, comp, shp, pt)
                                 for pt in pts for comp in comps)
                elif dim == sd:
                    # Interior dofs
                    Q, phis, nullspace_dim = self._interior_duals(ref_complex, degree)
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in phis)
                    self._reduced_dofs[dim] = nullspace_dim
                else:
                    # Facet dofs
                    Q, phis = map_duals(ref_el, dim, entity, mapping, Q_ref, Phis)
                    nodes.extend(IntegralMoment(ref_el, Q, phi, comp=comp, shp=shp)
                                 for phi in phis for comp in comps)
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
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")


class DivStokesDual(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        self.degree = degree
        nodes = []
        entity_ids = {}
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        Q = create_quadrature(ref_el, 2*degree)

        mapping = None
        for dim in sorted(top):
            entity_ids[dim] = {}
            for entity in sorted(top[dim]):
                entity_ids[dim][entity] = []

        # Vertex dofs
        for entity in top[0]:
            pt, = ref_el.make_points(0, entity, degree)
            nodes.append(PointEvaluation(ref_el, pt))

        if sd == 3:
            # Edge dofs
            Q_ref, Phis = jacobi_duals(ref_el, 0, degree, degree-2)
            for entity in top[1]:
                Q_edge, phis = map_duals(ref_el, 1, entity, mapping, Q_ref, Phis)
                nodes.extend(IntegralMoment(ref_el, Q_edge, phi) for phi in phis)

        # Interior dof
        nodes.append(IntegralMoment(ref_el, Q, numpy.ones(Q.get_weights().shape)))

        # Throwaway dofs
        B = Bernstein(ref_el, degree)
        ids = B.entity_dofs()

        indices = []
        for dim in sorted(top):
            for entity in sorted(top[dim]):
                if sd == 3 and dim == 1:
                    continue
                start = 1 if dim % sd == 0 else 0
                indices.extend(ids[dim][entity][start:])

        interior = ids[sd][0][0]
        phis = B.tabulate(0, Q.get_points())[(0,)*sd]
        phis -= phis[[interior]]
        nodes.extend(IntegralMoment(ref_el, Q, phi) for phi in phis[indices])

        entity_ids[sd][0] = list(range(len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)

    def get_indices(self, restriction_domain, take_closure=True):
        """Return the list of dofs with support on the given restriction domain.
        Allows for reduced Demkowicz elements, excluding the exterior
        derivative of the previous space in the de Rham complex.

        :arg restriction_domain: can be 'reduced', 'interior', 'vertex',
                                 'edge', 'face' or 'facet'
        :kwarg take_closure: Are we taking the closure of the restriction domain?
        """
        if restriction_domain == "reduced":
            sd = self.ref_el.get_spatial_dimension()
            top = self.ref_el.get_topology()
            num_bfs = 1 + len(top[0])
            if sd == 3:
                num_bfs += len(top[1]) * (self.degree-1)
            return list(range(num_bfs))
        else:
            return dual_set.DualSet.get_indices(self, restriction_domain, take_closure=take_closure)


class DivStokes(finite_element.CiarletElement):
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        if degree < 2*sd-1:
            raise ValueError(f"{type(self).__name__} elements only valid for k >= {2*sd-1}")

        poly_set = ONPolynomialSet(ref_el, degree, variant="bubble")
        dual = DivStokesDual(ref_el, degree)
        formdegree = sd  # n-form
        super().__init__(poly_set, dual, degree, formdegree)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import FIAT

    dim = 3
    degree = 6
    ref_el = FIAT.reference_element.symmetric_simplex(dim)
    fe = Stokes(ref_el, degree)
    # fe = MacroStokes(ref_el, degree)
    Q = create_quadrature(fe.ref_complex, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    df = DivStokes(ref_el, degree-1)
    df = FIAT.RestrictedElement(df, restriction_domain="reduced")
    print(df.tabulate(0, ref_el.vertices)[(0,)*dim])

    family = type(fe).__name__
    domains = (None, "reduced", "facet")
    fig, axes = plt.subplots(nrows=2, ncols=len(domains), figsize=(6*len(domains), 6*2))
    axes = axes.T.flat
    for domain in domains:
        if domain:
            fe = FIAT.RestrictedElement(fe, restriction_domain=domain)

        phi_at_qpts = fe.tabulate(1, Qpts)
        Veps = eps(phi_at_qpts)
        Vdiv = numpy.trace(Veps, axis1=1, axis2=2)
        Aeps = inner(Veps, Veps, Qwts)
        Adiv = inner(Vdiv, Vdiv, Qwts)

        title = f"{family}({degree}, {domain})"
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
