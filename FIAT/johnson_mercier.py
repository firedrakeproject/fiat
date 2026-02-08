from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.functional import (FrobeniusIntegralMoment, IntegralMomentOfTensorDivergence,
                             TensorBidirectionalIntegralMoment)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.nedelec_second_kind import NedelecSecondKind as N2curl
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini as N2div
from FIAT.restricted import RestrictedElement
import numpy


class JohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, variant=None, quad_scheme=None):
        if degree != 1:
            raise ValueError("Johnson-Mercier only defined for degree=1")
        if variant is not None:
            raise ValueError(f"Johnson-Mercier does not have the {variant} variant")
        ref_el = ref_complex.get_parent()
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # Face dofs: bidirectional (nn and nt) Legendre moments
        dim = sd - 1
        R = numpy.array([[0, 1], [-1, 0]])
        ref_facet = ref_el.construct_subelement(dim)
        Qref = parse_quadrature_scheme(ref_facet, 2*degree, quad_scheme)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree)
        phis = P.tabulate(Qref.get_points())[(0,) * dim]
        for f in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, f, Qref, avg=True)
            thats = ref_el.compute_tangents(dim, f)
            if sd == 2:
                nhat = numpy.dot(R, *thats)
            else:
                nhat = numpy.cross(*thats)
                thats = numpy.cross(nhat[None, :], thats, axis=1)

            nodes.extend(TensorBidirectionalIntegralMoment(ref_el, nhat, comp, Q, phi)
                         for phi in phis for comp in (nhat, *thats))
            entity_ids[dim][f].extend(range(cur, len(nodes)))

        cur = len(nodes)
        # Interior dofs: moments for each independent component
        n = list(map(ref_el.compute_scaled_normal, sorted(top[sd-1])))
        Q = parse_quadrature_scheme(ref_complex, 2*degree-1, quad_scheme)
        P = polynomial_set.ONPolynomialSet(ref_el, degree-1, scale="L2 piola")
        phis = P.tabulate(Q.get_points())[(0,) * sd]
        nodes.extend(TensorBidirectionalIntegralMoment(ref_el, n[i+1], n[j+1], Q, phi)
                     for phi in phis for i in range(sd) for j in range(i, sd))

        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class JohnsonMercier(finite_element.CiarletElement):
    """The Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None, quad_scheme=None):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = JohnsonMercierDualSet(ref_complex, degree, variant=variant, quad_scheme=quad_scheme)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)


def ReducedJohnsonMercierSpace(ref_el, degree):
    ref_complex = macro.AlfeldSplit(ref_el)
    poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()

    # Interior constraints: moments of divergence against the orthogonal complement of RBMs
    Q = create_quadrature(ref_complex, 2*(degree-1))
    qpts, qwts = Q.get_points(), Q.get_weights()

    N2 = N2curl(ref_el, degree)
    edofs = N2.entity_dofs()
    rbm_indices = [edofs[1][entity][0] for entity in edofs[1]]
    indices = numpy.setdiff1d(range(N2.space_dimension()), rbm_indices)

    N2_at_qpts = N2.tabulate(1, qpts)[(0,) * sd]
    rbms = N2_at_qpts[rbm_indices]
    ells = rbms * qwts[None, None, :]
    M = numpy.tensordot(ells, N2_at_qpts, axes=((1, 2), (1, 2)))
    C = numpy.linalg.solve(M[:, rbm_indices], M[:, indices])

    phis = N2_at_qpts[indices]
    phis -= numpy.tensordot(C, rbms, axes=(0, 0))
    nodes = [IntegralMomentOfTensorDivergence(ref_el, Q, phi) for phi in phis]

    # Facet constraints: moments against the orthogonal complement of RBMs
    ref_facet = ref_el.get_facet_element()
    Q = create_quadrature(ref_facet, 2*(degree-1))
    qpts, qwts = Q.get_points(), Q.get_weights()

    if sd == 2:
        P1 = polynomial_set.ONPolynomialSet(ref_facet, degree)
        phis = numpy.zeros((sd-1, sd-1, qpts.shape[-1]))
        phis[0] = P1.take([1]).tabulate(Q.get_points())[(0,)*(sd-1)]
    else:
        N2 = N2curl(ref_facet, degree)
        edofs = N2.entity_dofs()
        rbm_indices = [edofs[1][entity][0] for entity in edofs[1]]
        indices = numpy.setdiff1d(range(N2.space_dimension()), rbm_indices)

        N2_at_qpts = N2.tabulate(1, qpts)[(0,) * (sd-1)]
        rbms = N2_at_qpts[rbm_indices]
        ells = rbms * qwts[None, None, :]
        M = numpy.tensordot(ells, N2_at_qpts, axes=((1, 2), (1, 2)))
        C = numpy.linalg.solve(M[:, rbm_indices], M[:, indices])

        phis = N2_at_qpts[indices]
        phis -= numpy.tensordot(C, rbms, axes=(0, 0))

    Phis = phis
    for f in top[sd-1]:
        n = ref_el.compute_scaled_normal(f)
        # Map the BDM basis into the facet
        Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
        Jf = Qf.jacobian()
        phis = numpy.tensordot(Jf, Phis.transpose(1, 0, 2), (1, 0)).transpose(1, 0, 2)
        if sd == 3:
            # Normal moments against cross(n, RT)
            phis = numpy.cross(n[None, :, None], phis, axis=1)
        phis = phis[:, None, :] * n[None, :, None]
        nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, phi) for phi in phis)

    return polynomial_set.ConstrainedPolynomialSet(nodes, poly_set)


class ReducedJohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant=None, quad_scheme=None):
        if degree != 1:
            raise ValueError("Reduced Johnson-Mercier only defined for degree=1")
        if variant is not None:
            raise ValueError(f"Reduced Johnson-Mercier does not have the {variant} variant")

        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []
        # On each facet, let n be its normal.  We need to integrate
        # u.n against a Dubiner basis for P1
        # and u x n against a basis for lowest-order RT.
        ref_facet = ref_el.get_facet_element()
        Q = create_quadrature(ref_facet, degree+1)
        P1 = polynomial_set.ONPolynomialSet(ref_facet, 1)
        P1_at_qpts = P1.tabulate(Q.get_points())[(0,)*(sd - 1)]
        if sd == 2:
            # For 2D just take the constant
            RT_at_qpts = P1_at_qpts[:1, None, :]
        else:
            # Basis for lowest-order RT [(1, 0), (0, 1), (x, y)]
            RT_at_qpts = numpy.zeros((3, sd-1, P1_at_qpts.shape[-1]))
            RT_at_qpts[0, 0, :] = P1_at_qpts[0, None, :]
            RT_at_qpts[1, 1, :] = P1_at_qpts[0, None, :]
            RT_at_qpts[2, 0, :] = P1_at_qpts[1, None, :]
            RT_at_qpts[2, 1, :] = P1_at_qpts[2, None, :]

        for f in sorted(top[sd-1]):
            cur = len(nodes)
            n = ref_el.compute_scaled_normal(f)
            nn = numpy.outer(n, n)
            Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
            # Normal-normal moments against P1
            phis = P1_at_qpts[:, None, None, :] * nn[None, :, :, None]
            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, phi) for phi in phis)
            # Map the RT basis into the facet
            Jf = Qf.jacobian()
            phis = numpy.tensordot(Jf, RT_at_qpts.transpose(1, 0, 2), (1, 0)).transpose(1, 0, 2)
            if sd == 3:
                # Normal moments against cross(n, RT)
                phis = numpy.cross(n[None, :, None], phis, axis=1)
            phis = phis[:, None, :] * n[None, :, None]
            nodes.extend(FrobeniusIntegralMoment(ref_el, Qf, phi) for phi in phis)
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class ReducedJohnsonMercier(finite_element.CiarletElement):
    """The Reduced Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None, quad_scheme=None):
        poly_set = ReducedJohnsonMercierSpace(ref_el, degree)
        dual = ReducedJohnsonMercierDualSet(ref_el, degree, variant=variant, quad_scheme=quad_scheme)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
