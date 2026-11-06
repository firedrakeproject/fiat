from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.functional import (FrobeniusIntegralMoment, IntegralMomentOfTensorDivergence,
                             TensorBidirectionalIntegralMoment)
from FIAT.quadrature import FacetQuadratureRule
from FIAT.reference_element import make_affine_mapping
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

        # Face dofs: bidirectional (nn and nt) moments
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
                # Face moments of sigma.n against n P1 and t P1
                nhat = numpy.dot(R, *thats)
                components = (nhat, *thats)
            else:
                # Face moments of sigma.n against n P1 and (n x t_j) P1
                nhat = numpy.cross(*thats)
                ncrosst = numpy.cross(nhat[None, :], thats, axis=1)
                components = (nhat, *ncrosst)

            nodes.extend(TensorBidirectionalIntegralMoment(ref_el, nhat, comp, Q, phi)
                         for phi in phis for comp in components)
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


class ReducedJohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, variant=None, quad_scheme=None):
        if degree != 1:
            raise ValueError("Reduced Johnson-Mercier only defined for degree=1")
        if variant is not None:
            raise ValueError(f"Reduced Johnson-Mercier does not have the {variant} variant")
        ref_el = ref_complex.get_parent()
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        ref_facet = ref_el.get_facet_element()
        Q = parse_quadrature_scheme(ref_facet, degree+1, quad_scheme)
        P1 = polynomial_set.ONPolynomialSet(ref_facet, degree, scale="orthonormal")
        P1_at_qpts = P1.tabulate(Q.get_points())[(0,)*(sd - 1)]
        # Barycentric coordinates on the facet with respect to the tangents t_i = v_i - v_0
        bary = ref_facet.compute_barycentric_coordinates(Q.get_points()).T
        # Position relative to the facet barycenter is sum_i t_i mu_i
        mu = bary[1:] - 1/sd
        # nu_j = lambda_j - lambda_0 satisfies int mu_i nu_j = 0 for i != j
        nu = bary[1:] - bary[:1]

        # Facet degrees of freedom
        constraints = {}
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            n = ref_el.compute_scaled_normal(f)
            thats = ref_el.compute_tangents(sd-1, f)
            Qf = FacetQuadratureRule(ref_el, sd-1, f, Q, avg=True)
            # Normal-normal moments against P1
            nodes.extend(TensorBidirectionalIntegralMoment(ref_el, n, n, Qf, phi) for phi in P1_at_qpts)
            # Normal-tangential moments against RM(F): translations and rotations.
            # In 3D, the translations are the rotated tangents n x t_i, which map like
            # the rotation n x y.
            taus = numpy.cross(n, thats) if sd == 3 else thats
            nodes.extend(TensorBidirectionalIntegralMoment(ref_el, n, tau, Qf, P1_at_qpts[0]) for tau in taus)
            if sd == 3:
                y = numpy.dot(thats.T, mu)
                rot = numpy.cross(numpy.cross(*thats)[:, None], y, axis=0)
                nodes.append(FrobeniusIntegralMoment(ref_el, Qf, rot[:, None, :] * n[None, :, None]))
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

            # Normal-tangential moments against the complement of RM(F), the
            # symmetric fields t_i nu_j + t_j nu_i, whose moments give the facet strain
            phis = [numpy.outer(thats[i], nu[j]) + numpy.outer(thats[j], nu[i])
                    for i in range(sd-1) for j in range(i, sd-1)]
            constraints[f] = [FrobeniusIntegralMoment(ref_el, Qf, phi[:, None, :] * n[None, :, None])
                              for phi in phis]

        # The test functions of the constraints map contravariantly, as the L2
        # complements of the covariant spaces RM(F) and RM(T)
        mapping = [None] * len(nodes)

        # Facet constraints
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            nodes.extend(constraints[f])
            mapping.extend(["contravariant covariant piola"] * len(constraints[f]))
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Interior constraints: moments of divergence against the complement of RM(T)
        # in P0(Alfeld), spanned by S grad(lambda_i) on the subcell opposite to vertex i,
        # for each symmetric matrix S
        Q = parse_quadrature_scheme(ref_complex, 2*(degree-1), quad_scheme)
        qpts = Q.get_points()
        grad_bary = make_affine_mapping(ref_el.get_vertices(), numpy.eye(sd+1))[0]
        subcells = numpy.argmin(ref_el.compute_barycentric_coordinates(qpts), axis=1)
        grads = grad_bary[subcells].T
        cur = len(nodes)
        for i in range(sd):
            for j in range(i, sd):
                phi = numpy.zeros((sd, len(qpts)))
                phi[i] += grads[j]
                phi[j] += grads[i]
                nodes.append(IntegralMomentOfTensorDivergence(ref_el, Q, phi))
                mapping.append("contravariant piola")
        entity_ids[sd][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids, mapping=mapping)


class ReducedJohnsonMercier(finite_element.CiarletElement):
    """The Reduced Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None, quad_scheme=None):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = ReducedJohnsonMercierDualSet(ref_complex, degree, variant=variant, quad_scheme=quad_scheme)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
