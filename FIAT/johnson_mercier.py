from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.functional import (FrobeniusIntegralMoment, IntegralMomentOfDivergenceStrain,
                             IntegralMomentOfTractionStrain, TensorBidirectionalIntegralMoment)
from FIAT.quadrature import FacetQuadratureRule
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
        # The facet strain of a linear traction is constant
        Q0 = parse_quadrature_scheme(ref_facet, 0, quad_scheme)
        if sd == 3:
            # Position relative to the facet barycenter, sum_i t_i mu_i
            bary = ref_facet.compute_barycentric_coordinates(Q.get_points()).T
            mu = bary[1:] - 1/sd

        # Facet degrees of freedom
        strains = {}
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

            # The facet strain of the traction vanishes exactly when its
            # tangential part lies in RM(F)
            strains[f] = [IntegralMomentOfTractionStrain(ref_el, f, Q0, numpy.outer(thats[i], thats[j]) + numpy.outer(thats[j], thats[i]))
                          for i in range(sd-1) for j in range(i, sd-1)]

        for f in sorted(top[sd-1]):
            cur = len(nodes)
            nodes.extend(strains[f])
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Interior degrees of freedom: the strain of I_T div(sigma), which vanishes
        # exactly when div(sigma) lies in P_T RM(T)
        cur = len(nodes)
        I = numpy.eye(sd)
        nodes.extend(IntegralMomentOfDivergenceStrain(ref_complex, numpy.outer(I[i], I[j]) + numpy.outer(I[j], I[i]))
                     for i in range(sd) for j in range(i, sd))
        entity_ids[sd][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class ReducedJohnsonMercier(finite_element.CiarletElement):
    """The Reduced Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None, quad_scheme=None):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = ReducedJohnsonMercierDualSet(ref_complex, degree, variant=variant, quad_scheme=quad_scheme)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
