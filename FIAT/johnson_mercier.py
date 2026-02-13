from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.functional import TensorBidirectionalIntegralMoment
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
