from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.functional import IntegralMoment, FrobeniusIntegralMoment
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
import numpy


class JohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, variant=None):
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
        R = numpy.array([[0, 1], [-1, 0]])
        dim = sd - 1
        ref_facet = ref_el.construct_subelement(dim)
        Qref = create_quadrature(ref_facet, 2*degree)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree)
        phis = P.tabulate(Qref.get_points())[(0,) * dim]
        for facet in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, facet, Qref)
            thats = ref_el.compute_tangents(dim, facet)
            nhat = numpy.dot(R, *thats) if sd == 2 else numpy.cross(*thats)
            normal = nhat / Q.jacobian_determinant()

            uvecs = (nhat, *thats)
            comps = [numpy.outer(normal, uvec) for uvec in uvecs]
            nodes.extend(FrobeniusIntegralMoment(ref_el, Q, comp[:, :, None] * phi[None, None, :])
                         for phi in phis for comp in comps)
            entity_ids[dim][facet].extend(range(cur, len(nodes)))

        cur = len(nodes)
        if variant is None:
            # Interior dofs: moments for each independent component
            Q = create_quadrature(ref_complex, 2*degree-1)
            P = polynomial_set.ONPolynomialSet(ref_el, degree-1)
            phis = P.tabulate(Q.get_points())[(0,) * sd]
            nodes.extend(IntegralMoment(ref_el, Q, phi, comp=(i, j))
                         for j in range(sd) for i in range(j+1) for phi in phis)

        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class JohnsonMercier(finite_element.CiarletElement):
    """The Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = JohnsonMercierDualSet(ref_complex, degree, variant=variant)
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, mapping=mapping)
