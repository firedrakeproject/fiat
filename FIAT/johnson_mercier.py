from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.functional import FrobeniusIntegralMoment, IntegralMoment
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


class JohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, reduced=False):
        if degree != 1:
            raise ValueError("Johnson-Mercier only defined for degree=1")
        if reduced:
            raise NotImplementedError("TODO")
        ref_el = ref_complex.get_parent()
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        dim = sd - 1
        ref_facet = ref_el.construct_subelement(dim)
        Qref = create_quadrature(ref_facet, 2*degree)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree)
        phis = P.tabulate(Qref.get_points())[(0,) * dim]

        for facet in sorted(top[dim]):
            Q = FacetQuadratureRule(ref_el, dim, facet, Qref)
            Jdet = Q.jacobian_determinant()
            cur = len(nodes)
            tangents = ref_el.compute_normalized_tangents(dim, facet)
            normal = ref_el.compute_normal(facet)
            scaled_normal = normal / Jdet
            uvecs = (normal, *tangents)
            for uvec in uvecs:
                comp = uvec[:, None] * scaled_normal[None, :]
                cur_phis = comp[None, :, :, None] * phis[:, None, None, :]
                nodes.extend(FrobeniusIntegralMoment(ref_el, Q, phi) for phi in cur_phis)

            entity_ids[dim][facet].extend(range(cur, len(nodes)))

        Q = create_quadrature(ref_complex, 2*degree-1)
        P = polynomial_set.ONPolynomialSet(ref_el, degree-1)
        phis = P.tabulate(Q.get_points())[(0,) * sd]
        cur = len(nodes)
        nodes.extend(IntegralMoment(ref_el, Q, phi, comp=(i, j))
                     for j in range(sd) for i in range(j+1) for phi in phis)
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super(JohnsonMercierDualSet, self).__init__(nodes, ref_el, entity_ids)


class JohnsonMercier(finite_element.CiarletElement):
    """The Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, reduced=False):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = JohnsonMercierDualSet(ref_complex, degree, reduced=reduced)
        mapping = "double contravariant piola"
        super(JohnsonMercier, self).__init__(poly_set, dual, degree,
                                             mapping=mapping)
