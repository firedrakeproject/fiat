from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.check_format_variant import check_format_variant
from FIAT.functional import TensorBidirectionalIntegralMoment as BidirectionalMoment
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.restricted import RestrictedElement


class GLSDual(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        nodes = []
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # Face dofs: moments of normal-tangential components against a basis for P_k
        # Interior dofs: moments of normal-tangential components against a basis for P_{k-1}
        for dim in (sd-1, sd):
            q = degree + sd-1 - dim
            if q < 0:
                continue

            ref_facet = ref_el.construct_subelement(dim)
            Q_ref = create_quadrature(ref_facet, degree + q)
            P = polynomial_set.ONPolynomialSet(ref_facet, q, scale=1)
            phis = P.tabulate(Q_ref.get_points())[(0,) * dim]

            for entity in sorted(top[dim]):
                cur = len(nodes)
                Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                Jdet = Q.jacobian_determinant()
                for f in ref_el.get_connectivity()[(dim, sd-1)][entity]:
                    normal = ref_el.compute_scaled_normal(f)
                    tangents = ref_el.compute_tangents(sd-1, f)
                    n = normal / Jdet
                    nodes.extend(BidirectionalMoment(ref_el, t, n, Q, phi)
                                 for phi in phis for t in tangents)
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class GopalakrishnanLedererSchoberlSecondKind(finite_element.CiarletElement):
    """The GLS element used for the Mass-Conserving mixed Stress (MCS)
    formulation for Stokes flow with weakly imposed stress symmetry.

    GLS^2(k) is the space of trace-free polynomials of degree k with
    continuous normal-tangential components.

    Reference: https://doi.org/10.1137/19M1248960

    Notes
    -----
    This element does not include the bubbles required for inf-sup stability of
    the weak symmetry constraint.

    """
    def __init__(self, ref_el, degree, variant=None):

        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        assert variant == "integral"

        if splitting is not None:
            ref_el = splitting(ref_el)

        poly_set = polynomial_set.TracelessTensorPolynomialSet(ref_el, degree)
        dual = GLSDual(ref_el, degree)
        sd = ref_el.get_spatial_dimension()
        formdegree = (1, sd-1)
        mapping = "covariant contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)


def GopalakrishnanLedererSchoberlFirstKind(ref_el, degree, variant=None):
    """The GLS element used for the Mass-Conserving mixed Stress (MCS)
    formulation for Stokes flow.

    GLS^1(k) is the space of trace-free polynomials of degree k with
    continuous normal-tangential components of degree k-1.

    Reference: https://doi.org/10.1093/imanum/drz022
    """
    fe = GopalakrishnanLedererSchoberlSecondKind(ref_el, degree, variant=variant)
    entity_dofs = fe.entity_dofs()
    sd = ref_el.get_spatial_dimension()
    facet = ref_el.construct_subelement(sd-1)
    dimPkm1 = (sd-1)*expansions.polynomial_dimension(facet, degree-1)

    indices = []
    for f in sorted(entity_dofs[sd-1]):
        indices.extend(entity_dofs[sd-1][f][:dimPkm1])
    for cell in sorted(entity_dofs[sd]):
        indices.extend(entity_dofs[sd][cell])

    return RestrictedElement(fe, indices=indices)
