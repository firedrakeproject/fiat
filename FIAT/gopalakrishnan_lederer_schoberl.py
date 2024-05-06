from FIAT import finite_element, polynomial_set, dual_set, functional
import numpy

from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.expansions import polynomial_entity_ids


def mask_facet_ids(ref_el, dim, constrained_ids, mask):
    closure_ids = dual_set.make_entity_closure_ids(ref_el, constrained_ids)
    mask.fill(1)
    for facet in closure_ids[dim]:
        mask[facet][..., closure_ids[dim][facet]] = 0
    indices = numpy.flatnonzero(mask)
    return indices


def make_polynomial_sets(ref_el, degree):
    if degree == 1:
        raise NotImplementedError("TODO")
    sd = ref_el.get_spatial_dimension()
    phi = polynomial_set.TracelessTensorPolynomialSet(ref_el, degree, variant="bubble")
    expansion_set = phi.get_expansion_set()
    entity_ids = polynomial_entity_ids(ref_el, degree, continuity=expansion_set.continuity)
    mask = numpy.ones((phi.get_num_members(),), int).reshape(sd+1, sd-1, -1)

    # extract bubbles
    bubble_ids = mask_facet_ids(ref_el, sd-1, entity_ids, mask)
    P_bubble = phi.take(bubble_ids)

    # build constrained space with normal-tangential component in P_{k-1}
    constrained_ids = {}
    for dim in entity_ids:
        constrained_ids[dim] = {}
        if dim == 0 or dim == sd:
            for entity in entity_ids[dim]:
                constrained_ids[dim][entity] = []
        else:
            dimPkm1 = len(ref_el.make_points(dim, 0, degree-1))
            for entity in entity_ids[dim]:
                constrained_ids[dim][entity] = entity_ids[dim][entity][dimPkm1:]

    indices = mask_facet_ids(ref_el, sd-1, constrained_ids, mask)
    Sigma = phi.take(indices)
    return P_bubble, Sigma


class GLSDualSet(dual_set.DualSet):

    def __init__(self, ref_el, degree, bubbles):
        FIM = functional.FrobeniusIntegralMoment
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        nodes = []
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # Face dofs: bidirectional nt Legendre moments
        dim = sd - 1
        ref_facet = ref_el.construct_subelement(dim)
        Qref = create_quadrature(ref_facet, 2*degree-1)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree-1)
        phis = P.tabulate(Qref.get_points())[(0,) * dim]

        for facet in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, facet, Qref)
            Jdet = Q.jacobian_determinant()
            tangents = ref_el.compute_normalized_tangents(dim, facet)
            normal = ref_el.compute_normal(facet)
            normal /= numpy.linalg.norm(normal)
            scaled_normal = normal / Jdet
            comps = [numpy.outer(that, scaled_normal) for that in tangents]
            nodes.extend(FIM(ref_el, Q, comp[:, :, None] * phi[None, None, :])
                         for phi in phis for comp in comps)
            entity_ids[dim][facet].extend(range(cur, len(nodes)))

        # Interior dofs: moments against nt bubbles
        Q = create_quadrature(ref_el, 2*degree)
        phis = bubbles.tabulate(Q.get_points())[(0,) * sd]
        cur = len(nodes)
        nodes.extend(FIM(ref_el, Q, phi) for phi in phis)
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super(GLSDualSet, self).__init__(nodes, ref_el, entity_ids)


class GopalakrishnanLedererSchoberl(finite_element.CiarletElement):

    def __init__(self, ref_el, degree):
        bubbles, poly_set = make_polynomial_sets(ref_el, degree)
        dual = GLSDualSet(ref_el, degree, bubbles)
        mapping = "covariant contravariant piola"
        super(GopalakrishnanLedererSchoberl, self).__init__(poly_set, dual, degree, mapping=mapping)
