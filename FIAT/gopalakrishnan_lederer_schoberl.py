from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.functional import TensorBidirectionalIntegralMoment as BidirectionalMoment
from FIAT.restricted import RestrictedElement
import numpy


def TracelessTensorBasis(ref_el):
    """Return a basis for traceless tensors aligned with nt on each face of a reference element."""
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    dev = lambda S: S - (numpy.trace(S) / S.shape[0]) * numpy.eye(*S.shape)

    basis = numpy.zeros((len(top[sd-1]), sd-1, sd, sd))
    rts = ref_el.compute_tangents(sd, 0)
    rts = numpy.vstack((-sum(rts), rts))
    if sd == 2:
        R = numpy.array([[0, 1], [-1, 0]])
        for f in sorted(top[sd-1]):
            ids = [(f + s + 1) % (sd+1) for s in range(sd)]
            basis[f, 0] = dev(numpy.outer(rts[ids[0]], numpy.dot(R, rts[ids[1]])))
    else:
        for f in sorted(top[sd-1]):
            for i in range(sd-1):
                ids = [(f + (s+i) % sd + 1) % (sd+1) for s in range(sd)]
                basis[f, i] = dev(numpy.outer(rts[ids[0]], numpy.cross(*rts[ids[1:]])))
    return basis


class TracelessTensorPolynomialSet(polynomial_set.PolynomialSet):
    """Constructs an orthonormal basis for traceless-tensor-valued
    polynomials on a reference element.
    """
    def __init__(self, ref_el, degree, **kwargs):
        expansion_set = expansions.ExpansionSet(ref_el, **kwargs)

        sd = ref_el.get_spatial_dimension()
        shape = (sd, sd)
        num_exp_functions = expansion_set.get_num_members(degree)
        num_components = sd * sd - 1
        num_members = num_components * num_exp_functions
        embedded_degree = degree

        # set up coefficients for traceless tensors
        basis = TracelessTensorBasis(ref_el)
        coeffs = numpy.zeros((num_members, *shape, num_exp_functions))
        cur_bf = 0
        for S in basis.reshape(-1, *shape):
            for exp_bf in range(num_exp_functions):
                coeffs[cur_bf, :, :, exp_bf] = S
                cur_bf += 1

        super().__init__(ref_el, degree, embedded_degree, expansion_set, coeffs)


class GLSDual(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        nodes = []
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # Face dofs: moments of nt components against a basis for Pk
        ref_facet = ref_el.construct_subelement(sd-1)
        Qref = create_quadrature(ref_facet, 2*degree)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree)
        phis = P.tabulate(Qref.get_points())[(0,) * (sd-1)]
        for f in sorted(top[sd-1]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, sd-1, f, Qref)
            Jdet = Q.jacobian_determinant()
            normal = ref_el.compute_scaled_normal(f)
            tangents = ref_el.compute_tangents(sd-1, f)
            n = normal / Jdet
            nodes.extend(BidirectionalMoment(ref_el, t, n, Q, phi)
                         for phi in phis for t in tangents)
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Interior dofs: moments of nt components against a basis for P_{k-1}
        cur = len(nodes)
        Q = create_quadrature(ref_el, 2*degree-1)
        P = polynomial_set.ONPolynomialSet(ref_el, degree-1, scale="L2 piola")
        phis = P.tabulate(Q.get_points())[(0,) * sd]
        for f in sorted(top[sd-1]):
            n = ref_el.compute_scaled_normal(f)
            tangents = ref_el.compute_tangents(sd-1, f)
            nodes.extend(BidirectionalMoment(ref_el, t, n, Q, phi)
                         for phi in phis for t in tangents)
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class ExtendedGopalakrishnanLedererSchoberl(finite_element.CiarletElement):
    """The GLS element on the full space of trace-free polynomials.
    """
    def __init__(self, ref_el, degree):
        poly_set = TracelessTensorPolynomialSet(ref_el, degree)
        dual = GLSDual(ref_el, degree)
        sd = ref_el.get_spatial_dimension()
        formdegree = (1, sd-1)
        mapping = "covariant contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)


def GopalakrishnanLedererSchoberl(ref_el, degree):
    """The GLS element used for the MCS (Mass-Conserving mixed Stress) scheme.
    GLS(r) is the space of trace-free polynomials of degree r with
    continuous normal-tangential components of degree r-1.
    """
    fe = ExtendedGopalakrishnanLedererSchoberl(ref_el, degree)
    entity_dofs = fe.entity_dofs()
    sd = ref_el.get_spatial_dimension()
    dimPkm1 = (sd-1)*expansions.polynomial_dimension(ref_el.construct_subelement(sd-1), degree-1)
    indices = []
    for f in entity_dofs[sd-1]:
        indices.extend(entity_dofs[sd-1][f][:dimPkm1])
    indices.extend(entity_dofs[sd][0])
    return RestrictedElement(fe, indices=indices)
