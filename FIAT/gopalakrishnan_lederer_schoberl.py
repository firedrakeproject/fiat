from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
from FIAT.expansions import polynomial_entity_ids
from FIAT.functional import FrobeniusIntegralMoment as FIM
import numpy


def traceless_matrices(ref_el):
    """Returns a basis for traceless matrices on a reference element."""
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    verts = ref_el.get_vertices()
    v0 = numpy.array(verts[0])
    rts = [numpy.array(v1) - v0 for v1 in verts[1:]]
    rts.insert(0, -sum(rts))

    normalize = lambda u: u / numpy.linalg.norm(u)
    rts = list(map(normalize, rts))

    dev = lambda S: S - (numpy.trace(S) / S.shape[0]) * numpy.eye(*S.shape)
    basis = numpy.zeros((len(top[sd-1]), sd-1, sd, sd), "d")
    if sd == 2:
        R = numpy.array([[0, 1], [-1, 0]])
        for i in top[sd-1]:
            i1 = (i + 1) % (sd+1)
            i2 = (i + 2) % (sd+1)
            basis[i, 0] = dev(numpy.outer(rts[i1], R @ rts[i2]))
    elif sd == 3:
        for i in top[sd-1]:
            i1 = (i + 1) % (sd+1)
            i2 = (i + 2) % (sd+1)
            i3 = (i + 3) % (sd+1)
            for j in range(sd-1):
                if j > 0:
                    i1, i2, i3 = i2, i3, i1
                basis[i, j] = dev(numpy.outer(rts[i1], numpy.cross(rts[i2], rts[i3])))
    else:
        raise NotImplementedError("TODO")
    return basis


class TracelessTensorPolynomialSet(polynomial_set.PolynomialSet):
    """Constructs an orthonormal basis for traceless-tensor-valued
    polynomials on a reference element.
    """
    def __init__(self, ref_el, degree, size=None, **kwargs):
        expansion_set = expansions.ExpansionSet(ref_el, **kwargs)

        sd = ref_el.get_spatial_dimension()
        if size is None:
            size = sd

        shape = (size, size)
        num_exp_functions = expansion_set.get_num_members(degree)
        num_components = size * size - 1
        num_members = num_components * num_exp_functions
        embedded_degree = degree

        # set up coefficients for traceless tensors
        basis = traceless_matrices(ref_el)
        coeffs_shape = (num_members, *shape, num_exp_functions)
        coeffs = numpy.zeros(coeffs_shape, "d")
        cur_bf = 0
        for S in basis.reshape(-1, *shape):
            for exp_bf in range(num_exp_functions):
                coeffs[cur_bf, :, :, exp_bf] = S
                cur_bf += 1

        super().__init__(ref_el, degree, embedded_degree, expansion_set, coeffs)


def GLSSpace(ref_el, degree):
    """build constrained space with normal-tangential component in P_{k-1}"""
    sd = ref_el.get_spatial_dimension()
    P = TracelessTensorPolynomialSet(ref_el, degree, variant="bubble")
    expansion_set = P.get_expansion_set()
    if degree == 1:
        dimP1 = expansion_set.get_num_members(degree)
        coeffs = numpy.zeros((2*(sd+1)*(sd-1), sd+1, sd-1, dimP1))
        cur = 0
        for i, j in numpy.ndindex(coeffs.shape[1:3]):
            coeffs[cur, i, j, :] = 1
            cur += 1
            coeffs[cur, i, j, i] = 1
            cur += 1
        coeffs = coeffs.reshape(-1, P.get_num_members())
        coeffs = numpy.tensordot(coeffs, P.get_coeffs(), (1, 0))
        return polynomial_set.PolynomialSet(ref_el, degree, degree, expansion_set, coeffs)
    else:
        entity_ids = polynomial_entity_ids(ref_el, degree, continuity=expansion_set.continuity)
        mask = numpy.ones((P.get_num_members(),), int).reshape(sd+1, sd-1, -1)
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

        closure_ids = dual_set.make_entity_closure_ids(ref_el, constrained_ids)
        for facet in closure_ids[sd-1]:
            mask[facet, ..., closure_ids[sd-1][facet]] = 0
        indices = numpy.flatnonzero(mask)
        return P.take(indices)


class GLSDual(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        nodes = []
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # Face dofs: bidirectional nt Legendre moments
        ref_facet = ref_el.construct_subelement(sd-1)
        Qref = create_quadrature(ref_facet, 2*degree-1)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree-1)
        phis = P.tabulate(Qref.get_points())[(0,) * (sd-1)]

        for f in sorted(top[sd-1]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, sd-1, f, Qref)
            Jdet = Q.jacobian_determinant()
            tangents = ref_el.compute_normalized_tangents(sd-1, f)
            normal = ref_el.compute_scaled_normal(f)
            normal /= Jdet
            comps = [numpy.outer(that, normal) for that in tangents]
            nodes.extend(FIM(ref_el, Q, comp[:, :, None] * phi[None, None, :])
                         for phi in phis for comp in comps)
            entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        # Interior dofs: moments against nt bubbles
        cur = len(nodes)
        Q = create_quadrature(ref_el, 2*degree)
        P = polynomial_set.ONPolynomialSet(ref_el, degree-1)
        Phis = P.tabulate(Q.get_points())[(0,) * sd]
        x = ref_el.compute_barycentric_coordinates(Q.get_points())
        basis = traceless_matrices(ref_el)
        for i, j in numpy.ndindex(basis.shape[:2]):
            comp = basis[i, j]
            phis = comp[None, :, :, None] * x[None, None, None, :, i] * Phis[:, None, None, :]
            nodes.extend(FIM(ref_el, Q, phi) for phi in phis)
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class GopalakrishnanLedererSchoberl(finite_element.CiarletElement):

    def __init__(self, ref_el, degree):
        poly_set = GLSSpace(ref_el, degree)
        dual = GLSDual(ref_el, degree)
        mapping = "covariant contravariant piola"
        super().__init__(poly_set, dual, degree, mapping=mapping)
