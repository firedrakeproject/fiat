from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.functional import TensorBidirectionalIntegralMoment as BidirectionalMoment
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
import numpy


def inner(u, v, qwts):
    return numpy.tensordot(numpy.multiply(u, qwts), v,
                           axes=(range(1, u.ndim),
                                 range(1, v.ndim)))


def HLZSpace(ref_el, degree):
    sd = ref_el.get_spatial_dimension()
    k = degree - 1
    Q = create_quadrature(ref_el, 2*degree)
    qpts, qwts = Q.get_points(), Q.get_weights()
    x = qpts.T

    S = polynomial_set.ONSymTensorPolynomialSet(ref_el, k)
    T = polynomial_set.TracelessTensorPolynomialSet(ref_el, k+1)
    S_at_qpts = S.tabulate(qpts)[(0,)*sd]
    T_at_qpts = T.tabulate(qpts)[(0,)*sd]

    v = T_at_qpts
    SCrossX_at_qpts = numpy.cross(S_at_qpts, x[None, :, :], axis=-2)
    coeffs = numpy.linalg.solve(inner(v, v, qwts), inner(v, SCrossX_at_qpts, qwts))
    coeffs = numpy.tensordot(coeffs, T.get_coeffs(), axes=(0, 0))

    expansion_set = T.get_expansion_set()
    PCrossX = polynomial_set.PolynomialSet(ref_el, k + 1, k + 1,
                                           expansion_set,
                                           coeffs)
    dimPk = expansion_set.get_num_members(k)
    dimPkp1 = expansion_set.get_num_members(k+1)
    T = T.take([i+dimPkp1*j for j in range(sd**2-1) for i in range(dimPk)])
    return polynomial_set.polynomial_set_union_normalized(T, PCrossX)


class HLZDual(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        nodes = []
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # dim-facet dofs: moments of normal-tangential components against a basis for P_{k-dim}
        for dim in range(1, sd):
            if degree < dim:
                continue
            ref_facet = ref_el.construct_subelement(dim)
            Qref = create_quadrature(ref_facet, 2*degree-dim)
            P = polynomial_set.ONPolynomialSet(ref_facet, degree-dim)
            phis = P.tabulate(Qref.get_points())[(0,) * dim]
            for entity in sorted(top[dim]):
                cur = len(nodes)
                tangents = ref_el.compute_tangents(dim, entity)
                if dim == sd-1:
                    normals = ref_el.compute_scaled_normal(entity)[None, :]
                elif dim == 1:
                    normals = ref_el.compute_edge_normals(entity)
                else:
                    raise ValueError("Cannot compute normals")

                Q = FacetQuadratureRule(ref_el, dim, entity, Qref)
                normals /= Q.jacobian_determinant()
                nodes.extend(BidirectionalMoment(ref_el, t, n, Q, phi)
                             for phi in phis for t in tangents for n in normals)
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        # Interior dofs:
        cur = len(nodes)
        Q = create_quadrature(ref_el, 2*degree-1)
        P = polynomial_set.ONPolynomialSet(ref_el, degree-1, scale="L2 piola")
        phis = P.tabulate(Q.get_points())[(0,) * sd]
        # moments of normal-tangential components against a basis for P_{k-1}
        # get the tangent from two opposite edges, and
        # get the normal from the lowest face adjacent to each edge
        visit = []
        for e in sorted(top[1]):
            if any(v in visit for v in top[1][e]):
                continue
            face = min(f for f in top[2] if set(top[1][e]) < set(top[2][f]))
            n = ref_el.compute_scaled_normal(face)
            t = ref_el.compute_edge_tangent(e)
            nodes.extend(BidirectionalMoment(ref_el, t, n, Q, phi)
                         for phi in phis)
            visit.extend(top[1][e])

        # moments of normal-tangential components against a basis for P_{k-2}
        if degree > 1:
            dimPkm2 = expansions.polynomial_dimension(ref_el, degree-2)
            for f in sorted(top[sd-1]):
                n = ref_el.compute_scaled_normal(f)
                tangents = ref_el.compute_tangents(sd-1, f)
                nodes.extend(BidirectionalMoment(ref_el, t, n, Q, phi)
                             for phi in phis[:dimPkm2] for t in tangents)
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HuLinZhang(finite_element.CiarletElement):
    """
    HLZ(k) is the space of trace-free polynomials of degree k-1 + symeteric
    polynomials of degree k-1 cross x with continuous normal-tangential
    components.

    Reference: https://arxiv.org/abs/2311.15482
    """
    def __init__(self, ref_el, degree=1):
        poly_set = HLZSpace(ref_el, degree)
        dual = HLZDual(ref_el, degree)
        sd = ref_el.get_spatial_dimension()
        formdegree = (1, sd-1)
        mapping = "covariant contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
