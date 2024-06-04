from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.functional import IntegralMoment, FrobeniusIntegralMoment, IntegralMomentOfTensorDivergence
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.nedelec_second_kind import NedelecSecondKind as N2curl
import numpy


class JohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, variant=None):
        if degree != 1:
            raise ValueError("Johnson-Mercier only defined for degree=1")
        ref_el = ref_complex.get_parent()
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # Face dofs: bidirectional (nn and nt) Legendre moments
        dim = sd - 1
        ref_facet = ref_el.construct_subelement(dim)
        Qref = create_quadrature(ref_facet, 2*degree)
        P = polynomial_set.ONPolynomialSet(ref_facet, degree)
        phis = P.tabulate(Qref.get_points())[(0,) * dim]
        for facet in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, facet, Qref)
            Jdet = Q.jacobian_determinant()
            tangents = ref_el.compute_tangents(dim, facet)
            normal = ref_el.compute_normal(facet)
            normal /= numpy.linalg.norm(normal)
            scaled_normal = normal * Jdet
            uvecs = (scaled_normal, *tangents)
            comps = [numpy.outer(normal, uvec) for uvec in uvecs]
            nodes.extend(FrobeniusIntegralMoment(ref_el, Q, comp[:, :, None] * phi[None, None, :])
                         for phi in phis for comp in comps)
            entity_ids[dim][facet].extend(range(cur, len(nodes)))

        cur = len(nodes)
        if variant == "divergence":
            # Interior dofs: moments of divergence against the orthogonal complement of RBMs
            Q = create_quadrature(ref_complex, 2*(degree-1))
            qpts, qwts = Q.get_points(), Q.get_weights()

            N2 = N2curl(ref_el, degree)
            edofs = N2.entity_dofs()
            rbm_indices = [edofs[1][entity][0] for entity in edofs[1]]
            indices = numpy.setdiff1d(range(N2.space_dimension()), rbm_indices)

            N2_at_qpts = N2.tabulate(1, qpts)[(0,) * sd]
            rbms = N2_at_qpts[rbm_indices]
            ells = rbms * qwts[None, None, :]
            M = numpy.tensordot(ells, N2_at_qpts, axes=((1, 2), (1, 2)))
            C = numpy.linalg.solve(M[:, rbm_indices], M[:, indices])

            phis = N2_at_qpts[indices]
            phis -= numpy.tensordot(C, rbms, axes=(0, 0))
            nodes.extend(IntegralMomentOfTensorDivergence(ref_el, Q, phi) for phi in phis)
        else:
            # Interior dofs: moments for each independent component
            Q = create_quadrature(ref_complex, 2*degree-1)
            P = polynomial_set.ONPolynomialSet(ref_el, degree-1)
            phis = P.tabulate(Q.get_points())[(0,) * sd]
            nodes.extend(IntegralMoment(ref_el, Q, phi, comp=(i, j))
                         for j in range(sd) for i in range(j+1) for phi in phis)

        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super(JohnsonMercierDualSet, self).__init__(nodes, ref_el, entity_ids)


class JohnsonMercier(finite_element.CiarletElement):
    """The Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = JohnsonMercierDualSet(ref_complex, degree, variant=variant)
        mapping = "double contravariant piola"
        super(JohnsonMercier, self).__init__(poly_set, dual, degree,
                                             mapping=mapping)
