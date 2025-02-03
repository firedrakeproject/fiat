from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.functional import TensorBidirectionalIntegralMoment
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
import numpy


class JohnsonMercierDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, variant=None):
        if variant is not None:
            raise ValueError(f"Johnson-Mercier does not have the {variant} variant")
        ref_el = ref_complex.get_parent()
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        nodes = []

        # Exterior face dofs: bidirectional (nn and nt) Legendre moments
        dim = sd-1
        q = degree
        R = numpy.array([[0, 1], [-1, 0]])
        ref_facet = ref_el.construct_subelement(dim)
        Qref = create_quadrature(ref_facet, degree + q)
        P = polynomial_set.ONPolynomialSet(ref_facet, q)
        phis = P.tabulate(Qref.get_points())[(0,) * dim]
        for f in sorted(top[dim]):
            cur = len(nodes)
            Q = FacetQuadratureRule(ref_el, dim, f, Qref)
            thats = ref_el.compute_tangents(dim, f)
            nhat = numpy.dot(R, *thats) if sd == 2 else numpy.cross(*thats)
            normal = nhat / Q.jacobian_determinant()
            nodes.extend(TensorBidirectionalIntegralMoment(ref_el, normal, comp, Q, phi)
                         for phi in phis for comp in (nhat, *thats))
            entity_ids[dim][f].extend(range(cur, len(nodes)))

        cur = len(nodes)
        n = list(map(ref_complex.compute_scaled_normal, sorted(ref_complex.topology[sd-1])))

        # Interior facet dofs: bidirectional (nn and nt) Legendre moments
        for dim in range(sd):
            q = 0 if dim == 0 else degree - dim - 1
            if q < 0:
                continue
            ref_facet = ref_el.construct_subelement(dim)
            Qref = create_quadrature(ref_facet, degree + q)
            P = polynomial_set.ONPolynomialSet(ref_facet, q)
            phis = P.tabulate(Qref.get_points())[(0,) * dim]
            for f in ref_complex.get_interior_facets(dim):
                Q = FacetQuadratureRule(ref_complex, dim, f, Qref)
                scale = 1.0 / Q.jacobian_determinant()
                normals = [n[i] for i in ref_complex.connectivity[(dim, sd-1)][f]]
                nodes.extend(TensorBidirectionalIntegralMoment(ref_el, nf, nf, Q, phi * scale)
                             for phi in phis for nf in normals)

                thats = ref_complex.compute_tangents(dim, f)
                nodes.extend(TensorBidirectionalIntegralMoment(ref_el, nf, tf, Q, phi * scale)
                             for phi in phis for nf in normals[:sd-dim] for tf in thats)

        # Subcell dofs: Moments of unique components against a basis for P_{k-2}
        dim = sd
        q = degree - 2
        if q >= 0:
            normals = [n[i] for i in ref_complex.get_interior_facets(sd-1)]
            ref_facet = ref_el.construct_subelement(dim)
            Qref = create_quadrature(ref_facet, degree + q)
            P = polynomial_set.ONPolynomialSet(ref_facet, q)
            phis = P.tabulate(Qref.get_points())[(0,) * dim]
            for f in ref_complex.get_interior_facets(dim):
                Q = FacetQuadratureRule(ref_complex, dim, f, Qref)
                scale = 1.0 / Q.jacobian_determinant()
                nodes.extend(TensorBidirectionalIntegralMoment(ref_el, nf, nf, Q, phi * scale)
                             for phi in phis for nf in normals)

        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class JohnsonMercier(finite_element.CiarletElement):
    """The Johnson-Mercier finite element."""

    def __init__(self, ref_el, degree=1, variant=None):
        ref_complex = macro.AlfeldSplit(ref_el)
        poly_set = macro.HDivSymPolynomialSet(ref_complex, degree)
        dual = JohnsonMercierDualSet(ref_complex, degree, variant=variant)
        formdegree = ref_el.get_spatial_dimension() - 1
        mapping = "double contravariant piola"
        super().__init__(poly_set, dual, degree, formdegree, mapping=mapping)
