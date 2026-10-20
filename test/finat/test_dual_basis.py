from itertools import chain

import FIAT
import pytest
import numpy
import finat
import gem
import ufl
import finat.ufl
from finat.element_factory import create_element
from finat.enriched import as_enriched
from finat.fiat_elements import FiatElement
from finat.physically_mapped import (PhysicallyMappedElement, cartesian_transformations,
                                     in_range_of, adjugate_and_determinant)
from finat.point_set import PointSet, UnionPointSet
from finat.quadrature import QuadratureRule
from finat.quadrature_element import QuadratureElement
from gem.interpreter import evaluate
from FIAT import ufc_simplex


@pytest.mark.parametrize("dim", (2, 3))
def test_collapse_repeated_points(dim):
    # Construct CR using face moments with a composite lumped scheme
    # Here the quadrature points lie on the ridges and we expect the dual
    # to collapse repeated points
    cell = ufc_simplex(dim)
    CR = finat.CrouzeixRaviart(cell, 1, variant="integral", quad_scheme="powell-sabin,KMV(2)")
    Q, ps = CR.dual_basis
    points = ps.points

    expected = 74 if dim == 3 else 12
    assert len(points) == len(numpy.unique(numpy.round(points, decimals=7), axis=0))
    assert len(points) == expected

    # Enrich by CG with DOFs that overlay on top of the quadrature rule
    CG = finat.Lagrange(cell, dim, variant="chebyshev")
    F = finat.RestrictedElement(CG, "ridge")
    fe = finat.NodalEnrichedElement([F, CR])
    Q, ps = fe.dual_basis
    points = ps.points

    assert len(points) == len(numpy.unique(numpy.round(points, decimals=7), axis=0))
    assert len(points) == expected


def check_nodal(element):
    """Assert that applying the dual basis to the primal basis is the identity."""
    j = element.get_indices()
    zeta = element.get_value_indices()
    dim = element.cell.get_spatial_dimension()

    def tabulate(ps):
        table = element.basis_evaluation(0, ps)[(0,) * dim]
        return gem.ComponentTensor(gem.Indexed(table, j + zeta), zeta)

    expr, point_indices, indices = element.dual_evaluation(tabulate)
    if point_indices:
        expr = gem.IndexSum(expr, point_indices)
    result, = evaluate([gem.ComponentTensor(expr, indices + j)])
    n = element.space_dimension()
    assert numpy.allclose(result.arr.reshape(n, n), numpy.eye(n))


def check_dual_basis(element):
    """Assert that contracting the dual weights with the primal basis is the identity."""
    Q, x = element.dual_basis
    assert Q.shape == element.index_shape + element.value_shape
    assert set(Q.free_indices) == set(x.indices)
    summands = as_enriched(element)
    if summands is not None:
        assert len(x.points) == sum(len(e.dual_basis[1].points)
                                    for e in summands._summands)

    i = element.get_indices()
    j = element.get_indices()
    zeta = element.get_value_indices()
    dim = element.cell.get_spatial_dimension()
    table = element.basis_evaluation(0, x)[(0,) * dim]
    expr = gem.IndexSum(gem.Product(gem.Indexed(Q, i + zeta),
                                    gem.Indexed(table, j + zeta)),
                        x.indices + zeta)
    result, = evaluate([gem.ComponentTensor(expr, i + j)])
    n = element.space_dimension()
    assert numpy.allclose(result.arr.reshape(n, n), numpy.eye(n))


def restricted_lagrange_sum(cell):
    """A direct sum of a cubic Lagrange element restricted to its interior
    and to its facets."""
    fe = finat.Lagrange(cell, 3)
    return finat.EnrichedElement(
        [finat.RestrictedElement(fe, restriction_domain=domain)
         for domain in ("interior", "facet")], is_nodal_enriched=True)


def test_enriched_element_dual_basis():
    # The weights of a direct sum are block diagonal: each summand's weights
    # sit at its own offset in the union of the points, and are zero against
    # every other summand's points.
    enriched = restricted_lagrange_sum(ufc_simplex(2))

    assert isinstance(enriched.dual_basis[1], UnionPointSet)
    check_dual_basis(enriched)


@pytest.mark.parametrize("repeated", (False, True))
def test_quadrature_element_on_union_of_points(repeated: bool) -> None:
    # Firedrake interpolates through a quadrature space on the points of the
    # target's dual basis, which for a direct sum is a union.  That element
    # has to evaluate on each point set of the union, just as the sum does.
    cell = ufc_simplex(2)
    enriched = restricted_lagrange_sum(cell)

    _, ps = enriched.dual_basis
    if repeated:
        ps = UnionPointSet([ps, PointSet(ps.point_sets[0].points.copy())])
    # The weights are not used, this quadrature scheme is not for integration.
    rule = QuadratureRule(ps, numpy.full(len(ps.points), numpy.nan), ref_el=cell)
    element = QuadratureElement(cell, rule)
    # A vector-valued target interpolates through a wrapper of that element,
    # which is distributed over the sum just the same.
    vector = finat.TensorFiniteElement(element, (cell.get_spatial_dimension(),))
    for e in (element, vector):
        check_nodal(e)
        # Each summand evaluates on its own points, so the callable is called
        # once per point set of the union rather than once on the union.
        seen = []

        def fn(point_set, e=e):
            seen.append(point_set)
            return gem.Literal(numpy.zeros(e.value_shape))

        e.dual_evaluation(fn)
        assert seen == list(ps.point_sets)


@pytest.mark.parametrize("family", ("RTCE", "RTCF", "NCE", "NCF"))
@pytest.mark.parametrize("degree", (1, 2))
def test_hdivcurl_dual_basis(family, degree):
    # A union of points is a point set like any other, so a tensor product
    # tabulates on it by splitting the coordinates and sharing the point
    # index, and the weights contract against that tabulation.
    element = create_element(finat.ufl.FiniteElement(family, hdivcurl_cell(family), degree))
    check_dual_basis(element)


def test_non_nodal_enriched_element_has_no_dual_basis():
    # MINI: the linear basis functions are not zero at the bubble's point, so
    # the functionals of the sum are not dual to its basis.  The weights are
    # still block diagonal, but contracting them gives the functionals, not
    # the coefficients, so the sum must not offer them as a dual basis.
    cell = ufc_simplex(2)
    mini = finat.EnrichedElement([finat.Lagrange(cell, 1), finat.Bubble(cell, 3)])

    assert not mini.is_nodal_enriched
    with pytest.raises(NotImplementedError):
        mini.dual_basis
    with pytest.raises(NotImplementedError):
        mini.dual_evaluation(lambda x: gem.Literal(1.0))
    assert not mini.has_pointwise_dual_basis


def test_integrated_legendre_dual_evaluation():
    element = finat.IntegratedLegendre(ufc_simplex(2), 4)
    check_nodal(element)


def test_restricted_integrated_legendre_dual_basis() -> None:
    element = finat.RestrictedElement(
        finat.IntegratedLegendre(ufc_simplex(1), 5), restriction_domain="interior")
    check_dual_basis(element)


def test_restricted_integrated_legendre_quadrature_entity_numbering() -> None:
    cell = ufc_simplex(2)
    element = finat.RestrictedElement(
        finat.IntegratedLegendre(cell, 3), restriction_domain="interior")
    _, points = element.dual_basis
    rule = QuadratureRule(points, numpy.full(len(points.points), numpy.nan), ref_el=cell)
    quadrature = QuadratureElement(cell, rule)

    entity_dofs = quadrature.entity_dofs()
    assert [len(entity_dofs[0][entity]) for entity in sorted(entity_dofs[0])] == [1, 1, 1]


def test_enriched_element_dual_evaluation():
    enriched = restricted_lagrange_sum(ufc_simplex(2))

    fn = lambda x: gem.Literal(1.0)
    expr, point_indices, basis_indices = enriched.dual_evaluation(fn)
    assert isinstance(expr, gem.Indexed)
    assert isinstance(expr.children[0], gem.Concatenate)
    assert len(basis_indices) == 1
    assert basis_indices[0].extent == enriched.space_dimension()

    check_nodal(enriched)


def test_enriched_element_as_tensor_product_factor():
    # Restricting an element on a tensor product cell to its facets makes
    # the restriction of each factor a factor of the result.  Those factors
    # are themselves EnrichedElements, so the tensor product is the direct
    # sum of the products of their summands.
    interval = ufc_simplex(1)
    square = finat.TensorProductElement([finat.Lagrange(interval, 3)] * 2)
    restricted = finat.RestrictedElement(square, restriction_domain="facet")
    assert isinstance(restricted, finat.EnrichedElement)

    cube = finat.TensorProductElement([restricted, finat.Lagrange(interval, 3)])
    expanded = as_enriched(cube)
    assert len(expanded.elements) == len(restricted.elements) > 1
    assert sum(element.space_dimension() for element in expanded.elements) \
        == cube.space_dimension()
    check_nodal(cube)


@pytest.fixture(scope="module")
def hexahedron():
    line = finat.Lagrange(ufc_simplex(1), 1)
    return finat.TensorProductElement([finat.TensorProductElement([line, line]), line])


def coefficient_evaluation(element, ps, dofs):
    """Evaluate a coefficient at a point set, sum factorised as TSFC does."""
    beta = element.get_indices()
    zeta = element.get_value_indices()
    dim = element.cell.get_spatial_dimension()
    table = element.basis_evaluation(0, ps)[(0,) * dim]
    dofs = gem.Literal(dofs.reshape([index.extent for index in beta]))
    value = gem.Product(gem.Indexed(table, beta + zeta), gem.Indexed(dofs, beta))
    return gem.ComponentTensor(gem.optimise.contraction(gem.IndexSum(value, beta)), zeta)


def nodal_values(element, fn):
    """Dual evaluate fn against a nodal element, giving its values at the nodes."""
    expression, point_indices, basis_indices = element.dual_evaluation(fn)
    indices = point_indices + basis_indices
    result, = evaluate([gem.ComponentTensor(expression, indices)])
    return result.arr


@pytest.mark.parametrize("power", (2, 3, 4))
def test_dual_evaluation_of_powers(hexahedron, power):
    # Each evaluation contracts over the three tensor-product directions, so
    # a product of them carries more indices than one sum factorisation can
    # search.  The evaluations are already factorised, so keep them that way.
    numpy.random.seed(0)
    dofs = numpy.random.rand(hexahedron.space_dimension())

    def evaluation(ps):
        return coefficient_evaluation(hexahedron, ps, dofs)

    def monomial(ps):
        expression = evaluation(ps)
        for _ in range(power - 1):
            expression = gem.Product(expression, evaluation(ps))
        return expression

    assert numpy.allclose(nodal_values(hexahedron, monomial),
                          nodal_values(hexahedron, evaluation) ** power)


def test_dual_evaluation_of_coupled_evaluations(hexahedron):
    # Contracting the value indices of two evaluations couples them into a
    # single contraction, which no ordering of the factors can break up.
    element = finat.TensorFiniteElement(hexahedron, (3,))
    numpy.random.seed(0)
    dofs = numpy.random.rand(element.space_dimension())

    def evaluation(ps):
        return coefficient_evaluation(element, ps, dofs)

    def cubed(ps):
        u = evaluation(ps)
        i, j = gem.Index(extent=3), gem.Index(extent=3)
        square = gem.IndexSum(gem.Product(gem.Indexed(u, (i,)), gem.Indexed(u, (i,))), (i,))
        return gem.ComponentTensor(gem.Product(square, gem.Indexed(u, (j,))), (j,))

    values = nodal_values(element, evaluation)
    expected = numpy.einsum("...i,...i->...", values, values)[..., None] * values
    assert numpy.allclose(nodal_values(element, cubed), expected)


def test_direct_sum_must_be_the_first_factor():
    # A sum in a later factor interleaves with the factors before it, so its
    # summands do not stack along the flat basis index.
    interval = ufc_simplex(1)
    line = finat.Lagrange(interval, 3)
    restricted = finat.RestrictedElement(
        finat.TensorProductElement([line] * 2), restriction_domain="facet")
    with pytest.raises(NotImplementedError):
        as_enriched(finat.TensorProductElement([line, restricted]))


def hdivcurl_cell(family):
    if family.startswith("RTC"):
        return ufl.quadrilateral
    return ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)


@pytest.mark.parametrize("family", ("RTCE", "RTCF", "NCE", "NCF"))
@pytest.mark.parametrize("degree", (1, 2, 3))
def test_hdivcurl_dual_evaluation(family, degree):
    # On a hexahedron one factor of a summand is itself a direct sum, which
    # only stacks once the sum is brought out through the product and the
    # pullback around it.
    element = create_element(finat.ufl.FiniteElement(family, hdivcurl_cell(family), degree))
    check_nodal(element)


@pytest.mark.parametrize("family", ("RTCE", "RTCF", "NCE", "NCF"))
@pytest.mark.parametrize("domain", ("interior", "facet"))
def test_restricted_hdivcurl_dual_basis(family, domain):
    # Restriction selects disjoint subsets of the DoFs, so a restricted
    # H(div)/H(curl) element stays nodal even where the summands are not
    # orthogonal to each other, as several of them map to the same component.
    element = create_element(finat.ufl.FiniteElement(family, hdivcurl_cell(family), 2)[domain])
    check_nodal(element)

    # Each summand has a dual basis on its own points, and together they
    # account for every functional.  Bringing the sum outermost rewrites one
    # level at a time, so recurse to the summands that are not sums themselves.
    def summands(e):
        expanded = as_enriched(e)
        if expanded is None:
            return (e,)
        return tuple(chain.from_iterable(map(summands, expanded.elements)))

    elements = summands(element)
    assert len(elements) > 1
    assert sum(e.space_dimension() for e in elements) == element.space_dimension()
    for e in elements:
        Q, x = e.dual_basis
        assert Q.shape == e.index_shape + e.value_shape
        assert set(Q.free_indices) <= set(x.indices)
    assert len(element.dual_basis[1].points) \
        == sum(len(e.dual_basis[1].points) for e in elements)


def test_mixed_subelement_dual_evaluation():
    # A mixed element is the direct sum of its components, so each component
    # dual evaluates against its own slice of the flattened value vector.
    cell = ufc_simplex(2)
    scalar = finat.Lagrange(cell, 1)
    vector = finat.TensorFiniteElement(scalar, (2,))
    mixed = finat.MixedElement([scalar, vector])
    vector_component = mixed.elements[1]

    def fn(points):
        coordinates = points.expression
        return gem.ListTensor([
            gem.Literal(0.0),
            gem.Indexed(coordinates, (0,)),
            gem.Indexed(coordinates, (1,)),
        ])

    expression, point_indices, basis_indices = vector_component.dual_evaluation(fn)
    result, = evaluate([gem.ComponentTensor(gem.IndexSum(expression, point_indices),
                                            basis_indices)])

    assert numpy.allclose(result.arr, cell.vertices)


class MappedElement(PhysicallyMappedElement, FiatElement):
    pass


def test_pairing_respects_tolerance():
    """Near-divergence weights are not paired using NumPy's default tolerance."""
    ref_el = FIAT.ufc_simplex(2)
    node = FIAT.functional.Functional(
        ref_el, (2,), {},
        {(0.0, 0.0): [(1.0, (1, 0), (0,)),
                      (1.0 + 1.0e-6, (0, 1), (1,))]},
        "NearDivergence")
    assert node.orders == (1,)
    _, W = node.get_order_tensors(1)
    delta = numpy.eye(2).ravel()
    P = numpy.outer(delta, delta) / 2
    assert in_range_of(numpy.eye(2)[..., None], P, (0, 1), 1.0e-12)
    assert not in_range_of(W, P, (0, 1), 1.0e-12)


def transformation_matrix(G):
    coefficients, d = (result.arr for result in evaluate([gem.ListTensor(G.coefficients), G.denominator]))
    return numpy.tensordot(coefficients, G.matrices, axes=1) / d


@pytest.mark.parametrize("sd", [2, 3])
def test_pairing_transformation(sd):
    """The Kronecker product of a contravariant axis and a covariant axis maps the identity as the pair."""
    J = numpy.eye(sd) + numpy.random.default_rng(0).random((sd, sd))
    adjJ, detJ = adjugate_and_determinant(numpy.array([[gem.Literal(Jij) for Jij in row] for row in J], dtype=object))
    J = numpy.array([[gem.Literal(Jij) for Jij in row] for row in J], dtype=object)
    transformations = cartesian_transformations(J, adjJ, detJ)
    contravariant, covariant, pair = (transformation_matrix(transformations[mapping])
                                      for mapping in ("contravariant piola", "covariant piola", "L2 piola"))
    J, = evaluate([gem.ListTensor(J)])
    J = J.arr
    assert numpy.allclose(contravariant, J.T / numpy.linalg.det(J))
    assert numpy.allclose(covariant, numpy.linalg.inv(J))
    delta = numpy.eye(sd).reshape(-1, 1)
    assert numpy.allclose(pair, numpy.kron(contravariant, covariant) @ delta)
    assert numpy.allclose(pair, delta / numpy.linalg.det(J))


@pytest.mark.parametrize("scaled", [False, True])
def test_recombined_physical_dual(ref_to_phys, scaled_ref_to_phys, scaled):
    element = FIAT.Hermite(FIAT.ufc_simplex(2), 3)
    C = numpy.eye(element.space_dimension())
    for dofs in element.entity_dofs()[0].values():
        C[numpy.ix_(dofs, dofs)] = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    basis = element.get_nodal_basis().recombine(numpy.linalg.inv(C).T)
    recombined = FIAT.finite_element.CiarletElement(
        basis, element.get_dual_set(), 3, recombine_dual=True)
    assert numpy.allclose(recombined.get_dual_set().get_coeffs(), C)
    original = MappedElement(element)
    transformed = MappedElement(recombined)
    mapping = scaled_ref_to_phys[2][1] if scaled else ref_to_phys[2]
    Q = gem.Literal(numpy.eye(len(C)))
    B, BC, M = [result.arr for result in evaluate([
        original.dual_transformation(Q, mapping), transformed.dual_transformation(Q, mapping),
        transformed.basis_transformation(mapping)])]
    assert numpy.allclose(BC, C @ B @ numpy.linalg.inv(C))
    assert numpy.allclose(BC @ M.T, numpy.eye(len(C)))


@pytest.mark.parametrize("avg", [False, True])
def test_single_point_moment(ref_to_phys, avg):
    """A normal derivative moment at a single point is still an integral moment."""
    element = MappedElement(FIAT.Morley(FIAT.ufc_simplex(2)))
    element.avg = avg
    dual = element.fiat_equivalent.get_dual_set()
    nodes = dual.get_nodes()
    edge_dofs = list(chain.from_iterable(dual.get_entity_ids()[1].values()))
    for i in edge_dofs:
        points, _ = nodes[i].get_order_tensors(1)
        assert len(points) == 1
        assert isinstance(nodes[i], FIAT.functional.IntegralMoment)

    # The plain integral of the physical normal derivative is its average times the edge length.
    averaged = MappedElement(element.fiat_equivalent)
    Q = gem.Literal(numpy.eye(element.space_dimension()))
    B, Bavg = [result.arr for result in evaluate([
        element.dual_transformation(Q, ref_to_phys[2]), averaged.dual_transformation(Q, ref_to_phys[2])])]
    lengths = [1.0 if avg else ref_to_phys[2].phys_cell.volume_of_subcomplex(1, e) for e in range(3)]
    assert numpy.allclose(B[edge_dofs], numpy.multiply(lengths, Bavg[edge_dofs].T).T)


def test_plain_integral_facet_moments(ref_to_phys):
    element = FIAT.CrouzeixRaviart(FIAT.ufc_simplex(2), 1, variant="integral")
    mapped = MappedElement(element)
    mapped.avg = False
    B, M = [result.arr for result in evaluate([
        mapped.dual_transformation(gem.Literal(numpy.eye(3)), ref_to_phys[2]), mapped.basis_transformation(ref_to_phys[2])])]
    lengths = [ref_to_phys[2].phys_cell.volume_of_subcomplex(1, e) for e in range(3)]
    assert numpy.allclose(B, numpy.diag(lengths))
    assert numpy.allclose(B @ M.T, numpy.eye(3))
