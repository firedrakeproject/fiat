from itertools import chain

import pytest
import numpy
import finat
import gem
import ufl
import finat.ufl
from finat.element_factory import create_element
from finat.enriched import as_enriched
from finat.point_set import UnionPointSet
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


def test_enriched_element_dual_basis():
    # The weights of a direct sum are block diagonal: each summand's weights
    # sit at its own offset in the union of the points, and are zero against
    # every other summand's points.
    cell = ufc_simplex(2)
    fe = finat.Lagrange(cell, 3)
    enriched = finat.EnrichedElement(
        [finat.RestrictedElement(fe, restriction_domain=domain)
         for domain in ("interior", "facet")], is_nodal_enriched=True)

    assert isinstance(enriched.dual_basis[1], UnionPointSet)
    check_dual_basis(enriched)


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


def test_enriched_element_dual_evaluation():
    cell = ufc_simplex(2)
    fe = finat.Lagrange(cell, 3)

    fe1 = finat.RestrictedElement(fe, restriction_domain="interior")
    fe2 = finat.RestrictedElement(fe, restriction_domain="facet")
    enriched = finat.EnrichedElement([fe1, fe2], is_nodal_enriched=True)

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
    if family.startswith("RTC"):
        cell = ufl.quadrilateral
    else:
        cell = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
    element = create_element(finat.ufl.FiniteElement(family, cell, 2)[domain])
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
