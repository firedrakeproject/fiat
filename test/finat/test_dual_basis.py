import pytest
import numpy
import finat
import gem
import gem.driver
from FIAT import ufc_simplex
from gem.interpreter import evaluate


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


def test_enriched_element_dual_evaluation():
    cell = ufc_simplex(2)
    fe = finat.Lagrange(cell, 3)

    fe1 = finat.RestrictedElement(fe, restriction_domain="interior")
    fe2 = finat.RestrictedElement(fe, restriction_domain="facet")
    enriched = finat.EnrichedElement([fe1, fe2], is_nodal_enriched=True)

    # Check that calling dual_evaluation returns a valid Indexed expression
    fn = lambda x: gem.Literal(1.0)
    expr, point_indices, basis_indices = enriched.dual_evaluation(fn)
    assert isinstance(expr, gem.Indexed)
    assert isinstance(expr.children[0], gem.Concatenate)
    assert len(basis_indices) == 1
    assert basis_indices[0].extent == enriched.space_dimension()


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
    return gem.ComponentTensor(gem.driver.contraction(gem.IndexSum(value, beta)), zeta)


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
