import pytest
import numpy
import finat
import gem
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


def test_enriched_element_dual_evaluation():
    cell = ufc_simplex(2)
    fe = finat.Lagrange(cell, 3)

    fe1 = finat.RestrictedElement(fe, restriction_domain="interior")
    fe2 = finat.RestrictedElement(fe, restriction_domain="facet")
    enriched = finat.EnrichedElement([fe1, fe2], is_nodal_enriched=True)

    # Check that calling dual_evaluation returns a valid Indexed expression
    fn = lambda x: gem.Literal(1.0)
    expr, indices = enriched.dual_evaluation(fn)
    assert isinstance(expr, gem.Indexed)
    assert isinstance(expr.children[0], gem.Concatenate)
    assert len(indices) == 1
    assert indices[0].extent == enriched.space_dimension()


def test_mixed_subelement_dual_evaluation():
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

    expression, indices = vector_component.dual_evaluation(fn)
    values = gem.ComponentTensor(expression, indices)
    result, = gem.interpreter.evaluate([values])

    assert numpy.allclose(result.arr, cell.vertices)
