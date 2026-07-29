from itertools import chain

import pytest
import numpy
import finat
import gem
import ufl
import finat.ufl
from finat.element_factory import create_element
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

    expr, indices = element.dual_evaluation(tabulate)
    result, = evaluate([gem.ComponentTensor(expr, indices + j)])
    n = element.space_dimension()
    assert numpy.allclose(result.arr.reshape(n, n), numpy.eye(n))


def test_enriched_element_dual_evaluation():
    cell = ufc_simplex(2)
    fe = finat.Lagrange(cell, 3)

    fe1 = finat.RestrictedElement(fe, restriction_domain="interior")
    fe2 = finat.RestrictedElement(fe, restriction_domain="facet")
    enriched = finat.EnrichedElement([fe1, fe2], is_nodal_enriched=True)

    fn = lambda x: gem.Literal(1.0)
    expr, indices = enriched.dual_evaluation(fn)
    assert len(indices) == 1
    assert indices[0].extent == enriched.space_dimension()

    check_nodal(enriched)


def test_enriched_element_as_tensor_product_factor():
    # Restricting an element on a tensor product cell to its facets makes
    # the restriction of each factor a factor of the result.  Those factors
    # are themselves EnrichedElements, so the tensor product is the direct
    # sum of the products of their blocks.
    interval = ufc_simplex(1)
    square = finat.TensorProductElement([finat.Lagrange(interval, 3)] * 2)
    restricted = finat.RestrictedElement(square, restriction_domain="facet")
    assert isinstance(restricted, finat.EnrichedElement)

    cube = finat.TensorProductElement([restricted, finat.Lagrange(interval, 3)])
    assert len(cube.sub_elements) == len(restricted.sub_elements) > 1
    assert sum(element.space_dimension() for element in cube.sub_elements) \
        == cube.space_dimension()
    check_nodal(cube)


@pytest.mark.parametrize("family", ("RTCE", "RTCF", "NCE", "NCF"))
@pytest.mark.parametrize("domain", ("interior", "facet"))
def test_restricted_hdivcurl_dual_basis(family, domain):
    # Restriction selects disjoint subsets of the DoFs, so a restricted
    # H(div)/H(curl) element stays nodal even where the blocks are not
    # orthogonal to each other, as several of them map to the same component.
    if family.startswith("RTC"):
        cell = ufl.quadrilateral
    else:
        cell = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
    element = create_element(finat.ufl.FiniteElement(family, cell, 2)[domain])
    check_nodal(element)

    # Each sub-element has a dual basis on its own points, and together they
    # account for every functional: this is the path a TensorProductElement
    # takes to reach its factors.  An element decomposes one level at a time,
    # so recurse to the ones that are not themselves a direct sum.
    def summands(e):
        if e.sub_elements == (e,):
            return (e,)
        return tuple(chain.from_iterable(map(summands, e.sub_elements)))

    sub_elements = summands(element)
    assert len(sub_elements) > 1
    assert sum(e.space_dimension() for e in sub_elements) == element.space_dimension()
    for e in sub_elements:
        Q, x = e.dual_basis
        assert Q.shape == e.index_shape + e.value_shape
        assert set(Q.free_indices) <= set(x.indices)
    assert len(element.dual_point_set.points) \
        == sum(len(e.dual_basis[1].points) for e in sub_elements)
