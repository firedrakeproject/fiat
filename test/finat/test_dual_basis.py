import pytest
import numpy
import finat
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

@pytest.mark.parametrize("element,dim,order,expected", [
    (finat.Lagrange, 2, 1, ((0, 0),)),
    (finat.Lagrange, 3, 1, ((0, 0, 0),)),
    (finat.Hermite, 2, 3, ((0, 0), (0, 1), (1, 0))),
    (finat.Hermite, 3, 3, ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))),
    (finat.Argyris, 2, 5, ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))),
    (finat.Bell, 2, 5, ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))),
    (finat.ReducedHsiehCloughTocher, 2, 3, ((0, 0), (0, 1), (1, 0))),
    (finat.Morley, 2, 2, ((0, 0), (0, 1), (1, 0))),
    (finat.RaviartThomas, 2, 1, ((0, 0),))
])
def test_dual_basis_derivative_multiindices(element, dim, order, expected):
    cell = ufc_simplex(dim)
    fe = element(cell, order)
    assert fe._dual_basis_derivative_multiindices == expected