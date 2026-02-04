import pytest

import ufl
import finat.ufl
import finat
from finat.element_factory import create_element, supported_elements


@pytest.fixture(params=["BDM",
                        "BDFM",
                        "Lagrange",
                        "N1curl",
                        "N2curl",
                        "RT",
                        "Regge"])
def triangle_names(request):
    return request.param


@pytest.fixture
def ufl_element(triangle_names):
    return finat.ufl.FiniteElement(triangle_names, ufl.triangle, 2)


def test_triangle_basic(ufl_element):
    element = create_element(ufl_element)
    assert isinstance(element, supported_elements[ufl_element.family()])


@pytest.fixture
def ufl_vector_element(triangle_names):
    return finat.ufl.VectorElement(triangle_names, ufl.triangle, 2)


def test_triangle_vector(ufl_element, ufl_vector_element):
    scalar = create_element(ufl_element)
    vector = create_element(ufl_vector_element)

    assert isinstance(vector, finat.TensorFiniteElement)
    assert scalar == vector.base_element


@pytest.fixture(params=["CG", "DG", "DG L2"])
def tensor_name(request):
    return request.param


@pytest.fixture(params=[ufl.interval, ufl.triangle,
                        ufl.quadrilateral],
                ids=lambda x: x.cellname)
def ufl_A(request, tensor_name):
    return finat.ufl.FiniteElement(tensor_name, request.param, 1)


@pytest.fixture
def ufl_B(tensor_name):
    return finat.ufl.FiniteElement(tensor_name, ufl.interval, 1)


def test_tensor_prod_simple(ufl_A, ufl_B):
    tensor_ufl = finat.ufl.TensorProductElement(ufl_A, ufl_B)

    tensor = create_element(tensor_ufl)
    A = create_element(ufl_A)
    B = create_element(ufl_B)

    assert isinstance(tensor, finat.TensorProductElement)

    assert tensor.factors == (A, B)


@pytest.mark.parametrize(('family', 'expected_cls'),
                         [('P', finat.GaussLobattoLegendre),
                          ('DP', finat.GaussLegendre),
                          ('DP L2', finat.GaussLegendre)])
def test_interval_variant_default(family, expected_cls):
    ufl_element = finat.ufl.FiniteElement(family, ufl.interval, 3)
    assert isinstance(create_element(ufl_element), expected_cls)


@pytest.mark.parametrize(('family', 'variant', 'expected_cls'),
                         [('P', 'equispaced', finat.Lagrange),
                          ('P', 'spectral', finat.GaussLobattoLegendre),
                          ('DP', 'equispaced', finat.DiscontinuousLagrange),
                          ('DP', 'spectral', finat.GaussLegendre),
                          ('DP L2', 'equispaced', finat.DiscontinuousLagrange),
                          ('DP L2', 'spectral', finat.GaussLegendre)])
def test_interval_variant(family, variant, expected_cls):
    ufl_element = finat.ufl.FiniteElement(family, ufl.interval, 3, variant=variant)
    assert isinstance(create_element(ufl_element), expected_cls)


@pytest.mark.parametrize('cell', [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize('family,degree,quad_scheme',
                         [('CR', 1, 'default'),
                          ('CR', 1, 'KMV(1)'),
                          ('CR', 1, 'KMV(2)'),
                          ('CR', 1, 'KMV(2),powell-sabin')])
def test_quad_scheme(cell, family, degree, quad_scheme):
    ufl_element = finat.ufl.FiniteElement(family, cell, degree, variant="integral", quad_scheme=quad_scheme)
    fe = create_element(ufl_element)
    Q, ps = fe.dual_basis
    assert fe.space_dimension() == fe.cell.get_spatial_dimension() + 1
    if quad_scheme in {'KMV(1)', 'default'}:
        assert len(ps.points) == fe.space_dimension()
    else:
        assert len(ps.points) > fe.space_dimension()


def test_triangle_variant_spectral():
    ufl_element = finat.ufl.FiniteElement('DP', ufl.triangle, 2, variant='spectral')
    create_element(ufl_element)


def test_triangle_variant_spectral_l2():
    ufl_element = finat.ufl.FiniteElement('DP L2', ufl.triangle, 2, variant='spectral')
    create_element(ufl_element)


def test_quadrilateral_variant_spectral_q():
    element = create_element(finat.ufl.FiniteElement('Q', ufl.quadrilateral, 3, variant='spectral'))
    assert isinstance(element.product.factors[0], finat.GaussLobattoLegendre)
    assert isinstance(element.product.factors[1], finat.GaussLobattoLegendre)


def test_quadrilateral_bernstein():
    element = create_element(finat.ufl.FiniteElement('Bernstein', ufl.quadrilateral, 3))
    assert isinstance(element.product.factors[0], finat.Bernstein)
    assert isinstance(element.product.factors[1], finat.Bernstein)


def test_quadrilateral_variant_spectral_dq():
    element = create_element(finat.ufl.FiniteElement('DQ', ufl.quadrilateral, 1, variant='spectral'))
    assert isinstance(element.product.factors[0], finat.GaussLegendre)
    assert isinstance(element.product.factors[1], finat.GaussLegendre)


def test_quadrilateral_variant_spectral_dq_l2():
    element = create_element(finat.ufl.FiniteElement('DQ L2', ufl.quadrilateral, 1, variant='spectral'))
    assert isinstance(element.product.factors[0], finat.GaussLegendre)
    assert isinstance(element.product.factors[1], finat.GaussLegendre)


@pytest.mark.parametrize("cell, degree",
                         [(ufl.triangle, p) for p in range(1, 7)]
                         + [(ufl.tetrahedron, p) for p in range(1, 4)])
def test_kmv(cell, degree):
    ufl_element = finat.ufl.FiniteElement('KMV', cell, degree)
    finat_element = create_element(ufl_element)
    assert ufl_element.degree() == degree
    assert ufl_element.embedded_superdegree == finat_element.degree
    assert (finat_element.degree > degree) or (degree == 1)


def test_cache_hit(ufl_element):
    A = create_element(ufl_element)
    B = create_element(ufl_element)

    assert A is B


def test_cache_hit_vector(ufl_vector_element):
    A = create_element(ufl_vector_element)
    B = create_element(ufl_vector_element)

    assert A is B


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
