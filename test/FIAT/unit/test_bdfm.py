import pytest
import numpy
from FIAT import BrezziDouglasFortinMarini
from FIAT.reference_element import ufc_simplex
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.expansions import polynomial_dimension
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


@pytest.fixture(params=[2, 3])
def cell(request):
    dim = request.param
    K = ufc_simplex(dim)
    return K


def inner(u, v, qwts):
    return numpy.tensordot(numpy.multiply(u, qwts), v, axes=(tuple(range(1, u.ndim)), )*2)


@pytest.mark.parametrize("degree", (2, 4))
@pytest.mark.parametrize("variant", ("point", "integral", "integral(1)"))
def test_bdfm_space(cell, degree, variant):

    fe = BrezziDouglasFortinMarini(cell, degree, variant=variant)

    sd = cell.get_spatial_dimension()
    ref_face = cell.construct_subelement(sd-1)
    Q_face = create_quadrature(ref_face, 2*degree)

    dimPk = polynomial_dimension(ref_face, degree)
    dimPkm1 = polynomial_dimension(ref_face, degree-1)
    Phis = ONPolynomialSet(ref_face, degree)
    Phis = Phis.take(range(dimPkm1, dimPk))
    phis_at_qpts = Phis.tabulate(Q_face.get_points())[(0,)*(sd-1)]

    expected_dim = sd*polynomial_dimension(cell, degree) - (sd+1)*len(Phis)
    assert fe.space_dimension() == expected_dim

    for f in cell.topology[sd-1]:
        Q = FacetQuadratureRule(cell, sd-1, f, Q_face)
        vals = fe.tabulate(0, Q.get_points())[(0,)*sd]

        # Assert that the normal component is one degree lower
        n = cell.compute_normal(f)
        normal_trace = numpy.tensordot(vals, n, axes=(1, 0))
        normal_moments = inner(normal_trace, phis_at_qpts, Q.get_weights())
        assert numpy.allclose(normal_moments, 0)

        # Assert that the tangential component has the expected degree
        for t in cell.compute_tangents(sd-1, f):
            tangential_trace = numpy.tensordot(vals, t, axes=(1, 0))
            tangential_moments = inner(tangential_trace, phis_at_qpts, Q.get_weights())
            assert not numpy.allclose(tangential_moments, 0)
