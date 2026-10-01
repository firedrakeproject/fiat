import itertools

import numpy
import pytest

from FIAT.berchenko_kogan_gawlik import BerchenkoKoganGawlik
from FIAT.guzman_neilan import GuzmanNeilanFirstKindH1
from FIAT.quadrature_schemes import create_quadrature
from FIAT.rational import RationalExpansionSet, rational_integral
from FIAT.reference_element import ufc_simplex


def make_triangle(vertices=None):
    T = ufc_simplex(2)
    if vertices is not None:
        T.vertices = vertices
    return T


triangles = [None, ((0.1, 0.2), (1.3, 0.1), (0.4, 0.9)), ((0., 0.), (0., 1.), (2., 0.))]


def barycentric(T, pts):
    return T.compute_barycentric_coordinates(pts)


def gradients(T):
    verts = numpy.asarray(T.get_vertices())
    return numpy.linalg.inv(numpy.vstack((verts.T, numpy.ones(3))))[:, :2]


def interior_points(T, n=7):
    rng = numpy.random.default_rng(7)
    lam = rng.dirichlet(numpy.ones(3), n)
    return numpy.dot(lam, T.get_vertices())


@pytest.mark.parametrize("vertices", triangles)
def test_rational_derivatives(vertices):
    T = make_triangle(vertices)
    exponents = [[2, 1, 1, 2, 0, 0], [0, 2, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0]]
    es = RationalExpansionSet(T, exponents)
    pts = interior_points(T)
    tab = es._tabulate(0, pts, 2)
    h = 1E-5
    for k, e in enumerate(numpy.eye(2)):
        alpha = tuple(e.astype(int))
        lower = es._tabulate(0, pts - h*e, 1)
        upper = es._tabulate(0, pts + h*e, 1)
        for beta in lower:
            gamma = tuple(numpy.add(alpha, beta))
            assert numpy.allclose((upper[beta] - lower[beta]) / (2*h), tab[gamma], atol=1E-6)


@pytest.mark.parametrize("vertices", triangles)
@pytest.mark.parametrize("degree", (2, (4, 2), 4))
def test_rational_quadrature(vertices, degree):
    T = make_triangle(vertices)
    Q = create_quadrature(T, degree, scheme="rational")
    pts, wts = numpy.asarray(Q.get_points()), numpy.asarray(Q.get_weights())
    assert (wts > 0).all()
    lam = barycentric(T, pts)
    # Polynomial: the mean of lambda_0 lambda_1 is 1/12
    assert numpy.isclose(numpy.dot(wts, lam[:, 0] * lam[:, 1]), T.volume() / 12)
    # Single pole: the mean of lambda_1 lambda_2 / (1 - lambda_0)^2 is 1/6
    assert numpy.isclose(numpy.dot(wts, lam[:, 1] * lam[:, 2] / (1 - lam[:, 0])**2), T.volume() / 6)
    # Two poles: the mean of lambda_1 lambda_2 / ((1 - lambda_1) (1 - lambda_2)) is pi^2/3 - 3
    assert numpy.isclose(numpy.dot(wts, lam[:, 1] * lam[:, 2] / ((1 - lam[:, 1]) * (1 - lam[:, 2]))),
                         T.volume() * (numpy.pi**2 / 3 - 3))


def test_rational_integral():
    T = make_triangle()
    rng = numpy.random.default_rng(3)
    Q = create_quadrature(T, 40)
    pts, wts = numpy.asarray(Q.get_points()), numpy.asarray(Q.get_weights())
    lam = barycentric(T, pts)
    # Polynomials are integrated exactly by the default scheme
    for alpha in rng.integers(0, 5, (10, 3)):
        exact = rational_integral(T, [numpy.concatenate((alpha, (0, 0, 0)))])[0]
        assert numpy.isclose(exact, numpy.dot(wts, numpy.prod(lam ** alpha, axis=1)))
    # Non-integrable monomials are infinite
    assert numpy.isinf(rational_integral(T, [(1, 0, 0, 3, 0, 0)])[0])


@pytest.mark.parametrize("vertices", triangles)
def test_guzman_neilan_rational(vertices):
    T = make_triangle(vertices)
    fe = GuzmanNeilanFirstKindH1(T, 1, variant="rational")
    assert fe.space_dimension() == 12

    pts = create_quadrature(T, 6).get_points()
    tab = fe.tabulate(1, pts)

    # Contains P1^2
    lam = barycentric(T, pts)
    P1 = numpy.array([numpy.outer(e, l) for e in numpy.eye(2) for l in lam.T])
    coeffs = numpy.linalg.lstsq(tab[(0, 0)].reshape(12, -1).T, P1.reshape(6, -1).T, rcond=None)[0]
    assert numpy.allclose(numpy.dot(coeffs.T, tab[(0, 0)].reshape(12, -1)), P1.reshape(6, -1))

    # The divergence is constant
    div = tab[(1, 0)][:, 0] + tab[(0, 1)][:, 1]
    assert numpy.allclose(div, div[:, :1])

    # The trace on each edge is quadratic
    s = numpy.linspace(0, 1, 7)
    for edge, (i, j) in T.get_topology()[1].items():
        verts = numpy.asarray(T.get_vertices())
        edge_pts = numpy.outer(1 - s, verts[i]) + numpy.outer(s, verts[j])
        vals = fe.tabulate(0, edge_pts)[(0, 0)]
        fit = numpy.polynomial.polynomial.polyfit(s, vals.reshape(-1, len(s)).T, 2)
        assert numpy.allclose(numpy.polynomial.polynomial.polyval(s, fit), vals.reshape(-1, len(s)))


def blow_up_whitney_basis(T, k, pts):
    """The basis of (Berchenko-Kogan and Gawlik, 2024, Table 1)."""
    lam = barycentric(T, pts)
    G = gradients(T)
    edges = [T.get_topology()[1][e] for e in sorted(T.get_topology()[1])]
    if k == 0:
        return numpy.array([lam[:, i] * lam[:, j] / (1 - lam[:, i])
                            for edge in edges for i, j in itertools.permutations(edge)])
    phi = lambda i, j: lam[:, i, None] * G[j] - lam[:, j, None] * G[i]
    basis = [phi(i, j) for i, j in edges]
    for i in range(3):
        j, l = (m for m in range(3) if m != i)
        s = 1 - lam[:, i]
        basis.append((lam[:, i] * (1/s + 1/s**2))[:, None] * phi(j, l))
    return numpy.transpose(basis, (0, 2, 1))


@pytest.mark.parametrize("vertices", triangles)
@pytest.mark.parametrize("k", (0, 1))
def test_blow_up_whitney_basis(vertices, k):
    T = make_triangle(vertices)
    fe = BerchenkoKoganGawlik(T, k)
    pts = interior_points(T)
    assert numpy.allclose(fe.tabulate(0, pts)[(0, 0)], blow_up_whitney_basis(T, k, pts))


@pytest.mark.parametrize("vertices", triangles)
def test_blow_up_whitney_complex(vertices):
    T = make_triangle(vertices)
    pts = interior_points(T)
    V0, V1, V2 = (BerchenkoKoganGawlik(T, k).tabulate(1, pts) for k in range(3))

    # d of the 0-forms lies in the 1-forms
    grad = numpy.stack((V0[(1, 0)], V0[(0, 1)]), axis=1).reshape(6, -1)
    coeffs = numpy.linalg.lstsq(V1[(0, 0)].reshape(6, -1).T, grad.T, rcond=None)[0]
    assert numpy.allclose(numpy.dot(coeffs.T, V1[(0, 0)].reshape(6, -1)), grad)

    # d of the 1-forms is constant
    rot = V1[(1, 0)][:, 1] - V1[(0, 1)][:, 0]
    assert numpy.allclose(rot, rot[:, :1])
    assert numpy.allclose(V2[(0, 0)], 1 / T.volume())


def test_blow_up_whitney_mass():
    """Regression values for the 0-form mass matrix, as mean values."""
    T = make_triangle()
    Q = create_quadrature(T, (4, 2), scheme="rational")
    psi = blow_up_whitney_basis(T, 0, Q.get_points())
    wts = numpy.asarray(Q.get_weights()) / T.volume()
    mass = numpy.dot(psi * wts, psi.T)
    # Order: psi_012, psi_102 on edge 2; the row of psi_012 in the order 012, 021, 102, 120, 201, 210
    order = [4, 2, 5, 0, 3, 1]
    expected = [0.0556, 0.0278, 0.0399, 0.0157, 0.0157, 0.0121]
    assert numpy.allclose(mass[4][order], expected, atol=1E-4)


@pytest.mark.parametrize("element", (
    lambda T, degree: GuzmanNeilanFirstKindH1(T, degree, variant="rational"),
    lambda T, degree: BerchenkoKoganGawlik(T, 0, degree=degree),
    lambda T, degree: BerchenkoKoganGawlik(T, 1, degree=degree),
    lambda T, degree: BerchenkoKoganGawlik(T, 2, degree=degree),
))
@pytest.mark.parametrize("degree", (0, 2))
def test_unsupported_degree(element, degree):
    with pytest.raises(ValueError):
        element(make_triangle(), degree)


@pytest.mark.parametrize("vertices", triangles)
def test_blow_up_whitney_hdiv(vertices):
    T = make_triangle(vertices)
    pts = interior_points(T)
    curl = BerchenkoKoganGawlik(T, 1).tabulate(1, pts)
    div = BerchenkoKoganGawlik(T, 1, rotated=True).tabulate(1, pts)
    assert BerchenkoKoganGawlik(T, 1, rotated=True).mapping()[0] == "contravariant piola"
    # The basis is the rotated 1-form basis, (u_x, u_y) -> (u_y, -u_x)
    assert numpy.allclose(div[(0, 0)][:, 0], curl[(0, 0)][:, 1])
    assert numpy.allclose(div[(0, 0)][:, 1], -curl[(0, 0)][:, 0])
    # The divergence of the rotated proxy is the rot of the 1-form, a constant
    divergence = div[(1, 0)][:, 0] + div[(0, 1)][:, 1]
    assert numpy.allclose(divergence, curl[(1, 0)][:, 1] - curl[(0, 1)][:, 0])
    assert numpy.allclose(divergence, divergence[:, :1])
