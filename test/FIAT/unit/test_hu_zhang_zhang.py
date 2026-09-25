import numpy
import pytest

from FIAT import GuzmanNeilanFirstKindH1, Lagrange
from FIAT.hu_zhang_zhang import HuZhangZhang, curl
from FIAT.reference_element import ufc_simplex


def span_greater_equal(A, B):
    # span(A) >= span(B)
    _, residual, *_ = numpy.linalg.lstsq(A.reshape(A.shape[0], -1).T,
                                         B.reshape(B.shape[0], -1).T)
    return numpy.allclose(residual, 0)


def make_cell(distorted):
    cell = ufc_simplex(3)
    if distorted:
        cell.vertices = ((0, 0, 0), (1., 0.1, -0.37), (0.01, 0.987, -.23), (-0.1, -0.2, 1.38))
    return cell


def interior_points(fe):
    ref_complex = fe.get_reference_complex()
    sd = ref_complex.get_spatial_dimension()
    return [pt for cell in sorted(ref_complex.get_topology()[sd])
            for pt in ref_complex.make_points(sd, cell, fe.degree() + sd + 1)]


@pytest.fixture(params=(False, True), ids=("reference", "distorted"))
def cell(request):
    return make_cell(request.param)


@pytest.fixture
def hzz(cell):
    return HuZhangZhang(cell)


def test_hzz_dofs(hzz):
    assert hzz.space_dimension() == 26
    counts = {dim: [len(dofs) for dofs in hzz.entity_dofs()[dim].values()]
              for dim in hzz.entity_dofs()}
    assert counts == {0: [3]*4, 1: [1]*6, 2: [2]*4, 3: [0]}


def test_hzz_stokes_complex(cell, hzz):
    # grad(CG1) <= HZZ, curl(HZZ) <= GN, and the kernel of curl is exactly grad(CG1)
    pts = interior_points(hzz)
    num_ext = hzz.space_dimension()
    num_red = num_ext - 8
    V = hzz.tabulate(1, pts)
    curlV = curl(V)

    CG = Lagrange(cell, 1).tabulate(1, pts)
    gradCG = numpy.stack([CG[alpha] for alpha in sorted(CG) if sum(alpha) == 1], axis=1)
    assert span_greater_equal(V[(0, 0, 0)], gradCG)

    GN = GuzmanNeilanFirstKindH1(cell, 1)
    W = GN.tabulate(0, pts)[(0, 0, 0)]
    assert span_greater_equal(W, curlV)
    assert numpy.linalg.matrix_rank(curlV.reshape(num_ext, -1), tol=1E-8) == num_ext - 3

    # The reduced space has curl in the reduced Guzman-Neilan space
    assert span_greater_equal(W[:16], curlV[:num_red])


def test_hzz_tangential_conformity(cell, hzz):
    # Basis functions not associated with the closure of a face have zero tangential trace
    top = cell.get_topology()
    closure_dofs = hzz.entity_closure_dofs()
    for f in sorted(top[2]):
        others = [i for i in range(hzz.space_dimension()) if i not in closure_dofs[2][f]]
        pts = cell.make_points(2, f, 6)
        n = cell.compute_normal(f)
        vals = hzz.tabulate(0, pts)[(0, 0, 0)][others]
        assert numpy.allclose(numpy.cross(vals.transpose(0, 2, 1), n), 0)
