import pytest
import numpy as np

from FIAT import ufc_simplex, HuZhang, expansions
from FIAT.quadrature_schemes import create_quadrature


def test_dofs():
    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.asarray([(0.0, 0.0), (1.0, 0.25), (-0.75, 1.1)])
    HZ = HuZhang(T, 3)

    # check Kronecker property at vertices

    bases = [[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]]

    vert_vals = HZ.tabulate(0, T.vertices)[(0, 0)]
    for i in range(3):
        for j in range(3):
            assert np.allclose(vert_vals[3*i+j, :, :, i], bases[j])
            for k in (1, 2):
                assert np.allclose(vert_vals[3*i+j, :, :, (i+k) % 3], 0)

    # check edge moments
    Qline = create_quadrature(line, 6)

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    # n, n moments
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = HZ.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nnvals = np.zeros((30, nqpline))
        for i in range(30):
            for j in range(len(wts)):
                nnvals[i, j] = n @ vals[i, :, :, j] @ n

        nnmoments = np.zeros((30, 2))

        for bf in range(30):
            for k in range(nqpline):
                for m in (0, 1):
                    nnmoments[bf, m] += wts[k] * nnvals[bf, k] * linevals[m, k]

        for bf in range(30):
            if bf != HZ.dual.entity_ids[1][ed][0] and bf != HZ.dual.entity_ids[1][ed][2]:
                assert np.allclose(nnmoments[bf, :], 0)

    # n, t moments
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        t = T.compute_edge_tangent(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = HZ.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        ntvals = np.zeros((30, nqpline))
        for i in range(30):
            for j in range(len(wts)):
                ntvals[i, j] = n @ vals[i, :, :, j] @ t

        ntmoments = np.zeros((30, 2))

        for bf in range(30):
            for k in range(nqpline):
                for m in (0, 1):
                    ntmoments[bf, m] += wts[k] * ntvals[bf, k] * linevals[m, k]

        for bf in range(30):
            if bf != HZ.dual.entity_ids[1][ed][1] and bf != HZ.dual.entity_ids[1][ed][3]:
                assert np.allclose(ntmoments[bf, :], 0)

    # check internal dofs
    Q = create_quadrature(T, 6)
    qpvals = HZ.tabulate(0, Q.pts)[(0, 0)]
    const_moms = qpvals @ Q.wts
    assert np.allclose(const_moms[:21], 0)
    assert np.allclose(const_moms[24:], 0)
    assert np.allclose(const_moms[21:24, 0, 0], [1, 0, 0])
    assert np.allclose(const_moms[21:24, 0, 1], [0, 1, 0])
    assert np.allclose(const_moms[21:24, 1, 0], [0, 1, 0])
    assert np.allclose(const_moms[21:24, 1, 1], [0, 0, 1])


def frob(a, b):
    return a.ravel() @ b.ravel()


@pytest.mark.parametrize("variant", ("integral", "point"))
def test_projection(variant):
    T = ufc_simplex(2)
    T.vertices = np.asarray([(0.0, 0.0), (1.0, 0.0), (0.5, 2.1)])

    p = 3
    HZ = HuZhang(T, p, variant)

    Q = create_quadrature(T, 6)
    qpts = np.asarray(Q.pts)
    qwts = np.asarray(Q.wts)
    nqp = len(Q.wts)

    nbf = HZ.space_dimension() - 3 * (p-1)
    rhs_vals = np.zeros((1, 2, 2, nqp))

    bfvals = HZ.tabulate(0, qpts)[(0, 0)][:nbf, :, :, :]
    ells = np.multiply(bfvals, qwts)
    m = np.tensordot(ells, bfvals, (range(1, ells.ndim),)*2)

    assert np.linalg.cond(m) < 1.e12

    comps = [(0, 0), (0, 1), (1, 1)]

    # loop over monomials up to degree 2
    for deg in range(3):
        for jj in range(deg+1):
            ii = deg-jj
            for comp in comps:
                # set RHS (symmetrically) to be the monomial in
                # the proper component.
                rhs_vals[...] = 0
                rhs_vals[0][comp] = qpts[:, 0]**ii * qpts[:, 1]**jj
                rhs_vals[0][tuple(reversed(comp))] = rhs_vals[0][comp]

                b = np.tensordot(ells, rhs_vals, (range(1, ells.ndim),)*2)
                x = np.linalg.solve(m, b)

                sol_at_qpts = np.tensordot(x, bfvals, (0, 0))

                diff = (sol_at_qpts - rhs_vals)**2
                err = np.linalg.norm(np.tensordot(diff, qwts, (-1, -1))[0], "fro")
                assert np.sqrt(err) < 1.e-12
