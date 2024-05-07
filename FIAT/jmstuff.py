import numpy

import FIAT


# just 3d for now
def numerical_mapping(Tref, Tphys):
    V = numpy.eye(42, 42)
    J, b = FIAT.reference_element.make_affine_mapping(
        Tref.vertices, Tphys.vertices)
    detJ = numpy.linalg.det(J)

    # 3 linears per face and 3 components of sigma n -- 9 per face
    # plus 6 internal dofs

    for face in range(4):
        nhat = Tref.compute_normal(face)
        nhat /= numpy.linalg.norm(nhat)
        ehats = Tref.compute_tangents(2, face)
        rels = [numpy.linalg.norm(ehat) for ehat in ehats]
        thats = [a / b for a, b in zip(ehats, rels)]
        vf = Tref.volume_of_subcomplex(2, face)

        thingies = [numpy.cross(nhat, thats[1]),
                    numpy.cross(thats[0], nhat)]

        Jn = J @ nhat
        Jts = [J @ that for that in thats]
        Jthingies = [J @ thingy for thingy in thingies]

        mat = numpy.zeros((3, 3))
        mat[0, 0] = 1.0

        alphas = [rels[i] * Jn @ Jts[i] / detJ / vf / 2
                  for i in (0, 1)]
        betas = [Jthingies[0] @ Jts[i] / detJ / (thats[0] @ thingies[0])
                 for i in (0, 1)]
        gammas = [Jthingies[1] @ Jts[i] / detJ / (thats[1] @ thingies[1])
                  for i in (0, 1)]

        mat = numpy.array(
            [[1, 0, 0],
             [alphas[0], betas[0], gammas[0]],
             [alphas[1], betas[1], gammas[1]]])

        det = betas[0] * gammas[1] - betas[1] * gammas[0]

        matinv = numpy.zeros((3, 3))
        matinv[0, 0] = 1.0
        matinv[1, 0] = (alphas[1] * gammas[0] -
                        alphas[0] * gammas[1]) / det
        matinv[1, 1] = gammas[1] / det
        matinv[1, 2] = -gammas[0] / det
        matinv[2, 0] = (alphas[0] * betas[1] -
                        alphas[1] * betas[0]) / det
        matinv[2, 1] = -betas[1] / det
        matinv[2, 2] = betas[0] / det

        assert numpy.allclose(matinv, numpy.linalg.inv(mat))

        for p1 in range(3):
            offset = 9 * face + 3 * p1
            V[offset:offset+3, offset:offset+3] = numpy.linalg.inv(mat)
    return V


def mapping_by_proj(Tref, Tphys):
    J, b = FIAT.reference_element.make_affine_mapping(
        Tref.vertices, Tphys.vertices)
    detJ = numpy.linalg.det(J)

    Uref = FIAT.JohnsonMercier(Tref, 1)
    Uphys = FIAT.JohnsonMercier(Tphys, 1)

    Qref = FIAT.quadrature_schemes.create_quadrature(
        Uref.ref_complex, 2)

    Qphys = FIAT.quadrature_schemes.create_quadrature(
        Uphys.ref_complex, 2)
    z = (0, 0, 0)
    ref_vals = Uref.tabulate(0, Qref.get_points())[z]
    phys_vals = Uphys.tabulate(0, Qphys.get_points())[z]

    nbf = ref_vals.shape[0]
    print(f"{nbf}-dimensional space")
    M = numpy.zeros((nbf, nbf))
    Mmixed = numpy.zeros((nbf, nbf))

    nqp = len(Qref.get_points())
    phys_wts = Qphys.get_weights()

    ref_vals_piola = numpy.zeros(ref_vals.shape)
    for i in range(nbf):
        for k in range(nqp):
            ref_vals_piola[i, :, :, k] = \
                J @ ref_vals[i, :, :, k] @ J.T / detJ**2

    # mass matrix of Piola-mapped bfs (which aren't the physical ones!)
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nqp):
                M[i, j] += phys_wts[k] * numpy.tensordot(
                    ref_vals_piola[i, :, :, k],
                    ref_vals_piola[j, :, :, k])

    for i in range(nbf):
        for j in range(nbf):
            for k in range(nqp):
                Mmixed[i, j] += phys_wts[k] * numpy.tensordot(
                    phys_vals[i, :, :, k],
                    ref_vals_piola[j, :, :, k])

    boo = Mmixed @ numpy.linalg.inv(M)
    for i in range(nbf):
        for j in range(nbf):
            if abs(boo[i, j]) < 1.e-12:
                boo[i, j] = 0.0

    return boo.T


Tref = FIAT.ufc_simplex(3)
Tphys = FIAT.ufc_simplex(3)
Tphys.vertices = ((0, 0, 0), (1, 0.73, -.69), (0.37, 2, 0), (-1, 0.1, 1))

# get_mat()
V1 = numerical_mapping(Tref, Tphys)
V2 = mapping_by_proj(Tref, Tphys)

print(numpy.allclose(V1[:36, :36], V2[:36, :36]))
