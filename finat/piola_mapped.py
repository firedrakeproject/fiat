import numpy

from finat.fiat_elements import FiatElement
from finat.physically_mapped import adjugate, determinant, identity, PhysicallyMappedElement
from gem import Literal, ListTensor, Zero
from copy import deepcopy
from itertools import chain


def piola_inverse(fiat_cell, J, detJ):
    """Return the basis transformation of evaluation at a point.
    This simply inverts the Piola transform inv(J / detJ) = adj(J)."""
    sd = fiat_cell.get_spatial_dimension()
    Jnp = numpy.array([[J[i, j] for j in range(sd)] for i in range(sd)])
    return adjugate(Jnp)


def normal_tangential_edge_transform(fiat_cell, J, detJ, f):
    """Return the basis transformation of
    normal and tangential edge moments"""
    R = numpy.array([[0, 1], [-1, 0]])
    that = fiat_cell.compute_edge_tangent(f)
    that /= numpy.linalg.norm(that)
    nhat = R @ that
    Jn = J @ Literal(nhat)
    Jt = J @ Literal(that)
    alpha = Jn @ Jt
    beta = Jt @ Jt
    # Compute the last row of inv([[1, 0], [alpha/detJ, beta/detJ]])
    return (-1 * alpha / beta, detJ / beta)


def normal_tangential_face_transform(fiat_cell, J, detJ, f):
    """Return the basis transformation of
    normal and tangential face moments"""
    # Compute the reciprocal basis
    thats = fiat_cell.compute_tangents(2, f)
    nhat = numpy.cross(*thats)
    nhat /= numpy.dot(nhat, nhat)
    orths = numpy.cross(thats, nhat[None, :], axis=1)

    Jn = J @ Literal(nhat)
    Jthats = J @ Literal(thats.T)
    Jorths = J @ Literal(orths.T)
    A = Jthats.T @ Jorths
    B = Jn @ Jthats
    A = numpy.array([[A[i, j] for j in range(A.shape[1])] for i in range(A.shape[0])])
    B = numpy.array([B[i] for i in range(B.shape[0])])

    Q = numpy.dot(thats, thats.T)
    beta = determinant(A)
    alpha = Q @ (adjugate(A) @ B)
    return (alpha / beta, detJ / beta)


def normal_tangential_transform(fiat_cell, J, detJ, f):
    """Return the basis transformation of normal and tangential face moments

    :arg fiat_cell: a :class:`FIAT.reference_element.Cell`
    :arg J: the Jacobian of the coordinate transformation
    :arg detJ: the Jacobian determinant of the coordinate transformation
    :arg f: the face id.

    :returns: a 2-tuple of (Bnt, Btt) where
        Bnt is the numpy.ndarray of normal-tangential coefficients, and
        Btt is the tangential-tangential coefficient.
    """
    if fiat_cell.get_spatial_dimension() == 2:
        return normal_tangential_edge_transform(fiat_cell, J, detJ, f)
    else:
        return normal_tangential_face_transform(fiat_cell, J, detJ, f)


class PiolaBubbleElement(PhysicallyMappedElement, FiatElement):
    """A general class to transform Piola-mapped elements with normal facet bubbles."""
    def __init__(self, fiat_element):
        mapping, = set(fiat_element.mapping())
        if mapping != "contravariant piola":
            raise ValueError(f"{type(fiat_element).__name__} needs to be Piola mapped.")
        super().__init__(fiat_element)

        # On each facet we expect the normal dof followed by the tangential ones
        # The tangential dofs should be numbered last, and are constrained to be zero
        sd = self.cell.get_spatial_dimension()
        reduced_dofs = deepcopy(self._element.entity_dofs())
        reduced_dim = 0
        cur = reduced_dofs[sd-1][0][0]
        for entity in sorted(reduced_dofs[sd-1]):
            reduced_dim += len(reduced_dofs[sd-1][entity][1:])
            reduced_dofs[sd-1][entity] = [cur]
            cur += 1
        self._entity_dofs = reduced_dofs
        self._space_dimension = fiat_element.space_dimension() - reduced_dim

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        dofs = self.entity_dofs()
        bfs = self._element.entity_dofs()
        numdof = self.space_dimension()
        numbf = self._element.space_dimension()
        V = identity(numbf, numdof)

        # Undo the Piola transform for non-facet bubble basis functions
        nodes = self._element.get_dual_set().nodes
        Finv = piola_inverse(self.cell, J, detJ)
        for dim in dofs:
            if dim == sd-1:
                continue
            for e in sorted(dofs[dim]):
                k = 0
                while k < len(dofs[dim][e]):
                    cur = dofs[dim][e][k]
                    if len(nodes[cur].deriv_dict) > 0:
                        V[cur, cur] = detJ
                        k += 1
                    else:
                        s = dofs[dim][e][k:k+sd]
                        V[numpy.ix_(s, s)] = Finv
                        k += sd
        # Unpick the normal component for the facet bubbles
        for f in sorted(dofs[sd-1]):
            Bnt, Btt = normal_tangential_transform(self.cell, J, detJ, f)
            ndof, *tdofs = dofs[sd-1][f]
            nbf, *tbfs = bfs[sd-1][f]
            V[tbfs, ndof] = Bnt
            if len(tdofs) > 0:
                V[tbfs, tdofs] = Btt

        # Fix discrepancy between normal and tangential moments
        needs_facet_vertex_coupling = len(dofs[0][0]) > 0 and numbf > numdof
        if needs_facet_vertex_coupling:
            perp = lambda *t: numpy.array([t[0][1], -t[0][0]]) if len(t) == 1 else numpy.cross(*t)

            dim = max(d for d in range(sd-1) if len(dofs[d][0]) > 0)
            vdofs = chain.from_iterable(dofs[dim].values())
            vdofs = [i for i in vdofs if nodes[i].max_deriv_order == 0]
            fdofs = list(chain.from_iterable(dofs[sd-1].values()))

            T = numpy.full((len(fdofs), len(vdofs)), Zero(), dtype=object)
            for f in sorted(dofs[sd-1]):
                nhat = perp(*self.cell.compute_tangents(sd-1, f))
                Tfv = ((-1/sd) * nhat) @ Finv
                for v in self.cell.connectivity[(sd-1, dim)][f]:
                    curvdofs = [vdofs.index(i) for i in dofs[dim][v] if i in vdofs]
                    for fdof in dofs[sd-1][f]:
                        T[fdofs.index(fdof), curvdofs] = Tfv

            V[numdof:, vdofs] += V[numdof:, fdofs] @ T
        return ListTensor(V.T)
