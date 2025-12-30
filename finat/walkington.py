import FIAT
from FIAT.polynomial_set import mis

from gem import ListTensor, Literal, Power

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform, _normal_tangential_transform
from copy import deepcopy
import numpy


class Walkington(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Bell1969")
        super().__init__(FIAT.Walkington(cell, degree=degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:1]
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        # Jacobian at barycenter
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = identity(numbf, ndof)
        vorder = 2
        _vertex_transform(V, vorder, self.cell, coordinate_mapping)

        entity_dofs = self._element.entity_dofs()
        edges = self.cell.get_connectivity()[(2, 1)]
        for f in entity_dofs[2]:
            fdofs = entity_dofs[2][f]
            q = fdofs[0]
            c = fdofs[-2:]

            Rnn, Rnt, Jn = morley_transform(self.cell, J, detJ, f)
            V[q, q] = Rnn

            for j, e in enumerate(edges[f]):
                vid0, vid1 = (entity_dofs[0][v][0] for v in top[1][e])
                s = fdofs[1+j]
                Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e, face=f)

                # vertex points
                V[s, vid1] = 1/21 * Bnt
                V[s, vid0] = -1 * V[s, vid1]

                V[q, vid1] += Rnt[j]
                V[q, vid0] += Rnt[j]

                # TODO
                c = fdofs[-2:]
                Qnt = (Jn @ Jt) / (Jt @ Jt)
                V[c, vid1] += Qnt
                V[c, vid0] += Qnt
                c = []

                # vertex derivatives
                for i in range(sd):
                    V[s, vid1+i+1] = -1/42 * Bnt * Jt[i]
                    V[s, vid0+i+1] = V[s, vid1+1+i]

                    R1 = 1/5 * Rnt[j] * Jt[i]
                    V[q, vid1+i+1] -= R1
                    V[q, vid0+i+1] += R1

                    # TODO
                    V[c, vid1+i+1] += Jt[i] * Qnt
                    V[c, vid0+i+1] -= Jt[i] * Qnt
                c = []

                # second derivatives
                for i, alpha in enumerate(mis(sd, 2), start=sd+1):
                    ids = tuple(k for k, ak in enumerate(alpha) if ak)
                    a, b = ids[0], ids[-1]
                    tau = (1 + (a != b)) * Jt[a] * Jt[b]
                    V[s, vid1+i] = 1/252 * Bnt * tau
                    V[s, vid0+i] = -1 * V[s, vid1+i]

                    R2 = 1/60 * Rnt[j] * tau
                    V[q, vid1+i] += R2
                    V[q, vid0+i] += R2

                    # TODO
                    V[c, vid1+i] += Qnt * tau
                    V[c, vid0+i] += Qnt * tau


        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(entity_dofs[0]):
            vdofs = entity_dofs[0][v]
            V[:, vdofs[1:1+sd]] *= 1/h[v]
            V[:, vdofs[1+sd:]] *= 1/(h[v]*h[v])
        return ListTensor(V.T)

    # This wipes out the edge dofs.  FIAT gives a 65 DOF element
    # because we need some extra functions to help with transforming
    # under the edge constraint.  However, we only have an 45 DOF
    # element.
    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return 45


def morley_transform(cell, J, detJ, face):
    adjugate = lambda A: ListTensor([[A[1, 1], -1*A[1, 0]], [-1*A[0, 1], A[0, 0]]])
    sd = cell.get_spatial_dimension()
    thats = cell.compute_tangents(sd-1, face)
    nhat = numpy.cross(*thats)
    ahat = numpy.linalg.norm(nhat)
    nhat /= numpy.dot(nhat, nhat)

    nhat = Literal(nhat)
    Jn = J @ nhat
    Jt = J @ Literal(thats.T)
    Gnt = Jn.T @ Jt
    Gtt = Jt.T @ Jt
    detG = Gtt[0, 0]*Gtt[1, 1] - Gtt[0, 1]*Gtt[1, 0]
    area = Power(detG, Literal(0.5))

    Bnn = detJ / area
    Bnt = Gnt @ adjugate(Gtt) / detG
    Bnn *= ahat
    Bnt *= ahat
    Bnt = (-1*(Bnt[0] + Bnt[1]), Bnt[0], Bnt[1])
    return Bnn, Bnt, Jn
