import FIAT
import numpy

from FIAT.polynomial_set import mis
from gem import ListTensor, Zero

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform, _normal_tangential_transform
from finat.morley import morley_transform
from copy import deepcopy
from itertools import chain


class Walkington(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Walkington2010")
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

        # Evaluate nodal completion of the face constraints
        P = self._element.poly_set
        L = self._element.dual.nodal_completion
        coeffs = P.get_coeffs()
        tangential_dofs = numpy.dot(L.to_riesz(P), coeffs.T)
        tangential_dofs[abs(tangential_dofs) < 1e-10] = 0

        for f in entity_dofs[2]:
            Rnn, Rnt = morley_transform(self.cell, J, detJ, f)
            fdofs = entity_dofs[2][f]
            fid = fdofs[0]
            V[fid, fid] = Rnn
            for j, e in enumerate(edges[f]):
                Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e, face=f)
                vid0, vid1 = (entity_dofs[0][v][0] for v in top[1][e])
                eid = fdofs[1+j]

                # vertex points
                V[fid, vid1] += Rnt[j]
                V[fid, vid0] += Rnt[j]
                V[eid, vid1] = 1/21 * Bnt
                V[eid, vid0] = -1 * V[eid, vid1]

                # vertex derivatives
                for i in range(sd):
                    R1 = 1/5 * Rnt[j] * Jt[i]
                    V[fid, vid1+i+1] -= R1
                    V[fid, vid0+i+1] += R1
                    V[eid, vid1+i+1] = -1/42 * Bnt * Jt[i]
                    V[eid, vid0+i+1] = V[eid, vid1+1+i]

                # second derivatives
                for i, alpha in enumerate(mis(sd, 2), start=sd+1):
                    ids = tuple(k for k, ak in enumerate(alpha) if ak)
                    a, b = ids[0], ids[-1]
                    tau = (1 + (a != b)) * Jt[a] * Jt[b]

                    R2 = 1/60 * Rnt[j] * tau
                    V[fid, vid1+i] += R2
                    V[fid, vid0+i] += R2
                    V[eid, vid1+i] = 1/252 * Bnt * tau
                    V[eid, vid0+i] = -1 * V[eid, vid1+i]

            vids = list(chain.from_iterable(entity_dofs[0][v] for v in top[2][f]))
            # Recombine with nodal completion to satisfy the physical constraints
            C = tangential_dofs[L.entity_ids[2][f]]
            supp = numpy.unique(numpy.nonzero(C)[1])
            C = C.astype(object)
            C[C == 0] = Zero()

            CV = C[:, supp] @ V[numpy.ix_(supp, vids)]
            Gnt = numpy.asarray(Rnt[1:])
            c0, c1 = fdofs[-2:]
            V[c0, vids] = -1 * Gnt @ CV[[0, 1]]
            V[c1, vids] = -1 * Gnt @ CV[[1, 2]]

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
