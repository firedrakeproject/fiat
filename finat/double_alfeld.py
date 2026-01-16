from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.citations import cite
from finat.argyris import (_jet_transform, _vertex_transform,
                           _normal_tangential_transform)
from gem import ListTensor

import FIAT
import numpy
from math import comb


class DoubleAlfeld(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5, avg=True):
        cite("Alfeld1984")
        self.avg = avg
        super().__init__(FIAT.DoubleAlfeld(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        top = self.cell.topology
        sd = self.cell.get_spatial_dimension()
        entity_ids = self._element.entity_dofs()

        vorder = 2
        V = identity(self.space_dimension())
        _vertex_transform(V, vorder, self.cell, coordinate_mapping)

        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        Thetainv = _jet_transform(J, 2)

        ns = coordinate_mapping.physical_normals()
        ts = coordinate_mapping.physical_tangents()
        lens = coordinate_mapping.physical_edge_lengths()
        nhats = coordinate_mapping.reference_normals()
        thats = coordinate_mapping.normalized_reference_edge_tangents()

        n0 = self.degree - 5
        n1 = n0 + 1
        for e in top[1]:
            v0, v1 = top[1][e]
            vid0 = entity_ids[0][v0]
            vid1 = entity_ids[0][v1]
            eids = entity_ids[1][e]
            emoments = (eids[:n0], eids[n0:n0+n1], eids[n0+n1:])

            G = numpy.array([[u[e, j] for j in range(sd)] for u in (ns, ts)])
            Ghat = numpy.array([[u[e, j] for j in range(sd)] for u in (nhats, thats)])
            Gamma = _jet_transform(G, 2)
            Gammainvhat = _jet_transform(Ghat.T, 2)

            B2 = (Gammainvhat @ Thetainv) @ Gamma
            beta = B2[0, 1:] @ G / lens[e]

            Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e)
            if self.avg:
                Bnn = Bnn * lens[e]

            # first derivative moments
            for k, s1 in enumerate(emoments[1], start=1):
                # Derivative of Jacobi polynomial at the endpoints
                dP1 = comb(k + vorder, k-1) * (2*vorder+k+1)
                dP0 = (-1)**k * dP1

                V[s1, s1] = Bnn
                V[s1, vid0[0]] = dP0 * Bnt
                V[s1, vid1[0]] = dP1 * Bnt
                if k > 1:
                    s0 = emoments[0][k-2]
                    V[s1, s0] = -1 * Bnt

            # second derivative moments
            for k, s2 in enumerate(emoments[2]):
                # Jacobi polynomial at the endpoints
                P1 = comb(k + vorder, k)
                P0 = -(-1)**k * P1

                V[s2, s2] = B2[0, 0]
                V[s2, vid0[1:sd+1]] = P0 * beta
                V[s2, vid1[1:sd+1]] = P1 * beta
                if k > 0:
                    s1 = emoments[1][k-1]
                    V[s2, s1] = -2 * Bnt * V[s1, s1]
                    V[s2, vid0[0]] = -1 * Bnt * V[s1, vid0[0]]
                    V[s2, vid1[0]] = -1 * Bnt * V[s1, vid1[0]]
                if k > 1:
                    s0 = emoments[0][k-2]
                    V[s2, s0] = -1 * Bnt * V[s1, s0]

        # Now let's fix the scaling.
        h = coordinate_mapping.cell_size()
        for v in top[0]:
            vids = entity_ids[0][v]
            vjet = (vids[:1], vids[1:sd+1], vids[sd+1:])
            # scale the gradients and hessians
            V[:, vjet[1]] *= 1 / h[v]
            V[:, vjet[2]] *= 1 / (h[v] * h[v])

        for e in top[1]:
            eids = entity_ids[1][e]
            emoments = (eids[:n0], eids[n0:n0+n1], eids[n0+n1:])
            he = (1/len(top[1][e])) * sum(h[v] for v in top[1][e])
            # scale first and second derivative moments
            V[:, emoments[1]] *= 1 / he
            V[:, emoments[2]] *= 1 / (he * he)

        return ListTensor(V.T)
