from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform
from finat.citations import cite
from finat.argyris import _normal_tangential_transform
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
        Thetainv = hessian_transform(J)
        Jnp = numpy.array([[J[i, j] for j in range(sd)] for i in range(sd)])

        ns = coordinate_mapping.physical_normals()
        ts = coordinate_mapping.physical_tangents()
        lens = coordinate_mapping.physical_edge_lengths()
        nhats = coordinate_mapping.reference_normals()
        thats = coordinate_mapping.normalized_reference_edge_tangents()

        n0 = self.degree-5
        n1 = self.degree-4
        for e in top[1]:
            v0, v1 = top[1][e]
            vid0 = entity_ids[0][v0]
            vid1 = entity_ids[0][v1]
            eids = entity_ids[1][e]

            G = numpy.array([[u[e, j] for j in range(sd)] for u in (ns, ts)])
            Ghat = numpy.array([[u[e, j] for j in range(sd)] for u in (nhats, thats)])

            B1 = (Ghat @ Jnp.T) @ G.T
            alpha = B1[0, 1] / lens[e]

            Gamma = hessian_transform(G)
            Gammainvhat = hessian_transform(Ghat.T)
            B2 = (Gammainvhat @ Thetainv) @ Gamma
            beta = B2[0, 1:] @ G / lens[e]

            Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e)
            if self.avg:
                Bnn = Bnn * lens[e]

            # first derivative moments
            for k, eid in enumerate(eids[n0:(n0+n1)]):
                # Derivative of Jacobi polynomial at the endpoints
                P1 = comb(k+1 + vorder, k+1-1) * (2*vorder+k+1+1)
                P0 = -(-1)**k * P1

                V[eid, eid] = B1[0, 0]
                V[eid, vid0[0]] = P0*alpha
                V[eid, vid1[0]] = P1*alpha
                if k > 0:
                    V[eid, eids[k-1]] = -1 * alpha

            # second derivative moments
            for k, eid in enumerate(eids[(n0+n1):]):
                # Jacobi polynomial at the endpoints
                P1 = comb(k + vorder, k)
                P0 = -(-1)**k * P1
                V[eid, eid] = B2[0, 0]
                V[eid, vid0[1:sd+1]] = P0*beta
                V[eid, vid1[1:sd+1]] = P1*beta
                if k > 0:
                    P1 = comb(k + vorder, k-1) * (2*vorder+k+1)
                    P0 = -(-1)**(k+1) * P1

                    V[eid, eids[n0+k-1]] = -2*alpha*Bnn
                    V[eid, vid0[0]] = -P0*alpha*Bnt
                    V[eid, vid1[0]] = -P1*alpha*Bnt
                if k > 1:
                    V[eid, eids[k-2]] = alpha * alpha

        # Now let's fix the scaling.
        h = coordinate_mapping.cell_size()
        for v in top[0]:
            # This gets the vertex gradients
            vids = entity_ids[0][v][1:sd+1]
            V[:, vids] *= 1 / h[v]
            # this gets the vertex hessians
            vids = entity_ids[0][v][sd+1:]
            V[:, vids] *= 1 / (h[v]*h[v])

        for e in top[1]:
            he = (1/len(top[1][e])) * sum(h[v] for v in top[1][e])
            # scale first derivative moments
            eid = entity_ids[1][e][n0:(n0+n1)]
            V[:, eid] *= 1 / he
            # scale second derivative moments
            eid = entity_ids[1][e][(n0+n1):]
            V[:, eid] *= 1 / (he * he)

        return ListTensor(V.T)


def hessian_transform(J):
    return numpy.array(
        [[J[0, 0] * J[0, 0], J[0, 0] * J[1, 0] + J[0, 0] * J[1, 0], J[1, 0] * J[1, 0]],
         [J[0, 1] * J[0, 0], J[0, 1] * J[1, 0] + J[0, 0] * J[1, 1], J[1, 0] * J[1, 1]],
         [J[0, 1] * J[0, 1], J[0, 1] * J[1, 1] + J[0, 1] * J[1, 1], J[1, 1] * J[1, 1]]],
        dtype=object)
