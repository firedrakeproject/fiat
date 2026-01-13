import numpy

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform
from finat.citations import cite
import FIAT


class WuXuRobustH3NC(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=7):
        if degree != 7:
            raise ValueError("Degree must be 7 for robust Wu-Xu element")
        cite("WuXu2019")
        super().__init__(FIAT.WuXuRobustH3NC(cell))

    def basis_transformation(self, coordinate_mapping):
        top = self.cell.topology
        entity_ids = self._element.entity_dofs()
        V = identity(self.space_dimension())
        _vertex_transform(V, 1, self.cell, coordinate_mapping)

        J = coordinate_mapping.jacobian_at(self.cell.make_points(2, 0, 3)[0])
        J = numpy.array([[J[0, 0], J[0, 1]],
                         [J[1, 0], J[1, 1]]])
        [[dxdxhat, dxdyhat], [dydxhat, dydyhat]] = J

        Thetainv = numpy.array(
            [[dxdxhat*dxdxhat, 2 * dxdxhat * dydxhat, dydxhat*dydxhat],
             [dxdyhat * dxdxhat, dxdyhat * dydxhat + dxdxhat * dydyhat, dydxhat * dydyhat],
             [dxdyhat*dxdyhat, 2 * dxdyhat * dydyhat, dydyhat*dydyhat]])

        ns = coordinate_mapping.physical_normals()
        ts = coordinate_mapping.physical_tangents()
        lens = coordinate_mapping.physical_edge_lengths()

        nhats = coordinate_mapping.reference_normals()
        thats = numpy.zeros((3, 2), dtype=object)
        for e in range(3):
            tancur = self.cell.compute_normalized_edge_tangent(e)
            for i in range(2):
                thats[e, i] = Literal(tancur[i])

        B1s = numpy.zeros((3, 2, 2), dtype=object)
        B2s = numpy.zeros((3, 3, 3), dtype=object)
        betas = numpy.zeros((3, 2), dtype=object)

        for e in range(3):
            nx = ns[e, 0]
            ny = ns[e, 1]
            nhatx = nhats[e, 0]
            nhaty = nhats[e, 1]
            tx = ts[e, 0]
            ty = ts[e, 1]
            thatx = thats[e, 0]
            thaty = thats[e, 1]

            Gs = numpy.asarray([[nx, ny], [tx, ty]])
            Ghats = numpy.asarray([[nhatx, nhaty], [thatx, thaty]])

            B1s[e, :, :] = (Ghats @ (J.T @ Gs.T)) / lens[e]

            Gammas = numpy.asarray(
                [[nx*nx, 2*nx*tx, tx*tx],
                 [nx*ny, nx*ty+ny*tx, tx*ty],
                 [ny*ny, 2*ny*ty, ty*ty]])

            Gammainvhats = numpy.asarray(
                [[nhatx*nhatx, 2*nhatx*nhaty, nhaty*nhaty],
                 [nhatx*thatx, nhatx*thaty+nhaty*thatx, nhaty*thaty],
                 [thatx*thatx, 2*thatx*thaty, thaty*thaty]])

            B2s[e, :, :] = Gammainvhats @ (Thetainv @ Gammas)

            betas[e, 0] = (nx * B2s[e, 0, 1] + tx * B2s[e, 0, 2])/lens[e]
            betas[e, 1] = (ny * B2s[e, 0, 1] + ty * B2s[e, 0, 2])/lens[e]

        for e in top[1]:
            v0, v1 = top[1][e]
            vid0 = entity_ids[0][v0]
            vid1 = entity_ids[0][v1]

            eid = entity_ids[1][e][0]
            V[eid, eid] = B1s[e, 0, 0] * lens[e]
            V[eid, vid0[0]] = -1*B1s[e, 0, 1]
            V[eid, vid1[0]] = B1s[e, 0, 1]

            eid = entity_ids[1][e][1]
            V[eid, eid] = B2s[e, 0, 0]
            V[eid, vid0[1]] = -1*betas[e, 0]
            V[eid, vid0[2]] = -1*betas[e, 1]
            V[eid, vid1[1]] = betas[e, 0]
            V[eid, vid1[2]] = betas[e, 1]

        # Now let's fix the scaling.
        h = coordinate_mapping.cell_size()

        # This gets the vertex gradients
        for v in range(3):
            for k in range(2):
                V[:, 3*v+1+k] *= 1 / h[v]

        # this scales second derivative moments.  First should be ok.
        for e in range(3):
            v0id, v1id = [i for i in range(3) if i != e]
            for i in range(15):
                V[i, 12+e] = 2*V[i, 12+e] / (h[v0id] + h[v1id])

        return ListTensor(V.T)


class WuXuH3NC(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=4):
        if degree != 4:
            raise ValueError("Degree must be 4 for the Wu-Xu element")
        cite("WuXu2019")
        super().__init__(FIAT.WuXuH3NC(cell))

    def basis_transformation(self, coordinate_mapping):
        top = self.cell.topology
        entity_ids = self._element.entity_dofs()
        V = identity(self.space_dimension())
        _vertex_transform(V, 1, self.cell, coordinate_mapping)

        J = coordinate_mapping.jacobian_at(self.cell.make_points(2, 0, 3)[0])
        J = numpy.array([[J[0, 0], J[0, 1]],
                         [J[1, 0], J[1, 1]]])
        [[dxdxhat, dxdyhat], [dydxhat, dydyhat]] = J

        Thetainv = numpy.array(
            [[dxdxhat*dxdxhat, 2 * dxdxhat * dydxhat, dydxhat*dydxhat],
             [dxdyhat * dxdxhat, dxdyhat * dydxhat + dxdxhat * dydyhat, dydxhat * dydyhat],
             [dxdyhat*dxdyhat, 2 * dxdyhat * dydyhat, dydyhat*dydyhat]])

        ns = coordinate_mapping.physical_normals()
        ts = coordinate_mapping.physical_tangents()
        lens = coordinate_mapping.physical_edge_lengths()

        nhats = coordinate_mapping.reference_normals()
        thats = numpy.zeros((3, 2), dtype=object)
        for e in range(3):
            tancur = self.cell.compute_normalized_edge_tangent(e)
            for i in range(2):
                thats[e, i] = Literal(tancur[i])

        B2s = numpy.zeros((3, 3, 3), dtype=object)
        betas = numpy.zeros((3, 2), dtype=object)

        for e in range(3):
            nx = ns[e, 0]
            ny = ns[e, 1]
            nhatx = nhats[e, 0]
            nhaty = nhats[e, 1]
            tx = ts[e, 0]
            ty = ts[e, 1]
            thatx = thats[e, 0]
            thaty = thats[e, 1]

            Gammas = numpy.asarray(
                [[nx*nx, 2*nx*tx, tx*tx],
                 [nx*ny, nx*ty+ny*tx, tx*ty],
                 [ny*ny, 2*ny*ty, ty*ty]])

            Gammainvhats = numpy.asarray(
                [[nhatx*nhatx, 2*nhatx*nhaty, nhaty*nhaty],
                 [nhatx*thatx, nhatx*thaty+nhaty*thatx, nhaty*thaty],
                 [thatx*thatx, 2*thatx*thaty, thaty*thaty]])

            B2s[e, :, :] = Gammainvhats @ (Thetainv @ Gammas)

            betas[e, 0] = (nx * B2s[e, 0, 1] + tx * B2s[e, 0, 2])/lens[e]
            betas[e, 1] = (ny * B2s[e, 0, 1] + ty * B2s[e, 0, 2])/lens[e]

        for e in top[1]:
            v0, v1 = top[1][e]
            vid0 = entity_ids[0][v0]
            vid1 = entity_ids[0][v1]

            eid = entity_ids[1][e][0]
            V[eid, eid] = B2s[e, 0, 0]
            V[eid, vid0[1]] = -1*betas[e, 0]
            V[eid, vid0[2]] = -1*betas[e, 1]
            V[eid, vid1[1]] = betas[e, 0]
            V[eid, vid1[2]] = betas[e, 1]

        # Now let's fix the scaling.
        h = coordinate_mapping.cell_size()

        # This gets the vertex gradients
        for v in range(3):
            for k in range(2):
                V[:, 3*v+1+k] *= 1 / h[v]

        # this scales second derivative moments.  First should be ok.
        for e in range(3):
            v0id, v1id = [i for i in range(3) if i != e]
            for i in range(12):
                V[i, 9+e] = 2*V[i, 9+e] / (h[v0id] + h[v1id])

        return ListTensor(V.T)
