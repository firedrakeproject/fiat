from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform
from finat.citations import cite
from gem import ListTensor

import FIAT
import numpy


class WuXuRobustH3NC(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=7):
        if degree != 7:
            raise ValueError("Degree must be 7 for robust Wu-Xu element")
        cite("WuXu2019")
        super().__init__(FIAT.WuXuRobustH3NC(cell))

    def basis_transformation(self, coordinate_mapping):
        return wuxu_transformation(self, coordinate_mapping)


class WuXuH3NC(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=4):
        if degree != 4:
            raise ValueError("Degree must be 4 for the Wu-Xu element")
        cite("WuXu2019")
        super().__init__(FIAT.WuXuH3NC(cell))

    def basis_transformation(self, coordinate_mapping):
        return wuxu_transformation(self, coordinate_mapping)


def hessian_transform(J):
    return numpy.array(
        [[J[0, 0] * J[0, 0], J[0, 0] * J[1, 0] + J[0, 0] * J[1, 0], J[1, 0] * J[1, 0]],
         [J[0, 1] * J[0, 0], J[0, 1] * J[1, 0] + J[0, 0] * J[1, 1], J[1, 0] * J[1, 1]],
         [J[0, 1] * J[0, 1], J[0, 1] * J[1, 1] + J[0, 1] * J[1, 1], J[1, 1] * J[1, 1]]],
        dtype=object)


def wuxu_transformation(self, coordinate_mapping):
    top = self.cell.topology
    sd = self.cell.get_spatial_dimension()
    entity_ids = self._element.entity_dofs()

    V = identity(self.space_dimension())
    _vertex_transform(V, 1, self.cell, coordinate_mapping)

    bary, = self.cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)
    Thetainv = hessian_transform(J)
    J = numpy.array([[J[i, j] for j in range(sd)] for i in range(sd)])

    ns = coordinate_mapping.physical_normals()
    ts = coordinate_mapping.physical_tangents()
    lens = coordinate_mapping.physical_edge_lengths()
    nhats = coordinate_mapping.reference_normals()
    thats = coordinate_mapping.normalized_reference_edge_tangents()

    for e in top[1]:
        v0, v1 = top[1][e]
        vid0 = entity_ids[0][v0]
        vid1 = entity_ids[0][v1]

        G = numpy.array([[u[e, j] for j in range(sd)] for u in (ns, ts)])
        Ghat = numpy.array([[u[e, j] for j in range(sd)] for u in (nhats, thats)])

        if len(entity_ids[1][e]) > 1:
            # first derivative moments
            eid = entity_ids[1][e][0]
            B1 = (Ghat @ J.T) @ G.T
            alpha = B1[0, 1] / lens[e]
            V[eid, eid] = B1[0, 0]
            V[eid, vid0[0]] = -1*alpha
            V[eid, vid1[0]] = alpha

        # second derivative moments
        eid = entity_ids[1][e][-1]
        Gamma = hessian_transform(G)
        Gammainvhat = hessian_transform(Ghat.T)
        B2 = (Gammainvhat @ Thetainv) @ Gamma
        beta = B2[0, 1:] @ G / lens[e]
        V[eid, eid] = B2[0, 0]
        V[eid, vid0[1:]] = -1*beta
        V[eid, vid1[1:]] = beta

    # Now let's fix the scaling.
    h = coordinate_mapping.cell_size()

    # This gets the vertex gradients
    for v in top[0]:
        vids = entity_ids[0][v][1:]
        V[:, vids] *= 1 / h[v]

    # this scales second derivative moments.  First should be ok.
    for e in top[1]:
        eid = entity_ids[1][e][-1]
        he = (1/len(top[1][e])) * sum(h[v] for v in top[1][e])
        V[:, eid] *= 1 / (he * he)

    return ListTensor(V.T)
