import FIAT
import numpy

from gem import ListTensor, partial_indexed, Literal, Power, Zero

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement


class Morley(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("Morley1971")
            Citations().register("MingXu2006")
        super().__init__(FIAT.Morley(cell, degree=degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        # Jacobians at barycenter
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        V = identity(self.space_dimension())

        offset = len(top[sd-2])
        R = ListTensor([[Zero(), Literal(1)], [Literal(-1), Zero()]])
        if sd == 2:
            pel = coordinate_mapping.physical_edge_lengths()
            pts = coordinate_mapping.physical_tangents()
            for e in top[sd-1]:
                s = offset + e
                t = partial_indexed(pts, (e,))
                n = R @ t
                nhat = self.cell.compute_normal(e)
                Jn = J @ Literal(nhat)
                Bnn = Jn @ n
                Bnt = Jn @ t
                V[s, s] = Bnn
                v = list(top[sd-1][e])
                V[s, v] = Bnt / pel[e]
                V[s, v[0]] *= -1

        else:
            edges = self.cell.get_connectivity()[(sd-1, sd-2)]
            for face in top[sd-1]:
                s = offset + face
                thats = self.cell.compute_tangents(sd-1, face)
                nhat = numpy.cross(*thats)
                ahat = numpy.linalg.norm(nhat)

                nhat /= numpy.dot(nhat, nhat)
                Jn = J @ Literal(nhat)
                Jt = J @ Literal(thats.T)
                Gnt = Jn.T @ Jt
                Gtt = Jt.T @ Jt
                detG = Gtt[0, 0]*Gtt[1, 1] - Gtt[0, 1]*Gtt[1, 0]
                area = Power(detG, Literal(0.5))

                Bnn = detJ * (ahat / area)
                Bnt = Gnt @ R @ Gtt
                # Not sure where this factor comes from
                Bnt *= ahat / detG
                V[s, s] = Bnn
                V[s, list(edges[face])] = (Bnt[0] - Bnt[1], Bnt[1], -1*Bnt[0])

        # diagonal post-scaling to patch up conditioning
        h = coordinate_mapping.cell_size()
        for face in top[sd-1]:
            s = offset + face
            verts = top[sd-1][face]
            havg = sum(h[v] for v in verts) / len(verts)
            V[:, s] *= 1/havg

        return ListTensor(V.T)
