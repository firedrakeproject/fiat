import FIAT
import numpy

from gem import ListTensor, partial_indexed, Literal

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement


class Morley(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("Morley1971")
        super().__init__(FIAT.Morley(cell, degree=degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        edges = self.cell.get_connectivity()[(sd-1, sd-2)]
        # Jacobians at barycenter
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        pns = coordinate_mapping.physical_normals()
        pts = coordinate_mapping.physical_tangents()
        pel = coordinate_mapping.physical_edge_lengths()

        V = identity(self.space_dimension())

        offset = len(top[sd-2])

        if sd == 2:
            for i in top[sd-1]:
                e = list(edges[i])
                s = offset + i
                n = partial_indexed(pns, (i,))
                t = partial_indexed(pts, (i,))
                nhat = self.cell.compute_normal(i)
                Jn = J @ Literal(nhat)
                Bnn = n @ Jn
                Bnt = t @ Jn
                V[s, s] = Bnn
                V[s, e] = Bnt / pel[i]
                V[s, e[0]] *= -1

        else:
            for face in top[sd-1]:
                s = offset + face
                n = partial_indexed(pns, (face,))

                nhat = self.cell.compute_scaled_normal(face)
                ahat = numpy.linalg.norm(nhat)
                nhat /= ahat

                Jn = J @ Literal(nhat)
                Bnn = n @ Jn
                # Why 1/2?
                V[s, s] = 0.5*Bnn

                # Not sure where this comes from
                factor = 0.5*Bnn / (ahat * detJ)

                xf, = self.cell.make_points(2, face, 3)
                for edge in edges[face]:
                    t = partial_indexed(pts, (edge,))

                    xe, = self.cell.make_points(1, edge, 2)
                    out = numpy.array(xe) - numpy.array(xf)
                    that = self.cell.compute_edge_tangent(edge)
                    sign = numpy.sign(numpy.dot(numpy.cross(nhat, that), out))

                    # Bnt = dot(cross(n, t), J * nhat)
                    Bnt = sum((n[i]*t[j] - n[j]*t[i]) * Jn[k] for i, j, k in [(1, 2, 0), (2, 0, 1), (0, 1, 2)])
                    V[s, edge] = Bnt * pel[edge] * sign * factor

        # diagonal post-scaling to patch up conditioning
        h = coordinate_mapping.cell_size()
        for i in top[sd-1]:
            s = i + offset
            verts = top[sd-1][i]
            havg = sum(h[v] for v in verts) / len(verts)
            for i in range(sd-1):
                V[:, s] *= 1/havg

        return ListTensor(V.T)
