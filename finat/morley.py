import FIAT
import numpy

from gem import ListTensor, partial_indexed, Literal, Power, Zero

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
            for e in top[sd-1]:
                s = offset + e
                n = partial_indexed(pns, (e,))
                t = partial_indexed(pts, (e,))
                nhat = self.cell.compute_normal(e)
                Jn = J @ Literal(nhat)
                Bnn = Jn @ n
                Bnt = Jn @ t
                V[s, s] = Bnn
                v = list(top[sd-1][e])
                V[s, v] = Bnt / pel[e]
                V[s, v[0]] *= -1

        else:
            R = ListTensor([[Zero(), Literal(1)], [Literal(-1), Zero()]])
            edges = self.cell.get_connectivity()[(sd-1, sd-2)]
            for face in top[sd-1]:
                s = offset + face

                te = numpy.array(list(map(self.cell.compute_edge_tangent, edges[face])))
                nhat = -numpy.cross(*te[:2])
                ahat = numpy.linalg.norm(nhat)

                # Reciprocal basis
                nhat /= numpy.dot(nhat, nhat)
                thats = numpy.array([numpy.cross(te[0], nhat), numpy.cross(te[1], nhat)])

                Jt = J @ Literal(thats.T)
                Jn = J @ Literal(nhat)
                Jte = J @ Literal(te.T)

                A = Jte.T @ Jte
                area = Power(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0], Literal(0.5))

                detF = area / ahat
                Bnn = detJ / detF
                V[s, s] = Bnn

                Gte = Jt.T @ Jte
                Gnt = Jn @ Jt
                Bne = (Gnt @ R) @ Gte

                # Not sure where this comes from
                factor = 2*ahat / (detF * detF)
                for i, edge in enumerate(edges[face]):
                    # UFC convention alternates signs
                    sign = (-1) ** i
                    V[s, edge] = Bne[i] * sign * factor

        # diagonal post-scaling to patch up conditioning
        h = coordinate_mapping.cell_size()
        for i in top[sd-1]:
            s = i + offset
            verts = top[sd-1][i]
            havg = sum(h[v] for v in verts) / len(verts)
            for i in range(sd-1):
                V[:, s] *= 1/havg

        return ListTensor(V.T)
