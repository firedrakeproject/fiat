import FIAT

from gem import ListTensor, partial_indexed

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
        conn = self.cell.get_connectivity()
        # Jacobians at edge midpoints
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)

        rns = coordinate_mapping.reference_normals()
        pns = coordinate_mapping.physical_normals()
        pts = coordinate_mapping.physical_tangents()
        pel = coordinate_mapping.physical_edge_lengths()

        V = identity(self.space_dimension())

        offset = len(top[sd-2])
        for i in top[sd-1]:
            nhat = partial_indexed(rns, (i,))
            n = partial_indexed(pns, (i,))
            t = partial_indexed(pts, (i,))
            Bn = J @ nhat
            Bnn = n @ Bn
            Btn = t @ Bn

            s = i + offset
            c = list(conn[(sd-1, sd-2)][i])
            V[s, s] = Bnn
            V[s, c] = Btn / pel[i]
            V[s, c[0]] *= -1

        # diagonal post-scaling to patch up conditioning
        h = coordinate_mapping.cell_size()
        for i in top[sd-1]:
            s = i + offset
            V[:, s] *= 2 / sum(h[v] for v in top[sd-1][i])

        return ListTensor(V.T)
