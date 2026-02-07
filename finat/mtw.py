import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.piola_mapped import normal_tangential_edge_transform


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        cite("Mardal2002")
        super().__init__(FIAT.MardalTaiWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        entity_dofs = self.entity_dofs()
        if sd == 2:
            facet_transform = normal_tangential_edge_transform
        else:
            raise NotImplementedError

        # dim_P1 = sd
        # dim_Ned1 = (sd*(sd-1))//2
        # fdofs = dim_P1 + dim_Ned1

        ndof = self.space_dimension()
        V = identity(ndof, ndof)
        for f in sorted(entity_dofs[sd-1]):
            cur = entity_dofs[sd-1][f][0]
            V[cur, cur:cur+sd] = facet_transform(self.cell, J, detJ, f)[::-1]

        return ListTensor(V.T)
