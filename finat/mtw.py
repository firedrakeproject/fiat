import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.piola_mapped import normal_tangential_transform


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        if degree is None:
            degree = cell.get_spatial_dimension()+1
        cite("Mardal2002")
        super().__init__(FIAT.MardalTaiWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        V = identity(self.space_dimension())
        dimP1 = sd

        entity_dofs = self.entity_dofs()
        for f in sorted(entity_dofs[sd-1]):
            Bnt, Btt = normal_tangential_transform(self.cell, J, detJ, f)
            ndofs = entity_dofs[sd-1][f][:dimP1]
            tdofs = entity_dofs[sd-1][f][dimP1:]
            V[tdofs, tdofs] = Btt
            if sd == 2:
                V[tdofs, ndofs[0]] = Bnt
            else:
                V[tdofs[-1], ndofs[1:]] = Bnt
                V[tdofs[:-1], ndofs[0]] = Bnt

        return ListTensor(V.T)
