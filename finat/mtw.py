import FIAT
from math import comb
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.piola_mapped import normal_tangential_transform


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, order=1):
        if cell.get_spatial_dimension() == 2:
            cite("Mardal2002")
        else:
            cite("Xie2008")
        super().__init__(FIAT.MardalTaiWinther(cell, order=order))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        V = identity(self.space_dimension())
        q = self._element.order
        dimP1 = comb(1+sd-1, 1)
        dimPq = comb(q+sd-1, q)

        entity_dofs = self.entity_dofs()
        for f in sorted(entity_dofs[sd-1]):
            Bnt, Btt = normal_tangential_transform(self.cell, J, detJ, f)
            ndofs = entity_dofs[sd-1][f][:dimPq]
            tdofs = entity_dofs[sd-1][f][dimPq:]
            V[tdofs, tdofs] = Btt
            if sd == 2:
                V[tdofs, ndofs[0]] = Bnt
            else:
                V[tdofs[:-1], ndofs[0]] = Bnt
                V[tdofs[-1], ndofs[1:dimP1]] = Bnt

        return ListTensor(V.T)
