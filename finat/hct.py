import FIAT
from math import comb
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform, _edge_transform
from copy import deepcopy


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, avg=False):
        cite("Clough1965")
        if degree > 3:
            cite("Groselj2022")
        self.avg = avg
        super().__init__(FIAT.HsiehCloughTocher(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        if self.degree == 3:
            return super().basis_transformation(coordinate_mapping)

        V = identity(self.space_dimension())

        sd = self.cell.get_dimension()
        top = self.cell.get_topology()

        vorder = 1
        eorder = self.degree - 3
        voffset = comb(sd + vorder, vorder)
        _vertex_transform(V, vorder, self.cell, coordinate_mapping)
        _edge_transform(V, vorder, eorder, self.cell, coordinate_mapping, avg=self.avg)

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            s = voffset*v + 1
            V[:, s:s+sd] *= 1/h[v]
        return ListTensor(V.T)

    def dof_scale(self, node, dim, havg):
        return super().dof_scale(node, dim, havg) if dim == 0 else None


class ReducedHsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3):
        cite("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell, reduced=True))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = []
        self._entity_dofs = reduced_dofs

    # This wipes out the edge dofs.  FIAT gives a 12 DOF element
    # because we need some extra functions to help with transforming
    # under the edge constraint.  However, we only have a 9 DOF
    # element.
    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return 9
