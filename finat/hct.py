import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement
from copy import deepcopy


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, avg=False):
        cite("Clough1965")
        if degree > 3:
            cite("Groselj2022")
        self.avg = avg
        super().__init__(FIAT.HsiehCloughTocher(cell, degree))

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
