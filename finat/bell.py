import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement
from copy import deepcopy


class Bell(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Bell1969")
        super().__init__(FIAT.Bell(cell, degree=degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = []
        self._entity_dofs = reduced_dofs

    # This wipes out the edge dofs.  FIAT gives a 21 DOF element
    # because we need some extra functions to help with transforming
    # under the edge constraint.  However, we only have an 18 DOF
    # element.
    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return 18
