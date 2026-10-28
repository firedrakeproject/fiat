import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Bell(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Bell1969")
        super().__init__(FIAT.Bell(cell, degree=degree))

    # FIAT gives a 21 DOF element whose last 3 nodes constrain the
    # edges, and are used in the transformation.  However, we only have
    # an 18 DOF element.
    def space_dimension(self):
        return 18
