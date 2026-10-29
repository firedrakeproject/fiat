import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Walkington(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Walkington2010")
        super().__init__(FIAT.Walkington(cell, degree=degree))

    # FIAT gives a 65 DOF element whose last 20 nodes constrain the edges
    # and faces, and are used in the transformation.  However, we only
    # have a 45 DOF element.
    def space_dimension(self):
        return 45


class ReducedWalkington(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Walkington2010")
        super().__init__(FIAT.Walkington(cell, degree=degree, reduced=True))

    # FIAT gives a 64 DOF element whose last 24 nodes constrain the faces
    # and edges, and are used in the transformation.  However, we only
    # have a 40 DOF element.
    def space_dimension(self):
        return 40
