import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, avg=False):
        cite("Clough1965")
        if degree > 3:
            cite("Groselj2022")
        self.avg = avg
        super().__init__(FIAT.HsiehCloughTocher(cell, degree))


class ReducedHsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3):
        cite("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell, reduced=True))

    # FIAT gives a 12 DOF element whose last 3 nodes constrain the
    # edges, and are used in the transformation.  However, we only have
    # a 9 DOF element.
    def space_dimension(self):
        return 9
