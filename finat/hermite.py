import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, variant=None):
        cite("Ciarlet1972")
        super().__init__(FIAT.Hermite(cell, degree=degree, variant=variant))
