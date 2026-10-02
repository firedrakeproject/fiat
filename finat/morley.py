import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Morley(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        cite("Morley1971")
        cite("MingXu2006")
        super().__init__(FIAT.Morley(cell, degree=degree))
