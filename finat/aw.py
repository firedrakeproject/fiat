"""Implementation of the Arnold-Winther finite elements."""
import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class ArnoldWintherNC(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        cite("Arnold2003")
        super().__init__(FIAT.ArnoldWintherNC(cell, degree))

    def space_dimension(self):
        return 15


class ArnoldWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        cite("Arnold2002")
        super().__init__(FIAT.ArnoldWinther(cell, degree))

    def space_dimension(self):
        return 24
