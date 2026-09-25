import FIAT

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Stokes(PhysicallyMappedElement, FiatElement):
    """Pk^d"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.Stokes(cell, degree=degree))


class MacroStokes(PhysicallyMappedElement, FiatElement):
    """C0 Pk^d(Alfeld)"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.MacroStokes(cell, degree=degree))


class DivStokes(FiatElement):
    """Pk"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.DivStokes(cell, degree=degree))
