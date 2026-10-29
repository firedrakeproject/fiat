import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.JohnsonMercier(cell, degree, variant=variant, quad_scheme=quad_scheme))
