"""Implementation of the Hu-Zhang finite elements."""
import FIAT
from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class HuZhang(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3, variant=None, quad_scheme=None):
        cite("Hu2015")
        self.variant = "integral" if variant is None else variant
        super().__init__(FIAT.HuZhang(cell, degree, variant=variant, quad_scheme=quad_scheme))
