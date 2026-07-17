"""Implementation of the Hu-Zhang finite elements."""
import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.zany import PiolaPhysicallyMappedElement


class HuZhang(PiolaPhysicallyMappedElement, FiatElement):
    """The Hu-Zhang element.

    The basis transformation is derived automatically from the FIAT
    dual basis; see :class:`finat.zany.PiolaPhysicallyMappedElement`.
    """
    def __init__(self, cell, degree=3, variant=None, quad_scheme=None):
        cite("Hu2015")
        self.variant = variant
        super().__init__(FIAT.HuZhang(cell, degree, variant=variant, quad_scheme=quad_scheme))
