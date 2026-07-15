import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.zany import ZanyPhysicallyMappedElement


class JohnsonMercier(ZanyPhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    """The Johnson-Mercier element.

    The basis transformation is derived automatically from the FIAT
    dual basis; see :class:`finat.zany.ZanyPhysicallyMappedElement`.
    """
    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.JohnsonMercier(cell, degree, variant=variant,
                                             quad_scheme=quad_scheme))
