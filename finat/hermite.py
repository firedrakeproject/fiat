import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.zany import ScalarPhysicallyMappedElement


class Hermite(ScalarPhysicallyMappedElement, ScalarFiatElement):
    """The cubic Hermite element.

    The basis transformation is derived automatically from the FIAT
    dual basis; see :class:`finat.zany.ScalarPhysicallyMappedElement`.
    """
    def __init__(self, cell, degree=3):
        cite("Ciarlet1972")
        super().__init__(FIAT.CubicHermite(cell))
