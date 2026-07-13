import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.zany import ScalarPhysicallyMappedElement


class Morley(ScalarPhysicallyMappedElement, ScalarFiatElement):
    """The Morley element on simplices of any dimension.

    The basis transformation is derived automatically from the FIAT
    dual basis; see :class:`finat.zany.ScalarPhysicallyMappedElement`.
    """
    def __init__(self, cell, degree=2):
        cite("Morley1971")
        cite("MingXu2006")
        super().__init__(FIAT.Morley(cell, degree=degree))
