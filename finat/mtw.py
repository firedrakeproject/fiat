import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.zany import ZanyPhysicallyMappedElement


class MardalTaiWinther(ZanyPhysicallyMappedElement, FiatElement):
    """The Mardal-Tai-Winther element.

    The basis transformation is derived automatically from the FIAT
    dual basis; see :class:`finat.zany.ZanyPhysicallyMappedElement`.
    """
    def __init__(self, cell, order=1):
        if cell.get_spatial_dimension() == 2:
            cite("Mardal2002")
        else:
            cite("Xie2008")
        super().__init__(FIAT.MardalTaiWinther(cell, order=order))
