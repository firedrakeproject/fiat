import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, order=1):
        if cell.get_spatial_dimension() == 2:
            cite("Mardal2002")
        else:
            cite("Xie2008")
        super().__init__(FIAT.MardalTaiWinther(cell, order=order))
