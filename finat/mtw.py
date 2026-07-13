import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicalGeometry, PhysicallyMappedElement
from finat.zany import zany_basis_transformation


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    """The Mardal-Tai-Winther element.

    The basis transformation is derived automatically from the FIAT
    dual basis by :func:`finat.zany.zany_basis_transformation`.
    """
    def __init__(self, cell, order=1):
        if cell.get_spatial_dimension() == 2:
            cite("Mardal2002")
        else:
            cite("Xie2008")
        super().__init__(FIAT.MardalTaiWinther(cell, order=order))

    def basis_transformation(self, coordinate_mapping: PhysicalGeometry) -> ListTensor:
        return zany_basis_transformation(self._element, coordinate_mapping)
