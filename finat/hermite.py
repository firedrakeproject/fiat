import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicalGeometry, PhysicallyMappedElement
from finat.zany import zany_basis_transformation


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    """The cubic Hermite element.

    The basis transformation is derived automatically from the FIAT
    dual basis by :func:`finat.zany.zany_basis_transformation`.
    """
    def __init__(self, cell, degree=3):
        cite("Ciarlet1972")
        super().__init__(FIAT.CubicHermite(cell))

    def basis_transformation(self, coordinate_mapping: PhysicalGeometry) -> ListTensor:
        return zany_basis_transformation(self._element, coordinate_mapping)
