import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicalGeometry, PhysicallyMappedElement
from finat.zany import zany_basis_transformation


class Morley(PhysicallyMappedElement, ScalarFiatElement):
    """The Morley element on simplices of any dimension.

    The basis transformation is derived automatically from the FIAT
    dual basis by :func:`finat.zany.zany_basis_transformation`.
    """
    def __init__(self, cell, degree=2):
        cite("Morley1971")
        cite("MingXu2006")
        super().__init__(FIAT.Morley(cell, degree=degree))

    def basis_transformation(self, coordinate_mapping: PhysicalGeometry) -> ListTensor:
        return zany_basis_transformation(self._element, coordinate_mapping)
