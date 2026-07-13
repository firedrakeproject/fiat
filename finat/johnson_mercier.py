import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicalGeometry, PhysicallyMappedElement
from finat.zany import zany_basis_transformation


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    """The Johnson-Mercier element.

    The basis transformation is derived automatically from the FIAT
    dual basis by :func:`finat.zany.zany_basis_transformation`.
    """
    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.JohnsonMercier(cell, degree, variant=variant,
                                             quad_scheme=quad_scheme))

    def basis_transformation(self, coordinate_mapping: PhysicalGeometry) -> ListTensor:
        return zany_basis_transformation(self._element, coordinate_mapping)
