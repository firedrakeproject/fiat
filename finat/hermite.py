import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, variant=None):
        cite("Ciarlet1972")
        super().__init__(FIAT.Hermite(cell, degree=degree, variant=variant))


class ReducedHermite(PhysicallyMappedElement, ScalarFiatElement):
    """The physically mapped reduced cubic Kirchhoff triangle element.

    Parameters
    ----------
    cell : :class:`FIAT.reference_element.Cell`
        The reference triangle.
    degree : int, optional
        The polynomial degree.  Only degree 3 is supported.
    """

    def __init__(self, cell, degree=3):
        cite("Ciarlet1972")
        super().__init__(FIAT.Hermite(cell, degree=degree, reduced=True))
