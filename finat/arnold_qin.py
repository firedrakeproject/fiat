import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.piola_mapped import PiolaBubbleElement
from finat.zany import PiolaPhysicallyMappedElement


class ArnoldQin(FiatElement):
    def __init__(self, cell, degree=2):
        cite("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree))


class ReducedArnoldQin(PiolaPhysicallyMappedElement, PiolaBubbleElement):
    """The reduced Arnold-Qin element.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing tangential facet constraints of the extended element
    are dropped from the physical element by
    :meth:`~finat.piola_mapped.PiolaBubbleElement.space_dimension`.
    """
    def __init__(self, cell, degree=2):
        cite("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree, reduced=True))
