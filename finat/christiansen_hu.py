import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement
from finat.zany import PiolaPhysicallyMappedElement


class ChristiansenHu(PiolaPhysicallyMappedElement, PiolaBubbleElement):
    """The Christiansen-Hu element.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing tangential facet constraints of the extended element
    are dropped from the physical element by
    :meth:`~finat.piola_mapped.PiolaBubbleElement.space_dimension`.
    """
    def __init__(self, cell, degree=1):
        cite("ChristiansenHu2019")
        super().__init__(FIAT.ChristiansenHu(cell, degree))
