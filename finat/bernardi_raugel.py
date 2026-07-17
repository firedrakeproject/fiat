import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement
from finat.zany import PiolaPhysicallyMappedElement


class BernardiRaugel(PiolaPhysicallyMappedElement, PiolaBubbleElement):
    """The Bernardi-Raugel element.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing tangential facet constraints of the extended element
    are dropped from the physical element by
    :meth:`~finat.piola_mapped.PiolaBubbleElement.space_dimension`.
    """
    def __init__(self, cell, order=1, quad_scheme=None):
        cite("BernardiRaugel1985")
        super().__init__(FIAT.BernardiRaugel(cell, order=order, quad_scheme=quad_scheme))


class BernardiRaugelBubble(BernardiRaugel):
    """The normal facet bubbles of the Bernardi-Raugel element."""
    def __init__(self, cell, degree=None, quad_scheme=None):
        super().__init__(cell, order=0, quad_scheme=quad_scheme)
