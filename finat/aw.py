"""Implementation of the Arnold-Winther finite elements."""
import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.zany import PiolaPhysicallyMappedElement


class ArnoldWintherNC(PiolaPhysicallyMappedElement, FiatElement):
    """The nonconforming Arnold-Winther element.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing constraint functionals of the extended element are
    dropped from the physical element by :meth:`space_dimension`.
    """
    def __init__(self, cell, degree=2):
        cite("Arnold2003")
        super().__init__(FIAT.ArnoldWintherNC(cell, degree))

    def entity_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]},
                2: {0: [12, 13, 14]}}

    def space_dimension(self):
        return 15


class ArnoldWinther(PiolaPhysicallyMappedElement, FiatElement):
    """The conforming Arnold-Winther element.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing constraint functionals of the extended element are
    dropped from the physical element by :meth:`space_dimension`.
    """
    def __init__(self, cell, degree=3):
        cite("Arnold2002")
        super().__init__(FIAT.ArnoldWinther(cell, degree))

    def entity_dofs(self):
        return {0: {0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [6, 7, 8]},
                1: {0: [9, 10, 11, 12], 1: [13, 14, 15, 16], 2: [17, 18, 19, 20]},
                2: {0: [21, 22, 23]}}

    def space_dimension(self):
        return 24
