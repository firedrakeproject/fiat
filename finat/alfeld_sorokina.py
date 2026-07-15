import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.zany import PiolaPhysicallyMappedElement


class AlfeldSorokina(PiolaPhysicallyMappedElement, FiatElement):
    """The Alfeld-Sorokina C0 quadratic macroelement with C0 divergence.

    This element belongs to a Stokes complex, and is paired with CG1(Alfeld).
    The basis transformation is derived automatically from the FIAT dual
    basis; see :class:`finat.zany.PiolaPhysicallyMappedElement`.
    """
    def __init__(self, cell, degree=2):
        cite("AlfeldSorokina2016")
        super().__init__(FIAT.AlfeldSorokina(cell, degree))
