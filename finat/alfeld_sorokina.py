import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class AlfeldSorokina(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        cite("AlfeldSorokina2016")
        super().__init__(FIAT.AlfeldSorokina(cell, degree))
