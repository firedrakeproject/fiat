from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement
from finat.citations import cite

import FIAT


class BrambleZlamalC2(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=9, avg=True):
        cite("BrambleZlamal1970")
        self.avg = avg
        super().__init__(FIAT.BrambleZlamalC2(cell, degree))


class AlfeldC2(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5, avg=True):
        cite("Alfeld1984")
        self.avg = avg
        super().__init__(FIAT.AlfeldC2(cell, degree))
