import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.piola_mapped import PiolaBubbleElement


class ArnoldQin(FiatElement):
    def __init__(self, cell, degree=2):
        cite("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree))


class ReducedArnoldQin(PiolaBubbleElement):
    def __init__(self, cell, degree=2):
        cite("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree, reduced=True))
