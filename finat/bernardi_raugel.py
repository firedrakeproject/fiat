import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement


class BernardiRaugel(PiolaBubbleElement):
    def __init__(self, cell, order=1, quad_scheme=None):
        cite("BernardiRaugel1985")
        super().__init__(FIAT.BernardiRaugel(cell, order=order, quad_scheme=quad_scheme))


class BernardiRaugelBubble(BernardiRaugel):
    def __init__(self, cell, degree=None, quad_scheme=None):
        super().__init__(cell, order=0, quad_scheme=quad_scheme)
