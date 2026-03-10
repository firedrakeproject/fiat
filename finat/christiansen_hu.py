import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement


class ChristiansenHu(PiolaBubbleElement):
    def __init__(self, cell, degree=1):
        cite("ChristiansenHu2019")
        super().__init__(FIAT.ChristiansenHu(cell, degree))
