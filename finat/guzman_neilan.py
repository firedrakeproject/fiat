import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement


class GuzmanNeilanFirstKindH1(PiolaBubbleElement):
    """Pk^d enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, order=1):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanFirstKindH1(cell, order=order))


class GuzmanNeilanSecondKindH1(PiolaBubbleElement):
    """C0 Pk^d(Alfeld) enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, order=1):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanSecondKindH1(cell, order=order))


class GuzmanNeilanBubble(GuzmanNeilanFirstKindH1):
    """Modified Bernardi-Raugel bubbles that are C^0 P_dim(Alfeld) with constant divergence."""
    def __init__(self, cell, degree=None):
        super().__init__(cell, order=0)


class GuzmanNeilanH1div(PiolaBubbleElement):
    """Alfeld-Sorokina nodally enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, degree=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanH1div(cell, degree=degree))
