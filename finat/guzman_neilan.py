import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement


class GuzmanNeilanFirstKindH1(PiolaBubbleElement):
    """Pk^d enriched with Guzman-Neilan bubbles."""
    embedded_degree = FIAT.GuzmanNeilanFirstKindH1.embedded_degree

    def __init__(self, cell, order=1, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanFirstKindH1(cell, order=order, quad_scheme=quad_scheme))


class GuzmanNeilanSecondKindH1(PiolaBubbleElement):
    """C0 Pk^d(Alfeld) enriched with Guzman-Neilan bubbles."""
    embedded_degree = FIAT.GuzmanNeilanSecondKindH1.embedded_degree

    def __init__(self, cell, order=1, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanSecondKindH1(cell, order=order, quad_scheme=quad_scheme))


class GuzmanNeilanBubble(GuzmanNeilanFirstKindH1):
    """Modified Bernardi-Raugel bubbles that are C^0 P_dim(Alfeld) with constant divergence."""
    def __init__(self, cell, degree=None, quad_scheme=None):
        super().__init__(cell, order=0, quad_scheme=quad_scheme)

    @staticmethod
    def embedded_degree(cell, degree):
        return FIAT.GuzmanNeilanFirstKindH1.embedded_degree(cell, 0)


class GuzmanNeilanH1div(PiolaBubbleElement):
    """Alfeld-Sorokina nodally enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, degree=None, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanH1div(cell, degree=degree, quad_scheme=quad_scheme))

    @staticmethod
    def embedded_degree(cell, degree):
        return max(FIAT.AlfeldSorokina.embedded_degree(cell, 2),
                   FIAT.GuzmanNeilanFirstKindH1.embedded_degree(cell, 0))
