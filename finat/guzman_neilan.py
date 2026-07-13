import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.physically_mapped import PhysicalGeometry
from finat.piola_mapped import PiolaBubbleElement
from finat.zany import zany_basis_transformation


class GuzmanNeilanFirstKindH1(PiolaBubbleElement):
    """Pk^d enriched with Guzman-Neilan bubbles.

    The basis transformation is derived automatically from the FIAT
    dual basis by :func:`finat.zany.zany_basis_transformation`; the
    trailing tangential facet constraints of the extended element are
    dropped from the physical element.
    """
    def __init__(self, cell, order=1, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanFirstKindH1(cell, order=order, quad_scheme=quad_scheme))

    def basis_transformation(self, coordinate_mapping: PhysicalGeometry) -> ListTensor:
        return zany_basis_transformation(self._element, coordinate_mapping,
                                         ndof=self.space_dimension())


class GuzmanNeilanSecondKindH1(PiolaBubbleElement):
    """C0 Pk^d(Alfeld) enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, order=1, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanSecondKindH1(cell, order=order, quad_scheme=quad_scheme))


class GuzmanNeilanBubble(GuzmanNeilanFirstKindH1):
    """Modified Bernardi-Raugel bubbles that are C^0 P_dim(Alfeld) with constant divergence."""
    def __init__(self, cell, degree=None, quad_scheme=None):
        super().__init__(cell, order=0, quad_scheme=quad_scheme)


class GuzmanNeilanH1div(PiolaBubbleElement):
    """Alfeld-Sorokina nodally enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, degree=None, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanH1div(cell, degree=degree, quad_scheme=quad_scheme))
