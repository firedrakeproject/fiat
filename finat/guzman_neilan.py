import FIAT

from finat.citations import cite
from finat.piola_mapped import PiolaBubbleElement
from finat.zany import PiolaPhysicallyMappedElement


class GuzmanNeilanFirstKindH1(PiolaPhysicallyMappedElement, PiolaBubbleElement):
    """Pk^d enriched with Guzman-Neilan bubbles.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing tangential facet constraints of the extended element
    are dropped from the physical element by
    :meth:`~finat.piola_mapped.PiolaBubbleElement.space_dimension`.
    """
    def __init__(self, cell, order=1, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanFirstKindH1(cell, order=order, quad_scheme=quad_scheme))


class GuzmanNeilanSecondKindH1(PiolaPhysicallyMappedElement, PiolaBubbleElement):
    """C0 Pk^d(Alfeld) enriched with Guzman-Neilan bubbles.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing tangential facet constraints of the extended element
    are dropped from the physical element by
    :meth:`~finat.piola_mapped.PiolaBubbleElement.space_dimension`.
    """
    def __init__(self, cell, order=1, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanSecondKindH1(cell, order=order, quad_scheme=quad_scheme))


class GuzmanNeilanBubble(GuzmanNeilanFirstKindH1):
    """Modified Bernardi-Raugel bubbles that are C^0 P_dim(Alfeld) with constant divergence."""
    def __init__(self, cell, degree=None, quad_scheme=None):
        super().__init__(cell, order=0, quad_scheme=quad_scheme)


class GuzmanNeilanH1div(PiolaPhysicallyMappedElement, PiolaBubbleElement):
    """Alfeld-Sorokina nodally enriched with Guzman-Neilan bubbles.

    The basis transformation is derived automatically from the FIAT
    dual basis (see :class:`finat.zany.PiolaPhysicallyMappedElement`);
    the trailing tangential facet constraints of the extended element
    are dropped from the physical element by
    :meth:`~finat.piola_mapped.PiolaBubbleElement.space_dimension`.
    """
    def __init__(self, cell, degree=None, quad_scheme=None):
        cite("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanH1div(cell, degree=degree, quad_scheme=quad_scheme))
