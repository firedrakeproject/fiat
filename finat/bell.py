import FIAT
from copy import deepcopy

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.zany import ZanyPhysicallyMappedElement


class Bell(ZanyPhysicallyMappedElement, ScalarFiatElement):
    """The Bell element.

    FIAT provides the extended element on the full quintic space, with
    the cubic normal derivative constraints appended as extra edge
    nodes; the transformation of the extended element is derived
    automatically (see :class:`finat.zany.ZanyPhysicallyMappedElement`),
    and the constraint degrees of freedom are dropped from the physical
    element by overriding :meth:`space_dimension`.
    """
    def __init__(self, cell, degree=5):
        cite("Bell1969")
        super().__init__(FIAT.Bell(cell, degree=degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = []
        self._entity_dofs = reduced_dofs

    # The extended FIAT element has 21 basis functions, but the Bell
    # element only keeps the 18 vertex degrees of freedom.
    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return 18
