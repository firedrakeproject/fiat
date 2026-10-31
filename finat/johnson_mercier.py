import FIAT

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.JohnsonMercier(cell, degree, variant=variant, quad_scheme=quad_scheme))


class ReducedJohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.ReducedJohnsonMercier(cell, degree, variant=variant, quad_scheme=quad_scheme))
        # FIAT gives the normal-normal moments against P1 and the normal-tangential
        # moments against n x RT0 on each facet, followed by the constraint nodes
        # used in the transformation.
        sd = self.cell.get_spatial_dimension()
        self._space_dimension = (sd + 1) * (sd + (sd * (sd - 1)) // 2)

    def space_dimension(self):
        return self._space_dimension
