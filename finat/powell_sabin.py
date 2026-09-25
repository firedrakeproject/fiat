import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class QuadraticPowellSabin6(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        cite("PowellSabin1977")
        super().__init__(FIAT.QuadraticPowellSabin6(cell))


class QuadraticPowellSabin12(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2, avg=False):
        self.avg = avg
        cite("PowellSabin1977")
        super().__init__(FIAT.QuadraticPowellSabin12(cell))

    def dof_scale(self, node, dim, havg):
        return super().dof_scale(node, dim, havg) if dim == 0 else None
