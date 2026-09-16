from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement
from finat.citations import cite

import FIAT


class WuXuRobustH3NC(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=7):
        if degree != 7:
            raise ValueError("Degree must be 7 for robust Wu-Xu element")
        cite("WuXu2019")
        super().__init__(FIAT.WuXuRobustH3NC(cell))


class WuXuH3NC(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=4):
        if degree != 4:
            raise ValueError("Degree must be 4 for the Wu-Xu element")
        cite("WuXu2019")
        super().__init__(FIAT.WuXuH3NC(cell))
