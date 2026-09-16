import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Argyris(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5, variant=None, avg=False):
        cite("Argyris1968")
        if variant is None:
            variant = "integral"
        if variant == "point" and degree != 5:
            raise NotImplementedError("Degree must be 5 for 'point' variant of Argyris")
        fiat_element = FIAT.Argyris(cell, degree, variant=variant)
        self.avg = avg
        super().__init__(fiat_element)
