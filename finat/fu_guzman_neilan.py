"""H1(curl) element with automatic physical mapping."""

import FIAT
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class ReducedFuGuzmanNeilanH1curl(PhysicallyMappedElement, FiatElement):
    """The 54-dimensional quartic H1(curl) Alfeld tetrahedron.

    Parameters
    ----------
    cell : Cell
        A tetrahedron.
    degree : int, optional
        Polynomial degree, which must be four.
    """

    def __init__(self, cell, degree=4):
        super().__init__(FIAT.ReducedFuGuzmanNeilanH1curl(cell, degree))

    # FIAT gives an 86 DOF element whose last 32 nodes constrain the
    # boundary traces and curl, and are used in the transformation.
    # However, we only have a 54 DOF element.
    def space_dimension(self):
        return 54
