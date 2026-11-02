"""H(grad curl) element with automatic physical mapping."""

import FIAT
from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class HuZhangZhang(PhysicallyMappedElement, FiatElement):
    """The lowest order Hu-Zhang-Zhang H(grad curl) Alfeld tetrahedron.

    Parameters
    ----------
    cell : Cell
        A tetrahedron.
    k : int, optional
        The degree of the complete polynomials in the curl, which must be 1.
    """

    def __init__(self, cell, k=1):
        cite("HuZhangZhang2022")
        super().__init__(FIAT.HuZhangZhang(cell, k))

    # FIAT gives a 26 DOF element whose last 8 nodes constrain the curl, and are
    # used in the transformation.  We only count the curl at each vertex and a
    # circulation on each edge.
    def space_dimension(self):
        return 18
