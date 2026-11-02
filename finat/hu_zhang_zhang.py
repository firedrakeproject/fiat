"""H(grad curl) element with automatic physical mapping."""

import FIAT
from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class HuZhangZhang(PhysicallyMappedElement, FiatElement):
    """The Hu-Zhang-Zhang H(grad curl) Alfeld tetrahedron.

    Parameters
    ----------
    cell : Cell
        A tetrahedron.
    k : int, optional
        The degree of the complete polynomials in the curl, which must be 1 or 2.
    """

    def __init__(self, cell, k=1):
        cite("HuZhangZhang2022")
        super().__init__(FIAT.HuZhangZhang(cell, k))
        # FIAT gives an element whose last 8 nodes constrain the curl, and are used
        # in the transformation.  We only count the curl at each vertex, and on each
        # edge the tangential moments and the curl at interior lattice points.
        top = cell.get_topology()
        self._space_dimension = 3 * len(top[0]) + (k + 3 * (k - 1)) * len(top[1])

    def space_dimension(self):
        return self._space_dimension
