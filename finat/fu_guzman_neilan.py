"""H1(curl) element with automatic physical mapping."""

import FIAT
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class ReducedFuGuzmanNeilanH1curl(PhysicallyMappedElement, FiatElement):
    """The quartic H1(curl) Alfeld tetrahedron.

    Parameters
    ----------
    cell : Cell
        A tetrahedron.
    order : int, optional
        The embedded subdegree, which must be 2 or 3.
    """

    def __init__(self, cell, order=2):
        super().__init__(FIAT.ReducedFuGuzmanNeilanH1curl(cell, order))
        # FIAT gives an element whose last nodes constrain the boundary traces and
        # curl, and are used in the transformation.  We only count the dofs: a first
        # order jet per vertex, and a circulation per edge.  Order 3 adds the curl at
        # each edge midpoint, a normal moment per face, and a cell average.
        top = cell.get_topology()
        k = order - 2
        self._space_dimension = 12 * len(top[0]) + (1 + 3 * k) * len(top[1]) + k * len(top[2]) + k * len(top[3])

    def space_dimension(self):
        return self._space_dimension
