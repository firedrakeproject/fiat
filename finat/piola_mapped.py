"""Reduced Piola-mapped elements with normal facet bubbles."""
from copy import deepcopy

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class PiolaBubbleElement(PhysicallyMappedElement, FiatElement):
    """Dof-reduction wrapper for Piola-mapped elements with normal facet bubbles.

    The FIAT element is an extended element carrying tangential facet
    functionals as trailing constraints; this wrapper exposes only the
    normal facet bubble on each facet, following the reduced/constrained
    element convention (rectangular transformation, truncated to
    :meth:`space_dimension` columns).  The basis transformation itself is
    provided by :class:`finat.zany.PiolaPhysicallyMappedElement`, which
    concrete subclasses list first in their bases.
    """
    def __init__(self, fiat_element):
        mapping, = set(fiat_element.mapping())
        if mapping != "contravariant piola":
            raise ValueError(f"{type(fiat_element).__name__} needs to be Piola mapped.")
        super().__init__(fiat_element)

        # On each facet we expect the normal dof followed by the tangential ones
        # The tangential dofs should be numbered last, and are constrained to be zero
        sd = self.cell.get_spatial_dimension()
        reduced_dofs = deepcopy(self._element.entity_dofs())
        reduced_dim = 0
        cur = reduced_dofs[sd-1][0][0]
        for entity in sorted(reduced_dofs[sd-1]):
            reduced_dim += len(reduced_dofs[sd-1][entity][1:])
            reduced_dofs[sd-1][entity] = [cur]
            cur += 1
        self._entity_dofs = reduced_dofs
        self._space_dimension = fiat_element.space_dimension() - reduced_dim

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension
