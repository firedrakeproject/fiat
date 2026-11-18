from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class PiolaBubbleElement(PhysicallyMappedElement, FiatElement):
    """A general class to transform Piola-mapped elements with facet bubbles."""
    def __init__(self, fiat_element):
        mapping, = set(fiat_element.mapping())
        if mapping not in {"contravariant piola", "covariant piola"}:
            raise ValueError(f"{type(fiat_element).__name__} needs to be Piola mapped.")
        super().__init__(fiat_element)

        # On each facet we expose the first dof and constrain the remaining
        # facet bubbles to zero.
        sd = self.cell.get_spatial_dimension()
        facet_dofs = self._element.entity_dofs()[sd-1]
        reduced_dim = sum(len(dofs) - 1 for dofs in facet_dofs.values())
        self._space_dimension = fiat_element.space_dimension() - reduced_dim

    def space_dimension(self):
        return self._space_dimension
