import FIAT

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class Stokes(PhysicallyMappedElement, FiatElement):
    """Pk^d"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.Stokes(cell, degree=degree))


class MacroStokes(PhysicallyMappedElement, FiatElement):
    """C0 Pk^d(Alfeld)"""
    def __init__(self, cell, degree=None, quad_scheme=None):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        fiat_element = FIAT.MacroStokes(
            cell, degree=max(degree, sd), hierarchical=degree < sd,
            quad_scheme=quad_scheme)

        reduced_dim = None
        if degree < sd:
            # degree=k<sd means constrain edges and face tangents to degree k
            # The dofs already encode the constraints (see the RestrictedElementsNotes.tex in gn_multigrid repo)
            # finat expects the constrained dofs to appear last
            indices = FIAT.stokes.reduced_macro_stokes_indices(fiat_element, degree)
            kept = set(indices)
            constrained = [i for i in range(fiat_element.space_dimension()) if i not in kept]
            fiat_element = FIAT.RestrictedElement(fiat_element, indices=indices + constrained, sort=False)
            reduced_dim = len(indices)

        super().__init__(fiat_element)
        self._quad_scheme = quad_scheme

        self._space_dimension = fiat_element.space_dimension()
        self._entity_dofs = fiat_element.entity_dofs()
        if reduced_dim is not None:
            self._space_dimension = reduced_dim
            self._entity_dofs = {dim: {entity: [i for i in ids if i < reduced_dim]
                                       for entity, ids in entities.items()}
                                 for dim, entities in self._entity_dofs.items()}

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension


class DivStokes(FiatElement):
    """Pk"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.DivStokes(cell, degree=degree))
