import FIAT
import numpy
from gem import ListTensor

from finat.aw import _facet_transform
from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.piola_mapped import normal_tangential_edge_transform, normal_tangential_face_transform
from copy import deepcopy


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.JohnsonMercier(cell, degree, variant=variant, quad_scheme=quad_scheme))

    def basis_transformation(self, coordinate_mapping):
        V = identity(self.space_dimension())
        Vsub = _facet_transform(self.cell, 1, coordinate_mapping)
        m, n = Vsub.shape
        V[:m, :n] = Vsub
        return ListTensor(V.T)


class ReducedJohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued

    def __init__(self, cell, degree=1, variant=None, quad_scheme=None):
        cite("Gopalakrishnan2024")
        super().__init__(FIAT.ReducedJohnsonMercier(cell, degree, variant=variant, quad_scheme=quad_scheme))
        # On each facet we expect the normal dof followed by the tangential ones
        # The tangential dofs should be numbered last, and are constrained to be zero
        sd = self.cell.get_spatial_dimension()
        reduced_dofs = deepcopy(self._element.entity_dofs())
        reduced_dim = 0

        dimP1 = sd
        dimNed1 = (sd*(sd-1))//2
        r = dimP1 + dimNed1
        for entity in sorted(reduced_dofs[sd-1]):
            reduced_dim += len(reduced_dofs[sd-1][entity][r:])
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:r]

        for entity in sorted(reduced_dofs[sd]):
            reduced_dim += len(reduced_dofs[sd][entity])
            reduced_dofs[sd][entity] = []

        self._entity_dofs = reduced_dofs
        self._space_dimension = self._element.space_dimension() - reduced_dim

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        num_dofs = self.space_dimension()
        num_bfs = self._element.space_dimension()
        V = identity(num_bfs, num_dofs)
        dimP1 = sd
        dimNed1 = (sd*(sd-1))//2

        if sd == 2:
            transform = normal_tangential_edge_transform
        else:
            transform = normal_tangential_face_transform

        entity_dofs = self.entity_dofs()
        constraints = {f: list(range(num_dofs+f*dimNed1, num_dofs+(f+1)*dimNed1))
                       for f in entity_dofs[sd-1]}
        for f in sorted(entity_dofs[sd-1]):
            *Bnt, Btt = transform(self.cell, J, detJ, f)
            ndofs = entity_dofs[sd-1][f][:dimP1]
            tdofs = entity_dofs[sd-1][f][dimP1:]
            cdofs = constraints[f]

            V[tdofs, tdofs] = Btt
            if sd == 2:
                V[tdofs, ndofs[0]] = Bnt
                V[cdofs, ndofs[1]] = Bnt
            else:
                V[tdofs[:-1], ndofs[0]] = Bnt
                V[tdofs[-1], ndofs[1:]] = Bnt
                # FIXME
                V[cdofs[0], ndofs[1:]] = Bnt
                V[cdofs[1], ndofs[1:]] = Bnt
                V[cdofs[2], ndofs[1:]] = Bnt

        return ListTensor(V.T)
