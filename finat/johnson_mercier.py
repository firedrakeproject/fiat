import FIAT
from gem import ListTensor

from finat.aw import _facet_transform
from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.piola_mapped import normal_tangential_edge_transform, normal_tangential_face_transform


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

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        V = identity(self.space_dimension())
        dimP1 = sd

        if sd == 2:
            transform = normal_tangential_edge_transform
        else:
            transform = normal_tangential_face_transform

        entity_dofs = self.entity_dofs()
        for f in sorted(entity_dofs[sd-1]):
            *Bnt, Btt = transform(self.cell, J, detJ, f)
            ndofs = entity_dofs[sd-1][f][:dimP1]
            tdofs = entity_dofs[sd-1][f][dimP1:]
            V[tdofs, tdofs] = Btt
            if sd == 2:
                V[tdofs, ndofs[0]] = Bnt
            else:
                V[tdofs[-1], ndofs[1:]] = Bnt
                V[tdofs[:-1], ndofs[0]] = Bnt

        return ListTensor(V.T)
