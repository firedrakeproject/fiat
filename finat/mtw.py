import FIAT
import gem
import numpy

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.piola_mapped import normal_tangential_edge_transform, normal_tangential_face_transform


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        if degree is None:
            degree = cell.get_spatial_dimension()+1
        cite("Mardal2002")
        super().__init__(FIAT.MardalTaiWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        entity_dofs = self.entity_dofs()

        ndof = self.space_dimension()
        V = identity(ndof, ndof)
        dimP1 = sd
        if sd == 2:
            for f in sorted(entity_dofs[sd-1]):
                Bnt = normal_tangential_edge_transform(self.cell, J, detJ, f)
                ndofs = entity_dofs[sd-1][f][:dimP1]
                tdofs = entity_dofs[sd-1][f][dimP1:]

                V[tdofs[0], ndofs[0]] = Bnt[0]
                V[tdofs[0], tdofs[0]] = Bnt[1]
        else:
            for f in sorted(entity_dofs[sd-1]):
                Bnt = normal_tangential_face_transform(self.cell, J, detJ, f)
                ndofs = entity_dofs[sd-1][f][:dimP1]
                tdofs = entity_dofs[sd-1][f][dimP1:]

                thats = self.cell.compute_tangents(sd-1, f)
                nhat = numpy.cross(*thats)
                nhat /= numpy.dot(nhat, nhat)
                orths = numpy.array([numpy.cross(thats[1], nhat),
                                     numpy.cross(nhat, thats[0])])

                Jts = J @ gem.Literal(thats.T)
                Jorths = J @ gem.Literal(orths.T)
                A = Jorths.T @ Jts
                detA = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
                V[tdofs, tdofs] = detJ / detA

                Q = numpy.dot(thats, thats.T)
                Bnt = Q @ (Bnt[1][0], -1*Bnt[0][0])
                V[tdofs[:2], ndofs[0]] += Bnt
                V[tdofs[2], ndofs[1:]] += Bnt

        return gem.ListTensor(V.T)
