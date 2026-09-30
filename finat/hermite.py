import FIAT
import numpy
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, variant=None):
        cite("Ciarlet1972")
        super().__init__(FIAT.Hermite(cell, degree=degree, variant=variant))

    def basis_transformation(self, coordinate_mapping):
        vertices = self.cell.get_vertices()
        if self.cell.get_spatial_dimension() == 1:
            # The derivative along the tangent maps by the signed Jacobian
            # determinant, which carries the cell orientation on manifolds.
            Js = [ListTensor([[coordinate_mapping.detJ_at(vertex)]]) for vertex in vertices]
        else:
            Js = [coordinate_mapping.jacobian_at(vertex) for vertex in vertices]

        h = coordinate_mapping.cell_size()

        M = identity(self.space_dimension())

        entity_ids = self.entity_dofs()
        for i in entity_ids[0]:
            # skip the PointEvaluation DOF
            vids = entity_ids[0][i][1:]
            J = Js[i]
            Jnp = numpy.reshape([J[k] for k in numpy.ndindex(J.shape)], J.shape)
            M[numpy.ix_(vids, vids)] = Jnp * (1 / h[i])

        return ListTensor(M)
