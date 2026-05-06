import FIAT
import numpy
from gem import ListTensor, partial_indexed

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, variant=None):
        cite("Ciarlet1972")
        super().__init__(FIAT.CubicHermite(cell, degree=degree, variant=variant))

    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        pns = coordinate_mapping.physical_normals()
        h = coordinate_mapping.cell_size()

        M = identity(self.space_dimension())

        entity_ids = self.entity_dofs()
        for i in entity_ids[0]:
            # skip the PointEvaluation DOF
            vids = entity_ids[0][i][1:]
            J = Js[i]

            gdim, tdim = J.shape
            if gdim != tdim:
                assert tdim == 1
                J = partial_indexed(pns, (i,)) @ J

            Jnp = numpy.reshape([J[i] for i in numpy.ndindex(J.shape)], J.shape)
            M[numpy.ix_(vids, vids)] = Jnp * (1 / h[i])

        return ListTensor(M)
