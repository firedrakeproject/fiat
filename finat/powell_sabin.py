import FIAT
from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement


class QuadraticPowellSabin6(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        cite("PowellSabin1977")
        super().__init__(FIAT.QuadraticPowellSabin6(cell))

    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        h = coordinate_mapping.cell_size()

        d = self.cell.get_dimension()
        M = identity(self.space_dimension())

        cur = 0
        for i in range(d+1):
            cur += 1  # skip the vertex
            J = Js[i]
            for j in range(d):
                for k in range(d):
                    M[cur+j, cur+k] = J[j, k] / h[i]
            cur += d

        return ListTensor(M)


class QuadraticPowellSabin12(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2, avg=False):
        self.avg = avg
        cite("PowellSabin1977")
        super().__init__(FIAT.QuadraticPowellSabin12(cell))

    def dof_scale(self, node, dim, havg):
        return super().dof_scale(node, dim, havg) if dim == 0 else None
