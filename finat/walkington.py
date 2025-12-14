import FIAT
from FIAT.polynomial_set import mis
from math import comb

from gem import ListTensor

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform, _normal_tangential_transform
from copy import deepcopy


class Walkington(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        cite("Bell1969")
        super().__init__(FIAT.Walkington(cell, degree=degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:1]
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        # Jacobian at barycenter
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = identity(numbf, ndof)

        vorder = 2
        _vertex_transform(V, vorder, self.cell, coordinate_mapping)

        offset = ndof
        voffset = comb(sd + vorder, vorder)
        foffset = len(self._element.entity_dofs()[2][0]) - len(self.entity_dofs()[2][0])

        edges = self.cell.get_connectivity()[(2, 1)]
        for f in sorted(top[2]):
            q = len(top[0]) * voffset + f
            V[q, q] *= detJ

            for j, e in enumerate(edges[f]):
                s = offset + foffset * f + j

                v0id, v1id = (v * voffset for v in top[1][e])
                Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e, face=f)

                # vertex points
                V[s, v1id] = 1/21 * Bnt
                V[s, v0id] = -1 * V[s, v1id]

                # vertex derivatives
                for i in range(sd):
                    V[s, v1id+1+i] = -1/42 * Bnt * Jt[i]
                    V[s, v0id+1+i] = V[s, v1id+1+i]

                # second derivatives
                for i, alpha in enumerate(mis(sd, 2)):
                    ids = tuple(k for k, ak in enumerate(alpha) if ak)
                    a, b = ids[0], ids[-1]
                    tau = (1 + (a != b)) * Jt[a] * Jt[b]
                    V[s, v1id+sd+1+i] = 1/252 * Bnt * tau
                    V[s, v0id+sd+1+i] = -1 * V[s, v1id+sd+1+i]

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            s = voffset * v + 1
            V[:, s:s+sd] *= 1/h[v]
            V[:, s+sd:voffset*(v+1)] *= 1/(h[v]*h[v])

        return ListTensor(V.T)

    # This wipes out the edge dofs.  FIAT gives a 65 DOF element
    # because we need some extra functions to help with transforming
    # under the edge constraint.  However, we only have an 45 DOF
    # element.
    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return 45
