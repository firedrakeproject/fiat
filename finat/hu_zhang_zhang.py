from copy import deepcopy

import FIAT
import numpy
from gem import ListTensor, Zero

from finat.citations import cite
from finat.fiat_elements import FiatElement
from finat.guzman_neilan import GuzmanNeilanFirstKindH1
from finat.physically_mapped import identity, PhysicallyMappedElement
from FIAT.hu_zhang_zhang import curl_of


def apply_derivative_functionals(fiat_element, nodes):
    """Apply functionals defined through derivatives to the nodal basis of a FIAT element.

    :returns: an array of shape (len(nodes), fiat_element.space_dimension()).
    """
    pts = sorted(set(pt for ell in nodes for pt in ell.deriv_dict))
    index = {pt: i for i, pt in enumerate(pts)}
    order = max(ell.max_deriv_order for ell in nodes)
    tab = fiat_element.tabulate(order, pts)
    result = numpy.zeros((len(nodes), fiat_element.space_dimension()))
    for row, ell in zip(result, nodes):
        for pt, entries in ell.deriv_dict.items():
            for wt, alpha, comp in entries:
                row += wt * tab[alpha][(slice(None), *comp, index[pt])]
    return result


class HuZhangZhang(PhysicallyMappedElement, FiatElement):
    """The lowest-order Hu-Zhang-Zhang H(grad curl)-conforming macroelement.

    This is a gauge variant of the element of Hu, Zhang and Zhang: its shape
    functions differ from theirs by curl-free fields, while the degrees of
    freedom and the curl of the space are the same. See
    :class:`FIAT.hu_zhang_zhang.HuZhangZhang`.

    The curl of this element is the Guzman-Neilan element, and its degrees of
    freedom are Guzman-Neilan functionals composed with the curl, so the basis
    transformation is assembled from that of Guzman-Neilan.  The only new
    ingredient is that the normal face moments of the curl are not degrees of
    freedom: by Stokes' theorem they are sums of the tangential edge moments.
    """
    def __init__(self, cell, degree=1, quad_scheme=None):
        cite("HuZhangZhang2022")
        super().__init__(FIAT.HuZhangZhang(cell, degree, quad_scheme=quad_scheme))
        self._guzman_neilan = GuzmanNeilanFirstKindH1(cell, order=1, quad_scheme=quad_scheme)

        # The tangential face moments of the curl are constrained to be zero
        sd = self.cell.get_spatial_dimension()
        entity_dofs = deepcopy(self._element.entity_dofs())
        num_constraints = 0
        for f in entity_dofs[sd-1]:
            num_constraints += len(entity_dofs[sd-1][f])
            entity_dofs[sd-1][f] = []
        self._entity_dofs = entity_dofs
        self._space_dimension = self._element.space_dimension() - num_constraints

        # Normal face moments of the curl in terms of the tangential edge moments
        gn = self._guzman_neilan._element
        gn_nodes = gn.dual_basis()
        normal_nodes = [curl_of(gn_nodes[gn.entity_dofs()[sd-1][f][0]])
                        for f in sorted(entity_dofs[sd-1])]
        flux = apply_derivative_functionals(self._element, normal_nodes)
        edofs = [i for e in sorted(entity_dofs[1]) for i in entity_dofs[1][e]]
        others = [i for i in range(flux.shape[1]) if i not in edofs]
        if not numpy.allclose(flux[:, others], 0):
            raise RuntimeError("Normal face moments of the curl are not determined by the edge moments")
        flux = flux[:, edofs]
        flux[abs(flux) < 1E-12] = 0
        self._flux = flux

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        gn = self._guzman_neilan
        M = gn.basis_transformation(coordinate_mapping).array
        gn_dofs = gn.entity_dofs()
        gn_bfs = gn._element.entity_dofs()
        dofs = self.entity_dofs()
        bfs = self._element.entity_dofs()

        V = identity(self._element.space_dimension(), self.space_dimension())

        # Vertex curl values transform as the Guzman-Neilan vertex values,
        # coupling to the vertex and tangential face bubble basis functions
        bf_pairs = [(j, gj) for v in sorted(bfs[0])
                    for j, gj in zip(bfs[0][v], gn_bfs[0][v])]
        tbf_pairs = [(j, gj) for f in sorted(bfs[sd-1])
                     for j, gj in zip(bfs[sd-1][f], gn_bfs[sd-1][f][1:])]
        for v in sorted(dofs[0]):
            for i, gi in zip(dofs[0][v], gn_dofs[0][v]):
                for j, gj in bf_pairs + tbf_pairs:
                    V[j, i] = M[gi, gj]

        # The tangential face bubbles couple to the normal moments of the curl
        edofs = [i for e in sorted(dofs[1]) for i in dofs[1][e]]
        ndofs = [gn_dofs[sd-1][f][0] for f in sorted(dofs[sd-1])]
        for j, gj in tbf_pairs:
            for k, i in enumerate(edofs):
                terms = [M[gn_n, gj] * self._flux[g, k]
                         for g, gn_n in enumerate(ndofs)
                         if self._flux[g, k] != 0 and not isinstance(M[gn_n, gj], Zero)]
                if terms:
                    V[j, i] = sum(terms[1:], terms[0])
        return ListTensor(V.T)
