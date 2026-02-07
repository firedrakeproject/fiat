import pytest
import FIAT
import gem
import numpy as np
from finat.physically_mapped import PhysicalGeometry


class MyMapping(PhysicalGeometry):
    def __init__(self, ref_cell, phys_cell):
        self.ref_cell = ref_cell
        self.phys_cell = phys_cell

        self.A, self.b = FIAT.reference_element.make_affine_mapping(
            self.ref_cell.vertices,
            self.phys_cell.vertices)

    def cell_size(self):
        # Currently, just return 1 so we can compare FIAT dofs
        # to transformed dofs.
        return np.ones((len(self.ref_cell.vertices),))

    def detJ_at(self, point):
        return gem.Literal(np.linalg.det(self.A))

    def jacobian_at(self, point):
        return gem.Literal(self.A)

    def normalized_reference_edge_tangents(self):
        top = self.ref_cell.get_topology()
        return gem.Literal(
            np.asarray([self.ref_cell.compute_normalized_edge_tangent(i)
                        for i in sorted(top[1])]))

    def reference_normals(self):
        sd = self.ref_cell.get_spatial_dimension()
        top = self.ref_cell.get_topology()
        return gem.Literal(
            np.asarray([self.ref_cell.compute_normal(i)
                        for i in sorted(top[sd-1])]))

    def physical_normals(self):
        sd = self.phys_cell.get_spatial_dimension()
        top = self.phys_cell.get_topology()
        return gem.Literal(
            np.asarray([self.phys_cell.compute_normal(i)
                        for i in sorted(top[sd-1])]))

    def physical_tangents(self):
        top = self.phys_cell.get_topology()
        return gem.Literal(
            np.asarray([self.phys_cell.compute_normalized_edge_tangent(i)
                        for i in sorted(top[1])]))

    def physical_edge_lengths(self):
        top = self.phys_cell.get_topology()
        return gem.Literal(
            np.asarray([self.phys_cell.volume_of_subcomplex(1, i)
                        for i in sorted(top[1])]))

    def physical_points(self, ps, entity=None):
        prefs = ps.points
        A, b = self.A, self.b
        return gem.Literal(np.asarray([A @ x + b for x in prefs]))

    def physical_vertices(self):
        return gem.Literal(self.phys_cell.verts)


class ScaledMapping(MyMapping):

    def cell_size(self):
        # Firedrake interprets this as 2x the circumradius
        sd = self.phys_cell.get_spatial_dimension()
        top = self.phys_cell.get_topology()
        vol = self.phys_cell.volume()
        edges = [self.phys_cell.volume_of_subcomplex(1, i)
                 for i in sorted(top[1])]

        if sd == 1:
            cs = vol
        elif sd == 2:
            cs = np.prod(edges) / (2 * vol)
        elif sd == 3:
            edge_pairs = [edges[i] * edges[j] for i in top[1] for j in top[1]
                          if len(set(top[1][i] + top[1][j])) == len(top[0])]
            cs = 1.0 / (12 * vol)
            for k in range(4):
                s = [1]*len(edge_pairs)
                if k > 0:
                    s[k-1] = -1
                cs *= np.dot(s, edge_pairs) ** 0.5
        else:
            raise NotImplementedError(f"Cell size not implemented in {sd} dimensions")
        return np.asarray([cs for _ in sorted(top[0])])


def scaled_simplex(dim, scale):
    K = FIAT.ufc_simplex(dim)
    K.vertices = scale * np.array(K.vertices)
    return K


@pytest.fixture
def ref_el():
    K = {dim: FIAT.ufc_simplex(dim) for dim in (2, 3)}
    return K


@pytest.fixture
def phys_el():
    K = {dim: FIAT.ufc_simplex(dim) for dim in (2, 3)}
    K[2].vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    K[3].vertices = ((0, 0, 0),
                     (1., 0.1, -0.37),
                     (0.01, 0.987, -.23),
                     (-0.1, -0.2, 1.38))
    return K


@pytest.fixture
def ref_to_phys(ref_el, phys_el):
    return {dim: MyMapping(ref_el[dim], phys_el[dim]) for dim in ref_el}


@pytest.fixture
def scaled_ref_to_phys(ref_el):
    return {dim: [ScaledMapping(ref_el[dim], scaled_simplex(dim, 0.5**k)) for k in range(3)]
            for dim in ref_el}
