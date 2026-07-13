import pprint

import pytest
import FIAT
import gem
import numpy as np
from gem.interpreter import evaluate
from finat.physically_mapped import PhysicalGeometry, PhysicallyMappedElement


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


def make_unisolvent_points(element, interior=False):
    degree = element.degree()
    ref_complex = element.get_reference_complex()
    top = ref_complex.get_topology()
    pts = []
    if interior:
        dim = ref_complex.get_spatial_dimension()
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree+dim+1, variant="gll"))
    else:
        for dim in top:
            for entity in top[dim]:
                pts.extend(ref_complex.make_points(dim, entity, degree, variant="gll"))
    return pts


def check_zany_mapping(element, ref_to_phys, *args, **kwargs):
    phys_cell = ref_to_phys.phys_cell
    ref_cell = ref_to_phys.ref_cell
    phys_element = element(phys_cell, *args, **kwargs).fiat_equivalent
    finat_element = element(ref_cell, *args, **kwargs)

    ref_element = finat_element._element
    ref_cell = ref_element.get_reference_element()
    phys_cell = phys_element.get_reference_element()
    sd = ref_cell.get_spatial_dimension()

    shape = ref_element.value_shape()
    ref_pts = make_unisolvent_points(ref_element, interior=True)
    ref_vals = ref_element.tabulate(0, ref_pts)[(0,)*sd]

    phys_pts = make_unisolvent_points(phys_element, interior=True)
    phys_vals = phys_element.tabulate(0, phys_pts)[(0,)*sd]

    mapping = ref_element.mapping()[0]
    if mapping == "affine":
        ref_vals_piola = ref_vals
    else:
        # Piola map the reference elements
        J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                          phys_cell.vertices)
        K = []
        if "covariant" in mapping:
            K.append(np.linalg.inv(J).T)
        if "contravariant" in mapping:
            K.append(J / np.linalg.det(J))

        if len(shape) == 2:
            piola_map = lambda x: K[0] @ x @ K[-1].T
        else:
            piola_map = lambda x: K[0] @ x

        ref_vals_piola = np.zeros(ref_vals.shape)
        for i in range(ref_vals.shape[0]):
            for k in range(ref_vals.shape[-1]):
                ref_vals_piola[i, ..., k] = piola_map(ref_vals[i, ..., k])

    # Zany map the results
    num_bfs = phys_element.space_dimension()
    num_dofs = finat_element.space_dimension()
    if isinstance(finat_element, PhysicallyMappedElement):
        Mgem = finat_element.basis_transformation(ref_to_phys)
        M = evaluate([Mgem])[0].arr
        ref_vals_zany = np.tensordot(M, ref_vals_piola, (-1, 0))
    else:
        M = np.eye(num_dofs, num_bfs)
        ref_vals_zany = ref_vals_piola

    # Solve for the basis transformation and compare results
    Phi = ref_vals_piola.reshape(num_bfs, -1)
    phi = phys_vals.reshape(num_bfs, -1)
    Vh, residual, *_ = np.linalg.lstsq(Phi.T, phi.T)
    Mh = Vh.T
    Mh = Mh[:num_dofs]
    tol = 1E-10
    Mh[abs(Mh) < tol] = 0
    M[abs(M) < tol] = 0

    delta = M.T - Mh.T
    delta[abs(delta) < tol] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        error = delta / Mh.T
    error[error != error] = 0
    error[delta == 0] = 0
    error[abs(error) < tol] = 0

    inds = tuple(map(np.unique, np.nonzero(error)))
    error = error[np.ix_(*inds)]
    error[error != 0] += 1

    pp = pprint.PrettyPrinter(width=140, compact=True)
    assert np.allclose(residual, 0), pp.pformat((np.round(error, 8).tolist(), *inds))
    assert np.allclose(ref_vals_zany, phys_vals[:num_dofs]), pp.pformat((np.round(error, 8).tolist(), *inds))


@pytest.fixture(name="check_zany_mapping")
def check_zany_mapping_fixture():
    return check_zany_mapping
