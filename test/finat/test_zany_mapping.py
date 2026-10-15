import FIAT
import finat
import gem
import numpy as np
import pytest
import pprint

from gem.interpreter import evaluate
from gem.cost import operation_count
from gem.flop_count import count_flops
from gem.impero_utils import compile_gem
from gem.node import traversal
from gem.optimise import contraction
from finat.physically_mapped import MappedTabulation, PhysicallyMappedElement

from .conftest import MyMapping


def test_entity_mapped_tabulation():
    """Check a rectangular entity map and its coefficient contraction."""
    coefficient = gem.Variable("coefficient", ())
    matrix = gem.ListTensor(np.asarray([
        [gem.Literal(1.0), gem.Zero(), coefficient],
        [gem.Zero(), gem.Literal(1.0), gem.Zero()],
    ], dtype=object))
    table_values = np.arange(1.0, 7.0).reshape(3, 2)
    table = gem.Literal(table_values)

    mapped_tabulation = MappedTabulation(matrix, {None: table}, {0: {0: [0, 1], 1: [2]}})
    i, j = gem.indices(2)
    mapped = contraction(gem.Indexed(mapped_tabulation[None], (i, j)))
    products = [node for node in traversal((mapped,)) if isinstance(node, gem.Product)]
    assert sum(map(operation_count, products)) <= 4

    actual, = evaluate(
        [gem.ComponentTensor(mapped, (i, j))],
        {coefficient: np.asarray(2.0)},
    )
    expected = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]) \
        @ table_values
    assert np.array_equal(actual.arr, expected)

    weights = gem.Variable("weights", (2,))
    value = contraction(gem.IndexSum(
        gem.Product(gem.Indexed(weights, (i,)),
                    gem.Indexed(mapped_tabulation[None], (i, j))), (i,)))
    output = gem.Indexed(gem.Variable("output", (2,)), (j,))
    # Transform the coefficient weights before looping over components.
    assert count_flops(compile_gem([(output, value)], (j,))) <= 13
    assert not any(isinstance(node, gem.Conditional) for node in traversal((value,)))

    actual, = evaluate(
        [gem.ComponentTensor(value, (j,))],
        {coefficient: np.asarray(2.0), weights: np.asarray([1.0, 10.0])},
    )
    assert np.array_equal(actual.arr, np.asarray([1.0, 10.0]) @ expected)


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
@pytest.mark.parametrize("restriction_indices", [None, [4, 1, 0]])
def test_entity_mapped_tabulation_restriction(shape, restriction_indices):
    """Entity couplings retain basis order, tensor components and point indices."""
    matrix_values = np.asarray([
        [1, 2, 0, 0, 3, 0, 0],
        [0, 4, 5, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 8],
        [0, 0, 0, 0, 9, 0, 10],
    ], dtype=float)
    scale = gem.Variable("scale", ())
    matrix = gem.ListTensor([[scale * value if value else gem.Zero()
                              for value in row] for row in matrix_values])
    retained = list(range(5)) if restriction_indices is None else restriction_indices
    reference_entities = {0: {0: [0, 1], 1: [2]}, 1: {0: [3, 4, 5, 6]}}
    q = gem.Index(extent=3)
    values = np.random.default_rng(42).normal(size=(3, 7, *shape))
    table = gem.partial_indexed(gem.Literal(values), (q,))
    mapped = MappedTabulation(matrix, {None: table}, reference_entities,
                              restriction_indices=restriction_indices)
    result, = evaluate([mapped[None]], {scale: np.asarray(1.5)})
    expected = np.einsum("ij,qj...->qi...", 1.5*matrix_values[retained], values)
    assert result.fids == (q,)
    assert np.allclose(result.arr, expected)
    weights = gem.Variable("weights", (len(retained),))
    i, *tail = gem.indices(1 + len(shape))
    value = contraction(gem.IndexSum(gem.Product(
        gem.Indexed(weights, (i,)), gem.Indexed(mapped[None], (i, *tail))), (i,)))
    coefficients = np.random.default_rng(19).normal(size=len(retained))
    actual, = evaluate([gem.ComponentTensor(value, tuple(tail))],
                       {scale: np.asarray(1.5), weights: coefficients})
    assert actual.fids == (q,)
    assert np.allclose(actual.arr, np.einsum("i,qi...->q...", coefficients, expected))
    assert mapped[None] is mapped[None]
    assert list(mapped) == [None]
    assert len(mapped) == 1
    with pytest.raises(KeyError):
        mapped["missing"]


@pytest.mark.parametrize("element", [finat.Argyris, finat.Bell, finat.HsiehCloughTocher])
def test_entity_mapped_tabulation_triangular(ref_to_phys, element):
    """Keep all couplings from vertex jets into facet functionals."""
    mapping = ref_to_phys[2]
    element = element(mapping.ref_cell)
    table = gem.Variable("reference", (element._element.space_dimension(),))
    result = element.map_tabulation({None: table}, mapping)[None]
    reference = np.random.default_rng(11).normal(size=table.shape)
    matrix, = evaluate([element.basis_transformation(mapping)])
    actual, = evaluate([result], {table: reference})
    assert np.allclose(actual.arr, matrix.arr @ reference)
    physical_dims = {dof: dim for dim, entities in element.entity_dofs().items()
                     for dofs in entities.values() for dof in dofs}
    reference_dims = {dof: dim for dim, entities in element._element.entity_dofs().items()
                      for dofs in entities.values() for dof in dofs}
    assert all(physical_dims[row] <= reference_dims[column]
               for row, column in zip(*np.nonzero(matrix.arr)))


class TranslatedMapping(MyMapping):
    """Translate the Jacobian afresh on every call, as a form compiler does."""

    def jacobian_at(self, point):
        i, j, k = gem.indices(3)
        product = gem.Product(gem.Indexed(gem.Literal(self.A), (i, k)),
                              gem.Indexed(gem.Literal(np.eye(len(self.A))), (k, j)))
        return gem.ComponentTensor(gem.IndexSum(product, (k,)), (i, j))


def test_mapped_tabulations_share_transformations(ref_to_phys):
    """Separately translated transformations share the same GEM expressions."""
    mapping = TranslatedMapping(ref_to_phys[2].ref_cell, ref_to_phys[2].phys_cell)
    element = finat.Hermite(mapping.ref_cell, 3)
    ps = finat.point_set.PointSet([[0.25, 0.25]])
    values = element.basis_evaluation(0, ps, coordinate_mapping=mapping)
    gradients = element.basis_evaluation(1, ps, coordinate_mapping=mapping)
    assert values[(0, 0)] == gradients[(0, 0)]


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


@pytest.mark.parametrize("element, degree", [
    *((finat.Hermite, k) for k in range(3, 6)),
])
def test_C1_interval(ref_to_phys, element, degree):
    check_zany_mapping(element, ref_to_phys[1], degree)


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.Hermite,
                         finat.Bell,
                         finat.WuXuH3NC,
                         finat.WuXuRobustH3NC,
                         ])
def test_C1_triangle(ref_to_phys, element):
    check_zany_mapping(element, ref_to_phys[2])


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.Hermite,
                         finat.Walkington,
                         ])
def test_C1_tetrahedron(ref_to_phys, element):
    check_zany_mapping(element, ref_to_phys[3])


@pytest.mark.parametrize("element", [
                         finat.QuadraticPowellSabin6,
                         finat.QuadraticPowellSabin12,
                         finat.ReducedHsiehCloughTocher,
                         ])
def test_C1_macroelements(ref_to_phys, element):
    kwargs = {}
    if element == finat.QuadraticPowellSabin12:
        kwargs = dict(avg=True)
    check_zany_mapping(element, ref_to_phys[2], **kwargs)


@pytest.mark.parametrize("element, degree", [
    *((finat.Argyris, k) for k in range(5, 8)),
    *((finat.HsiehCloughTocher, k) for k in range(3, 6)),
    *((finat.AlfeldC2, k) for k in range(5, 7)),
    *((finat.BrambleZlamalC2, k) for k in range(9, 11)),
])
def test_high_order_Ck_elements(ref_to_phys, element, degree):
    check_zany_mapping(element, ref_to_phys[2], degree, avg=True)


def test_argyris_point(ref_to_phys):
    check_zany_mapping(finat.Argyris, ref_to_phys[2], variant="point")


zany_piola_elements = {
    2: [
        finat.ReducedArnoldQin,
        finat.ArnoldWinther,
        finat.ArnoldWintherNC,
    ],
    3: [
        finat.MardalTaiWinther,
        finat.BernardiRaugel,
        finat.BernardiRaugelBubble,
        finat.AlfeldSorokina,
        finat.ChristiansenHu,
        finat.JohnsonMercier,
        finat.GuzmanNeilanFirstKindH1,
        finat.GuzmanNeilanSecondKindH1,
        finat.GuzmanNeilanBubble,
        finat.GuzmanNeilanH1div,
    ],
}


@pytest.mark.parametrize("dimension, element", [
    *((2, e) for e in zany_piola_elements[2]),
    *((2, e) for e in zany_piola_elements[3]),
    *((3, e) for e in zany_piola_elements[3]),
])
def test_piola(ref_to_phys, element, dimension):
    check_zany_mapping(element, ref_to_phys[dimension])


@pytest.mark.parametrize("dimension, element, degree", [
    (3, finat.MardalTaiWinther, 2),
    (3, finat.GuzmanNeilanFirstKindH1, 2),
])
def test_high_order_stokes_elements(ref_to_phys, element, dimension, degree):
    check_zany_mapping(element, ref_to_phys[dimension], degree)


@pytest.mark.parametrize("element, degree, variant", [
    *((finat.HuZhang, k, v) for v in ("integral", "point") for k in range(3, 6)),
])
def test_piola_triangle_high_order(ref_to_phys, element, degree, variant):
    check_zany_mapping(element, ref_to_phys[2], degree, variant)


@pytest.mark.parametrize("element, degree", [
                         *((finat.Regge, k) for k in range(3)),
                         *((finat.HellanHerrmannJohnson, k) for k in range(3)),
                         *((finat.GopalakrishnanLedererSchoberlFirstKind, k) for k in range(1, 4)),
                         *((finat.GopalakrishnanLedererSchoberlSecondKind, k) for k in range(0, 3)),
                         ])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("variant", [None, "alfeld"])
def test_affine(ref_to_phys, element, degree, variant, dimension):
    check_zany_mapping(element, ref_to_phys[dimension], degree, variant=variant)


@pytest.mark.parametrize("element", [finat.BrezziDouglasMarini, finat.NedelecSecondKind])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("variant", [None, "iso"])
def test_macro_piola(ref_to_phys, element, degree, variant, dimension):
    check_zany_mapping(element, ref_to_phys[dimension], degree, variant=variant)
