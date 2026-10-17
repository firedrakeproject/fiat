import FIAT
import gem
import numpy as np
import pytest

from gem.interpreter import evaluate
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


class MappedElement(PhysicallyMappedElement, FiatElement):
    pass


def test_divergence_detection_respects_tolerance():
    """Near-divergence tensors are not classified using NumPy's default tolerance."""
    ref_el = FIAT.ufc_simplex(2)
    node = FIAT.functional.Functional(
        ref_el, (2,), {},
        {(0.0, 0.0): [(1.0, (1, 0), (0,)),
                      (1.0 + 1.0e-6, (0, 1), (1,))]},
        "NearDivergence")

    _, _, mappings = node.as_tensor(mapping="contravariant piola", tol=1.0e-12)

    assert mappings == ("contravariant piola", "derivative")


@pytest.mark.parametrize("scaled", [False, True])
def test_recombined_physical_dual(ref_to_phys, scaled_ref_to_phys, scaled):
    element = FIAT.Hermite(FIAT.ufc_simplex(2), 3)
    C = np.eye(element.space_dimension())
    for dofs in element.entity_dofs()[0].values():
        C[np.ix_(dofs, dofs)] = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    basis = element.get_nodal_basis().recombine(np.linalg.inv(C).T)
    recombined = FIAT.finite_element.CiarletElement(
        basis, element.get_dual_set(), 3, recombine_dual=True)
    assert np.allclose(recombined.get_dual_set().get_coeffs(), C)
    original = MappedElement(element)
    transformed = MappedElement(recombined)
    mapping = scaled_ref_to_phys[2][1] if scaled else ref_to_phys[2]
    Q = gem.Literal(np.eye(len(C)))
    B, BC, M = [result.arr for result in evaluate([
        original.dual_transformation(Q, mapping), transformed.dual_transformation(Q, mapping),
        transformed.basis_transformation(mapping)])]
    assert np.allclose(BC, C @ B @ np.linalg.inv(C))
    assert np.allclose(BC @ M.T, np.eye(len(C)))


@pytest.mark.parametrize("avg", [False, True])
def test_single_point_moment(avg):
    cell = FIAT.ufc_simplex(2)
    Q = FIAT.quadrature_schemes.create_quadrature(cell.construct_subelement(1), 0)
    node = FIAT.functional.IntegralMomentOfNormalDerivative(cell, 0, Q, np.ones(1))
    ids = {0: {v: [] for v in cell.get_topology()[0]},
           1: {e: [0] if e == 0 else [] for e in cell.get_topology()[1]}, 2: {0: []}}
    dual = FIAT.dual_set.DualSet([node], cell, ids)
    points, coefficients, axes = dual.as_tensor(["affine"])
    assert len(points[0]) == 1
    assert dual.node_mapping[0] == "frame"
    assert isinstance(node, FIAT.functional.IntegralMoment)
    assert dual.get_scaling_exponents(coefficients, axes, avg=avg)[0] == int(avg)


def test_explicit_integral_convention(ref_to_phys):
    element = FIAT.CrouzeixRaviart(FIAT.ufc_simplex(2), 1, variant="integral")
    dual = element.get_dual_set()
    dual = FIAT.dual_set.DualSet(dual.nodes, dual.ref_el, dual.entity_ids,
                                 node_mapping=["frame"] * len(dual))
    element = FIAT.finite_element.CiarletElement(element.get_nodal_basis(), dual, 1)
    mapped = MappedElement(element)
    mapped.avg = False
    B, M = [result.arr for result in evaluate([
        mapped.dual_transformation(gem.Literal(np.eye(3)), ref_to_phys[2]), mapped.basis_transformation(ref_to_phys[2])])]
    lengths = [ref_to_phys[2].phys_cell.volume_of_subcomplex(1, e) for e in range(3)]
    assert np.allclose(B, np.diag(lengths))
    assert np.allclose(B @ M.T, np.eye(3))
