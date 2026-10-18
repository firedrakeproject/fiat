from itertools import chain

import FIAT
import gem
import numpy as np
import pytest

from gem.interpreter import evaluate
from finat.fiat_elements import FiatElement
from finat.physically_mapped import (PhysicallyMappedElement, _cartesian_pullbacks, _duality_projector,
                                     _in_range, adjugate_and_determinant)


class MappedElement(PhysicallyMappedElement, FiatElement):
    pass


def test_duality_pairing_respects_tolerance():
    """Near-divergence weights are not paired using NumPy's default tolerance."""
    ref_el = FIAT.ufc_simplex(2)
    node = FIAT.functional.Functional(
        ref_el, (2,), {},
        {(0.0, 0.0): [(1.0, (1, 0), (0,)),
                      (1.0 + 1.0e-6, (0, 1), (1,))]},
        "NearDivergence")
    _, W = node.weight_tensors[1]
    P = _duality_projector(2)
    assert _in_range(np.eye(2)[..., None], (0, 1), P, 1.0e-12)
    assert not _in_range(W, (0, 1), P, 1.0e-12)


def pullback_matrix(pullback):
    coefficients, matrices = zip(*pullback.numerator())
    coefficients, d = (result.arr for result in evaluate([gem.ListTensor(coefficients), pullback.denominator]))
    return sum(c * C for c, C in zip(coefficients, matrices)) / d


@pytest.mark.parametrize("sd", [2, 3])
def test_duality_pullback(sd):
    """Pairing a contravariant axis with a derivative axis only splits the Kronecker product."""
    J = np.eye(sd) + np.random.default_rng(0).random((sd, sd))
    adjJ, detJ = adjugate_and_determinant(np.array([[gem.Literal(Jij) for Jij in row] for row in J], dtype=object))
    J = np.array([[gem.Literal(Jij) for Jij in row] for row in J], dtype=object)
    pullbacks = _cartesian_pullbacks(J, adjJ, detJ)
    contravariant, derivative, pair = (pullback_matrix(pullbacks[mapping])
                                       for mapping in ("contravariant piola", "derivative", "duality"))
    J, = evaluate([gem.ListTensor(J)])
    J = J.arr
    assert np.allclose(contravariant, J.T / np.linalg.det(J))
    assert np.allclose(derivative, np.linalg.inv(J))
    assert np.allclose(pair, np.kron(contravariant, derivative))
    P = _duality_projector(sd)
    assert np.allclose(pair @ P, P / np.linalg.det(J))


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
    """A normal derivative moment at a single point is still an integral moment."""
    element = MappedElement(FIAT.Morley(FIAT.ufc_simplex(2)))
    element.avg = avg
    dual = element._element.get_dual_set()
    for i in chain.from_iterable(dual.entity_ids[1].values()):
        points, _ = dual.nodes[i].weight_tensors[1]
        assert len(points) == 1
        assert dual.node_mapping[i] == "frame"
        assert isinstance(dual.nodes[i], FIAT.functional.IntegralMoment)
        assert element._scaling_exponents[i] == int(avg)


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
