import FIAT
import finat
import numpy as np
import pytest

from gem.interpreter import evaluate
from finat.functional import DERIVATIVE, FunctionalData


@pytest.mark.parametrize("dimension", [2, 3])
def test_functional_from_fiat(dimension):
    """The coefficient tensors of the Morley nodes are recovered from the
    FIAT functional dictionaries."""
    cell = FIAT.ufc_simplex(dimension)
    element = FIAT.Morley(cell)
    entity_ids = element.entity_dofs()
    nodes = element.dual_basis()

    for i in entity_ids[dimension - 2][0]:
        ell = FunctionalData.from_fiat(nodes[i])
        assert ell.mappings == ()
        assert ell.coefficients.shape == (len(ell.points),)

    for entity in entity_ids[dimension - 1]:
        for i in entity_ids[dimension - 1][entity]:
            ell = FunctionalData.from_fiat(nodes[i])
            assert ell.mappings == (DERIVATIVE,)
            normal = cell.compute_normal(entity)
            direction = ell.coefficients.sum(axis=0)
            cosine = direction @ normal
            assert np.isclose(abs(cosine), np.linalg.norm(direction) * np.linalg.norm(normal))


auto_elements = [finat.Morley, finat.Hermite]


@pytest.mark.parametrize("element", auto_elements)
@pytest.mark.parametrize("dimension", [2, 3])
def test_conditioning_scaling(ref_to_phys, scaled_ref_to_phys, element, dimension):
    """Derivative dofs are rescaled by cell size to the power of the
    derivative order."""
    scaled = scaled_ref_to_phys[dimension][-1]
    # the same geometric mapping with unit cell size
    unit = type(ref_to_phys[dimension])(scaled.ref_cell, scaled.phys_cell)
    finat_element = element(scaled.ref_cell)

    Ms = evaluate([finat_element.basis_transformation(scaled)])[0].arr
    Mu = evaluate([finat_element.basis_transformation(unit)])[0].arr

    h = scaled.cell_size()[0]
    assert not np.isclose(h, 1)
    orders = [node.max_deriv_order for node in finat_element._element.dual_basis()]
    expected = Mu * np.asarray([h**-order for order in orders])[:, None]
    assert np.allclose(Ms, expected)
