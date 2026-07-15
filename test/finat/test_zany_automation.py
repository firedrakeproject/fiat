import FIAT
import finat
import numpy as np
import pytest

from gem.interpreter import evaluate
from finat.functional import PhysicallyMappedFunctional
from finat.fiat_elements import FiatElement
from finat.zany import ZanyPhysicallyMappedElement


@pytest.mark.parametrize("dimension", [2, 3])
def test_functional_from_fiat(dimension):
    """The symbolic PhysicallyMappedFunctional recovers order, weights and
    direction numerically from the FIAT functional dictionaries."""
    cell = FIAT.ufc_simplex(dimension)
    element = FIAT.Morley(cell)
    entity_ids = element.entity_dofs()
    nodes = element.dual_basis()

    for i in entity_ids[dimension - 2][0]:
        ell = PhysicallyMappedFunctional.from_fiat(nodes[i])
        assert ell.order == 0
        assert ell.direction is None

    for entity in entity_ids[dimension - 1]:
        for i in entity_ids[dimension - 1][entity]:
            ell = PhysicallyMappedFunctional.from_fiat(nodes[i])
            assert ell.order == 1
            normal = cell.compute_normal(entity)
            cosine = ell.direction @ normal
            assert np.isclose(abs(cosine), np.linalg.norm(normal))


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


def test_unsupported_mapping(ref_to_phys):
    """ZanyPhysicallyMappedElement rejects an unrecognized pullback."""
    mapping = ref_to_phys[2]

    class BogusZany(ZanyPhysicallyMappedElement, FiatElement):
        pass

    fiat_element = FIAT.Nedelec(mapping.ref_cell, 1)
    fiat_element._mapping = "nonsense"
    element = BogusZany(fiat_element)
    with pytest.raises(NotImplementedError):
        element.basis_transformation(mapping)
