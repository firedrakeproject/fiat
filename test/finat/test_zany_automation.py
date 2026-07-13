import finat
import numpy as np
import pytest

from gem.interpreter import evaluate
from finat.zany import zany_basis_transformation


class AutoMorley(finat.Morley):
    """Morley element with automatically derived basis transformation."""
    def basis_transformation(self, coordinate_mapping):
        return zany_basis_transformation(self._element, coordinate_mapping)


auto_elements = {
    AutoMorley: finat.Morley,
}


@pytest.mark.parametrize("element", auto_elements)
@pytest.mark.parametrize("dimension", [2, 3])
def test_auto_transformation(check_zany_mapping, ref_to_phys, element, dimension):
    check_zany_mapping(element, ref_to_phys[dimension])


@pytest.mark.parametrize("element", auto_elements)
@pytest.mark.parametrize("dimension", [2, 3])
def test_auto_matches_handcoded(scaled_ref_to_phys, element, dimension):
    handcoded = auto_elements[element]
    for mapping in scaled_ref_to_phys[dimension]:
        cell = mapping.ref_cell
        Ma = evaluate([element(cell).basis_transformation(mapping)])[0].arr
        Mh = evaluate([handcoded(cell).basis_transformation(mapping)])[0].arr
        assert np.allclose(Ma, Mh, atol=1e-14)
