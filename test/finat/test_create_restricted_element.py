import pytest
import ufl
from finat.ufl import FiniteElement, RestrictedElement, VectorElement, TensorElement, MixedElement

sub_elements = [
    FiniteElement("CG", ufl.triangle, 1),
    FiniteElement("BDM", ufl.triangle, 2),
    FiniteElement("DG", ufl.interval, 2, variant="spectral")
]

sub_ids = [
    "CG(1)",
    "BDM(2)",
    "DG(2,spectral)"
]


@pytest.mark.parametrize("sub_element", sub_elements, ids=sub_ids)
@pytest.mark.parametrize("shape", (1, 2, (2, 3)), ids=("1", "2", "(2,3)"))
def test_create_restricted_vector_or_tensor_element(shape, sub_element):
    """Check that RestrictedElement returns a nested element
    for mixed, vector, and tensor elements.
    """
    if not isinstance(shape, int):
        make_element = lambda elem: TensorElement(elem, shape=shape)
    else:
        make_element = lambda elem: VectorElement(elem, dim=shape)

    tensor = make_element(sub_element)
    expected = make_element(RestrictedElement(sub_element, "interior"))

    assert RestrictedElement(tensor, "interior") == expected


@pytest.mark.parametrize("sub_elements", [sub_elements, sub_elements[-1:]],
                         ids=(f"nsubs={len(sub_elements)}", "nsubs=1"))
def test_create_restricted_mixed_element(sub_elements):
    """Check that RestrictedElement returns a nested element
    for mixed, vector, and tensor elements.
    """
    mixed = MixedElement(sub_elements)
    expected = MixedElement([elem["facet"] for elem in sub_elements])
    assert mixed["facet"] == expected
