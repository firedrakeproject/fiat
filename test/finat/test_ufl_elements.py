import pytest
import numpy
import ufl
import finat.ufl


@pytest.fixture
def domain():
    cell = ufl.triangle
    return ufl.Mesh(finat.ufl.VectorElement(finat.ufl.FiniteElement("Lagrange", cell, 1)))


def test_extract_subelement_component(domain):
    cell = domain.ufl_cell()
    V = finat.ufl.VectorElement(finat.ufl.FiniteElement("Lagrange", cell, 2))
    Q = finat.ufl.FiniteElement("Lagrange", cell, 1)
    Z = V * Q

    # ufl.FunctionSpace now requires a MeshSequence
    mixed_mesh = ufl.MeshSequence([domain]*len(Z.sub_elements))
    space = ufl.FunctionSpace(mixed_mesh, Z)
    test = ufl.TestFunction(space)

    for i in range(3):
        expr = test[i]
        _, multiindex = expr.ufl_operands
        subindex, _ = Z.extract_subelement_component(multiindex, domain)
        sub_elem = Z.sub_elements[subindex]
        assert sub_elem is (Q if i == 2 else V)


@pytest.mark.parametrize("family", ("Lagrange", "N1curl"))
def test_symmetry(domain, family):
    cell = domain.ufl_cell()
    Q = finat.ufl.FiniteElement(family, cell, 1)
    V = finat.ufl.VectorElement(Q)
    T = finat.ufl.TensorElement(Q)

    mapping = Q.mapping()
    for element in (Q, V, T):
        symmetry = element.symmetry(domain=domain)
        assert isinstance(symmetry, dict)
        assert len(symmetry) == 0

        if mapping == "identity":
            symmetry = element.symmetry()
            assert isinstance(symmetry, dict)
            assert len(symmetry) == 0

    S = finat.ufl.TensorElement(Q, symmetry=True)
    value_size = numpy.prod(Q.reference_value_shape)
    phys_size = numpy.prod(S.pullback.physical_value_shape(S, domain))
    ref_size = numpy.prod(S.reference_value_shape)
    assert phys_size > ref_size
    symmetry_size = (phys_size - ref_size) // value_size

    symmetry = S.symmetry(domain=domain)
    assert isinstance(symmetry, dict)
    assert len(symmetry) == symmetry_size

    if mapping == "identity":
        symmetry = S.symmetry()
        assert isinstance(symmetry, dict)
        assert len(symmetry) == symmetry_size
