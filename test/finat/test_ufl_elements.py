import ufl
import finat.ufl


def test_extract_subelement_component():
    cell = ufl.triangle
    domain = ufl.Mesh(finat.ufl.VectorElement(finat.ufl.FiniteElement("Lagrange", cell, 1)))

    V = finat.ufl.VectorElement(finat.ufl.FiniteElement("Lagrange", cell, 2))
    Q = finat.ufl.FiniteElement("Lagrange", cell, 1)
    Z = V * Q

    space = ufl.FunctionSpace(domain, Z)
    test = ufl.TestFunction(space)

    for i in range(3):
        expr = test[i]
        _, multiindex = expr.ufl_operands
        subindex, _ = Z.extract_subelement_component(domain, multiindex)
        sub_elem = Z.sub_elements[subindex]
        assert sub_elem is (Q if i == 2 else V)
