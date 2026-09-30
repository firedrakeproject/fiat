import pytest
import ufl
from finat.ufl import (EnrichedElement, FiniteElement, HDivElement,
                       MixedElement, TensorElement, TensorProductElement,
                       VectorElement)


def test_reconstruct_degree_plain_element():
    """A plain FiniteElement's degree is set absolutely."""
    CG3 = FiniteElement("Lagrange", ufl.triangle, 3)
    assert CG3.reconstruct(degree=1) == FiniteElement("Lagrange", ufl.triangle, 1)


def test_reconstruct_degree_mixed_preserves_offset():
    """Reconstructing a heterogeneous MixedElement (e.g. Taylor-Hood) at a lower
    degree must shift every sub-element by the same amount, not clobber them
    all to the same absolute degree.
    """
    CG4 = FiniteElement("Lagrange", ufl.triangle, 4)
    CG2 = FiniteElement("Lagrange", ufl.triangle, 2)
    TH = MixedElement(CG4, CG2)

    coarse = TH.reconstruct(degree=3)

    CG3 = FiniteElement("Lagrange", ufl.triangle, 3)
    CG1 = FiniteElement("Lagrange", ufl.triangle, 1)
    assert coarse == MixedElement(CG3, CG1)


def test_reconstruct_degree_enriched_preserves_offset():
    """Same offset-preservation requirement for EnrichedElement."""
    CG4 = FiniteElement("Lagrange", ufl.triangle, 4)
    CG2 = FiniteElement("Lagrange", ufl.triangle, 2)
    E = EnrichedElement(CG4, CG2)

    coarse = E.reconstruct(degree=3)

    CG3 = FiniteElement("Lagrange", ufl.triangle, 3)
    CG1 = FiniteElement("Lagrange", ufl.triangle, 1)
    assert coarse == EnrichedElement(CG3, CG1)


def test_reconstruct_degree_vector_and_tensor():
    """VectorElement/TensorElement wrap a single repeated sub-element, so an
    absolute degree is already offset-preserving.
    """
    CG4 = FiniteElement("Lagrange", ufl.triangle, 4)
    CG3 = FiniteElement("Lagrange", ufl.triangle, 3)

    V = VectorElement(CG4)
    assert V.reconstruct(degree=3) == VectorElement(CG3)

    T = TensorElement(CG4)
    assert T.reconstruct(degree=3) == TensorElement(CG3)


def test_reconstruct_degree_tensor_product_preserves_offset():
    """TensorProductElement factors at different degrees must each shift by
    the same amount, matching the sum-of-factor-degrees embedded_superdegree
    convention used by this class.
    """
    CG3 = FiniteElement("Lagrange", ufl.interval, 3)
    DG1 = FiniteElement("Discontinuous Lagrange", ufl.interval, 1)
    tp = TensorProductElement(CG3, DG1)

    coarse = tp.reconstruct(degree=2)

    CG2 = FiniteElement("Lagrange", ufl.interval, 2)
    DG0 = FiniteElement("Discontinuous Lagrange", ufl.interval, 0)
    assert coarse == TensorProductElement(CG2, DG0)


def test_reconstruct_degree_ncf_nested_composition():
    """NCF is EnrichedElement(HDivElement(TensorProductElement(...)), ...), i.e.
    a real, practically-used deeply nested composite with heterogeneous
    sub-degrees at every level (built via RTCF, itself another EnrichedElement
    of TensorProductElements). Reconstructing it at a lower degree must land
    exactly on the directly-constructed lower-degree element.
    """
    cell = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
    fine = FiniteElement("NCF", cell, 3)
    coarse = fine.reconstruct(degree=2)
    assert coarse == FiniteElement("NCF", cell, 2)


def test_reconstruct_degree_hdiv_delegates_to_wrapee():
    """HDivElement (and similarly HCurlElement/WithMapping/BrokenElement/
    RestrictedElement) wrap a single sub-element, so reconstruct(degree=...)
    is expected to delegate straight through without any shift bookkeeping
    of its own.
    """
    tp3 = TensorProductElement(FiniteElement("Lagrange", ufl.interval, 3),
                               FiniteElement("Discontinuous Lagrange", ufl.interval, 1))
    hdiv = HDivElement(tp3)

    coarse = hdiv.reconstruct(degree=2)

    tp2 = TensorProductElement(FiniteElement("Lagrange", ufl.interval, 2),
                               FiniteElement("Discontinuous Lagrange", ufl.interval, 0))
    assert coarse == HDivElement(tp2)


@pytest.mark.parametrize("modifier", (TensorProductElement, MixedElement, EnrichedElement))
def test_reconstruct_degree_tuple_sets_explicit_degrees(modifier):
    """Passing degree as a list/tuple gives the explicit per-sub-element
    degrees directly, instead of shifting each sub-element by a common
    offset.
    """
    CG4 = FiniteElement("Lagrange", ufl.interval, 4)
    DG2 = FiniteElement("Discontinuous Lagrange", ufl.interval, 2)
    composite = modifier(CG4, DG2)

    explicit = composite.reconstruct(degree=(1, 3))

    CG1 = FiniteElement("Lagrange", ufl.interval, 1)
    DG3 = FiniteElement("Discontinuous Lagrange", ufl.interval, 3)
    assert explicit == modifier(CG1, DG3)


@pytest.mark.parametrize("degree", (1, 2, 3))
def test_reconstruct_degree_no_kwarg_is_noop(degree):
    """Reconstructing without a degree kwarg must leave every sub-element's
    degree untouched, for both plain and composite elements.
    """
    CG = FiniteElement("Lagrange", ufl.triangle, degree)
    DG = FiniteElement("Discontinuous Lagrange", ufl.triangle, max(degree - 1, 0))
    mixed = MixedElement(CG, DG)
    assert mixed.reconstruct() == mixed
