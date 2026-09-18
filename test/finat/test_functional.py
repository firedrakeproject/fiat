import FIAT

from FIAT.functional import Functional

from finat.functional import DERIVATIVE, FunctionalData


def test_divergence_detection_respects_tolerance():
    """Near-divergence tensors are not classified using NumPy's default tolerance."""
    ref_el = FIAT.ufc_simplex(2)
    node = Functional(
        ref_el, (2,), {},
        {(0.0, 0.0): [(1.0, (1, 0), (0,)),
                      (1.0 + 1.0e-6, (0, 1), (1,))]},
        "NearDivergence")

    functional = FunctionalData.from_fiat(node, mapping="contravariant piola", tol=1.0e-12)

    assert functional.mappings == ("contravariant piola", DERIVATIVE)
