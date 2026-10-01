import numpy
import pytest

import finat
import finat.ufl
import gem
import ufl
from finat.element_factory import create_element
from finat.point_set import PointSet
from FIAT.reference_element import ufc_simplex


@pytest.mark.parametrize("element", (finat.BerchenkoKoganGawlikH1, finat.BerchenkoKoganGawlikHCurl,
                                     finat.BerchenkoKoganGawlikHDiv, finat.BerchenkoKoganGawlikL2))
def test_basis_evaluation(element):
    cell = ufc_simplex(2)
    fe = element(cell)
    pts = numpy.array([[0.2, 0.3], [0.6, 0.1], [0.1, 0.7]])
    expected = fe.fiat_equivalent.tabulate(1, pts)
    result = fe.basis_evaluation(1, PointSet(pts))
    for alpha, table in expected.items():
        value, = gem.interpreter.evaluate([result[alpha]])
        value = numpy.broadcast_to(value.arr, (len(pts), *table.shape[:-1]))
        assert numpy.allclose(numpy.moveaxis(value, 0, -1), table)


@pytest.mark.parametrize("family,element,mapping", (("BKGH1", finat.BerchenkoKoganGawlikH1, "affine"),
                                                    ("BKGHCurl", finat.BerchenkoKoganGawlikHCurl, "covariant piola"),
                                                    ("BKGHDiv", finat.BerchenkoKoganGawlikHDiv, "contravariant piola"),
                                                    ("BKGL2", finat.BerchenkoKoganGawlikL2, "affine")))
def test_create_element(family, element, mapping):
    fe = create_element(finat.ufl.FiniteElement(family, ufl.triangle, 1))
    assert isinstance(fe, element)
    assert fe.mapping == mapping
