import pytest

import FIAT
import finat
import gem
import numpy


@pytest.fixture(params=[1, 2, 3])
def cell(request):
    dim = request.param
    return FIAT.ufc_simplex(dim)


@pytest.mark.parametrize('degree', [1, 2])
def test_cellwise_constant(cell, degree):
    dim = cell.get_spatial_dimension()
    element = finat.Lagrange(cell, degree)
    index = gem.Index()
    point = gem.partial_indexed(gem.Variable('X', (17, dim)), (index,))

    order = 2
    for alpha, table in element.point_evaluation(order, point).items():
        if sum(alpha) < degree:
            assert table.free_indices == (index,)
        else:
            assert table.free_indices == ()


@pytest.mark.parametrize("element,degree", [
    (finat.HsiehCloughTocher, 3),
    (finat.Argyris, 5),
    (finat.MardalTaiWinther, 3),
])
def test_point_evaluation_zany(ref_to_phys, element, degree):
    dim = 2
    ref_cell = ref_to_phys[dim].ref_cell
    phys_cell = ref_to_phys[dim].phys_cell
    A, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices, phys_cell.vertices)

    ref_pt = numpy.array([0.2, 0.3])
    phys_pt = numpy.dot(A, ref_pt) + b

    finat_kwargs = {}
    if element in {finat.HsiehCloughTocher, finat.Argyris}:
        finat_kwargs["avg"] = True

    order = 0
    point = gem.Literal(ref_pt)
    ref_element = element(ref_cell, degree, **finat_kwargs)
    result = ref_element.point_evaluation(order, point, coordinate_mapping=ref_to_phys[dim])

    phys_element = element(phys_cell, degree, **finat_kwargs).fiat_equivalent
    expected = phys_element.tabulate(order, phys_pt)

    num_dof = ref_element.space_dimension()
    for alpha in result:

        ref_val, = gem.interpreter.evaluate([result[alpha]])

        if phys_element.mapping()[0] == "covariant piola":
            val = numpy.tensordot(ref_val.arr, A, (-1, 0))
        elif phys_element.mapping()[0] == "contravariant piola":
            detA = numpy.linalg.det(A)
            val = numpy.tensordot(ref_val.arr, A/detA, (-1, 1))
        else:
            val = ref_val.arr

        assert numpy.allclose(val, expected[alpha][:num_dof])


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
