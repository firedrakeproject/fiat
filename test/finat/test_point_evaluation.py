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
    (finat.MardalTaiWinther, 1),
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


@pytest.mark.parametrize('degree', [1, 4])
def test_duffy_evaluation(cell, degree):
    from FIAT.expansions import lexicographic_multiindices
    from finat.point_set import PointSet, CollapsedTensorProductPointSet

    dim = cell.get_spatial_dimension()
    element = finat.Legendre(cell, degree, variant="integral")

    # Unequal point counts per axis, including the collapsed vertex eta=1
    factors = [PointSet(numpy.linspace(0, 1, 3 + axis)[:, None]) for axis in range(dim)]
    ps = CollapsedTensorProductPointSet(factors)
    multiindex, duffy = element.duffy_evaluation(1, ps)

    dense_ps = PointSet(ps.points)
    expected = element.basis_evaluation(1, dense_ps)
    assert expected.keys() == duffy.keys()

    # dof j's lattice coordinates, per FIAT.hierarchical.LegendreDual's
    # lattice-lexicographic dof order.
    flat_of = {tuple(row): j for j, row in enumerate(lexicographic_multiindices(dim, degree))}
    lattice_shape = (degree + 1,) * dim
    for alpha, table in expected.items():
        exp, = gem.interpreter.evaluate([table])
        exp = exp.broadcast(dense_ps.indices)
        act, = gem.interpreter.evaluate([duffy[alpha]])
        act = act.broadcast(multiindex + ps.indices).reshape(*lattice_shape, -1)
        for index in numpy.ndindex(lattice_shape):
            if sum(index) > degree:
                assert numpy.allclose(act[index], 0.0)
            else:
                assert numpy.allclose(act[index], exp[:, flat_of[index]],
                                      rtol=1E-10, atol=1E-12)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
