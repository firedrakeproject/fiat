"""Structural sparsity with an opaque Jacobian: no geometric cancellation."""

from math import comb

import FIAT
import finat
import gem
import numpy as np
import pytest


class SymbolicMapping:
    def __init__(self, sd):
        self.J = gem.Variable("J", (sd, sd))
        self.h = gem.Variable("h", (sd + 1,))

    def jacobian_at(self, point):
        return self.J

    def cell_size(self):
        return self.h


def transformations(element, sd):
    dual = finat.PhysicallyMappedDualSet(element._element, range(element._element.space_dimension()), avg=element.avg)
    mapping = SymbolicMapping(sd)
    return dual.dual_transformation(mapping).array, dual.basis_transformation(mapping).array.T


def nonzeros(row):
    return {j for j, entry in enumerate(row) if not isinstance(entry, gem.Zero)}


@pytest.mark.parametrize("element, degree, first_degree", [
    (finat.HsiehCloughTocher, 3, 3),
    (finat.HsiehCloughTocher, 4, 3),
    (finat.HsiehCloughTocher, 5, 3),
    (finat.Argyris, 5, 5),
    (finat.Argyris, 6, 5),
    (finat.Argyris, 7, 5),
])
def test_edge_moment_sparsity(element, degree, first_degree):
    """Integration by parts couples a normal moment to its endpoints and matching trace moment."""
    cell = FIAT.ufc_simplex(2)
    element = element(cell, degree, avg=True)
    B, V = transformations(element, 2)
    dofs = element._element.entity_dofs()
    k = degree - first_degree
    for edge, edge_dofs in dofs[1].items():
        endpoints = {dofs[0][v][0] for v in cell.get_topology()[1][edge]}
        for mode, i in enumerate(edge_dofs[:k + 1]):
            # Integral p_i d_t f = [p_i f] - Integral p_i' f.
            expected = {i} | endpoints
            if mode:
                expected.add(edge_dofs[k + mode])
            assert nonzeros(B[i]) == expected
            assert nonzeros(V[i]) == expected
        for i in edge_dofs[k + 1:]:
            assert nonzeros(B[i]) == {i}
            assert nonzeros(V[i]) == {i}


@pytest.mark.parametrize("sd, element, degree", [
    (2, finat.Hermite, 3),
    (3, finat.Hermite, 3),
    (2, finat.Argyris, 5),
    (2, finat.AlfeldC2, 5),
    (2, finat.BrambleZlamalC2, 9),
    (3, finat.Walkington, 5),
])
def test_vertex_jet_sparsity(sd, element, degree):
    """An affine chain rule mixes only derivatives of the same order at the same vertex."""
    element = element(FIAT.ufc_simplex(sd), degree)
    B, V = transformations(element, sd)
    for dofs in element._element.entity_dofs()[0].values():
        for i in dofs:
            order = element._element.dual_basis()[i].max_deriv_order
            expected = {j for j in dofs if element._element.dual_basis()[j].max_deriv_order == order}
            assert nonzeros(B[i]) == expected
            assert nonzeros(V[i]) == expected


@pytest.mark.parametrize("sd, order", [(2, 1), (3, 1), (3, 2)])
def test_piola_facet_sparsity(sd, order):
    """MTW normal moments are invariant; each tangential moment couples only to matching normal weights."""
    element = finat.MardalTaiWinther(FIAT.ufc_simplex(sd), order)
    B, V = transformations(element, sd)
    nnormal = comb(order + sd - 1, sd - 1)
    for dofs in element._element.entity_dofs()[sd - 1].values():
        for i in dofs[:nnormal]:
            assert nonzeros(B[i]) == {i}
            assert nonzeros(V[i]) == {i}
        for mode, i in enumerate(dofs[nnormal:]):
            # Constant tangents pair with the constant normal moment;
            # the 3D radial tangent pairs with the two linear normal moments.
            normal_dofs = dofs[1:3] if mode == 2 else dofs[:1]
            expected = {i, *normal_dofs}
            assert nonzeros(B[i]) == expected
            assert nonzeros(V[i]) == expected


@pytest.mark.parametrize("sd, element, degree", [
    (2, finat.Bell, 5),
    (2, finat.ReducedHsiehCloughTocher, 3),
    (2, finat.MardalTaiWinther, 1),
    (3, finat.MardalTaiWinther, 2),
    (2, finat.ArnoldWinther, 3),
    (3, finat.Walkington, 5),
])
def test_entity_closure_sparsity(sd, element, degree):
    """Back-substitution introduces no couplings outside entity closures, including constraints."""
    element = element(FIAT.ufc_simplex(sd), degree)
    B, V = transformations(element, sd)
    closures = element._element.entity_closure_dofs()
    for dim, entities in element._element.entity_dofs().items():
        for entity, dofs in entities.items():
            allowed = set(closures[dim][entity])
            for i in dofs:
                assert nonzeros(B[i]) <= allowed
                assert nonzeros(V[i]) <= allowed

    ndof = element.space_dimension()
    M = element.basis_transformation(SymbolicMapping(sd)).array
    assert M.shape == (ndof, element._element.space_dimension())
    assert np.array_equal([[isinstance(entry, gem.Zero) for entry in row] for row in M],
                          [[isinstance(entry, gem.Zero) for entry in row] for row in V[:, :ndof].T])
