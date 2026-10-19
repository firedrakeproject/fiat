# Copyright (C) 2015-2016 Jan Blechta
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy as np
import pytest

from FIAT.reference_element import LINE, ReferenceElement
from FIAT.reference_element import Point, UFCInterval, UFCTriangle, UFCTetrahedron, ufc_simplex
from FIAT.lagrange import Lagrange
from FIAT.discontinuous_lagrange import DiscontinuousLagrange   # noqa: F401
from FIAT.discontinuous_taylor import DiscontinuousTaylor       # noqa: F401
from FIAT.P0 import P0                                          # noqa: F401
from FIAT.crouzeix_raviart import CrouzeixRaviart               # noqa: F401
from FIAT.raviart_thomas import RaviartThomas                   # noqa: F401
from FIAT.discontinuous_raviart_thomas import DiscontinuousRaviartThomas  # noqa: F401
from FIAT.bernstein import Bernstein  # noqa: F401
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini      # noqa: F401
from FIAT.mixed import MixedElement
from FIAT.nedelec import Nedelec                                # noqa: F401
from FIAT.nedelec_second_kind import NedelecSecondKind          # noqa: F401
from FIAT.regge import Regge                                    # noqa: F401
from FIAT.hdiv_trace import HDivTrace, map_to_reference_facet   # noqa: F401
from FIAT.hellan_herrmann_johnson import HellanHerrmannJohnson  # noqa: F401
from FIAT.gopalakrishnan_lederer_schoberl import GopalakrishnanLedererSchoberlFirstKind  # noqa: F401
from FIAT.gopalakrishnan_lederer_schoberl import GopalakrishnanLedererSchoberlSecondKind  # noqa: F401
from FIAT.brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini  # noqa: F401
from FIAT.gauss_legendre import GaussLegendre                   # noqa: F401
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre    # noqa: F401
from FIAT.restricted import RestrictedElement                   # noqa: F401
from FIAT.tensor_product import TensorProductElement            # noqa: F401
from FIAT.tensor_product import FlattenedDimensions             # noqa: F401
from FIAT.hdivcurl import Hdiv, Hcurl                           # noqa: F401
from FIAT.mardal_tai_winther import MardalTaiWinther            # noqa: F401
from FIAT.arnold_winther import ArnoldWinther, ArnoldWintherNC  # noqa: F401
from FIAT.hu_zhang import HuZhang                               # noqa: F401
from FIAT.bernardi_raugel import BernardiRaugel                 # noqa: F401
from FIAT.argyris import Argyris                                # noqa: F401
from FIAT.hermite import Hermite                                # noqa: F401
from FIAT.morley import Morley                                  # noqa: F401
from FIAT.hct import HsiehCloughTocher                          # noqa: F401
from FIAT.c2_elements import AlfeldC2, BrambleZlamalC2          # noqa: F401
from FIAT.alfeld_sorokina import AlfeldSorokina                 # noqa: F401
from FIAT.arnold_qin import ArnoldQin                           # noqa: F401
from FIAT.christiansen_hu import ChristiansenHu                 # noqa: F401
from FIAT.guzman_neilan import GuzmanNeilanFirstKindH1          # noqa: F401
from FIAT.guzman_neilan import GuzmanNeilanSecondKindH1         # noqa: F401
from FIAT.johnson_mercier import JohnsonMercier                 # noqa: F401
from FIAT.bubble import Bubble, FacetBubble                     # noqa: F401
from FIAT.enriched import EnrichedElement                       # noqa: F401
from FIAT.nodal_enriched import NodalEnrichedElement
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen    # noqa: F401
from FIAT.histopolation import Histopolation                    # noqa: F401
from FIAT.wuxu import WuXuH3NC, WuXuRobustH3NC                  # noqa: F401
from FIAT import dual_set, functional, polynomial_set, quadrature_schemes
from FIAT.walkington import Walkington                         # noqa: F401

P = Point()
I = UFCInterval()  # noqa: E741
T = UFCTriangle()
S = UFCTetrahedron()


def test_integral_moment_point_and_weight_accessors():
    """Integral moment accessors agree with the public point dictionary."""
    cell = ufc_simplex(2)
    quadrature = quadrature_schemes.create_quadrature(cell, 4)
    values = 1 + quadrature.get_points().sum(axis=1)
    moment = functional.IntegralMoment(cell, quadrature, values)

    point_dict = moment.get_point_dict()
    assert moment.points_by_order == {0: tuple(point_dict)}
    assert np.allclose(moment.weights_by_order[0],
                       [entries[0][0] for entries in point_dict.values()])


def contract_functional(node, derivative, sd):
    """Contract a functional's public point and weight accessors."""
    value = 0
    for order, points in node.points_by_order.items():
        weights = node.weights_by_order[order]
        values = np.empty_like(weights)
        for q, point in enumerate(points):
            for index in np.ndindex((sd,) * order):
                alpha = tuple(np.bincount(index, minlength=sd))
                values[(..., *index, q)] = derivative(np.asarray(point), alpha)
        value += np.sum(weights * values)
    return value


@pytest.mark.parametrize("sd", [2, 3])
@pytest.mark.parametrize("kind", ["directional", "component", "mixed", "divergence", "tensor divergence"])
def test_derivative_moment(sd, kind):
    cell = ufc_simplex(sd)
    quadrature = quadrature_schemes.create_quadrature(cell, 6)
    points = quadrature.get_points()
    f = 1 + points.sum(axis=1)
    a = np.arange(1, sd + 1, dtype=float)
    n = a / np.linalg.norm(a)
    t = a[::-1]
    if kind in {"directional", "component", "mixed"}:
        directions = (n, t) if kind == "mixed" else (n,)
        shape = (sd,) if kind == "component" else ()
        comp = (sd - 1,) if shape else ()
        node = functional.IntegralMomentOfDerivative(cell, quadrature, f, *directions,
                                                     comp=comp, shp=shape)
        order = len(directions)
        expected = f * (6 if order == 2 else 3) * (points @ a)**(3 - order)
        expected *= np.prod([a @ direction for direction in directions])

        def derivative(point, alpha):
            value = (6 if order == 2 else 3) * (point @ a)**(3 - order)
            value *= np.prod(a**alpha)
            if shape:
                result = np.zeros(shape)
                result[comp] = value
                return result
            return np.asarray(value)
    elif kind == "divergence":
        shape = (sd,)
        node = functional.IntegralMomentOfDivergence(cell, quadrature, f)
        expected = f * 2 * points.sum(axis=1)

        def derivative(point, alpha):
            return 2 * point * np.asarray(alpha)
    else:
        shape = (sd, sd)
        test_values = a[:, None] * f
        node = functional.IntegralMomentOfTensorDivergence(cell, quadrature, test_values)
        expected = (sd + 1) * (points @ a) * f

        def derivative(point, alpha):
            return np.outer(alpha, point) + np.outer(point, alpha)

    value = contract_functional(node, derivative, sd)
    assert np.allclose(value, expected @ quadrature.get_weights())
    assert isinstance(node, functional.IntegralMomentOfDerivative)

    ids = {dim: {entity: [] for entity in entities} for dim, entities in cell.get_topology().items()}
    ids[sd][0] = [0]
    dual = dual_set.DualSet([node], cell, ids)
    poly = polynomial_set.ONPolynomialSet(cell, 3, shape=shape)
    assert np.allclose(dual.to_riesz(poly)[0], node.to_riesz(poly))


@pytest.mark.parametrize("sd", [2, 3])
def test_normal_derivative_moment(sd):
    cell = ufc_simplex(sd)
    quadrature = quadrature_schemes.create_quadrature(cell.construct_subelement(sd - 1), 4)
    a = np.arange(1, sd + 1, dtype=float)
    for facet in cell.get_topology()[sd - 1]:
        node = functional.IntegralMomentOfNormalDerivative(
            cell, facet, quadrature, np.ones(len(quadrature.get_points())))
        value = np.einsum("dq,d->", node.weights_by_order[1], a)
        expected = quadrature.get_weights().sum() * (a @ cell.compute_normal(facet))
        assert np.allclose(value, expected)


@pytest.mark.parametrize("element, degree", [
    (Argyris, 6), (HsiehCloughTocher, 4), (Walkington, 5),
    (ArnoldWinther, 3), (MardalTaiWinther, 1), (Nedelec, 3),
    (BrezziDouglasMarini, 2)])
def test_integral_moment_points_and_weights(element, degree):
    """Public point and weight accessors describe integral moment nodes."""
    sd = 3 if element in {Walkington, Nedelec} else 2
    kwargs = {"variant": "integral"} if element in {Nedelec, BrezziDouglasMarini} else {}
    nodes = element(ufc_simplex(sd), degree, **kwargs).get_dual_set().get_nodes()
    moments = [node for node in nodes if isinstance(node, functional.IntegralMoment)]
    assert moments
    for node in moments:
        points_by_order = node.points_by_order
        weights_by_order = node.weights_by_order
        assert set(points_by_order) == set(weights_by_order)
        for order, points in points_by_order.items():
            assert weights_by_order[order].shape[-1] == len(points)
            assert all(len(point) == sd for point in points)
        point_dict = node.get_point_dict()
        if point_dict:
            assert points_by_order[0] == tuple(point_dict)
            expected = np.zeros_like(weights_by_order[0])
            for q, entries in enumerate(point_dict.values()):
                for weight, component in entries:
                    component = component if isinstance(component, tuple) else (component,)
                    expected[(*component, q)] += weight
            assert np.allclose(weights_by_order[0], expected)


@pytest.mark.parametrize("sd", [2, 3])
def test_uncontracted_riesz(sd):
    """Inserting a map between the derivative axes changes the direction of the derivative."""
    cell = ufc_simplex(sd)
    quadrature = quadrature_schemes.create_quadrature(cell, 4)
    f = 1 + quadrature.get_points().sum(axis=1)
    n = np.arange(1, sd + 1, dtype=float)
    G = np.random.default_rng(0).random((sd, sd))
    ids = {dim: {entity: [] for entity in entities} for dim, entities in cell.get_topology().items()}
    ids[sd][0] = [0]
    poly = polynomial_set.ONPolynomialSet(cell, 3)

    dual = dual_set.DualSet([functional.IntegralMomentOfDerivative(cell, quadrature, f, n)], cell, ids)
    R = dual.to_uncontracted_riesz(poly)[1][0]
    mapped = dual_set.DualSet([
        functional.IntegralMomentOfDerivative(cell, quadrature, f, G.T @ n)], cell, ids)
    assert np.allclose(np.einsum("bgk,bg->k", R, G), mapped.to_riesz(poly)[0])
    assert np.allclose(np.einsum("bbk->k", R), dual.to_riesz(poly)[0])


def test_basis_derivatives_scaling():
    "Regression test for issue #9"
    class Interval(ReferenceElement):

        def __init__(self, a, b):
            verts = ((a,), (b,))
            edges = {0: (0, 1)}
            topology = {0: {0: (0,), 1: (1,)},
                        1: edges}
            super(Interval, self).__init__(LINE, verts, topology)

    random.seed(42)
    for i in range(26):
        a = 1000.0*(random.random() - 0.5)
        b = 1000.0*(random.random() - 0.5)
        a, b = min(a, b), max(a, b)

        interval = Interval(a, b)
        element = Lagrange(interval, 1)

        points = [(a,), (0.5*(a+b),), (b,)]
        tab = element.get_nodal_basis().tabulate(points, 2)

        # first basis function
        assert np.isclose(tab[(0,)][0][0], 1.0)
        assert np.isclose(tab[(0,)][0][1], 0.5)
        assert np.isclose(tab[(0,)][0][2], 0.0)
        # second basis function
        assert np.isclose(tab[(0,)][1][0], 0.0)
        assert np.isclose(tab[(0,)][1][1], 0.5)
        assert np.isclose(tab[(0,)][1][2], 1.0)

        # first and second derivatives
        D = 1.0 / (b - a)
        for p in range(len(points)):
            assert np.isclose(tab[(1,)][0][p], -D)
            assert np.isclose(tab[(1,)][1][p], +D)
            assert np.isclose(tab[(2,)][0][p], 0.0)
            assert np.isclose(tab[(2,)][1][p], 0.0)


xfail_impl = lambda element: pytest.param(element, marks=pytest.mark.xfail(strict=True, raises=NotImplementedError))
elements = [
    "Lagrange(I, 1)",
    "Lagrange(I, 2)",
    "Lagrange(I, 3)",
    "Lagrange(T, 1)",
    "Lagrange(T, 2)",
    "Lagrange(T, 3)",
    "Lagrange(S, 1)",
    "Lagrange(S, 2)",
    "Lagrange(S, 3)",
    "P0(I)",
    "P0(T)",
    "P0(S)",
    "DiscontinuousLagrange(P, 0)",
    "DiscontinuousLagrange(I, 0)",
    "DiscontinuousLagrange(I, 1)",
    "DiscontinuousLagrange(I, 2)",
    "DiscontinuousLagrange(T, 0)",
    "DiscontinuousLagrange(T, 1)",
    "DiscontinuousLagrange(T, 2)",
    "DiscontinuousLagrange(S, 0)",
    "DiscontinuousLagrange(S, 1)",
    "DiscontinuousLagrange(S, 2)",
    "DiscontinuousTaylor(I, 0)",
    "DiscontinuousTaylor(I, 1)",
    "DiscontinuousTaylor(I, 2)",
    "DiscontinuousTaylor(T, 0)",
    "DiscontinuousTaylor(T, 1)",
    "DiscontinuousTaylor(T, 2)",
    "DiscontinuousTaylor(S, 0)",
    "DiscontinuousTaylor(S, 1)",
    "DiscontinuousTaylor(S, 2)",
    "CrouzeixRaviart(I, 1)",
    "CrouzeixRaviart(T, 1)",
    "CrouzeixRaviart(S, 1)",
    "CrouzeixRaviart(T, 3)",
    "CrouzeixRaviart(T, 5)",
    "CrouzeixRaviart(I, 1, variant='point')",
    "CrouzeixRaviart(T, 1, variant='point')",
    "CrouzeixRaviart(S, 1, variant='point')",
    "CrouzeixRaviart(T, 3, variant='point')",
    "CrouzeixRaviart(T, 5, variant='point')",
    "RaviartThomas(I, 1)",
    "RaviartThomas(I, 2)",
    "RaviartThomas(I, 3)",
    "RaviartThomas(T, 1)",
    "RaviartThomas(T, 2)",
    "RaviartThomas(T, 3)",
    "RaviartThomas(S, 1)",
    "RaviartThomas(S, 2)",
    "RaviartThomas(S, 3)",
    'RaviartThomas(I, 1, variant="integral")',
    'RaviartThomas(I, 2, variant="integral")',
    'RaviartThomas(I, 3, variant="integral")',
    'RaviartThomas(T, 1, variant="integral")',
    'RaviartThomas(T, 2, variant="integral")',
    'RaviartThomas(T, 3, variant="integral")',
    'RaviartThomas(S, 1, variant="integral")',
    'RaviartThomas(S, 2, variant="integral")',
    'RaviartThomas(S, 3, variant="integral")',
    'RaviartThomas(I, 1, variant="integral(1)")',
    'RaviartThomas(I, 2, variant="integral(1)")',
    'RaviartThomas(I, 3, variant="integral(1)")',
    'RaviartThomas(T, 1, variant="integral(1)")',
    'RaviartThomas(T, 2, variant="integral(1)")',
    'RaviartThomas(T, 3, variant="integral(1)")',
    'RaviartThomas(S, 1, variant="integral(1)")',
    'RaviartThomas(S, 2, variant="integral(1)")',
    'RaviartThomas(S, 3, variant="integral(1)")',
    'RaviartThomas(I, 1, variant="point")',
    'RaviartThomas(I, 2, variant="point")',
    'RaviartThomas(I, 3, variant="point")',
    'RaviartThomas(T, 1, variant="point")',
    'RaviartThomas(T, 2, variant="point")',
    'RaviartThomas(T, 3, variant="point")',
    'RaviartThomas(S, 1, variant="point")',
    'RaviartThomas(S, 2, variant="point")',
    'RaviartThomas(S, 3, variant="point")',
    "DiscontinuousRaviartThomas(T, 1)",
    "DiscontinuousRaviartThomas(T, 2)",
    "DiscontinuousRaviartThomas(T, 3)",
    "DiscontinuousRaviartThomas(S, 1)",
    "DiscontinuousRaviartThomas(S, 2)",
    "DiscontinuousRaviartThomas(S, 3)",
    "BrezziDouglasMarini(T, 1)",
    "BrezziDouglasMarini(T, 2)",
    "BrezziDouglasMarini(T, 3)",
    "BrezziDouglasMarini(S, 1)",
    "BrezziDouglasMarini(S, 2)",
    "BrezziDouglasMarini(S, 3)",
    'BrezziDouglasMarini(T, 1, variant="integral")',
    'BrezziDouglasMarini(T, 2, variant="integral")',
    'BrezziDouglasMarini(T, 3, variant="integral")',
    'BrezziDouglasMarini(S, 1, variant="integral")',
    'BrezziDouglasMarini(S, 2, variant="integral")',
    'BrezziDouglasMarini(S, 3, variant="integral")',
    'BrezziDouglasMarini(T, 1, variant="integral(1)")',
    'BrezziDouglasMarini(T, 2, variant="integral(1)")',
    'BrezziDouglasMarini(T, 3, variant="integral(1)")',
    'BrezziDouglasMarini(S, 1, variant="integral(1)")',
    'BrezziDouglasMarini(S, 2, variant="integral(1)")',
    'BrezziDouglasMarini(S, 3, variant="integral(1)")',
    'BrezziDouglasMarini(T, 1, variant="point")',
    'BrezziDouglasMarini(T, 2, variant="point")',
    'BrezziDouglasMarini(T, 3, variant="point")',
    'BrezziDouglasMarini(S, 1, variant="point")',
    'BrezziDouglasMarini(S, 2, variant="point")',
    'BrezziDouglasMarini(S, 3, variant="point")',
    "Nedelec(T, 1)",
    "Nedelec(T, 2)",
    "Nedelec(T, 3)",
    "Nedelec(S, 1)",
    "Nedelec(S, 2)",
    "Nedelec(S, 3)",
    'Nedelec(T, 1, variant="integral")',
    'Nedelec(T, 2, variant="integral")',
    'Nedelec(T, 3, variant="integral")',
    'Nedelec(S, 1, variant="integral")',
    'Nedelec(S, 2, variant="integral")',
    'Nedelec(S, 3, variant="integral")',
    'Nedelec(T, 1, variant="integral(1)")',
    'Nedelec(T, 2, variant="integral(1)")',
    'Nedelec(T, 3, variant="integral(1)")',
    'Nedelec(S, 1, variant="integral(1)")',
    'Nedelec(S, 2, variant="integral(1)")',
    'Nedelec(S, 3, variant="integral(1)")',
    'Nedelec(T, 1, variant="point")',
    'Nedelec(T, 2, variant="point")',
    'Nedelec(T, 3, variant="point")',
    'Nedelec(S, 1, variant="point")',
    'Nedelec(S, 2, variant="point")',
    'Nedelec(S, 3, variant="point")',
    "NedelecSecondKind(T, 1)",
    "NedelecSecondKind(T, 2)",
    "NedelecSecondKind(T, 3)",
    "NedelecSecondKind(S, 1)",
    "NedelecSecondKind(S, 2)",
    "NedelecSecondKind(S, 3)",
    'NedelecSecondKind(T, 1, variant="integral")',
    'NedelecSecondKind(T, 2, variant="integral")',
    'NedelecSecondKind(T, 3, variant="integral")',
    'NedelecSecondKind(S, 1, variant="integral")',
    'NedelecSecondKind(S, 2, variant="integral")',
    'NedelecSecondKind(S, 3, variant="integral")',
    'NedelecSecondKind(T, 1, variant="integral(1)")',
    'NedelecSecondKind(T, 2, variant="integral(1)")',
    'NedelecSecondKind(T, 3, variant="integral(1)")',
    'NedelecSecondKind(S, 1, variant="integral(1)")',
    'NedelecSecondKind(S, 2, variant="integral(1)")',
    'NedelecSecondKind(S, 3, variant="integral(1)")',
    'NedelecSecondKind(T, 1, variant="point")',
    'NedelecSecondKind(T, 2, variant="point")',
    'NedelecSecondKind(T, 3, variant="point")',
    'NedelecSecondKind(S, 1, variant="point")',
    'NedelecSecondKind(S, 2, variant="point")',
    'NedelecSecondKind(S, 3, variant="point")',
    "Regge(T, 0)",
    "Regge(T, 1)",
    "Regge(T, 2)",
    "Regge(S, 0)",
    "Regge(S, 1)",
    "Regge(S, 2)",
    "Regge(T, 1, variant='point')",
    "Regge(S, 1, variant='point')",
    "HellanHerrmannJohnson(T, 0)",
    "HellanHerrmannJohnson(T, 1)",
    "HellanHerrmannJohnson(T, 2)",
    "HellanHerrmannJohnson(S, 0)",
    "HellanHerrmannJohnson(S, 1)",
    "HellanHerrmannJohnson(S, 2)",
    "HellanHerrmannJohnson(T, 1, variant='point')",
    "HellanHerrmannJohnson(S, 1, variant='point')",
    "GopalakrishnanLedererSchoberlFirstKind(T, 1)",
    "GopalakrishnanLedererSchoberlFirstKind(T, 2)",
    "GopalakrishnanLedererSchoberlFirstKind(T, 3)",
    "GopalakrishnanLedererSchoberlFirstKind(S, 1)",
    "GopalakrishnanLedererSchoberlFirstKind(S, 2)",
    "GopalakrishnanLedererSchoberlFirstKind(S, 3)",
    "GopalakrishnanLedererSchoberlSecondKind(T, 0)",
    "GopalakrishnanLedererSchoberlSecondKind(T, 1)",
    "GopalakrishnanLedererSchoberlSecondKind(T, 2)",
    "GopalakrishnanLedererSchoberlSecondKind(S, 0)",
    "GopalakrishnanLedererSchoberlSecondKind(S, 1)",
    "GopalakrishnanLedererSchoberlSecondKind(S, 2)",
    "BrezziDouglasFortinMarini(T, 2)",
    "BrezziDouglasFortinMarini(T, 3)",
    "BrezziDouglasFortinMarini(S, 2)",
    "BrezziDouglasFortinMarini(T, 2, variant='point')",
    "GaussLegendre(I, 0)",
    "GaussLegendre(I, 1)",
    "GaussLegendre(I, 2)",
    "GaussLegendre(T, 0)",
    "GaussLegendre(T, 1)",
    "GaussLegendre(T, 2)",
    "GaussLegendre(S, 0)",
    "GaussLegendre(S, 1)",
    "GaussLegendre(S, 2)",
    "GaussLobattoLegendre(I, 1)",
    "GaussLobattoLegendre(I, 2)",
    "GaussLobattoLegendre(I, 3)",
    "GaussLobattoLegendre(T, 1)",
    "GaussLobattoLegendre(T, 2)",
    "GaussLobattoLegendre(T, 3)",
    "GaussLobattoLegendre(S, 1)",
    "GaussLobattoLegendre(S, 2)",
    "GaussLobattoLegendre(S, 3)",
    "Histopolation(I, 0)",
    "Histopolation(I, 1)",
    "Histopolation(I, 2)",
    "Bubble(I, 2)",
    "Bubble(I, 2, 'integral')",
    "Bubble(T, 3)",
    "Bubble(T, 3, 'integral')",
    "Bubble(S, 4)",
    "Bubble(S, 4, 'integral')",
    "FacetBubble(T, 2)",
    "FacetBubble(T, 2, 'integral')",
    "FacetBubble(S, 3)",
    "FacetBubble(S, 3, 'integral')",
    "RestrictedElement(Lagrange(I, 2), restriction_domain='facet')",
    "RestrictedElement(Lagrange(T, 2), restriction_domain='vertex')",
    "RestrictedElement(Lagrange(T, 3), restriction_domain='facet')",
    "NodalEnrichedElement(Lagrange(I, 1), Bubble(I, 2))",
    "NodalEnrichedElement(Lagrange(T, 1), Bubble(T, 3))",
    "NodalEnrichedElement(Lagrange(S, 1), Bubble(S, 4))",
    "NodalEnrichedElement("
    "    RaviartThomas(T, 1),"
    "    RestrictedElement(RaviartThomas(T, 2), restriction_domain='interior')"
    ")",
    "NodalEnrichedElement("
    "    Regge(S, 1),"
    "    RestrictedElement(Regge(S, 2), restriction_domain='interior')"
    ")",
    "Argyris(T, 5, 'point')",
    "Argyris(T, 5, 'integral')",
    "Argyris(T, 6, 'integral')",
    "WuXuH3NC(T, 4)",
    "WuXuRobustH3NC(T, 7)",
    "Hermite(I)",
    "Hermite(I, 4)",
    "Hermite(T)",
    "Hermite(S)",
    "Morley(T)",
    "Morley(S)",
    "BernardiRaugel(T)",
    "BernardiRaugel(S)",
    "MardalTaiWinther(T, 1)",
    "MardalTaiWinther(S, 1)",
    "MardalTaiWinther(S, 2)",
    "ArnoldWintherNC(T, 2)",
    "ArnoldWinther(T, 3)",
    "HuZhang(T, 3)",
    "HuZhang(T, 4)",
    "HuZhang(T, 3, 'point')",
    "HuZhang(T, 4, 'point')",
    "KongMulderVeldhuizen(T,1)",
    "KongMulderVeldhuizen(T,2)",
    "KongMulderVeldhuizen(T,3)",
    "KongMulderVeldhuizen(T,4)",
    "KongMulderVeldhuizen(T,5)",
    "KongMulderVeldhuizen(T,6)",
    "BrambleZlamalC2(T, 9)",
    "BrambleZlamalC2(T, 10)",

    # Macroelements
    "Lagrange(T, 1, 'iso')",
    "Lagrange(T, 1, 'alfeld')",
    "Lagrange(T, 2, 'alfeld')",
    "DiscontinuousLagrange(T, 1, 'alfeld')",
    "HsiehCloughTocher(T)",
    "AlfeldC2(T, 5)",
    "AlfeldC2(T, 6)",
    "JohnsonMercier(T)",
    "JohnsonMercier(S)",
    "AlfeldSorokina(T)",
    "AlfeldSorokina(S)",
    "ArnoldQin(T, reduced=False)",
    "ArnoldQin(T, reduced=True)",
    "ChristiansenHu(T)",
    "ChristiansenHu(S)",
    "GuzmanNeilanFirstKindH1(T, 1)",
    "GuzmanNeilanFirstKindH1(S, 1)",
    "GuzmanNeilanFirstKindH1(S, 2)",
    "GuzmanNeilanSecondKindH1(T, 1)",
    "GuzmanNeilanSecondKindH1(S, 1)",
    "GuzmanNeilanSecondKindH1(S, 2)",
    "NodalEnrichedElement(GuzmanNeilanFirstKindH1(S, 0), AlfeldSorokina(S))",

    # MixedElement made of nodal elements should be nodal, but its API
    # is currently just broken.
    xfail_impl("MixedElement(["
               "    DiscontinuousLagrange(T, 1),"
               "    RaviartThomas(T, 2)"
               "])"),

    # Following element do not bother implementing get_nodal_basis
    # so the test would need to be rewritten using tabulate
    xfail_impl("TensorProductElement(DiscontinuousLagrange(I, 1), Lagrange(I, 2))"),
    xfail_impl("Hdiv(TensorProductElement(DiscontinuousLagrange(I, 1), Lagrange(I, 2)))"),
    xfail_impl("Hcurl(TensorProductElement(DiscontinuousLagrange(I, 1), Lagrange(I, 2)))"),
    xfail_impl("HDivTrace(T, 1)"),
    xfail_impl("EnrichedElement("
               "Hdiv(TensorProductElement(Lagrange(I, 1), DiscontinuousLagrange(I, 0))), "
               "Hdiv(TensorProductElement(DiscontinuousLagrange(I, 0), Lagrange(I, 1)))"
               ")"),
    xfail_impl("EnrichedElement("
               "Hcurl(TensorProductElement(Lagrange(I, 1), DiscontinuousLagrange(I, 0))), "
               "Hcurl(TensorProductElement(DiscontinuousLagrange(I, 0), Lagrange(I, 1)))"
               ")"),
    # Following elements are checked using tabulate
    xfail_impl("HDivTrace(T, 0)"),
    xfail_impl("HDivTrace(T, 1)"),
    xfail_impl("HDivTrace(T, 2)"),
    xfail_impl("HDivTrace(T, 3)"),
    xfail_impl("HDivTrace(S, 0)"),
    xfail_impl("HDivTrace(S, 1)"),
    xfail_impl("HDivTrace(S, 2)"),
    xfail_impl("HDivTrace(S, 3)"),
    xfail_impl("TensorProductElement(Lagrange(I, 1), Lagrange(I, 1))"),
    xfail_impl("TensorProductElement(Lagrange(I, 2), Lagrange(I, 2))"),
    xfail_impl("TensorProductElement(TensorProductElement(Lagrange(I, 1), Lagrange(I, 1)), Lagrange(I, 1))"),
    xfail_impl("TensorProductElement(TensorProductElement(Lagrange(I, 2), Lagrange(I, 2)), Lagrange(I, 2))"),
    xfail_impl("FlattenedDimensions(TensorProductElement(Lagrange(I, 1), Lagrange(I, 1)))"),
    xfail_impl("FlattenedDimensions(TensorProductElement(Lagrange(I, 2), Lagrange(I, 2)))"),
    xfail_impl("FlattenedDimensions(TensorProductElement(FlattenedDimensions(TensorProductElement(Lagrange(I, 1), Lagrange(I, 1))), Lagrange(I, 1)))"),
    xfail_impl("FlattenedDimensions(TensorProductElement(FlattenedDimensions(TensorProductElement(Lagrange(I, 2), Lagrange(I, 2))), Lagrange(I, 2)))"),
]


@pytest.mark.parametrize('element', elements)
def test_nodality(element):
    """Check that generated elements are nodal, i.e. nodes evaluated
    on basis functions give Kronecker delta
    """
    # Instantiate element lazily
    element = eval(element)

    # Fetch primal and dual basis
    poly_set = element.get_nodal_basis()
    dual_set = element.get_dual_set()
    assert poly_set.get_reference_element() >= dual_set.get_reference_element()

    # Get coeffs of primal and dual bases w.r.t. expansion set
    coeffs_poly = poly_set.get_coeffs()
    coeffs_dual = dual_set.to_riesz(poly_set)
    assert coeffs_poly.shape == coeffs_dual.shape

    # Check nodality
    for i in range(coeffs_dual.shape[0]):
        for j in range(coeffs_poly.shape[0]):
            assert np.isclose(
                coeffs_dual[i].flatten().dot(coeffs_poly[j].flatten()),
                1.0 if i == j else 0.0
            )


@pytest.mark.parametrize('elements', [
    (Lagrange(I, 2), Bubble(I, 2)),
    (Lagrange(T, 3), Bubble(T, 3)),
    (Lagrange(S, 4), Bubble(S, 4)),
    (Lagrange(I, 1), Lagrange(I, 1)),
    (Lagrange(I, 1), Bubble(I, 2), Bubble(I, 2)),
])
def test_illposed_nodal_enriched(elements):
    """Check that nodal enriched element fails on ill-posed
    (non-unisolvent) case
    """
    with pytest.raises(np.linalg.LinAlgError):
        NodalEnrichedElement(*elements)


def test_empty_bubble():
    "Check that bubble of too low degree fails"
    with pytest.raises(RuntimeError):
        Bubble(I, 1)
    with pytest.raises(RuntimeError):
        Bubble(T, 2)
    with pytest.raises(RuntimeError):
        Bubble(S, 3)


@pytest.mark.parametrize('elements', [
    (Lagrange(I, 2), Lagrange(I, 1), Bubble(I, 2)),
    (Lagrange(I, 3, variant="gll"), Lagrange(I, 1),
     RestrictedElement(Lagrange(I, 3, variant="gll"), restriction_domain="interior")),
    (RaviartThomas(T, 2),
     RestrictedElement(RaviartThomas(T, 2), restriction_domain='facet'),
     RestrictedElement(RaviartThomas(T, 2), restriction_domain='interior')),
])
def test_nodal_enriched_implementation(elements):
    """Following element pair should be the same.
    This might be fragile to dof reordering but works now.
    """

    e0 = elements[0]
    e1 = NodalEnrichedElement(*elements[1:])

    for attr in ["degree",
                 "get_reference_element",
                 "entity_dofs",
                 "entity_closure_dofs",
                 "get_formdegree",
                 "mapping",
                 "num_sub_elements",
                 "space_dimension",
                 "value_shape",
                 "is_nodal",
                 ]:
        assert getattr(e0, attr)() == getattr(e1, attr)()
    assert np.allclose(e0.get_coeffs(), e1.get_coeffs())
    assert np.allclose(e0.dmats(), e1.dmats())
    assert np.allclose(e0.get_dual_set().to_riesz(e0.get_nodal_basis()),
                       e1.get_dual_set().to_riesz(e1.get_nodal_basis()))


def test_mixed_is_nodal():
    element = MixedElement([DiscontinuousLagrange(T, 1), RaviartThomas(T, 2)])

    assert element.is_nodal()


def test_mixed_is_not_nodal():
    element = MixedElement([
        EnrichedElement(
            RaviartThomas(T, 1),
            RestrictedElement(RaviartThomas(T, 2), restriction_domain="interior")
        ),
        DiscontinuousLagrange(T, 1)
    ])

    assert not element.is_nodal()


@pytest.mark.parametrize('element', [
    "TensorProductElement(Lagrange(I, 1), Lagrange(I, 1))",
    "TensorProductElement(Lagrange(I, 2), Lagrange(I, 2))",
    "TensorProductElement(TensorProductElement(Lagrange(I, 1), Lagrange(I, 1)), Lagrange(I, 1))",
    "TensorProductElement(TensorProductElement(Lagrange(I, 2), Lagrange(I, 2)), Lagrange(I, 2))",
    "FlattenedDimensions(TensorProductElement(Lagrange(I, 1), Lagrange(I, 1)))",
    "FlattenedDimensions(TensorProductElement(Lagrange(I, 2), Lagrange(I, 2)))",
    "FlattenedDimensions(TensorProductElement(FlattenedDimensions(TensorProductElement(Lagrange(I, 1), Lagrange(I, 1))), Lagrange(I, 1)))",
    "FlattenedDimensions(TensorProductElement(FlattenedDimensions(TensorProductElement(Lagrange(I, 2), Lagrange(I, 2))), Lagrange(I, 2)))",
])
def test_nodality_tabulate(element):
    """Check that certain elements (which do no implement
    get_nodal_basis) are nodal too, by tabulating at nodes
    (assuming nodes are point evaluation)
    """
    # Instantiate element
    element = eval(element)

    # Get nodes coordinates
    nodes_coords = []
    for node in element.dual_basis():
        # Assume point evaluation
        (coords, weights), = node.get_point_dict().items()
        assert weights == [(1.0, ())]

        nodes_coords.append(coords)

    # Check nodality
    for j, x in enumerate(nodes_coords):
        basis, = element.tabulate(0, (x,)).values()
        for i in range(len(basis)):
            assert np.isclose(basis[i], 1.0 if i == j else 0.0)


@pytest.mark.parametrize('element', [
    "HDivTrace(T, 0)",
    "HDivTrace(T, 1)",
    "HDivTrace(T, 2)",
    "HDivTrace(T, 3, 'spectral')",
    "HDivTrace(S, 0)",
    "HDivTrace(S, 1)",
    "HDivTrace(S, 2)",
    "HDivTrace(S, 3, 'spectral')",
])
def test_facet_nodality_tabulate(element):
    """Check that certain elements (which do no implement get_nodal_basis)
    are nodal too, by tabulating facet-wise at nodes (assuming nodes
    are point evaluation)
    """
    # Instantiate element
    element = eval(element)

    # Dof/Node coordinates and respective facet
    nodes_coords = []

    # Iterate over facet degrees of freedom
    entity_dofs = element.dual.entity_ids
    facet_dim = sorted(entity_dofs.keys())[-2]
    facet_dofs = entity_dofs[facet_dim]
    dofs = element.dual_basis()
    vertices = element.ref_el.vertices

    for (facet, indices) in facet_dofs.items():
        for i in indices:
            node = dofs[i]
            # Assume point evaluation
            (coords, weights), = node.get_point_dict().items()
            assert weights == [(1.0, ())]

            # Map dof coordinates to reference element due to
            # HdivTrace interface peculiarity
            ref_coords, = map_to_reference_facet((coords,), vertices, facet)
            nodes_coords.append((facet, ref_coords))

    # Check nodality
    for j, (facet, x) in enumerate(nodes_coords):
        basis, = element.tabulate(0, (x,), entity=(facet_dim, facet)).values()
        for i in range(len(basis)):
            assert np.isclose(basis[i], 1.0 if i == j else 0.0)


@pytest.mark.parametrize('element', [
    'Nedelec(S, 3, variant="integral(-1)")',
    'NedelecSecondKind(S, 3, variant="integral(-1)")'
])
def test_error_quadrature_degree(element):
    with pytest.raises(ValueError):
        eval(element)


@pytest.mark.parametrize('element', [
    'DiscontinuousLagrange(P, 1)',
    'GaussLegendre(P, 1)'
])
def test_error_point_high_order(element):
    with pytest.raises(ValueError):
        eval(element)


@pytest.mark.parametrize('element', [
    'DiscontinuousLagrange(I, 0)',
    'DiscontinuousLagrange(T, 0)',
    'DiscontinuousLagrange(I, 1)',
    'DiscontinuousLagrange(T, 1)',
    'Lagrange(I, 1)',
    'Lagrange(T, 1)',
    'Bernstein(I, 1)',
    'Bernstein(T, 1)',
])
def test_single_point_tabulation(element):
    L = eval(element)
    sd = L.ref_el.get_spatial_dimension()
    point = (0.0,) * sd

    T1 = L.tabulate(0, point)[(0,) * sd]
    assert T1.shape == (L.space_dimension(),)

    T2 = L.tabulate(0, [point])[(0,) * sd]
    assert T2.shape == (L.space_dimension(), 1)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
