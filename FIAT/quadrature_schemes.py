"""Quadrature schemes on cells

This module generates quadrature schemes on reference cells that integrate
exactly a polynomial of a given degree using a specified scheme.

Scheme options are:

  scheme="default"

  scheme="canonical" (collapsed Gauss scheme)

Background on the schemes:

  Keast rules for tetrahedra:
    Keast, P. Moderate-degree tetrahedral quadrature formulas, Computer
    Methods in Applied Mechanics and Engineering 55(3):339-348, 1986.
    http://dx.doi.org/10.1016/0045-7825(86)90059-9

  Xiao-Gimbutas rules for simplices:
    Xiao, H., and Gimbutas, Z. A numerical algorithm for the construction of
    efficient quadrature rules in two and higher dimensions, Computers &
    mathematics with applications 59(2): 663-676, 2010.
"""

# Copyright (C) 2011 Garth N. Wells
# Copyright (C) 2016 Miklos Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# First added:  2011-04-19
# Last changed: 2011-04-19

import numpy

from FIAT.quadrature import (QuadratureRule, FacetQuadratureRule, make_quadrature,
                             make_tensor_product_quadrature, map_quadrature)
from FIAT.reference_element import (HEXAHEDRON, QUADRILATERAL, TENSORPRODUCT,
                                    TETRAHEDRON, TRIANGLE, UFCTetrahedron,
                                    UFCTriangle, symmetric_simplex)
from FIAT.macro import MacroQuadratureRule


def create_quadrature(ref_el, degree, scheme="default", entity=None):
    """
    Generate quadrature rule for given reference element that will integrate a
    polynomial of order 'degree' exactly.

    For low-degree polynomials on triangles (<=50) and tetrahedra (<=15), this uses
    hard-coded rules, otherwise it falls back to a collapsed Gauss scheme on
    simplices.  On tensor-product cells, it is a tensor-product quadrature rule
    of the subcells.

    :arg ref_el: The FIAT cell to create the quadrature for.
    :arg degree: The degree of polynomial that the rule should integrate exactly.
    :kwarg scheme: The quadrature scheme, can be choosen from ["default", "canonical", "KMV"]
        "default" -> hard-coded scheme for low degree and collapsed Gauss scheme for high degree,
        "canonical" -> collapsed Gauss scheme,
        "KMV" -> spectral lumped scheme for low degree (<=5 on triangles, <=3 on tetrahedra).
    :kwarg entity: A tuple of entity dimension and entity id specifying the
        integration domain. If not provided, the domain is the entire cell.
    """
    if entity is not None:
        dimension, entity_id = entity
        sub_el = ref_el.construct_subelement(dimension)
        Q_ref = create_quadrature(sub_el, degree, scheme=scheme)
        return FacetQuadratureRule(ref_el, dimension, entity_id, Q_ref)

    if ref_el.is_macrocell():
        dimension = ref_el.get_dimension()
        sub_el = ref_el.construct_subelement(dimension)
        Q_ref = create_quadrature(sub_el, degree, scheme=scheme)
        return MacroQuadratureRule(ref_el, Q_ref)

    if ref_el.get_shape() == TENSORPRODUCT:
        try:
            degree = tuple(degree)
        except TypeError:
            degree = (degree,) * len(ref_el.cells)

        assert len(ref_el.cells) == len(degree)
        quad_rules = [create_quadrature(c, d, scheme)
                      for c, d in zip(ref_el.cells, degree)]
        return make_tensor_product_quadrature(*quad_rules)

    if ref_el.get_shape() in [QUADRILATERAL, HEXAHEDRON]:
        return create_quadrature(ref_el.product, degree, scheme)

    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)

    if scheme == "default":
        if ref_el.get_shape() == TRIANGLE:
            return _triangle_scheme(ref_el, degree)
        elif ref_el.get_shape() == TETRAHEDRON:
            return _tetrahedron_scheme(ref_el, degree)
        else:
            return _fiat_scheme(ref_el, degree)
    elif scheme == "canonical":
        return _fiat_scheme(ref_el, degree)
    elif scheme == "KMV":  # Kong-Mulder-Veldhuizen scheme
        return _kmv_lump_scheme(ref_el, degree)
    else:
        raise ValueError("Unknown quadrature scheme: %s." % scheme)


def _fiat_scheme(ref_el, degree):
    """Get quadrature scheme from FIAT interface"""

    # Number of points per axis for exact integration
    num_points_per_axis = (degree + 1 + 1) // 2

    # Create and return FIAT quadrature rule
    return make_quadrature(ref_el, num_points_per_axis)


def _kmv_lump_scheme(ref_el, degree):
    """Specialized quadrature schemes for P < 6 for KMV simplical elements."""

    sd = ref_el.get_spatial_dimension()
    # set the unit element
    if sd == 2:
        T = UFCTriangle()
    elif sd == 3:
        T = UFCTetrahedron()
    else:
        raise ValueError("Dimension not supported")

    if degree == 1:
        x = ref_el.vertices
        w = numpy.arange(sd + 1, dtype=numpy.float64)
        if sd == 2:
            w[:] = 1.0 / 6.0
        elif sd == 3:
            w[:] = 1.0 / 24.0
        else:
            raise ValueError("Dimension not supported")
    elif degree == 2:
        if sd == 2:
            x = list(ref_el.vertices)
            for e in range(3):
                x.extend(ref_el.make_points(1, e, 2))  # edge midpoints
            x.extend(ref_el.make_points(2, 0, 3))  # barycenter
            w = numpy.arange(7, dtype=numpy.float64)
            w[0:3] = 1.0 / 40.0
            w[3:6] = 1.0 / 15.0
            w[6] = 9.0 / 40.0
        elif sd == 3:
            x = list(ref_el.vertices)
            x.extend(
                [
                    (0.0, 0.50, 0.50),
                    (0.50, 0.0, 0.50),
                    (0.50, 0.50, 0.0),
                    (0.0, 0.0, 0.50),
                    (0.0, 0.50, 0.0),
                    (0.50, 0.0, 0.0),
                ]
            )
            # in facets
            x.extend(
                [
                    (0.33333333333333337, 0.3333333333333333, 0.3333333333333333),
                    (0.0, 0.3333333333333333, 0.3333333333333333),
                    (0.3333333333333333, 0.0, 0.3333333333333333),
                    (0.3333333333333333, 0.3333333333333333, 0.0),
                ]
            )
            # in the cell
            x.extend([(1 / 4, 1 / 4, 1 / 4)])
            w = numpy.arange(15, dtype=numpy.float64)
            w[0:4] = 17.0 / 5040.0
            w[4:10] = 2.0 / 315.0
            w[10:14] = 9.0 / 560.0
            w[14] = 16.0 / 315.0
        else:
            raise ValueError("Dimension not supported")

    elif degree == 3:
        if sd == 2:
            alpha = 0.2934695559090401
            beta = 0.2073451756635909
            x = list(ref_el.vertices)
            x.extend(
                [
                    (1 - alpha, alpha),
                    (alpha, 1 - alpha),
                    (0.0, 1 - alpha),
                    (0.0, alpha),
                    (alpha, 0.0),
                    (1 - alpha, 0.0),
                ]  # edge points
            )
            x.extend(
                [(beta, beta), (1 - 2 * beta, beta), (beta, 1 - 2 * beta)]
            )  # points in center of cell
            w = numpy.arange(12, dtype=numpy.float64)
            w[0:3] = 0.007436456512410291
            w[3:9] = 0.02442084061702551
            w[9:12] = 0.1103885289202054
        elif sd == 3:
            x = list(ref_el.vertices)
            x.extend(
                [
                    (0, 0.685789657581967, 0.314210342418033),
                    (0, 0.314210342418033, 0.685789657581967),
                    (0.314210342418033, 0, 0.685789657581967),
                    (0.685789657581967, 0, 0.314210342418033),
                    (0.685789657581967, 0.314210342418033, 0.0),
                    (0.314210342418033, 0.685789657581967, 0.0),
                    (0, 0, 0.685789657581967),
                    (0, 0, 0.314210342418033),
                    (0, 0.314210342418033, 0.0),
                    (0, 0.685789657581967, 0.0),
                    (0.314210342418033, 0, 0.0),
                    (0.685789657581967, 0, 0.0),
                ]
            )  # 12 points on edges of facets (0-->1-->2)

            x.extend(
                [
                    (0.21548220313557542, 0.5690355937288492, 0.21548220313557542),
                    (0.21548220313557542, 0.21548220313557542, 0.5690355937288492),
                    (0.5690355937288492, 0.21548220313557542, 0.21548220313557542),
                    (0.0, 0.5690355937288492, 0.21548220313557542),
                    (0.0, 0.21548220313557542, 0.5690355937288492),
                    (0.0, 0.21548220313557542, 0.21548220313557542),
                    (0.5690355937288492, 0.0, 0.21548220313557542),
                    (0.21548220313557542, 0.0, 0.5690355937288492),
                    (0.21548220313557542, 0.0, 0.21548220313557542),
                    (0.5690355937288492, 0.21548220313557542, 0.0),
                    (0.21548220313557542, 0.5690355937288492, 0.0),
                    (0.21548220313557542, 0.21548220313557542, 0.0),
                ]
            )  # 12 points (3 points on each facet, 1st two parallel to edge 0)
            alpha = 1 / 6
            x.extend(
                [
                    (alpha, alpha, 0.5),
                    (0.5, alpha, alpha),
                    (alpha, 0.5, alpha),
                    (alpha, alpha, alpha),
                ]
            )  # 4 points inside the cell
            w = numpy.arange(32, dtype=numpy.float64)
            w[0:4] = 0.00068688236002531922325120561367839
            w[4:16] = 0.0015107814913526136472998739890272
            w[16:28] = 0.0050062894680040258624242888174649
            w[28:32] = 0.021428571428571428571428571428571
        else:
            raise ValueError("Dimension not supported")
    elif degree == 4:
        if sd == 2:
            alpha = 0.2113248654051871  # 0.2113248654051871
            betas = [
                0.4247639617258106,
                0.130791593829745
            ]
            x = list(ref_el.vertices)
            for e in range(3):
                x.extend(ref_el.make_points(1, e, 2))  # edge midpoints
            x.extend(
                [
                    (1 - alpha, alpha),
                    (alpha, 1 - alpha),
                    (0.0, 1 - alpha),
                    (0.0, alpha),
                    (alpha, 0.0),
                    (1 - alpha, 0.0),
                ]  # edge points
            )
            for beta in betas:
                x.extend(
                    [(beta, beta), (1 - 2 * beta, beta), (beta, 1 - 2 * beta)]
                )  # points in center of cell
            w = numpy.arange(18, dtype=numpy.float64)
            w[0:3] = 0.003174603174603175
            w[3:6] = 0.0126984126984127
            w[6:12] = 0.01071428571428571
            w[12:15] = 0.07878121446939182
            w[15:18] = 0.05058386489568756
        else:
            raise ValueError("Dimension not supported")

    elif degree == 5:
        if sd == 2:
            alphas = [
                0.3632980741536860e-00,
                0.1322645816327140e-00
            ]
            betas = [
                0.4578368380791611e-00,
                0.2568591072619591e-00,
                0.5752768441141011e-01
            ]
            gamma = 0.7819258362551702e-01
            delta = 0.2210012187598900e-00
            x = list(ref_el.vertices)
            for alpha in alphas:
                x.extend(
                    [
                        (1 - alpha, alpha),
                        (alpha, 1 - alpha),
                        (0.0, 1 - alpha),
                        (0.0, alpha),
                        (alpha, 0.0),
                        (1 - alpha, 0.0),
                    ]  # edge points
                )
            for beta in betas:
                x.extend(
                    [(beta, beta), (1 - 2 * beta, beta), (beta, 1 - 2 * beta)]
                )  # points in center of cell
            x.extend(
                [
                    (gamma, delta),
                    (1 - gamma - delta, delta),
                    (gamma, 1 - gamma - delta),
                    (delta, gamma),
                    (1 - gamma - delta, gamma),
                    (delta, 1 - gamma - delta),
                ]  # edge points
            )
            w = numpy.arange(30, dtype=numpy.float64)
            w[0:3] = 0.7094239706792450e-03
            w[3:9] = 0.6190565003676629e-02
            w[9:15] = 0.3480578640489211e-02
            w[15:18] = 0.3453043037728279e-01
            w[18:21] = 0.4590123763076286e-01
            w[21:24] = 0.1162613545961757e-01
            w[24:30] = 0.2727857596999626e-01
        else:
            raise ValueError("Dimension not supported")

    elif degree == 6:
        if sd == 2:
            x = list(ref_el.vertices)
            episilon = 5.00000000000000e-1
            alphas = [
                8.29411811106452e-2,
                2.68649695592714e-1,
            ]
            betas = [
                4.68059729056814e-1,
                7.93088545089875e-2,
                3.92931636618867e-1,
            ]
            gammas = [
                2.48172758709406e-1,
                1.56582066033687e-1,
            ]
            deltas = [
                6.99812197147049e-1,
                2.43089592364562e-1,
            ]
            weights = [
                5.35113520281665e-4,
                4.29435346026293e-3,
                3.02990950926060e-3,
                3.16396316646563e-3,
                2.43035184285235e-2,
                1.66312091329395e-2,
                3.42178857644876e-2,
                1.73480160090330e-2,
                1.98004044953264e-2,
            ]

            x.extend(
                [
                    (episilon, episilon),
                    (0.0, episilon),
                    (episilon, 0.0),
                ]
            )  # edge midpoints (class 2 points)
            for alpha in alphas:
                x.extend(
                    [
                        (1 - alpha, alpha),
                        (alpha, 1 - alpha),
                        (0.0, 1 - alpha),
                        (0.0, alpha),
                        (alpha, 0.0),
                        (1 - alpha, 0.0),
                    ]  # edge points (sets of class 3 points)
                )

            for beta in betas:
                x.extend(
                    [
                        (beta, beta),
                        (1 - 2 * beta, beta),
                        (beta, 1 - 2 * beta)
                    ]  # interior points on bisector (sets of class 5 points)
                )

            for gamma, delta in zip(gammas, deltas):
                x.extend(
                    [
                        (gamma, delta),
                        (1 - gamma - delta, delta),
                        (gamma, 1 - gamma - delta),
                        (delta, gamma),
                        (1 - gamma - delta, gamma),
                        (delta, 1 - gamma - delta),
                    ]  # interior points (sets of class 6 points)
                )

            w = numpy.arange(39, dtype=numpy.float64)
            w[0:3] = weights[0]  # class 1 points (vertices)
            w[3:6] = weights[1]  # class 2 points
            w[6:12] = weights[2]  # class 3 points
            w[12:18] = weights[3]
            w[18:21] = weights[4]  # class 5 points
            w[21:24] = weights[5]
            w[24:27] = weights[6]
            w[27:33] = weights[7]  # class 6 points
            w[33:39] = weights[8]

        else:
            raise ValueError("Dimension not supported")
    # Return scheme
    return QuadratureRule(T, x, w)


def xg_scheme(ref_el, degree):
    """A (nearly) Gaussian simplicial quadrature with very few quadrature nodes,
    available for low-to-moderate orders.

    Raises `ValueError` if no quadrature rule for the requested parameters is available.

    See

        H. Xiao and Z. Gimbutas, "A numerical algorithm for the construction of
        efficient quadrature rules in two and higher dimensions," Computers &
        Mathematics with Applications, vol. 59, no. 2, pp. 663-676, 2010.
        http://dx.doi.org/10.1016/j.camwa.2009.10.027
    """
    dim = ref_el.get_spatial_dimension()
    if dim == 2:
        from FIAT.xg_quad_data import triangle_table as table
    elif dim == 3:
        from FIAT.xg_quad_data import tetrahedron_table as table
    else:
        raise ValueError(f"Xiao-Gambutas rule not availale for {dim} dimensions.")
    try:
        order_table = table[degree]
    except KeyError:
        raise ValueError(f"Xiao-Gambutas rule not availale for degree {degree}.")

    pts_ref = order_table["points"]
    wts_ref = order_table["weights"]
    Ref1 = symmetric_simplex(dim)
    pts, wts = map_quadrature(pts_ref, wts_ref, Ref1, ref_el)
    return QuadratureRule(ref_el, pts, wts)


def _triangle_scheme(ref_el, degree):
    """Return a quadrature scheme on a triangle of specified order. Falls
    back on canonical rule for higher orders."""
    if degree == 0 or degree == 1:
        # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
        x = numpy.array([[1.0/3.0, 1.0/3.0]])
        w = numpy.array([0.5])
    elif degree == 2:
        # Scheme from Strang and Fix, 3 points, degree of precision 2
        x = numpy.array([[1.0/6.0, 1.0/6.0],
                         [1.0/6.0, 2.0/3.0],
                         [2.0/3.0, 1.0/6.0]])
        w = numpy.zeros((3,), dtype=numpy.float64)
        w.fill(1.0/6.0)
    elif degree == 3:
        # Scheme from Strang and Fix, 6 points, degree of precision 3
        x = numpy.array([[0.659027622374092, 0.231933368553031],
                         [0.659027622374092, 0.109039009072877],
                         [0.231933368553031, 0.659027622374092],
                         [0.231933368553031, 0.109039009072877],
                         [0.109039009072877, 0.659027622374092],
                         [0.109039009072877, 0.231933368553031]])
        w = numpy.zeros((6,), dtype=numpy.float64)
        w.fill(1.0/12.0)
    else:
        try:
            # Get Xiao-Gambutas scheme
            return xg_scheme(ref_el, degree)
        except ValueError:
            # Get canonical scheme
            return _fiat_scheme(ref_el, degree)

    # Return scheme
    x, w = map_quadrature(x, w, UFCTriangle(), ref_el)
    return QuadratureRule(ref_el, x, w)


def _tetrahedron_scheme(ref_el, degree):
    """Return a quadrature scheme on a tetrahedron of specified
    degree. Falls back on canonical rule for higher orders"""
    if degree == 0 or degree == 1:
        # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
        x = numpy.array([[1.0/4.0, 1.0/4.0, 1.0/4.0]])
        w = numpy.array([1.0/6.0])
    elif degree == 2:
        # Scheme from Zienkiewicz and Taylor, 4 points, degree of precision 2
        a, b = 0.585410196624969, 0.138196601125011
        x = numpy.array([[a, b, b],
                         [b, a, b],
                         [b, b, a],
                         [b, b, b]])
        w = numpy.zeros((4,), dtype=numpy.float64)
        w.fill(1.0/24.0)
    else:
        try:
            # Get Xiao-Gambutas scheme
            return xg_scheme(ref_el, degree)
        except ValueError:
            # Get canonical scheme
            return _fiat_scheme(ref_el, degree)

    # Return scheme
    x, w = map_quadrature(x, w, UFCTetrahedron(), ref_el)
    return QuadratureRule(ref_el, x, w)
