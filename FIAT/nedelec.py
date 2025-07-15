# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (polynomial_set, expansions, dual_set,
                  finite_element, functional, macro)
import numpy
from itertools import chain
from FIAT.check_format_variant import check_format_variant
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule


def NedelecSpace2D(ref_el, degree):
    """Constructs a basis for the 2d H(curl) space of the first kind
    which is (P_{degree-1})^2 + P_{degree-1} rot( x )"""
    sd = ref_el.get_spatial_dimension()
    if sd != 2:
        raise Exception("NedelecSpace2D requires 2d reference element")

    k = degree - 1
    vec_Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, k + 1)
    dimPk = expansions.polynomial_dimension(ref_el, k)
    dimPkm1 = expansions.polynomial_dimension(ref_el, k - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk_from_Pkp1 = vec_Pkp1.take(vec_Pk_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1)
    PkH = Pkp1.take(list(range(dimPkm1, dimPk)))

    Q = create_quadrature(ref_el, 2 * (k + 1))
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    PkH_at_Qpts = PkH.tabulate(Qpts)[(0,) * sd]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[(0,) * sd]

    CrossX = numpy.dot(numpy.array([[0.0, 1.0], [-1.0, 0.0]]), Qpts.T)
    PkHCrossX_at_Qpts = PkH_at_Qpts[:, None, :] * CrossX[None, :, :]
    PkHCrossX_coeffs = numpy.dot(numpy.multiply(PkHCrossX_at_Qpts, Qwts),
                                 Pkp1_at_Qpts.T)
    PkHcrossX = polynomial_set.PolynomialSet(ref_el,
                                             k + 1,
                                             k + 1,
                                             vec_Pkp1.get_expansion_set(),
                                             PkHCrossX_coeffs)

    return polynomial_set.polynomial_set_union_normalized(vec_Pk_from_Pkp1,
                                                          PkHcrossX)


def NedelecSpace3D(ref_el, degree):
    """Constructs a nodal basis for the 3d first-kind Nedelec space"""
    sd = ref_el.get_spatial_dimension()
    if sd != 3:
        raise Exception("NedelecSpace3D requires 3d reference element")

    k = degree - 1
    vec_Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, k + 1)
    dimPk = expansions.polynomial_dimension(ref_el, k)
    dimPkm1 = expansions.polynomial_dimension(ref_el, k - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk = vec_Pkp1.take(vec_Pk_indices)

    vec_Pke_indices = list(chain(*(range(i * dimPkp1 + dimPkm1, i * dimPkp1 + dimPk)
                                   for i in range(sd))))
    vec_Pke = vec_Pkp1.take(vec_Pke_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1)

    Q = create_quadrature(ref_el, 2 * (k + 1))
    Qpts, Qwts = Q.get_points(), Q.get_weights()

    Pke_qpts = vec_Pke.tabulate(Qpts)[(0,) * sd]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[(0,) * sd]

    x = Qpts.T
    PkCrossX_at_Qpts = numpy.cross(Pke_qpts, x[None, :, :], axis=1)
    PkCrossXcoeffs = numpy.dot(numpy.multiply(PkCrossX_at_Qpts, Qwts), Pkp1_at_Qpts.T)

    PkCrossX = polynomial_set.PolynomialSet(ref_el,
                                            k + 1,
                                            k + 1,
                                            vec_Pkp1.get_expansion_set(),
                                            PkCrossXcoeffs)
    return polynomial_set.polynomial_set_union_normalized(vec_Pk, PkCrossX)


class NedelecDual(dual_set.DualSet):
    """Dual basis for first-kind Nedelec."""

    def __init__(self, ref_el, degree, variant, interpolant_deg):
        nodes = []
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()

        entity_ids = {}
        # set to empty
        for dim in top:
            entity_ids[dim] = {}
            for entity in top[dim]:
                entity_ids[dim][entity] = []

        if variant == "integral":
            # edge nodes are \int_F v\cdot t p ds where p \in P_{q-1}(edge)
            # degree is q - 1

            # face nodes are \int_F v\cdot p dA where p \in P_{q-2}(f)^3 with p \cdot n = 0 (cmp. Monk)
            # these are equivalent to dofs from Fenics book defined by
            # \int_F v\times n \cdot p ds where p \in P_{q-2}(f)^2
            for dim in range(1, sd):
                phi_deg = degree - dim
                if phi_deg >= 0:
                    facet = ref_el.construct_subelement(dim)
                    Q_ref = create_quadrature(facet, interpolant_deg + phi_deg)
                    Pqmd = polynomial_set.ONPolynomialSet(facet, phi_deg, (dim,))
                    Phis = Pqmd.tabulate(Q_ref.get_points())[(0,) * dim]
                    Phis = numpy.transpose(Phis, (0, 2, 1))

                    for entity in top[dim]:
                        cur = len(nodes)
                        Q = FacetQuadratureRule(ref_el, dim, entity, Q_ref)
                        Jdet = Q.jacobian_determinant()
                        R = numpy.array(ref_el.compute_tangents(dim, entity))
                        phis = numpy.dot(Phis, R / Jdet)
                        phis = numpy.transpose(phis, (0, 2, 1))
                        nodes.extend(functional.FrobeniusIntegralMoment(ref_el, Q, phi)
                                     for phi in phis)
                        entity_ids[dim][entity] = list(range(cur, len(nodes)))

        elif variant == "point":
            for i in top[1]:
                cur = len(nodes)
                # points to specify P_k on each edge
                pts_cur = ref_el.make_points(1, i, degree + 1)
                nodes.extend(functional.PointEdgeTangentEvaluation(ref_el, i, pt)
                             for pt in pts_cur)
                entity_ids[1][i] = list(range(cur, len(nodes)))

            if sd > 2 and degree > 1:  # face tangents
                for i in top[2]:  # loop over faces
                    cur = len(nodes)
                    pts_cur = ref_el.make_points(2, i, degree + 1)
                    nodes.extend(functional.PointFaceTangentEvaluation(ref_el, i, k, pt)
                                 for k in range(2)  # loop over tangents
                                 for pt in pts_cur  # loop over points
                                 )
                    entity_ids[2][i] = list(range(cur, len(nodes)))

        # internal nodes. These are \int_T v \cdot p dx where p \in P_{q-d}^3(T)
        phi_deg = degree - sd
        if phi_deg >= 0:
            if interpolant_deg is None:
                interpolant_deg = degree

            cell = ref_el.construct_subelement(sd)
            Q_ref = create_quadrature(cell, interpolant_deg + phi_deg)
            Pqmd = polynomial_set.ONPolynomialSet(cell, phi_deg)
            Phis = Pqmd.tabulate(Q_ref.get_points())[(0,) * sd]

            for entity in top[sd]:
                Q = FacetQuadratureRule(ref_el, sd, entity, Q_ref)
                cur = len(nodes)
                nodes.extend(functional.IntegralMoment(ref_el, Q, phi, (d,), (sd,))
                             for d in range(sd)
                             for phi in Phis)
                entity_ids[sd][entity] = list(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Nedelec(finite_element.CiarletElement):
    """
    Nedelec finite element

    :arg ref_el: The reference element.
    :arg degree: The degree.
    :arg variant: optional variant specifying the types of nodes.

    variant can be chosen from ["point", "integral", "integral(q)"]
    "point" -> dofs are evaluated by point evaluation. Note that this variant
    has suboptimal convergence order in the H(curl)-norm
    "integral" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence.
    "integral(q)" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence plus q. You might want to choose a high
    quadrature degree to make sure that expressions will be interpolated
    exactly. This is important when you want to have (nearly) curl-preserving
    interpolation.
    """

    def __init__(self, ref_el, degree, variant=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)

        if ref_el.is_macrocell():
            base_element = Nedelec(ref_el.get_parent(), degree)
            poly_set = macro.MacroPolynomialSet(ref_el, base_element)
        elif ref_el.get_spatial_dimension() == 3:
            poly_set = NedelecSpace3D(ref_el, degree)
        elif ref_el.get_spatial_dimension() == 2:
            poly_set = NedelecSpace2D(ref_el, degree)
        else:
            raise Exception("Not implemented")
        dual = NedelecDual(ref_el, degree, variant, interpolant_deg)
        formdegree = 1  # 1-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="covariant piola")
