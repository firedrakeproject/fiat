# -*- coding: utf-8 -*-
"""Implementation of the Hu-Zhang finite elements."""

# Copyright (C) 2024 by Francis Aznaran (University of Notre Dame)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONSymTensorPolynomialSet, ONPolynomialSet
from FIAT.functional import (
    PointwiseInnerProductEvaluation as InnerProduct,
    FrobeniusIntegralMoment as FIM,
    IntegralMomentOfTensorDivergence,
    IntegralLegendreNormalNormalMoment,
    IntegralLegendreNormalTangentialMoment,
    IntegralLegendreTangentialTangentialMoment,
    )

from FIAT.quadrature import make_quadrature

from FIAT.bubble import Bubble

import numpy


class HuZhangDual(DualSet):
    def __init__(self, cell, degree):
        dofs = []
        dof_ids = {}
        dof_ids[0] = {0: [], 1: [], 2: []}
        dof_ids[1] = {0: [], 1: [], 2: []}
        dof_ids[2] = {0: []}

        dof_cur = 0

        # vertex dofs
        vs = cell.get_vertices()
        e1 = numpy.array([1.0, 0.0])
        e2 = numpy.array([0.0, 1.0])
        basis = [(e1, e1), (e1, e2), (e2, e2)]

        dof_cur = 0

        for entity_id in range(3):
            node = tuple(vs[entity_id])
            for (v1, v2) in basis:
                dofs.append(InnerProduct(cell, v1, v2, node))
            dof_ids[0][entity_id] = list(range(dof_cur, dof_cur + 3))
            dof_cur += 3

        # edge dofs now
        # moments of normal . sigma against degree p - 2.
        for entity_id in range(3):
            #for order in (0, degree - 1): #### NB this should also have been range() back with AW!
            for order in range(degree - 1):
                dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, order + degree),
                         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, order + degree)]
            # NB, mom_deg should actually be order + degree <= 2*degree, but in AW have 6 = 2*degree
            dof_ids[1][entity_id] = list(range(dof_cur, dof_cur + 2*(degree - 1)))
            dof_cur += 2*(degree - 1)

        # internal dofs
        Q = make_quadrature(cell, 2*(degree + 1))

        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrices

        # Copying DOFs of Nedelec of 2nd kind (moments against RT)
        qs = Q.get_points()
        # Create Lagrange bubble nodal basis
        CGbubbles = Bubble(cell, degree)
        phi = CGbubbles.get_nodal_basis()

        # Evaluate Lagrange bubble basis at quadrature points
        
        for (v1, v2) in basis:
            v1v2t = numpy.outer(v1, v2)
            #phi = [phi[i]*v1v2t for i in len(phi)]
            fatqp = numpy.zeros((2, 2, len(Q.pts)))
            phiatqpts = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            for k in range(len(Q.pts)):
                fatqp[:, :, k] = v1v2t
                #temp = phiatqpts[k, :]
                #fatqp[:, :, k] = temp.reshape((2, 2))
                phi_at_qs[:, :, k] = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            phi_at_qs = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            dofs.append([FIM(cell, Q, phi_at_qs[i, :]) for i in range(len(phi_at_qs))])
            dofs.append(FIM(cell, Q, fatqp))
        dof_ids[2][0] = list(range(dof_cur, dof_cur + 3*(degree + 1)))
        dof_cur += 3*(degree + 1)

        #for entity_id in range(3):
        #    for order in range(1, degree):
        #        dofs += [IntegralLegendreTangentialTangentialMoment(cell, entity_id, order, degree*2)]

        dof_ids[2][0] = list(range(dof_cur, dof_cur + round(3*degree*(degree - 1)/2)))
        dof_cur += round(3*degree*(degree - 1)/2)

#        # Constraint dofs
#
#        Q = make_quadrature(cell, 5)
#
#        onp = ONPolynomialSet(cell, 2, (2,))
#        pts = Q.get_points()
#        onpvals = onp.tabulate(pts)[0, 0]
#
#        for i in list(range(3, 6)) + list(range(9, 12)):
#            dofs.append(IntegralMomentOfTensorDivergence(cell, Q,
#                                                         onpvals[i, :, :]))
#
#        dof_ids[2][0] += list(range(dof_cur, dof_cur + 6))

        super(HuZhangDual, self).__init__(dofs, cell, dof_ids)


class HuZhang(CiarletElement):
    """The definition of the Hu-Zhang element.
    """
    def __init__(self, cell, degree):
        Ps = ONSymTensorPolynomialSet(cell, degree)
        Ls = HuZhangDual(cell, degree)
        mapping = "double contravariant piola"
        super(HuZhang, self).__init__(Ps, Ls, degree, mapping = mapping)
