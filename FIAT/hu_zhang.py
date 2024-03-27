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

from FIAT.bubble import Bubble, FacetBubble # each of these is for the interior DOFs

import numpy


class HuZhangDual(DualSet):
    def __init__(self, cell, degree):
        p = degree # This just makes some code below easier to read
        dofs = []
        dof_ids = {}
        dof_ids[0] = {0: [], 1: [], 2: []}
        dof_ids[1] = {0: [], 1: [], 2: []}
        dof_ids[2] = {0: []}

        #dof_cur = 0

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
        # moments of normal component of sigma against degree p - 2.
        for entity_id in range(3):
            #for order in (0, p - 1): #### NB this should also have been range() back with AW!
            #for order in range(p - 1):
            for order in range(2):
                #dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, order + p),
                #         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, order + p)]
                dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, 6),
                         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, 6)]
            # NB, mom_deg should actually be order + p <= 2p, but in AW have 6 = 2p
            #dof_ids[1][entity_id] = list(range(dof_cur, dof_cur + 2*(p - 1)))
            dof_ids[1][entity_id] = list(range(dof_cur, dof_cur + 4))
            #dof_cur += 2*(p - 1)
            dof_cur += 4

        # internal dofs
        #Q = make_quadrature(cell, 2*(p + 1))
        #Q = make_quadrature(cell, p) # p points -> exactly integrate polys of degree 2p + 1 -> in particular a product of two degree p things, which is what this DOF is
        Q = make_quadrature(cell, 3) # In lowest order case I think integration of the product of 2 cubic tensors

        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrices

        # Copying DOFs of Nedelec of 2nd kind (moments against RT)
        qs = Q.get_points()
        # Create Lagrange bubble nodal basis
        #CGbubbles = Bubble(cell, p)
        CGbubbles = Bubble(cell, 3)
        phi = CGbubbles.get_nodal_basis()

        # Evaluate Lagrange bubble basis at quadrature points
        # Copying AWc rather than AWnc internal DOFs, since latter has 4 nested for loops
        
        for (v1, v2) in basis:
            v1v2t = numpy.outer(v1, v2)
            #phi_times_matrix = [phi[i]*v1v2t for i in len(phi)]
            fatqp = numpy.zeros((2, 2, len(Q.pts)))
            #phiatqpts = numpy.outer(phi_times_matrix.tabulate(qs)[(0,) * 2], v1v2t)
            phiatqpts = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            #print(len(Q.pts))
            for k in range(len(Q.pts)):
                #fatqp[:, :, k] = v1v2t
                temp = phiatqpts[k, :]
                fatqp[:, :, k] = temp.reshape((2, 2))
                #phi_at_qs[:, :, k] = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            #phi_at_qs = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            #dofs.append([FIM(cell, Q, phi_at_qs[i, :]) for i in range(len(phi_at_qs))])
            dofs.append(FIM(cell, Q, fatqp))
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + round(3*(p - 1)*(p - 2)/2))))
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + 6))
        #dof_cur += round(3*(p - 1)*(p - 2)/2)
        #dof_cur += 3

        for entity_id in range(3):
        #    for order in range(1, p):
            for order in range(1, 3):
        #        dofs += [IntegralLegendreTangentialTangentialMoment(cell, entity_id, order, 2*p)]
                dofs += [IntegralLegendreTangentialTangentialMoment(cell, entity_id, order, 6)]

        #dof_ids[2][0] = list(range(dof_cur, dof_cur + 3*(p - 1))
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + 6))
        #dof_cur += 3*(p - 1)
        #dof_cur += 6

        # More internal dofs: evaluation of interior-of-edge Lagrange functions, inner product with tt^T for each edge. Note these are evaluated on the edge, but not shared between cells (hence internal).
        # Could instead do via moments against edge bubbles.
        #CGEdgeBubbles = FaceBubble()
        #for entity_id in range(3):
            

        # This counting below can be done here, or above for one type of internal DOF at a time
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + round(3*p*(p - 1)/2)))
        dof_ids[2][0] = list(range(dof_cur, dof_cur + 9))
        #dof_cur += round(3*p*(p - 1)/2)
        dof_cur += 9

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

        #print(dof_cur)
        #print(dof_ids)

        super(HuZhangDual, self).__init__(dofs, cell, dof_ids)


class HuZhang(CiarletElement):
    """The definition of the Hu-Zhang element.
    """
    def __init__(self, cell, degree):
        Ps = ONSymTensorPolynomialSet(cell, degree)
        Ls = HuZhangDual(cell, degree)
        mapping = "double contravariant piola"
        super(HuZhang, self).__init__(Ps, Ls, degree, mapping = mapping)
