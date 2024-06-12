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
from FIAT import polynomial_set
from FIAT.quadrature_schemes import create_quadrature
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
            #for order in (0, p - 1): 
            for order in range(p - 1): #### NB this should also have been range() back with AW!
            #for order in range(2):
                dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, order + p),
                         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, order + p)]
                #dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, 6),
                #         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, 6)]
            # NB, mom_deg should actually be order + p <= 2p, but in AW have 6 = 2p
            dof_ids[1][entity_id] = list(range(dof_cur, dof_cur + 2*(p - 1)))
            #dof_ids[1][entity_id] = list(range(dof_cur, dof_cur + 4))
            dof_cur += 2*(p - 1)
            #dof_cur += 4

        # internal dofs
        #Q = make_quadrature(cell, 2*(p + 1))
        #Q = make_quadrature(cell, p) # p points -> exactly integrate polys of degree 2p + 1 -> in particular a product of two degree p things, which is what this DOF is
        #Q = make_quadrature(cell, 3) # In lowest order case I think integration of the product of 2 cubic tensors
        Q = create_quadrature(cell, 2*p)

        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrices

        # Copying DOFs of Nedelec of 2nd kind (moments against RT)
        qs = Q.get_points()
        # Create Lagrange bubble nodal basis
        CGbubbles = Bubble(cell, p)
        #CGbubbles = Bubble(cell, 3)
        phi = CGbubbles.get_nodal_basis()

        # Evaluate Lagrange bubble basis at quadrature points
        # Copying AWc rather than AWnc internal DOFs, since latter has 4 nested for loops
        
        for (v1, v2) in basis:
            v1v2t = numpy.outer(v1, v2)
            #phi_times_matrix = [phi[i]*v1v2t for i in range(phi.get_num_members())]
            #fatqp = numpy.zeros((2, 2, len(Q.pts)))
            Fatqp = numpy.zeros((2, 2, phi.get_num_members()))
            #phiatqpts = numpy.outer(phi_times_matrix.tabulate(qs)[(0,) * 2], v1v2t)
            phiatqpts = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            #print("length of Q.pts", len(Q.pts))
            dim_of_bubbles = phi.get_num_members()
            for j in range(dim_of_bubbles):
                fatQP = numpy.zeros((2, 2, len(Q.pts)))
                # Each DOF here is somehow independent of len(Q.pts)
                num_q_pts = len(Q.pts)
                for k in range(num_q_pts):
                    #fatQP[:, :, k] = phiatqpts[j*k:(j + 1)*k, :]
                    #temp = phiatqpts[j*dim_of_bubbles:(j + 1)*dim_of_bubbles, :]
                    #temp = phiatqpts[j*k, :]
                    #temp = phiatqpts[j*num_q_pts + k, :]
                    # NOTE depends how entries of phiatqpts are ordered
                    temp = phiatqpts[k*dim_of_bubbles + j, :]
                    #print("note: ", temp.shape)
                    fatQP[:, :, k] = temp.reshape((2, 2))
                #fatqp[:, :, k] = v1v2t
            #    temp = phiatqpts[k, :]
                #fatqp[:, :, k] = temp.reshape((2, 2))
                #phi_at_qs[:, :, k] = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
                #dofs.append(FIM(cell, Q, fatQP))
            phi_at_qs = numpy.outer(phi.tabulate(qs)[(0,) * 2], v1v2t)
            #phi_at_qs = numpy.outer(phi.tabulate(qs)[(0,) * 0], v1v2t)
            #print((phi.tabulate(qs)[(0,) * 2]).shape)
            #print(phi_at_qs.shape)
            #print(len(phi_at_qs))
            #print(len(phi_at_qs[0, :]))
            #print(phi.get_num_members(), "bubbles")
            #dofs.append([FIM(cell, Q, phi_at_qs[i, :]) for i in range(len(phi_at_qs))])
            #dofs.append([FIM(cell, Q, ) for i in range(phi.get_num_members())])
                #Temp = temp.reshape((2, 2))
                #dofs.append(FIM(cell, Q, Temp))
                #dofs.append(FIM(cell, qs, Temp))
            #print(len(FIM(cell, Q, fatqp)))
            #dofs.append(FIM(cell, Q, fatqp))

       # Alternative bubble-internal DOFs: just evaluate at the interior points
        interior_points = cell.make_points(2, 0, p) # I presume the only cell has entity_id = 0
        num_interior = len(interior_points)
        #print("Num interior =", num_interior)
        for K in range(num_interior):
            #print(interior_points[K])
            for (v1, v2) in basis:
               #v1v2t = numpy.outer(v1, v2)
               dofs.append(InnerProduct(cell, v1, v2, interior_points[K])) 

        dof_ids[2][0] = list(range(dof_cur, dof_cur + round(3*(p - 1)*(p - 2)/2)))
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + 6))
        #dof_cur += round(3*(p - 1)*(p - 2)/2)
        #dof_cur += 3

        # More internal dofs: evaluation of interior-of-edge Lagrange functions, inner product with tt^T for each edge. Note these are evaluated on the edge, but not shared between cells (hence internal).
        #ts = cell.compute_tangents(
        # Copying BDM
        #facet = cell.get_facet_element()
        #Q = create_quadrature(facet, p)
        #Q = make_quadrature(cell, p) # p points -> exactly integrate polys of degree 2p + 1 -> in particular a product of two degree p things, which is what this DOF is
        #qs = Q.get_points()
        #Pp = polynomial_set.ONPolynomialSet(facet, p) 
        #Pp = polynomial_set.ONPolynomialSet(cell, p) 
        #Pp_at_qpts = Pp.tabulate(qs)[(0,) * 2]
        #print(Pp_at_qpts.shape)
        #dim_of_Pp = Pp.get_num_members() # i.e. don't have to call get_nodal_basis() on Pp and then call get_num_members() on that
        #print(dim_of_Pp)
        for entity_id in range(3):
            pts = cell.make_points(1, entity_id, p + 2) # Gives p + 1 points. Strange that have to add 2 to the degree? Since p + 1 points determines P_p, not p_{p + 2}?
            #pts = cell.make_points(1, entity_id, p) # Could just take p - 1 points, which arises from passing p here
            #print(len(pts), "hi")
            t = cell.compute_edge_tangent(entity_id)
            #dofs += [InnerProduct(cell, t, t, pt) for pt in pts]
            #ttT = numpy.outer(t, t)
            #Q = create_quadrature(entity_id, 2*p - 1) # Since this should give (p - 1) points
            #qs = Q.get_points()
            #test_fns_at_qpts = numpy.outer(Pp_at_qpts, ttT)
            #num_evaluation_pts = len(qs)
            #for order in range(1, p):
        #    for order in range(1, 3):
                ## MISTAKE Frobenius inner product with tt^T is not the same as tangential-tangential moment
                #dofs += [IntegralLegendreTangentialTangentialMoment(cell, entity_id, order, 2*p)]
        #        dofs += [IntegralLegendreTangentialTangentialMoment(cell, entity_id, order, 6)]
                #fatQP = numpy.zeros((2, 2, len(Q.pts)))
                #num_q_pts = len(Q.pts)
                #for k in range(num_q_pts):
                #    temp = test_fns_at_qpts[order*num_q_pts + k, :]
                #    # NOTE depends how entries of phiatqpts are ordered, exactly in analogy to the other internal DOFs
                #    #temp = test_fns_at_qpts[k*dim_of_bubbles + order, :]
                #    #print("note: ", temp.shape)
                #    fatQP[:, :, k] = temp.reshape((2, 2))
                #dofs.append(FIM(cell, Q, fatQP))
            P = 0
            for i in range(1, p):
            #for i in range(len(pts)):
                dofs.append(InnerProduct(cell, t, t, pts[i]))
                P += 1
            #for J in range(p + 1):
            #    print(pts[J])
            #print(" ")
            #print(P, "here")
        dof_ids[2][0] = list(range(dof_cur, dof_cur + 3*(p - 1)))
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + 6))
        #dof_cur += 3*(p - 1)
        #dof_cur += 6

        # Could instead do the tt^T internal dofs via moments against edge bubbles.
        #CGEdgeBubbles = FaceBubble()
        #for entity_id in range(3):

        ### NEW complex-based DOFs: airy of fns from the left, div against fns on the right
            
        # This counting below can be done here, or above for one type of internal DOF at a time
        dof_ids[2][0] = list(range(dof_cur, dof_cur + round(3*p*(p - 1)/2)))
        #dof_ids[2][0] = list(range(dof_cur, dof_cur + 9))
        dof_cur += round(3*p*(p - 1)/2)
        #dof_cur += 9

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
        #print(len(dofs))

        super(HuZhangDual, self).__init__(dofs, cell, dof_ids)

class HuZhang(CiarletElement):
    """The definition of the Hu-Zhang element.
    """
    def __init__(self, cell, degree):
        Ps = ONSymTensorPolynomialSet(cell, degree)
        Ls = HuZhangDual(cell, degree)
        mapping = "double contravariant piola"
        super(HuZhang, self).__init__(Ps, Ls, degree, mapping = mapping)
