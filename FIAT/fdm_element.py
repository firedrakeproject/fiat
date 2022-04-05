# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import abc
import numpy

from FIAT import finite_element, polynomial_set, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.barycentric_interpolation import barycentric_interpolation
from FIAT.lagrange import make_entity_permutations
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
from FIAT.P0 import P0Dual


def sym_eig(A, B):
    """
    A numpy-only implementation of `scipy.linalg.eigh`
    """
    L = numpy.linalg.cholesky(B)
    Linv = numpy.linalg.inv(L)
    C = numpy.dot(Linv, numpy.dot(A, Linv.T))
    Z, W = numpy.linalg.eigh(C)
    V = numpy.dot(Linv.T, W)
    return Z, V


class FDMDual(dual_set.DualSet):
    """The dual basis for 1D CG elements with FDM shape functions."""
    def __init__(self, ref_el, degree, kind=1):
        # Construct the FDM dual basis for given kind
        # kind =-1: diagonalize (v, u) and (int(v), int(u)) with no BCs
        # kind = 0: diagonalize (v, u) and (v', u') with no BCs
        # kind = 1: diagonalize (v, u) and (v', u') with point evaluation BCs
        # kind = 2: diagonalize (v, u) and (v'', u'') with point evaluation and derivative BCs

        # Define the generalized eigenproblem on a GLL element
        gll_degree = degree - min(kind, 0)
        gll = GaussLobattoLegendre(ref_el, gll_degree)
        xhat = numpy.array([list(x.get_point_dict().keys())[0][0] for x in gll.dual_basis()])
        E = numpy.eye(gll.space_dimension())

        bdof = []
        idof = slice(0, E.shape[0]+1)
        if kind > 0:
            # Add BC nodes
            bc_nodes = []
            for x in ref_el.get_vertices():
                bc_nodes.append([functional.PointEvaluation(ref_el, x),
                                 *[functional.PointDerivative(ref_el, x, [alpha]) for alpha in range(1, kind)]])
            bc_nodes[1].reverse()
            k = len(bc_nodes[0])
            idof = slice(k, -k)
            bdof = list(range(-k, k))
            bdof = bdof[k:] + bdof[:k]
            # Tabulate the BC nodes
            constraints = gll.tabulate(kind-1, ref_el.get_vertices())
            C = numpy.column_stack(list(constraints.values()))
            perm = list(range(len(bdof)))
            perm = perm[::2] + perm[-1::-2]
            C = C[:, perm].T
            # Tabulate the basis that splits the DOFs into interior and bcs
            E[bdof, idof] = -C[:, idof]
            E[bdof, :] = numpy.dot(numpy.linalg.inv(C[:, bdof]), E[bdof, :])
        else:
            bc_nodes = [[], []]

        # Assemble the constrained Galerkin matrices on the reference cell
        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, gll.space_dimension())
        phi = gll.tabulate(max(1, kind), rule.get_points())
        E0 = numpy.dot(E.T, phi[(0, )])
        Ek = numpy.dot(E.T, phi[(max(1, kind), )])
        B = numpy.dot(numpy.multiply(E0, rule.get_weights()), E0.T)
        A = numpy.dot(numpy.multiply(Ek, rule.get_weights()), Ek.T)

        # Eigenfunctions in the constrained basis
        S = numpy.eye(A.shape[0])
        if S.shape[0] > len(bdof):
            lam, Sii = sym_eig(A[idof, idof], B[idof, idof])
            S[idof, idof] = Sii
            S[idof, bdof] = numpy.dot(Sii, numpy.dot(Sii.T, -B[idof, bdof]))

        if kind < 0:
            # Take the derivative of the eigenbasis and normalize
            idof = lam > 1.0E-12
            S = numpy.multiply(S, numpy.sqrt(1.0E0/lam))
            basis = numpy.dot(S.T, Ek)
            self.gll_points = numpy.array(rule.get_points()).flatten()
            self.gll_tabulation = basis[idof]
        else:
            basis = numpy.dot(S.T, E0)
            # Eigenfunctions in the Lagrange basis
            self.gll_points = xhat
            self.gll_tabulation = numpy.dot(S.T, E.T)

        # Interpolate eigenfunctions onto the quadrature points
        nodes = bc_nodes[0] + [functional.IntegralMoment(ref_el, rule, f) for f in basis[idof]] + bc_nodes[1]

        if kind > 0:
            entity_ids = {0: {0: [0], 1: [degree]},
                          1: {0: list(range(1, degree))}}
            entity_permutations = {}
            entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
            entity_permutations[1] = {0: make_entity_permutations(1, degree - 1)}
        else:
            entity_ids = {0: {0: [], 1: []},
                          1: {0: list(range(0, degree+1))}}
            entity_permutations = {}
            entity_permutations[0] = {0: {0: []}, 1: {0: []}}
            entity_permutations[1] = {0: make_entity_permutations(1, degree + 1)}
        super(FDMDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class FDMFiniteElement(finite_element.CiarletElement):
    """1D element that diagonalizes certain problems under certain BCs."""

    @property
    @abc.abstractmethod
    def _kind(self):
        pass

    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        if degree == 0:
            dual = P0Dual(ref_el)
        else:
            dual = FDMDual(ref_el, degree, kind=self._kind)
        formdegree = 0 if self._kind > 0 else 1
        super(FDMFiniteElement, self).__init__(poly_set, dual, degree, formdegree)

    def tabulate(self, order, points, entity=None):
        # This overrides the default with a more numerically stable algorithm
        if hasattr(self.dual, "gll_points"):
            if entity is None:
                entity = (self.ref_el.get_dimension(), 0)

            entity_dim, entity_id = entity
            transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
            xsrc = self.dual.gll_points
            xdst = numpy.array(list(map(transform, points))).flatten()
            tabulation = barycentric_interpolation(xsrc, xdst, order=order)
            for key in tabulation:
                tabulation[key] = numpy.dot(self.dual.gll_tabulation, tabulation[key])
            return tabulation
        else:
            return super(FDMFiniteElement, self).tabulate(order, points, entity)


class FDMLagrange(FDMFiniteElement):
    """1D CG element with interior shape functions that diagonalize the Laplacian."""
    _kind = 1


class FDMHermite(FDMFiniteElement):
    """1D CG element with interior shape functions that diagonalize the biharmonic operator."""
    _kind = 2


class FDMDiscontinuous(FDMFiniteElement):
    """1D DG element with shape functions that diagonalize the Laplacian."""
    _kind = 0


class FDMDerivative(FDMFiniteElement):
    """1D DG element with the derivate of the shape functions that diagonalize the Laplacian."""
    _kind = -1
