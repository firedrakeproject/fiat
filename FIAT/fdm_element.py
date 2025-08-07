# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import abc
import numpy

from FIAT import dual_set, finite_element, functional, quadrature
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.reference_element import LINE
from FIAT.P0 import P0


def sym_eig(A, B):
    """
    A numpy-only implementation of `scipy.linalg.eigh`
    """
    Linv = numpy.linalg.inv(numpy.linalg.cholesky(B))
    C = numpy.dot(Linv, numpy.dot(A, Linv.T))
    Z, V = numpy.linalg.eigh(C, "U")
    V = numpy.dot(Linv.T, V)
    return Z, V


def tridiag_eig(A, B):
    """
    Same as sym_eig, but assumes that A is already diagonal and B tri-diagonal
    """
    a = numpy.reciprocal(A.diagonal())
    numpy.sqrt(a, out=a)
    C = numpy.multiply(a, B)
    numpy.multiply(C, a[:, None], out=C)
    Z, V = numpy.linalg.eigh(C, "U")
    numpy.reciprocal(Z, out=Z)
    numpy.multiply(numpy.sqrt(Z), V, out=V)
    numpy.multiply(V, a[:, None], out=V)
    # Reorder by increasing eigenvalue
    return Z[::-1], V[:, ::-1]


class FDMDual(dual_set.DualSet):
    """The dual basis for 1D elements with FDM shape functions."""
    def __init__(self, ref_el, degree, bc_order=1, formdegree=0, orthogonalize=False):
        # Define the generalized eigenproblem on a reference element
        P = ONPolynomialSet(ref_el, degree + formdegree, variant="bubble")
        Pdim = len(P)
        # Apply even / odd reordering on edge bubbles
        P = P.take([*range(2), *range(2, Pdim, 2), *range(3, Pdim, 2)])
        self.poly_set = P

        if bc_order == 1 and formdegree == 0:
            rule = quadrature.GaussLobattoLegendreQuadratureLineRule(ref_el, Pdim+1)
        else:
            rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, Pdim)
        self.rule = rule

        solve_eig = sym_eig
        if bc_order == 1:
            solve_eig = tridiag_eig

        # Tabulate the BC nodes
        if bc_order == 0:
            C = numpy.empty((0, Pdim), "d")
        else:
            constraints = P.tabulate(ref_el.get_vertices(), bc_order-1)
            C = numpy.transpose(numpy.column_stack(list(constraints.values())))
        bdof = slice(None, C.shape[0])
        idof = slice(C.shape[0], None)

        # Coefficients of the vertex and interior modes
        E = numpy.eye(Pdim)
        E[bdof, idof] = -C[:, idof]
        E[bdof, :] = numpy.linalg.solve(C[:, bdof], E[bdof, :])

        # Assemble the constrained Galerkin matrices on the reference cell
        k = max(1, bc_order)
        phi = P.tabulate(rule.get_points(), k)
        wts = rule.get_weights()
        E0 = numpy.dot(E.T, phi[(0, )])
        Ek = numpy.dot(E.T, phi[(k, )])
        B = numpy.dot(numpy.multiply(E0, wts), E0.T)
        A = numpy.dot(numpy.multiply(Ek, wts), Ek.T)

        # Eigenfunctions in the constrained basis
        S = numpy.eye(A.shape[0])
        lam = numpy.ones((A.shape[0],))
        if S.shape[0] > C.shape[0]:
            lam[idof], Sii = solve_eig(A[idof, idof], B[idof, idof])
            S[idof, idof] = Sii
            S[idof, bdof] = numpy.dot(Sii, numpy.dot(Sii.T, -B[idof, bdof]))

        if orthogonalize:
            Abb = numpy.dot(S[:, bdof].T, numpy.dot(A, S[:, bdof]))
            Bbb = numpy.dot(S[:, bdof].T, numpy.dot(B, S[:, bdof]))
            _, Qbb = sym_eig(Abb, Bbb)
            S[:, bdof] = numpy.dot(S[:, bdof], Qbb)

        if formdegree == 0:
            # Tabulate eigenbasis
            basis = numpy.dot(S.T, E0)
        else:
            # Tabulate the derivative of the eigenbasis and normalize
            if bc_order == 0:
                idof = lam > 1.0E-12
                lam[~idof] = 1.0E0
            numpy.reciprocal(lam, out=lam)
            numpy.sqrt(lam, out=lam)
            numpy.multiply(S, lam, out=S)
            basis = numpy.dot(S.T, Ek)

        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []
        if formdegree == 0:
            if orthogonalize:
                idof = slice(None)
            elif bc_order > 0:
                # Vertex dofs -- jet evaluation
                for v in sorted(top[0]):
                    cur = len(nodes)
                    x, = ref_el.make_points(0, v, 0)
                    nodes.append(functional.PointEvaluation(ref_el, x))
                    nodes.extend(functional.PointDerivative(ref_el, x, (alpha, ))
                                 for alpha in range(1, bc_order))
                    entity_ids[0][v].extend(range(cur, len(nodes)))

        elif bc_order > 0:
            basis[bdof] = numpy.sqrt(1.0E0 / ref_el.volume())
            idof = slice(formdegree, None)

        # Interior dofs -- moments against eigenfunctions
        cur = len(nodes)
        nodes.extend(functional.IntegralMoment(ref_el, rule, f) for f in basis[idof])
        entity_ids[sd][0].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class FDMFiniteElement(finite_element.CiarletElement):
    """1D element that diagonalizes bilinear forms with BCs."""

    _orthogonalize = False

    @property
    @abc.abstractmethod
    def _bc_order(self):
        pass

    @property
    @abc.abstractmethod
    def _formdegree(self):
        pass

    def __new__(cls, ref_el, degree):
        if cls._formdegree == 1 and degree == 0:
            return P0(ref_el)
        return super().__new__(cls)

    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("%s is only defined in one dimension." % type(self))
        dual = FDMDual(ref_el, degree, bc_order=self._bc_order,
                       formdegree=self._formdegree, orthogonalize=self._orthogonalize)
        if self._formdegree == 0:
            poly_set = dual.poly_set
        else:
            lr = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
            poly_set = LagrangePolynomialSet(ref_el, lr.get_points())
        super().__init__(poly_set, dual, degree, self._formdegree)


class FDMLagrange(FDMFiniteElement):
    """1D CG element with interior shape functions that diagonalize the Laplacian."""
    _bc_order = 1
    _formdegree = 0


class FDMDiscontinuousLagrange(FDMFiniteElement):
    """1D DG element with derivatives of interior CG FDM shape functions."""
    _bc_order = 1
    _formdegree = 1


class FDMQuadrature(FDMFiniteElement):
    """1D DG element with interior CG FDM shape functions and orthogonalized vertex modes."""
    _bc_order = 1
    _formdegree = 0
    _orthogonalize = True


class FDMBrokenH1(FDMFiniteElement):
    """1D DG element with shape functions that diagonalize the Laplacian."""
    _bc_order = 0
    _formdegree = 0


class FDMBrokenL2(FDMFiniteElement):
    """1D DG element with the derivates of DG FDM shape functions."""
    _bc_order = 0
    _formdegree = 1


class FDMHermite(FDMFiniteElement):
    """1D CG element with interior shape functions that diagonalize the biharmonic operator."""
    _bc_order = 2
    _formdegree = 0
