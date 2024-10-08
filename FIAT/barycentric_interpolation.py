# Copyright (C) 2021 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy
from FIAT import reference_element, expansions, polynomial_set
from FIAT.functional import index_iterator


def get_lagrange_points(nodes):
    """Extract singleton point for each node."""
    points = []
    for node in nodes:
        pt, = node.get_point_dict()
        points.append(pt)
    return points


def barycentric_interpolation(nodes, wts, dmat, pts, order=0):
    """Evaluates a Lagrange basis on a line reference element
    via the second barycentric interpolation formula. See Berrut and Trefethen (2004)
    https://doi.org/10.1137/S0036144502417715 Eq. (4.2) & (9.4)
    """
    if pts.dtype == object:
        from sympy import simplify
        sp_simplify = numpy.vectorize(simplify)
    else:
        sp_simplify = lambda x: x
    phi = numpy.add.outer(-nodes, pts.flatten())
    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.reciprocal(phi, out=phi)
        numpy.multiply(phi, wts[:, None], out=phi)
        numpy.multiply(1.0 / numpy.sum(phi, axis=0), phi, out=phi)
    phi[phi != phi] = 1.0
    phi = phi.reshape(-1, *pts.shape[:-1])

    phi = sp_simplify(phi)
    results = {(0,): phi}
    for r in range(1, order+1):
        phi = sp_simplify(numpy.dot(dmat, phi))
        results[(r,)] = phi
    return results


def make_dmat(x):
    """Returns Lagrange differentiation matrix and barycentric weights
    associated with x[j]."""
    dmat = numpy.add.outer(-x, x)
    numpy.fill_diagonal(dmat, 1.0)
    wts = numpy.prod(dmat, axis=0)
    numpy.reciprocal(wts, out=wts)
    numpy.divide(numpy.divide.outer(wts, wts), dmat, out=dmat)
    numpy.fill_diagonal(dmat, dmat.diagonal() - numpy.sum(dmat, axis=0))
    return dmat, wts


class LagrangeLineExpansionSet(expansions.LineExpansionSet):
    """Lagrange polynomial expansion set for given points the line."""
    def __init__(self, ref_el, pts):
        self.points = pts
        self.x = numpy.array(pts, dtype="d").flatten()
        self.cell_node_map = expansions.compute_cell_point_map(ref_el, pts, unique=False)
        self.dmats = [None for _ in self.cell_node_map]
        self.weights = [None for _ in self.cell_node_map]
        self.nodes = [None for _ in self.cell_node_map]
        for cell, ibfs in self.cell_node_map.items():
            self.nodes[cell] = self.x[ibfs]
            self.dmats[cell], self.weights[cell] = make_dmat(self.nodes[cell])

        self.degree = max(len(wts) for wts in self.weights)-1
        self.recurrence_order = self.degree + 1
        super().__init__(ref_el)
        self.continuity = None if len(self.x) == sum(len(xk) for xk in self.nodes) else "C0"

    def get_num_members(self, n):
        return len(self.points)

    def get_cell_node_map(self, n):
        return self.cell_node_map

    def get_points(self):
        return self.points

    def get_dmats(self, degree, cell=0):
        return [self.dmats[cell].T]

    def _tabulate_on_cell(self, n, pts, order=0, cell=0, direction=None):
        return barycentric_interpolation(self.nodes[cell], self.weights[cell], self.dmats[cell], pts, order=order)


class LagrangePolynomialSet(polynomial_set.PolynomialSet):

    def __init__(self, ref_el, pts, shape=tuple()):
        if ref_el.get_shape() != reference_element.LINE:
            raise ValueError("Invalid reference element type.")

        expansion_set = LagrangeLineExpansionSet(ref_el, pts)
        degree = expansion_set.degree
        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel(shape)
            num_components = numpy.prod(flat_shape)
        num_exp_functions = expansion_set.get_num_members(degree)
        num_members = num_components * num_exp_functions
        embedded_degree = degree

        # set up coefficients
        if shape == tuple():
            coeffs = numpy.eye(num_members, dtype="d")
        else:
            coeffs_shape = (num_members, *shape, num_exp_functions)
            coeffs = numpy.zeros(coeffs_shape, "d")
            # use functional's index_iterator function
            cur_bf = 0
            for idx in index_iterator(shape):
                for exp_bf in range(num_exp_functions):
                    cur_idx = (cur_bf, *idx, exp_bf)
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        super().__init__(ref_el, degree, embedded_degree, expansion_set, coeffs)
