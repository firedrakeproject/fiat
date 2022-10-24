# Copyright (C) 2021 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy
from FIAT import expansions, polynomial_set


def make_dmat(xhat):
    D = numpy.add.outer(-xhat, xhat)
    numpy.fill_diagonal(D, 1.0E0)
    w = numpy.prod(D, axis=0)
    numpy.reciprocal(w, out=w)
    numpy.divide(numpy.divide.outer(w, w), D, out=D)
    numpy.fill_diagonal(D, D.diagonal() - numpy.sum(D, axis=0))
    return D, w


class LagrangeLineExpansionSet(expansions.LineExpansionSet):
    """1D Lagrange nodal basis via the second barycentric interpolation formula

    See Berrut and Trefethen (2004) https://doi.org/10.1137/S0036144502417715 Eq. (4.2) & (9.4)
    """

    def __init__(self, ref_el, pts):
        self.nodes = numpy.array(pts).flatten()
        self.dmat, self.weights = make_dmat(self.nodes)
        expansions.LineExpansionSet.__init__(self, ref_el)

    def get_num_members(self, n):
        return len(self.nodes)

    def tabulate(self, n, pts):
        assert n == len(self.nodes)-1
        xdst = numpy.array(pts).flatten()
        results = numpy.add.outer(-self.nodes, xdst)
        idx = numpy.argwhere(numpy.isclose(results, 0.0E0, 0.0E0))
        results[idx[:, 0], idx[:, 1]] = 1.0E0
        numpy.reciprocal(results, out=results)

        results *= self.weights[:, None]
        results[:, idx[:, 1]] = 0.0E0
        results[idx[:, 0], idx[:, 1]] = 1.0E0
        numpy.multiply(1.0E0 / numpy.sum(results, axis=0), results, out=results)
        return results

    def tabulate_derivative(self, n, pts):
        return numpy.dot(sefl.dmat, self.tabulate(n, pts))


class LagrangePolynomialSet(polynomial_set.PolynomialSet):

    def __init__(self, ref_el, pts, shape=tuple()):
        degree = len(pts) - 1
        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel(shape)
            num_components = numpy.prod(flat_shape)
        num_exp_functions = expansions.polynomial_dimension(ref_el, degree)
        num_members = num_components * num_exp_functions
        embedded_degree = degree
        expansion_set = LagrangeLineExpansionSet(ref_el, pts)

        # set up coefficients
        if shape == tuple():
            coeffs = numpy.eye(num_members)
        else:
            coeffs_shape = tuple([num_members] + list(shape) + [num_exp_functions])
            coeffs = numpy.zeros(coeffs_shape, "d")
            # use functional's index_iterator function
            cur_bf = 0
            for idx in index_iterator(shape):
                n = expansions.polynomial_dimension(ref_el, embedded_degree)
                for exp_bf in range(n):
                    cur_idx = tuple([cur_bf] + list(idx) + [exp_bf])
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        dmats = [numpy.transpose(expansion_set.dmat)]
        polynomial_set.PolynomialSet.__init__(self, ref_el, degree, embedded_degree,
                                              expansion_set, coeffs, dmats)
