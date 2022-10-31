# Copyright (C) 2021 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy
from FIAT.expansions import LineExpansionSet, polynomial_dimension
from FIAT.polynomial_set import PolynomialSet


def make_dmat(x):
    """returns Lagrange differentiation matrix and barycentric weights 
    associated with x[j]."""
    dmat = numpy.add.outer(-x, x)
    numpy.fill_diagonal(dmat, 1.0)
    wts = numpy.prod(dmat, axis=0)
    numpy.reciprocal(wts, out=wts)
    numpy.divide(numpy.divide.outer(wts, wts), dmat, out=dmat)
    numpy.fill_diagonal(dmat, dmat.diagonal() - numpy.sum(dmat, axis=0))
    return dmat, wts


class LagrangeLineExpansionSet(LineExpansionSet):
    """Evaluates a 1D Lagrange nodal basis on a line reference element
    via the second barycentric interpolation formula. See Berrut and Trefethen (2004) 
    https://doi.org/10.1137/S0036144502417715 Eq. (4.2) & (9.4)
    """

    def __init__(self, ref_el, pts):
        self.nodes = numpy.array(pts).flatten()
        self.dmat, self.weights = make_dmat(self.nodes)
        LineExpansionSet.__init__(self, ref_el)

    def get_num_members(self, n):
        return len(self.nodes)

    def tabulate(self, n, pts):
        assert n == len(self.nodes)-1
        results = numpy.add.outer(-self.nodes, numpy.array(pts).flatten())
        with numpy.errstate(divide='ignore', invalid='ignore'):
            numpy.reciprocal(results, out=results)
            numpy.multiply(results, self.weights[:, None], out=results)
            numpy.multiply(1.0 / numpy.sum(results, axis=0), results, out=results)

        results[results != results] = 1.0
        if results.dtype == object:
            from sympy import simplify
            results = numpy.array(list(map(simplify, results)))
        return results

    def tabulate_derivative(self, n, pts):
        return numpy.dot(sefl.dmat, self.tabulate(n, pts))


class LagrangePolynomialSet(PolynomialSet):

    def __init__(self, ref_el, pts, shape=tuple()):
        degree = len(pts) - 1
        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel(shape)
            num_components = numpy.prod(flat_shape)
        num_exp_functions = polynomial_dimension(ref_el, degree)
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
                n = polynomial_dimension(ref_el, embedded_degree)
                for exp_bf in range(n):
                    cur_idx = tuple([cur_bf] + list(idx) + [exp_bf])
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        dmats = [numpy.transpose(expansion_set.dmat)]
        PolynomialSet.__init__(self, ref_el, degree, embedded_degree,
                               expansion_set, coeffs, dmats)
