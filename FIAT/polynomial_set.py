# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# polynomial sets
# basic interface:
# -- defined over some reference element
# -- need to be able to tabulate (jets)
# -- type of entry: could by scalar, numpy array, or object-value
#    (such as symmetric tensors, as long as they can be converted <-->
#    with 1d arrays)
# Don't need the "Polynomial" class we had before, provided that
# we have an interface for defining sets of functionals (moments against
# an entire set of polynomials)

import numpy
from itertools import chain
from FIAT import expansions
from FIAT.functional import index_iterator


def mis(m, n):
    """Returns all m-tuples of nonnegative integers that sum up to n."""
    if m == 1:
        return [(n,)]
    elif n == 0:
        return [(0,) * m]
    else:
        return [(n - i,) + foo
                for i in range(n + 1)
                for foo in mis(m - 1, i)]


# We order coeffs by C_{i,j,k}
# where i is the index into the polynomial set,
# j may be an empty tuple (scalar polynomials)
#   or else a vector/tensor
# k is the expansion function
# so if I have all bfs at a given point x in an array bf,
# then dot(coeffs, bf) gives the array of bfs
class PolynomialSet(object):
    """Implements a set of polynomials as linear combinations of an
    expansion set over a reference element.
    ref_el: the reference element
    degree: an order labeling the space
    embedded degree: the degree of polynomial expansion basis that
         must be used to evaluate this space
    coeffs: A numpy array containing the coefficients of the expansion
         basis for each member of the set.  Coeffs is ordered by
         coeffs[i,j,k] where i is the label of the member, k is
         the label of the expansion function, and j is a (possibly
         empty) tuple giving the index for a vector- or tensor-valued
         function.
    """

    def __init__(self, ref_el, degree, embedded_degree, expansion_set, coeffs):
        self.ref_el = ref_el
        self.num_members = coeffs.shape[0]
        self.degree = degree
        self.embedded_degree = embedded_degree
        self.expansion_set = expansion_set
        self.coeffs = coeffs
        self.dmats = []

    def tabulate_new(self, pts):
        return numpy.dot(self.coeffs,
                         self.expansion_set.tabulate(self.embedded_degree, pts))

    def tabulate(self, pts, jet_order=0):
        """Returns the values of the polynomial set."""
        base_vals = self.expansion_set._tabulate_jet(self.embedded_degree, pts, order=jet_order)
        D = self.ref_el.get_spatial_dimension()
        result = {}
        for i in range(jet_order + 1):
            alphas = mis(D, i)
            for alpha in alphas:
                result[alpha] = numpy.dot(self.coeffs, base_vals[alpha])
        return result

    def get_expansion_set(self):
        return self.expansion_set

    def get_coeffs(self):
        return self.coeffs

    def get_num_members(self):
        return self.num_members

    def get_degree(self):
        return self.degree

    def get_embedded_degree(self):
        return self.embedded_degree

    def get_dmats(self):
        if len(self.dmats) == 0:
            self.dmats = self.expansion_set.get_dmats(self.embedded_degree)
        return self.dmats

    def get_reference_element(self):
        return self.ref_el

    def get_shape(self):
        """Returns the shape of phi(x), where () corresponds to
        scalar (2,) a vector of length 2, etc"""
        return self.coeffs.shape[1:-1]

    def take(self, items):
        """Extracts subset of polynomials given by items."""
        new_coeffs = numpy.take(self.get_coeffs(), items, 0)
        return PolynomialSet(self.ref_el, self.degree, self.embedded_degree,
                             self.expansion_set, new_coeffs)


class ONPolynomialSet(PolynomialSet):
    """Constructs an orthonormal basis out of expansion set by having an
    identity matrix of coefficients.  Can be used to specify ON bases
    for vector- and tensor-valued sets as well.

    """

    def __init__(self, ref_el, degree, shape=tuple(), scale=None, variant=None):

        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel(shape)
            num_components = numpy.prod(flat_shape)
        num_exp_functions = expansions.polynomial_dimension(ref_el, degree)
        num_members = num_components * num_exp_functions
        embedded_degree = degree
        expansion_set = expansions.ExpansionSet(ref_el, scale=scale, variant=variant)

        # set up coefficients
        if shape == tuple():
            coeffs = numpy.eye(num_members)
        else:
            coeffs_shape = (num_members, *shape, num_exp_functions)
            coeffs = numpy.zeros(coeffs_shape, "d")
            # use functional's index_iterator function
            cur_bf = 0
            for idx in index_iterator(shape):
                n = expansions.polynomial_dimension(ref_el, embedded_degree)
                for exp_bf in range(n):
                    cur_idx = (cur_bf, *idx, exp_bf)
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        super(ONPolynomialSet, self).__init__(ref_el, degree, embedded_degree,
                                              expansion_set, coeffs)


def project(f, U, Q):
    """Computes the expansion coefficients of f in terms of the members of
    a polynomial set U.  Numerical integration is performed by
    quadrature rule Q.

    """
    pts = Q.get_points()
    wts = Q.get_weights()
    f_at_qps = [f(x) for x in pts]
    U_at_qps = U.tabulate(pts)
    coeffs = numpy.array([sum(wts * f_at_qps * phi) for phi in U_at_qps])
    return coeffs


def form_matrix_product(mats, alpha):
    """Forms product over mats[i]**alpha[i]"""
    m = mats[0].shape[0]
    result = numpy.eye(m)
    for i in range(len(alpha)):
        for j in range(alpha[i]):
            result = numpy.dot(mats[i], result)
    return result


def polynomial_set_union_normalized(A, B):
    """Given polynomial sets A and B, constructs a new polynomial set
    whose span is the same as that of span(A) union span(B).  It may
    not contain any of the same members of the set, as we construct a
    span via SVD.

    """
    new_coeffs = numpy.array(list(A.coeffs) + list(B.coeffs))
    func_shape = new_coeffs.shape[1:]
    if len(func_shape) == 1:
        (u, sig, vt) = numpy.linalg.svd(new_coeffs)
        num_sv = len([s for s in sig if abs(s) > 1.e-10])
        coeffs = vt[:num_sv]
    else:
        new_shape0 = new_coeffs.shape[0]
        new_shape1 = numpy.prod(func_shape)
        newshape = (new_shape0, new_shape1)
        nc = numpy.reshape(new_coeffs, newshape)
        (u, sig, vt) = numpy.linalg.svd(nc, 1)
        num_sv = len([s for s in sig if abs(s) > 1.e-10])

        coeffs = numpy.reshape(vt[:num_sv], (num_sv,) + func_shape)

    return PolynomialSet(A.get_reference_element(),
                         A.get_degree(),
                         A.get_embedded_degree(),
                         A.get_expansion_set(),
                         coeffs)


class ONSymTensorPolynomialSet(PolynomialSet):
    """Constructs an orthonormal basis for symmetric-tensor-valued
    polynomials on a reference element.

    """

    def __init__(self, ref_el, degree, size=None, scale=None):

        sd = ref_el.get_spatial_dimension()
        if size is None:
            size = sd

        shape = (size, size)
        num_exp_functions = expansions.polynomial_dimension(ref_el, degree)
        num_components = size * (size + 1) // 2
        num_members = num_components * num_exp_functions
        embedded_degree = degree
        expansion_set = expansions.ExpansionSet(ref_el, scale=scale)

        # set up coefficients for symmetric tensors
        coeffs_shape = (num_members, *shape, num_exp_functions)
        coeffs = numpy.zeros(coeffs_shape, "d")
        cur_bf = 0
        for i, j in index_iterator(shape):
            if i == j:
                for exp_bf in range(num_exp_functions):
                    coeffs[cur_bf, i, j, exp_bf] = 1.0
                    cur_bf += 1
            elif i < j:
                for exp_bf in range(num_exp_functions):
                    coeffs[cur_bf, i, j, exp_bf] = 1.0
                    coeffs[cur_bf, j, i, exp_bf] = 1.0
                    cur_bf += 1

        super(ONSymTensorPolynomialSet, self).__init__(ref_el, degree, embedded_degree,
                                                       expansion_set, coeffs)


def make_bubbles(ref_el, degree, shape=()):
    """Construct a polynomial set with bubbles up to the given degree.

    """
    dim = ref_el.get_spatial_dimension()
    poly_set = ONPolynomialSet(ref_el, degree, shape=shape, scale="L2 piola", variant="integral")
    degrees = chain(range(dim + 1, degree+1, 2), range(dim + 2, degree+1, 2))

    if dim == 1:
        indices = list(degrees)
    else:
        idx = (expansions.morton_index2, expansions.morton_index3)[dim-2]
        indices = []
        for p in degrees:
            for alpha in mis(dim, p):
                if alpha[0] > 1 and min(alpha[1:]) > 0:
                    indices.append(idx(*alpha))

    if shape != ():
        ncomp = numpy.prod(shape)
        dimPk = poly_set.get_num_members() // ncomp
        indices = list((numpy.array(indices)[:, None] + dimPk * numpy.arange(ncomp)[None, :]).flat)
    poly_set = poly_set.take(indices)
    return poly_set
