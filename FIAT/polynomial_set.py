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

    def tabulate_new(self, pts):
        return numpy.dot(self.coeffs,
                         self.expansion_set.tabulate(self.embedded_degree, pts))

    def tabulate(self, pts, jet_order=0):
        """Returns the values of the polynomial set."""
        base_vals = self.expansion_set._tabulate(self.embedded_degree, pts, order=jet_order)
        result = {alpha: numpy.dot(self.coeffs, base_vals[alpha]) for alpha in base_vals}
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

    def get_dmats(self, cell=0):
        return self.expansion_set.get_dmats(self.embedded_degree, cell=cell)

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

    def __len__(self):
        return self.num_members


class ONPolynomialSet(PolynomialSet):
    """Constructs an orthonormal basis out of expansion set by having an
    identity matrix of coefficients.  Can be used to specify ON bases
    for vector- and tensor-valued sets as well.
    """
    def __init__(self, ref_el, degree, shape=tuple(), **kwargs):
        expansion_set = expansions.ExpansionSet(ref_el, **kwargs)
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
            coeffs = numpy.eye(num_members)
        else:
            coeffs = numpy.zeros((num_members, *shape, num_exp_functions))
            cur = 0
            exp_bf = range(num_exp_functions)
            for idx in index_iterator(shape):
                cur_bf = range(cur, cur+num_exp_functions)
                coeffs[(cur_bf, *idx, exp_bf)] = 1.0
                cur += num_exp_functions

        super().__init__(ref_el, degree, embedded_degree, expansion_set, coeffs)


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


def spanning_basis(A, nullspace=False, rtol=1e-10):
    """Construct a basis that spans the rows of A via SVD.
    """
    Aflat = A.reshape(A.shape[0], -1)
    u, sig, vt = numpy.linalg.svd(Aflat, full_matrices=True)
    atol = rtol * (sig[0] + 1)
    num_sv = len([s for s in sig if abs(s) > atol])
    basis = vt[num_sv:] if nullspace else vt[:num_sv]
    return numpy.reshape(basis, (-1, *A.shape[1:]))


def polynomial_set_union_normalized(A, B):
    """Given polynomial sets A and B, constructs a new polynomial set
    whose span is the same as that of span(A) union span(B).  It may
    not contain any of the same members of the set, as we construct a
    span via SVD.
    """
    assert A.get_reference_element() == B.get_reference_element()
    new_coeffs = construct_new_coeffs(A.get_reference_element(), A, B)

    deg = max(A.get_degree(), B.get_degree())
    em_deg = max(A.get_embedded_degree(), B.get_embedded_degree())
    coeffs = spanning_basis(new_coeffs)
    return PolynomialSet(A.get_reference_element(),
                         deg,
                         em_deg,
                         A.get_expansion_set(),
                         coeffs)


def construct_new_coeffs(ref_el, A, B):
    """
    Constructs new coefficients for the set union of A and B
    If A and B are discontinuous and do not have the same degree the smaller one
    is extended to match the larger.

    This does not handle the case that A and B have continuity but not the same degree.
    """

    if A.get_expansion_set().continuity != B.get_expansion_set().continuity:
        raise ValueError("Continuity of expansion sets does not match.")

    if A.get_embedded_degree() != B.get_embedded_degree() and A.get_expansion_set().continuity is None:
        higher = A if A.get_embedded_degree() > B.get_embedded_degree() else B
        lower = B if A.get_embedded_degree() > B.get_embedded_degree() else A

        diff = higher.coeffs.shape[-1] - lower.coeffs.shape[-1]

        # pad only the 0th axis with the difference in size
        padding = [(0, 0) for i in range(len(lower.coeffs.shape) - 1)] + [(0, diff)]
        embedded_coeffs = numpy.pad(lower.coeffs, padding)

        new_coeffs = numpy.concatenate((embedded_coeffs, higher.coeffs), axis=0)
    elif A.get_embedded_degree() == B.get_embedded_degree():
        new_coeffs = numpy.concatenate((A.coeffs, B.coeffs), axis=0)
    else:
        raise NotImplementedError("Extending of coefficients is not implemented for PolynomialSets with continuity and different degrees")
    return new_coeffs


class ONSymTensorPolynomialSet(PolynomialSet):
    """Constructs an orthonormal basis for symmetric-tensor-valued
    polynomials on a reference element.
    """
    def __init__(self, ref_el, degree, size=None, **kwargs):
        expansion_set = expansions.ExpansionSet(ref_el, **kwargs)

        sd = ref_el.get_spatial_dimension()
        if size is None:
            size = sd

        shape = (size, size)
        num_exp_functions = expansion_set.get_num_members(degree)
        num_components = size * (size + 1) // 2
        num_members = num_components * num_exp_functions
        embedded_degree = degree

        # set up coefficients for symmetric tensors
        coeffs = numpy.zeros((num_members, *shape, num_exp_functions))
        cur = 0
        exp_bf = range(num_exp_functions)
        for i, j in index_iterator(shape):
            if i > j:
                continue
            cur_bf = range(cur, cur+num_exp_functions)
            coeffs[cur_bf, i, j, exp_bf] = 1.0
            coeffs[cur_bf, j, i, exp_bf] = 1.0
            cur += num_exp_functions

        super().__init__(ref_el, degree, embedded_degree, expansion_set, coeffs)


class TracelessTensorPolynomialSet(PolynomialSet):
    """Constructs an orthonormal basis for traceless-tensor-valued
    polynomials on a reference element.
    """
    def __init__(self, ref_el, degree, size=None, **kwargs):
        expansion_set = expansions.ExpansionSet(ref_el, **kwargs)

        sd = ref_el.get_spatial_dimension()
        if size is None:
            size = sd

        shape = (size, size)
        num_exp_functions = expansion_set.get_num_members(degree)
        num_components = size * size - 1
        num_members = num_components * num_exp_functions
        embedded_degree = degree

        # set up coefficients for traceless tensors
        coeffs = numpy.zeros((num_members, *shape, num_exp_functions))
        cur = 0
        exp_bf = range(num_exp_functions)
        for i, j in index_iterator(shape):
            if i == size-1 and j == size-1:
                continue
            cur_bf = range(cur, cur+num_exp_functions)
            coeffs[cur_bf, i, j, exp_bf] = 1.0
            if i == j:
                coeffs[cur_bf, -1, -1, exp_bf] = -1.0
            cur += num_exp_functions

        super().__init__(ref_el, degree, embedded_degree, expansion_set, coeffs)


def make_bubbles(ref_el, degree, codim=0, shape=(), scale="L2 piola"):
    """Construct a polynomial set with codim bubbles up to the given degree.
    """
    poly_set = ONPolynomialSet(ref_el, degree, shape=shape, scale=scale, variant="bubble")
    if ref_el.get_spatial_dimension() == 0:
        return poly_set

    entity_ids = expansions.polynomial_entity_ids(ref_el, degree, continuity="C0")
    sd = ref_el.get_spatial_dimension()
    dim = sd - codim
    indices = list(chain(*entity_ids[dim].values()))
    if shape != ():
        ncomp = numpy.prod(shape, dtype=int)
        dimPk = poly_set.get_num_members() // ncomp
        indices = list((numpy.array(indices)[:, None] + dimPk * numpy.arange(ncomp)[None, :]).flat)
    poly_set = poly_set.take(indices)
    return poly_set
