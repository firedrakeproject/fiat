# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Principal orthogonal expansion functions as defined by Karniadakis
and Sherwin.  These are parametrized over a reference element so as
to allow users to get coordinates that they want."""

import numpy
import math
from FIAT import reference_element, jacobi


def morton_index2(p, q=0):
    return (p + q) * (p + q + 1) // 2 + q


def morton_index3(p, q=0, r=0):
    return (p + q + r)*(p + q + r + 1)*(p + q + r + 2)//6 + (q + r)*(q + r + 1)//2 + r


def jrc(a, b, n):
    """Jacobi recurrence coefficients"""
    an = (2*n+1+a+b)*(2*n+2+a+b) / (2*(n+1)*(n+1+a+b))
    bn = (a+b)*(a-b)*(2*n+1+a+b) / (2*(n+1)*(n+1+a+b)*(2*n+a+b))
    cn = (n+a)*(n+b)*(2*n+2+a+b) / ((n+1)*(n+1+a+b)*(2*n+a+b))
    return an, bn, cn


def integrated_jrc(a, b, n):
    """Integrated Jacobi recurrence coefficients"""
    if n == 1:
        an = (a + b + 2) / 2
        bn = (a - 3*b - 2) / 2
        cn = 0.0
    else:
        an, bn, cn = jrc(a-1, b+1, n-1)
    return an, bn, cn


def pad_coordinates(ref_pts, embedded_dim):
    """Pad reference coordinates by appending -1.0."""
    return tuple(ref_pts) + (-1.0, )*(embedded_dim - len(ref_pts))


def pad_jacobian(A, embedded_dim):
    """Pad coordinate mapping Jacobian by appending zero rows."""
    A = numpy.pad(A, [(0, embedded_dim - A.shape[0]), (0, 0)])
    return tuple(row[..., None] for row in A)


def jacobi_factors(x, y, z, dx, dy, dz):
    fb = 0.5 * (y + z)
    fa = x + (fb + 1.0)
    fc = fb ** 2
    dfa = dfb = dfc = None
    if dx is not None:
        dfb = 0.5 * (dy + dz)
        dfa = dx + dfb
        dfc = 2 * fb * dfb
    return fa, fb, fc, dfa, dfb, dfc


def dubiner_recurrence(dim, n, order, ref_pts, Jinv, scale, variant=None):
    """Tabulate a Dubiner expansion set using the recurrence from (Kirby 2010).

    :arg dim: The spatial dimension of the simplex.
    :arg n: The polynomial degree.
    :arg order: The maximum order of differenation.
    :arg ref_pts: An ``ndarray`` with the coordinates on the default (-1, 1)^d simplex.
    :arg Jinv: The inverse of the Jacobian of the coordinate mapping from the default simplex.
    :arg scale: A scale factor that sets the first member of expansion set.
    :arg variant: Choose between the default (None) orthogonal basis,
                  'integral' for integrated Jacobi polynomials,
                  or 'dual' for the L2-duals of the integrated Jacobi polynomials.

    :returns: A tuple with tabulations of the expansion set and its derivatives.
    """
    if order > 2:
        raise ValueError("Higher order derivatives not supported")
    if variant not in [None, "integral", "dual"]:
        raise ValueError(f"Invalid variant {variant}")

    if variant == "integral":
        scale = -scale
    if n == 0:
        # Always return 1 for n=0 to make regression tests pass
        scale = 1.0

    num_members = math.comb(n + dim, dim)
    results = tuple([None] * num_members for i in range(order+1))
    phi, dphi, ddphi = results + (None,) * (2-order)

    outer = lambda x, y: x[:, None, ...] * y[None, ...]
    sym_outer = lambda x, y: outer(x, y) + outer(y, x)

    pad_dim = dim + 2
    dX = pad_jacobian(Jinv, pad_dim)
    phi[0] = sum((ref_pts[i] - ref_pts[i] for i in range(dim)), scale)
    if dphi is not None:
        dphi[0] = (phi[0] - phi[0]) * dX[0]
    if ddphi is not None:
        ddphi[0] = outer(dphi[0], dX[0])
    if dim == 0 or n == 0:
        return results
    if dim > 3 or dim < 0:
        raise ValueError("Invalid number of spatial dimensions")

    beta = 1 if variant == "dual" else 0
    coefficients = integrated_jrc if variant == "integral" else jrc
    X = pad_coordinates(ref_pts, pad_dim)
    idx = (lambda p: p, morton_index2, morton_index3)[dim-1]
    for codim in range(dim):
        # Extend the basis from codim to codim + 1
        fa, fb, fc, dfa, dfb, dfc = jacobi_factors(*X[codim:codim+3], *dX[codim:codim+3])
        ddfc = 2 * outer(dfb, dfb)
        for sub_index in reference_element.lattice_iter(0, n, codim):
            # handle i = 1
            icur = idx(*sub_index, 0)
            inext = idx(*sub_index, 1)

            if variant == "integral":
                alpha = 2 * sum(sub_index)
                a = b = -0.5
            else:
                alpha = 2 * sum(sub_index) + len(sub_index)
                if variant == "dual":
                    alpha += 1 + len(sub_index)
                a = 0.5 * (alpha + beta) + 1.0
                b = 0.5 * (alpha - beta)

            factor = a * fa - b * fb
            phi[inext] = factor * phi[icur]
            if dphi is not None:
                dfactor = a * dfa - b * dfb
                dphi[inext] = factor * dphi[icur] + phi[icur] * dfactor
                if ddphi is not None:
                    ddphi[inext] = factor * ddphi[icur] + sym_outer(dphi[icur], dfactor)

            # general i by recurrence
            for i in range(1, n - sum(sub_index)):
                iprev, icur, inext = icur, inext, idx(*sub_index, i + 1)
                a, b, c = coefficients(alpha, beta, i)
                factor = a * fa - b * fb
                phi[inext] = factor * phi[icur] - c * (fc * phi[iprev])
                if dphi is None:
                    continue
                dfactor = a * dfa - b * dfb
                dphi[inext] = (factor * dphi[icur] + phi[icur] * dfactor -
                               c * (fc * dphi[iprev] + phi[iprev] * dfc))
                if ddphi is None:
                    continue
                ddphi[inext] = (factor * ddphi[icur] + sym_outer(dphi[icur], dfactor) -
                                c * (fc * ddphi[iprev] + sym_outer(dphi[iprev], dfc) + phi[iprev] * ddfc))

        # normalize
        d = codim + 1
        shift = 1 if variant == "dual" else 0
        for index in reference_element.lattice_iter(0, n+1, d):
            icur = idx(*index)
            if variant is not None:
                p = index[-1] + shift
                alpha = 2 * (sum(index[:-1]) + d * shift) - 1
                norm2 = 1.0
                if p > 0 and p + alpha > 0:
                    norm2 = (p + alpha) * (2*p + alpha) / p
                    norm2 *= (2*d+1) / (2*d)
            else:
                norm2 = (2*sum(index) + d) / d
            scale = math.sqrt(norm2)
            for result in results:
                result[icur] *= scale

    # recover facet bubbles
    if variant == "integral":
        icur = 0
        for result in results:
            result[icur] *= -1
        for inext in range(1, dim+1):
            for result in results:
                result[icur] -= result[inext]

        if dim == 2:
            for i in range(2, n+1):
                icur = idx(0, i)
                iprev = idx(1, i-1)
                for result in results:
                    result[icur] -= result[iprev]

        elif dim == 3:
            for i in range(2, n+1):
                for j in range(0, n+1-i):
                    icur = idx(0, i, j)
                    iprev = idx(1, i-1, j)
                    for result in results:
                        result[icur] -= result[iprev]

                icur = idx(0, 0, i)
                iprev0 = idx(1, 0, i-1)
                iprev1 = idx(0, 1, i-1)
                for result in results:
                    result[icur] -= result[iprev0]
                    result[icur] -= result[iprev1]

    return results


def xi_triangle(eta):
    """Maps from [-1,1]^2 to the (-1,1) reference triangle."""
    eta1, eta2 = eta
    xi1 = 0.5 * (1.0 + eta1) * (1.0 - eta2) - 1.0
    xi2 = eta2
    return (xi1, xi2)


def xi_tetrahedron(eta):
    """Maps from [-1,1]^3 to the -1/1 reference tetrahedron."""
    eta1, eta2, eta3 = eta
    xi1 = 0.25 * (1. + eta1) * (1. - eta2) * (1. - eta3) - 1.
    xi2 = 0.5 * (1. + eta2) * (1. - eta3) - 1.
    xi3 = eta3
    return xi1, xi2, xi3


class ExpansionSet(object):
    def __new__(cls, *args, **kwargs):
        """Returns an ExpansionSet instance appopriate for the given
        reference element."""
        if cls is not ExpansionSet:
            return super(ExpansionSet, cls).__new__(cls)
        try:
            ref_el = args[0]
            expansion_set = {
                reference_element.POINT: PointExpansionSet,
                reference_element.LINE: LineExpansionSet,
                reference_element.TRIANGLE: TriangleExpansionSet,
                reference_element.TETRAHEDRON: TetrahedronExpansionSet,
            }[ref_el.get_shape()]
            return expansion_set(*args, **kwargs)
        except KeyError:
            raise ValueError("Invalid reference element type.")

    def __init__(self, ref_el, scale=None, variant=None):
        self.ref_el = ref_el
        self.variant = variant
        dim = ref_el.get_spatial_dimension()
        self.base_ref_el = reference_element.default_simplex(dim)
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
        self._dmats_cache = {}
        if scale is None:
            scale = math.sqrt(1.0 / self.base_ref_el.volume())
        elif isinstance(scale, str):
            scale = scale.lower()
            if scale == "orthonormal":
                scale = math.sqrt(1.0 / ref_el.volume())
            elif scale == "l2 piola":
                scale = 1.0 / ref_el.volume()
        self.scale = scale

    def get_num_members(self, n):
        D = self.ref_el.get_spatial_dimension()
        return math.comb(n + D, D)

    def _mapping(self, pts):
        if isinstance(pts, numpy.ndarray) and len(pts.shape) == 2:
            return numpy.dot(self.A, pts) + self.b[:, None]
        else:
            m1, m2 = self.A.shape
            return [sum((self.A[i, j] * pts[j] for j in range(m2)), self.b[i])
                    for i in range(m1)]

    def _tabulate(self, n, pts, order=0):
        """A version of tabulate() that also works for a single point.
        """
        D = self.ref_el.get_spatial_dimension()
        return dubiner_recurrence(D, n, order, self._mapping(pts), self.A, self.scale, variant=self.variant)

    def get_dmats(self, degree):
        """Returns a numpy array with the expansion coefficients dmat[k, j, i]
        of the gradient of each member of the expansion set:
            d/dx_k phi_j = sum_i dmat[k, j, i] phi_i.
        """
        cache = self._dmats_cache
        key = degree
        try:
            return cache[key]
        except KeyError:
            pass
        if degree == 0:
            return cache.setdefault(key, numpy.zeros((self.ref_el.get_spatial_dimension(), 1, 1), "d"))

        pts = reference_element.make_lattice(self.ref_el.get_vertices(), degree, variant="gl")
        v, dv = self._tabulate(degree, numpy.transpose(pts), order=1)
        dv = numpy.transpose(dv, (1, 2, 0))
        dmats = numpy.linalg.solve(numpy.transpose(v), dv)
        return cache.setdefault(key, dmats)

    def _tabulate_jet(self, degree, pts, order=0):
        from FIAT.polynomial_set import mis
        D = self.ref_el.get_spatial_dimension()
        lorder = min(2, order)
        vals = self._tabulate(degree, numpy.transpose(pts), order=lorder)
        result = {(0,) * D: numpy.array(vals[0])}
        for r in range(1, 1+lorder):
            vr = numpy.transpose(vals[r], tuple(range(1, r+1)) + (0, r+1))
            for indices in numpy.ndindex(vr.shape[:r]):
                alpha = tuple(map(indices.count, range(D)))
                if alpha not in result:
                    result[alpha] = vr[indices]

        def distance(alpha, beta):
            return sum(ai != bi for ai, bi in zip(alpha, beta))

        # Only use dmats if order > lorder
        for i in range(lorder + 1, order + 1):
            dmats = self.get_dmats(degree)
            alphas = mis(D, i)
            for alpha in alphas:
                base_alpha = next(a for a in result if sum(a) == i-1 and distance(alpha, a) == 1)
                vals = result[base_alpha]
                for dmat, start, end in zip(dmats, base_alpha, alpha):
                    for j in range(start, end):
                        vals = numpy.dot(dmat.T, vals)
                result[alpha] = vals
        return result

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        results, = self._tabulate(n, numpy.transpose(pts))
        return numpy.array(results)

    def tabulate_derivatives(self, n, pts):
        vals, deriv_vals = self._tabulate(n, numpy.transpose(pts), order=1)
        # Create the ordinary data structure.
        D = self.ref_el.get_spatial_dimension()
        data = [[(vals[i][j], [deriv_vals[i][r][j] for r in range(D)])
                 for j in range(len(vals[0]))]
                for i in range(len(vals))]
        return data

    def tabulate_jet(self, n, pts, order=1):
        vals = self._tabulate_jet(n, pts, order=order)
        # Create the ordinary data structure.
        D = self.ref_el.get_spatial_dimension()
        v0 = vals[(0,)*D]
        data = [v0]
        for r in range(1, order+1):
            vr = numpy.zeros((D,)*r + v0.shape, dtype=v0.dtype)
            for index in numpy.ndindex(vr.shape[:r]):
                vr[index] = vals[tuple(map(index.count, range(D)))]
            data.append(vr.transpose((r, r+1) + tuple(range(r))))
        return data


class PointExpansionSet(ExpansionSet):
    """Evaluates the point basis on a point reference element."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 0:
            raise ValueError("Must have a point")
        super(PointExpansionSet, self).__init__(ref_el, **kwargs)

    def tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j]) = 1.0."""
        assert n == 0
        return numpy.ones((1, len(pts)))


class LineExpansionSet(ExpansionSet):
    """Evaluates the Legendre basis on a line reference element."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 1:
            raise Exception("Must have a line")
        super(LineExpansionSet, self).__init__(ref_el, **kwargs)

    def _tabulate(self, n, pts, order=0):
        """Returns a tuple of (vals, derivs) such that
        vals[i,j] = phi_i(pts[j]), derivs[i,j] = D vals[i,j]."""
        if self.variant is not None:
            return super(LineExpansionSet, self)._tabulate(n, pts, order=order)

        xs = self._mapping(pts).T
        results = []
        scale = self.scale * numpy.sqrt(2 * numpy.arange(n+1) + 1)
        for k in range(order+1):
            v = numpy.zeros((n + 1, len(xs)), xs.dtype)
            if n >= k:
                v[k:] = jacobi.eval_jacobi_batch(k, k, n-k, xs)
            for p in range(n + 1):
                v[p] *= scale[p]
                scale[p] *= 0.5 * (p + k + 1) * self.A[0, 0]
            shape = v.shape
            shape = shape[:1] + (1,) * k + shape[1:]
            results.append(v.reshape(shape))
        return tuple(results)


class TriangleExpansionSet(ExpansionSet):
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception("Must have a triangle")
        super(TriangleExpansionSet, self).__init__(ref_el, **kwargs)


class TetrahedronExpansionSet(ExpansionSet):
    """Collapsed orthonormal polynomial expansion on a tetrahedron."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception("Must be a tetrahedron")
        super(TetrahedronExpansionSet, self).__init__(ref_el, **kwargs)


def polynomial_dimension(ref_el, degree):
    """Returns the dimension of the space of polynomials of degree no
    greater than degree on the reference element."""
    if ref_el.get_shape() == reference_element.POINT:
        if degree > 0:
            raise ValueError("Only degree zero polynomials supported on point elements.")
        return 1
    elif ref_el.get_shape() == reference_element.LINE:
        return max(0, degree + 1)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return max((degree + 1) * (degree + 2) // 2, 0)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return max(0, (degree + 1) * (degree + 2) * (degree + 3) // 6)
    else:
        raise ValueError("Unknown reference element type.")
