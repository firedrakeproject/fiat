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
    :arg order: The maximum order of differentiation.
    :arg ref_pts: An ``ndarray`` with the coordinates on the default (-1, 1)^d simplex.
    :arg Jinv: The inverse of the Jacobian of the coordinate mapping from the default simplex.
    :arg scale: A scale factor that sets the first member of expansion set.
    :arg variant: Choose between the default (None) orthogonal basis,
                  'bubble' for integrated Jacobi polynomials,
                  or 'dual' for the L2-duals of the integrated Jacobi polynomials.

    :returns: A tuple with tabulations of the expansion set and its derivatives.
    """
    if order > 2:
        raise ValueError("Higher order derivatives not supported")
    if variant not in [None, "bubble", "dual"]:
        raise ValueError(f"Invalid variant {variant}")

    if variant == "bubble":
        scale = -scale

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
    coefficients = integrated_jrc if variant == "bubble" else jrc
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

            if variant == "bubble":
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
                norm2 = (0.5 + d) / d
                if p > 0 and p + alpha > 0:
                    norm2 *= (p + alpha) * (2*p + alpha) / p
            else:
                norm2 = (2*sum(index) + d) / d
            scale = math.sqrt(norm2)
            for result in results:
                result[icur] *= scale
    return results


def C0_basis(dim, n, tabulations):
    """Modify a tabulation of a hierarchical basis to enforce C0-continuity.

    :arg dim: The spatial dimension of the simplex.
    :arg n: The polynomial degree.
    :arg tabulations: An iterable tabulations of the hierarchical basis.

    :returns: A tuple of tabulations of the C0 basis.
    """
    idx = (lambda p: p, morton_index2, morton_index3)[dim-1]
    # Recover facet bubbles
    for phi in tabulations:
        icur = 0
        phi[icur] *= -1
        for inext in range(1, dim+1):
            phi[icur] -= phi[inext]
        if dim == 2:
            for i in range(2, n+1):
                phi[idx(0, i)] -= phi[idx(1, i-1)]
        elif dim == 3:
            for i in range(2, n+1):
                for j in range(0, n+1-i):
                    phi[idx(0, i, j)] -= phi[idx(1, i-1, j)]
                icur = idx(0, 0, i)
                phi[icur] -= phi[idx(0, 1, i-1)]
                phi[icur] -= phi[idx(1, 0, i-1)]

    # Reorder by dimension and entity on the reference simplex
    dofs = list(range(dim+1))
    if dim == 1:
        dofs.extend(range(2, n+1))
    elif dim == 2:
        dofs.extend(idx(1, i-1) for i in range(2, n+1))
        dofs.extend(idx(0, i) for i in range(2, n+1))
        dofs.extend(idx(i, 0) for i in range(2, n+1))

        dofs.extend(idx(i, j) for j in range(1, n+1) for i in range(2, n-j+1))
    else:
        dofs.extend(idx(0, 1, i-1) for i in range(2, n+1))
        dofs.extend(idx(1, 0, i-1) for i in range(2, n+1))
        dofs.extend(idx(1, i-1, 0) for i in range(2, n+1))
        dofs.extend(idx(0, 0, i) for i in range(2, n+1))
        dofs.extend(idx(0, i, 0) for i in range(2, n+1))
        dofs.extend(idx(i, 0, 0) for i in range(2, n+1))

        dofs.extend(idx(1, i-1, j) for j in range(1, n+1) for i in range(2, n-j+1))
        dofs.extend(idx(0, i, j) for j in range(1, n+1) for i in range(2, n-j+1))
        dofs.extend(idx(i, 0, j) for j in range(1, n+1) for i in range(2, n-j+1))
        dofs.extend(idx(i, j, 0) for j in range(1, n+1) for i in range(2, n-j+1))

        dofs.extend(idx(i, j, k) for k in range(1, n+1) for j in range(1, n-k+1) for i in range(2, n-j-k+1))

    return tuple([phi[i] for i in dofs] for phi in tabulations)


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
        """Returns an ExpansionSet instance appropriate for the given
        reference element."""
        if cls is not ExpansionSet:
            return super().__new__(cls)
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
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        base_ref_el = reference_element.default_simplex(sd)
        base_verts = base_ref_el.get_vertices()

        self.affine_mappings = [reference_element.make_affine_mapping(
                                ref_el.get_vertices_of_subcomplex(top[sd][cell]),
                                base_verts) for cell in top[sd]]
        if scale is None:
            scale = math.sqrt(1.0 / base_ref_el.volume())
        self.scale = scale
        self.variant = variant
        self.continuity = "C0" if variant == "bubble" else None
        self.recurrence_order = 2
        self._dmats_cache = {}
        self._cell_node_map_cache = {}

    def reconstruct(self, ref_el=None, scale=None, variant=None):
        """Reconstructs this ExpansionSet with modified arguments."""
        return ExpansionSet(ref_el or self.ref_el,
                            scale=scale or self.scale,
                            variant=variant or self.variant)

    def get_scale(self, n, cell=0):
        scale = self.scale
        sd = self.ref_el.get_spatial_dimension()
        if isinstance(scale, str):
            vol = self.ref_el.volume_of_subcomplex(sd, cell)
            scale = scale.lower()
            if scale == "orthonormal":
                scale = math.sqrt(1.0 / vol)
            elif scale == "l2 piola":
                scale = 1.0 / vol
        elif n == 0 and sd > 1 and len(self.affine_mappings) == 1:
            # return 1 for n=0 to make regression tests pass
            scale = 1
        return scale

    def get_num_members(self, n):
        return polynomial_dimension(self.ref_el, n, self.continuity)

    def get_cell_node_map(self, n):
        try:
            return self._cell_node_map_cache[n]
        except KeyError:
            cell_node_map = polynomial_cell_node_map(self.ref_el, n, self.continuity)
            return self._cell_node_map_cache.setdefault(n, cell_node_map)

    def _tabulate_on_cell(self, n, pts, order=0, cell=0, direction=None):
        """Returns a dict of tabulations such that
        tabulations[alpha][i, j] = D^alpha phi_i(pts[j])."""
        from FIAT.polynomial_set import mis
        lorder = min(order, self.recurrence_order)
        A, b = self.affine_mappings[cell]
        ref_pts = numpy.add(numpy.dot(pts, A.T), b).T
        Jinv = A if direction is None else numpy.dot(A, direction)[:, None]
        sd = self.ref_el.get_spatial_dimension()
        scale = self.get_scale(n, cell=cell)
        phi = dubiner_recurrence(sd, n, lorder, ref_pts, Jinv,
                                 scale, variant=self.variant)
        if self.continuity == "C0":
            phi = C0_basis(sd, n, phi)

        # Pack linearly independent components into a dictionary
        result = {(0,) * sd: numpy.asarray(phi[0])}
        for r in range(1, len(phi)):
            vr = numpy.transpose(phi[r], tuple(range(1, r+1)) + (0, r+1))
            for indices in numpy.ndindex(vr.shape[:r]):
                alpha = tuple(map(indices.count, range(sd)))
                if alpha not in result:
                    result[alpha] = vr[indices]

        def distance(alpha, beta):
            return sum(ai != bi for ai, bi in zip(alpha, beta))

        # Only use dmats if tabulate failed
        for i in range(len(phi), order + 1):
            dmats = self.get_dmats(n, cell=cell)
            for alpha in mis(sd, i):
                base_alpha = next(a for a in result if sum(a) == i-1 and distance(alpha, a) == 1)
                vals = result[base_alpha]
                for dmat, start, end in zip(dmats, base_alpha, alpha):
                    for j in range(start, end):
                        vals = numpy.dot(dmat.T, vals)
                result[alpha] = vals
        return result

    def _tabulate(self, n, pts, order=0):
        """A version of tabulate() that also works for a single point."""
        pts = numpy.asarray(pts)
        unique = self.continuity is not None and order == 0
        cell_point_map = compute_cell_point_map(self.ref_el, pts, unique=unique)
        phis = {cell: self._tabulate_on_cell(n, pts[ipts], order, cell=cell)
                for cell, ipts in cell_point_map.items()}

        if not self.ref_el.is_macrocell():
            return phis[0]

        if pts.dtype == object:
            # If binning is undefined, scale by the characteristic function of each subcell
            Xi = compute_partition_of_unity(self.ref_el, pts, unique=unique)
            for cell, phi in phis.items():
                for alpha in phi:
                    phi[alpha] *= Xi[cell]
        elif not unique:
            # If binning is not unique, divide by the multiplicity of each point
            mult = numpy.zeros(pts.shape[:-1])
            for cell, ipts in cell_point_map.items():
                mult[ipts] += 1
            for cell, ipts in cell_point_map.items():
                phi = phis[cell]
                for alpha in phi:
                    phi[alpha] /= mult[None, ipts]

        # Insert subcell tabulations into the corresponding submatrices
        idx = lambda *args: args if args[-1] is Ellipsis else numpy.ix_(*args)
        num_phis = self.get_num_members(n)
        cell_node_map = self.get_cell_node_map(n)
        result = {}
        base_phi = tuple(phis.values())[0]
        for alpha in base_phi:
            dtype = base_phi[alpha].dtype
            result[alpha] = numpy.zeros((num_phis, *pts.shape[:-1]), dtype=dtype)
            for cell in cell_point_map:
                ibfs = cell_node_map[cell]
                ipts = cell_point_map[cell]
                result[alpha][idx(ibfs, ipts)] += phis[cell][alpha]
        return result

    def tabulate_normal_jumps(self, n, ref_pts, facet, order=0):
        """Tabulates the normal derivative jumps on reference points on a facet.

        :arg n: the polynomial degree.
        :arg ref_pts: an iterable of points on the reference facet.
        :arg facet: the facet id.
        :kwarg order: the order of differentiation.

        :returns: a numpy array of tabulations of normal derivative jumps.
        """
        sd = self.ref_el.get_spatial_dimension()
        transform = self.ref_el.get_entity_transform(sd-1, facet)
        pts = transform(ref_pts)
        cell_point_map = compute_cell_point_map(self.ref_el, pts, unique=False)
        cell_node_map = self.get_cell_node_map(n)

        num_phis = self.get_num_members(n)
        results = numpy.zeros((order+1, num_phis, *pts.shape[:-1]))

        for cell in cell_point_map:
            ipts = cell_point_map[cell]
            ibfs = cell_node_map[cell]
            normal = self.ref_el.compute_normal(facet, cell=cell)
            side = numpy.dot(normal, self.ref_el.compute_normal(facet))
            phi = self._tabulate_on_cell(n, pts[ipts], order, cell=cell)
            v0 = phi[(0,)*sd]
            for r in range(order+1):
                vr = numpy.zeros((sd,)*r + v0.shape, dtype=v0.dtype)
                for index in numpy.ndindex(vr.shape[:r]):
                    vr[index] = phi[tuple(map(index.count, range(sd)))]
                for _ in range(r):
                    vr = numpy.tensordot(normal, vr, axes=(0, 0))

                indices = numpy.ix_(ibfs, ipts)
                if r % 2 == 0 and side < 0:
                    results[r][indices] -= vr
                else:
                    results[r][indices] += vr
        return results

    def tabulate_jumps(self, n, points, order=0):
        """Tabulates derivative jumps on given points.

        :arg n: the polynomial degree.
        :arg points: an iterable of points on the cell complex.
        :kwarg order: the order of differentiation.

        :returns: a dictionary of tabulations of derivative jumps across interior facets.
        """

        from FIAT.polynomial_set import mis
        sd = self.ref_el.get_spatial_dimension()
        num_members = self.get_num_members(n)
        cell_node_map = self.get_cell_node_map(n)
        cell_point_map = compute_cell_point_map(self.ref_el, points, unique=False)

        num_jumps = 0
        facet_point_map = {}
        for facet in self.ref_el.get_interior_facets(sd-1):
            try:
                cells = self.ref_el.connectivity[(sd-1, sd)][facet]
                ipts = list(set.intersection(*(set(cell_point_map[c]) for c in cells)))
                if ipts != ():
                    facet_point_map[facet] = ipts
                    num_jumps += len(ipts)
            except KeyError:
                pass

        derivs = {cell: self._tabulate_on_cell(n, points, order=order, cell=cell)
                  for cell in cell_point_map}

        jumps = {}
        for r in range(order+1):
            cur = 0
            alphas = mis(sd, r)
            jumps[r] = numpy.zeros((num_members, len(alphas) * num_jumps))
            for facet, ipts in facet_point_map.items():
                c0, c1 = self.ref_el.connectivity[(sd-1, sd)][facet]
                for alpha in alphas:
                    ijump = range(cur, cur + len(ipts))
                    jumps[r][numpy.ix_(cell_node_map[c1], ijump)] += derivs[c1][alpha][:, ipts]
                    jumps[r][numpy.ix_(cell_node_map[c0], ijump)] -= derivs[c0][alpha][:, ipts]
                    cur += len(ipts)
        return jumps

    def get_dmats(self, degree, cell=0):
        """Returns a numpy array with the expansion coefficients dmat[k, j, i]
        of the gradient of each member of the expansion set:
            d/dx_k phi_j = sum_i dmat[k, j, i] phi_i.
        """
        from FIAT.polynomial_set import mis
        key = (degree, cell)
        cache = self._dmats_cache
        try:
            return cache[key]
        except KeyError:
            pass
        if degree == 0:
            return cache.setdefault(key, numpy.zeros((self.ref_el.get_spatial_dimension(), 1, 1), "d"))

        D = self.ref_el.get_dimension()
        top = self.ref_el.get_topology()
        verts = self.ref_el.get_vertices_of_subcomplex(top[D][cell])
        pts = reference_element.make_lattice(verts, degree, variant="gl")
        v = self._tabulate_on_cell(degree, pts, order=1, cell=cell)
        dv = [numpy.transpose(v[alpha]) for alpha in mis(D, 1)]
        dmats = numpy.linalg.solve(numpy.transpose(v[(0,) * D]), dv)
        return cache.setdefault(key, dmats)

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        sd = self.ref_el.get_spatial_dimension()
        return self._tabulate(n, pts)[(0,) * sd]

    def tabulate_derivatives(self, n, pts):
        from FIAT.polynomial_set import mis
        vals = self._tabulate(n, pts, order=1)
        # Create the ordinary data structure.
        sd = self.ref_el.get_spatial_dimension()
        v = vals[(0,) * sd]
        dv = [vals[alpha] for alpha in mis(sd, 1)]
        data = [[(v[i, j], [vi[i, j] for vi in dv])
                 for j in range(v.shape[1])]
                for i in range(v.shape[0])]
        return data

    def tabulate_jet(self, n, pts, order=1):
        vals = self._tabulate(n, pts, order=order)
        # Create the ordinary data structure.
        sd = self.ref_el.get_spatial_dimension()
        v0 = vals[(0,) * sd]
        data = [v0]
        for r in range(1, order+1):
            vr = numpy.zeros((sd,) * r + v0.shape, dtype=v0.dtype)
            for index in numpy.ndindex(vr.shape[:r]):
                vr[index] = vals[tuple(map(index.count, range(sd)))]
            data.append(vr.transpose((r, r+1) + tuple(range(r))))
        return data

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.ref_el == other.ref_el and
                self.continuity == other.continuity)


class PointExpansionSet(ExpansionSet):
    """Evaluates the point basis on a point reference element."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 0:
            raise ValueError("Must have a point")
        super().__init__(ref_el, **kwargs)

    def _tabulate_on_cell(self, n, pts, order=0, cell=0, direction=None):
        """Returns a dict of tabulations such that
        tabulations[alpha][i, j] = D^alpha phi_i(pts[j])."""
        assert n == 0 and order == 0
        return {(): numpy.ones((1, len(pts)))}


class LineExpansionSet(ExpansionSet):
    """Evaluates the Legendre basis on a line reference element."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 1:
            raise Exception("Must have a line")
        super().__init__(ref_el, **kwargs)

    def _tabulate_on_cell(self, n, pts, order=0, cell=0, direction=None):
        """Returns a dict of tabulations such that
        tabulations[alpha][i, j] = D^alpha phi_i(pts[j])."""
        if self.variant is not None:
            return super()._tabulate_on_cell(n, pts, order=order, cell=cell, direction=direction)

        A, b = self.affine_mappings[cell]
        Jinv = A[0, 0] if direction is None else numpy.dot(A, direction)
        xs = numpy.add(numpy.dot(pts, A.T), b)
        results = {}
        scale = self.get_scale(n, cell=cell) * numpy.sqrt(2 * numpy.arange(n+1) + 1)
        for k in range(order+1):
            v = numpy.zeros((n + 1, *xs.shape[:-1]), xs.dtype)
            if n >= k:
                v[k:] = jacobi.eval_jacobi_batch(k, k, n-k, xs)
            for p in range(n + 1):
                v[p] *= scale[p]
                scale[p] *= 0.5 * (p + k + 1) * Jinv
            results[(k,)] = v
        return results


class TriangleExpansionSet(ExpansionSet):
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception("Must have a triangle")
        super().__init__(ref_el, **kwargs)


class TetrahedronExpansionSet(ExpansionSet):
    """Collapsed orthonormal polynomial expansion on a tetrahedron."""
    def __init__(self, ref_el, **kwargs):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception("Must be a tetrahedron")
        super().__init__(ref_el, **kwargs)


def polynomial_dimension(ref_el, n, continuity=None):
    """Returns the dimension of the space of polynomials of degree no
    greater than n on the reference complex."""
    if ref_el.get_shape() == reference_element.POINT:
        if n > 0:
            raise ValueError("Only degree zero polynomials supported on point elements.")
        return 1
    top = ref_el.get_topology()

    if isinstance(continuity, dict):
        space_dimension = sum(len(continuity[dim][0]) * len(top[dim]) for dim in top)
    elif continuity == "C0":
        space_dimension = sum(math.comb(n - 1, dim) * len(top[dim]) for dim in top)
    else:
        dim = ref_el.get_spatial_dimension()
        space_dimension = math.comb(n + dim, dim) * len(top[dim])
    return space_dimension


def polynomial_entity_ids(ref_el, n, continuity=None):
    """Maps entites of a cell complex to members of a polynomial basis.

    :arg ref_el: a SimplicialComplex.
    :arg n: the polynomial degree of the expansion set.
    :arg continuity: the continuity of the expansion set.
    :returns: a dict of dicts mapping dimension and entity id to basis functions.
    """
    top = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()
    entity_ids = {}
    cur = 0
    for dim in sorted(top):
        if isinstance(continuity, dict):
            dofs, = set(len(continuity[dim][entity]) for entity in continuity[dim])
        elif continuity == "C0":
            dofs = math.comb(n - 1, dim)
        else:
            # DG numbering
            dofs = math.comb(n + dim, dim) if dim == sd else 0
        entity_ids[dim] = {}
        for entity in sorted(top[dim]):
            entity_ids[dim][entity] = list(range(cur, cur + dofs))
            cur += dofs
    return entity_ids


def polynomial_cell_node_map(ref_el, n, continuity=None):
    """Maps cells on a simplicial complex to members of a polynomial basis.

    :arg ref_el: a SimplicialComplex.
    :arg n: the polynomial degree of the expansion set.
    :arg continuity: the continuity of the expansion set.
    :returns: a numpy array mapping cell id to basis functions supported on that cell.
    """
    top = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()

    entity_ids = polynomial_entity_ids(ref_el, n, continuity)
    ref_entity_ids = polynomial_entity_ids(ref_el.construct_subelement(sd), n, continuity)

    num_cells = len(top[sd])
    dofs_per_cell = sum(len(ref_entity_ids[dim][entity])
                        for dim in ref_entity_ids for entity in ref_entity_ids[dim])
    cell_node_map = numpy.zeros((num_cells, dofs_per_cell), dtype=int)
    conn = ref_el.get_cell_connectivity()
    for cell in top[sd]:
        for dim in top:
            for ref_entity, entity in enumerate(conn[cell][dim]):
                ref_dofs = ref_entity_ids[dim][ref_entity]
                cell_node_map[cell, ref_dofs] = entity_ids[dim][entity]
    return cell_node_map


def compute_cell_point_map(ref_el, pts, unique=True, tol=1E-12):
    """Maps cells on a simplicial complex to points.
    Points outside the complex are binned to the nearest cell.

    :arg ref_el: a SimplicialComplex.
    :arg pts: an iterable of physical points on the complex.
    :kwarg unique: Are we assigning a unique cell to points on facets?
    :kwarg tol: the absolute tolerance.
    :returns: a dict mapping cell id to the point ids nearest to that cell.
    """
    top = ref_el.get_topology()
    sd = ref_el.get_spatial_dimension()
    if len(top[sd]) == 1:
        return {0: Ellipsis}

    pts = numpy.asarray(pts)
    if pts.dtype == object:
        return {cell: Ellipsis for cell in sorted(top[sd])}

    # The distance to the nearest cell is equal to the distance to the parent cell
    best = ref_el.get_parent().distance_to_point_l1(pts, rescale=True)
    tol = best + tol

    cell_point_map = {}
    for cell in sorted(top[sd]):
        # Bin points based on l1 distance
        pts_near_cell = ref_el.distance_to_point_l1(pts, entity=(sd, cell), rescale=True) < tol
        if len(pts_near_cell.shape) == 0:
            # singleton case
            if pts_near_cell:
                cell_point_map[cell] = Ellipsis
                if unique:
                    break
        else:
            if unique:
                for other in cell_point_map.values():
                    pts_near_cell[other] = False
            ipts = numpy.where(pts_near_cell)[0]
            if len(ipts) > 0:
                cell_point_map[cell] = ipts
    return cell_point_map


def compute_partition_of_unity(ref_el, pt, unique=True, tol=1E-12):
    """Computes the partition of unity functions for each subcell.

    :arg ref_el: a SimplicialComplex.
    :arg pt: a physical point on the complex.
    :kwarg unique: Are we assigning a unique cell to points on facets?
    :kwarg tol: the absolute tolerance.
    :returns: a list of (weighted) characteristic functions for each subcell.
    """
    from sympy import Piecewise
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    # assert singleton point
    pt = pt.reshape((sd,))

    # The distance to the nearest cell is equal to the distance to the parent cell
    best = ref_el.get_parent().distance_to_point_l1(pt, rescale=True)
    tol = best + tol

    # Compute characteristic function of each subcell
    otherwise = []
    masks = []
    for cell in sorted(top[sd]):
        # Bin points based on l1 distance
        pt_near_cell = ref_el.distance_to_point_l1(pt, entity=(sd, cell), rescale=True) < tol
        masks.append(Piecewise(*otherwise, (1.0, pt_near_cell), (0.0, True)))
        if unique:
            otherwise.append((0.0, pt_near_cell))
    # If the point is on a facet, divide the characteristic function by the facet multiplicity
    if not unique:
        mult = sum(masks)
        masks = [m / mult for m in masks]
    return masks
