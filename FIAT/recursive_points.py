# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2023

import numpy

"""
@article{isaac2020recursive,
  title={Recursive, parameter-free, explicitly defined interpolation nodes for simplices},
  author={Isaac, Tobin},
  journal={SIAM Journal on Scientific Computing},
  volume={42},
  number={6},
  pages={A4046--A4062},
  year={2020},
  publisher={SIAM}
}
"""


def multiindex_equal(d, isum, imin=0):
    """A generator for d-tuple multi-indices whose sum is isum and minimum is imin.
    """
    if d <= 0:
        return
    imax = isum - (d - 1) * imin
    if imax < imin:
        return
    for i in range(imin, imax):
        for a in multiindex_equal(d - 1, isum - i, imin=imin):
            yield a + (i,)
    yield (imin,) * (d - 1) + (imax,)


class RecursivePointSet(object):
    """Family of points on the unit interval.  This class essentially is a
    lazy-evaluate-and-cache dictionary: the user passes a routine to evaluate
    entries for unknown keys """

    def __init__(self, f):
        self._f = f
        self._cache = {}

    def interval_points(self, degree):
        try:
            return self._cache[degree]
        except KeyError:
            x = self._f(degree)
            if x is None:
                x_ro = x
            else:
                x_ro = numpy.array(x).flatten()
                x_ro.setflags(write=False)
            return self._cache.setdefault(degree, x_ro)

    def _recursive(self, alpha):
        """The barycentric (d-1)-simplex coordinates for a
        multiindex alpha of length d and sum n, based on a 1D node family."""
        d = len(alpha)
        n = sum(alpha)
        b = numpy.zeros((d,), dtype="d")
        xn = self.interval_points(n)
        if xn is None:
            return b
        if d == 2:
            b[:] = xn[list(alpha)]
            return b
        weight = 0.0
        for i in range(d):
            w = xn[n - alpha[i]]
            alpha_noti = alpha[:i] + alpha[i+1:]
            br = self._recursive(alpha_noti)
            b[:i] += w * br[:i]
            b[i+1:] += w * br[i:]
            weight += w
        b /= weight
        return b

    def recursive_points(self, vertices, order, interior=0):
        X = numpy.array(vertices)
        get_point = lambda alpha: tuple(numpy.dot(self._recursive(alpha), X))
        return list(map(get_point, multiindex_equal(len(vertices), order, interior)))

    def make_points(self, ref_el, dim, entity_id, order):
        """Constructs a lattice of points on the entity_id:th
        facet of dimension dim.  Order indicates how many points to
        include in each direction."""
        if dim == 0:
            return (ref_el.get_vertices()[entity_id], )
        elif 0 < dim < ref_el.get_spatial_dimension():
            entity_verts = \
                ref_el.get_vertices_of_subcomplex(
                    ref_el.get_topology()[dim][entity_id])
            return self.recursive_points(entity_verts, order, 1)
        elif dim == ref_el.get_spatial_dimension():
            return self.recursive_points(ref_el.get_vertices(), order, 1)
        else:
            raise ValueError("illegal dimension")


def make_recursive_point_set(family):
    from FIAT import quadrature, reference_element
    ref_el = reference_element.UFCInterval()
    if family == "equispaced":
        f = lambda n: numpy.linspace(0.0, 1.0, n + 1)
    elif family == "dg_equispaced":
        f = lambda n: numpy.linspace(1.0/(n+2.0), (n+1.0)/(n+2.0), n + 1)
    elif family == "gl":
        lr = quadrature.GaussLegendreQuadratureLineRule
        f = lambda n: lr(ref_el, n + 1).pts
    elif family == "gll":
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule
        f = lambda n: lr(ref_el, n + 1).pts if n else None
    else:
        raise ValueError("Invalid node family %s" % family)
    return RecursivePointSet(f)


if __name__ == "__main__":
    from FIAT import reference_element
    from matplotlib import pyplot as plt
    ref_el = reference_element.ufc_simplex(2)
    h = numpy.sqrt(3)
    s = 2*h/3
    ref_el.vertices = [(0, s), (-1.0, s-h), (1.0, s-h)]
    x = numpy.array(ref_el.vertices + ref_el.vertices[:1])
    plt.plot(x[:, 0], x[:, 1], "k")

    order = 7
    rule = "gll"
    dg_rule = "gl"

    # rule = "equispaced"
    # dg_rule = "dg_equispaced"

    family = make_recursive_point_set(rule)
    dg_family = make_recursive_point_set(dg_rule)

    for d in range(1, 4):
        print(family.make_points(reference_element.ufc_simplex(d), d, 0, d))

    topology = ref_el.get_topology()
    for dim in topology:
        for entity in topology[dim]:
            pts = family.make_points(ref_el, dim, entity, order)
            if len(pts):
                x = numpy.array(pts)
                for r in range(1, 3):
                    th = r * (2*numpy.pi)/3
                    ct = numpy.cos(th)
                    st = numpy.sin(th)
                    Q = numpy.array([[ct, -st], [st, ct]])
                    x = numpy.concatenate([x, numpy.dot(x, Q)])
                plt.scatter(x[:, 0], x[:, 1])

    x0 = 2.0
    h = -h
    s = 2*h/3
    ref_el = reference_element.ufc_simplex(2)
    ref_el.vertices = [(x0, s), (x0-1.0, s-h), (x0+1.0, s-h)]

    x = numpy.array(ref_el.vertices + ref_el.vertices[:1])
    d = len(ref_el.vertices)
    x0 = sum(x[:d])/d
    plt.plot(x[:, 0], x[:, 1], "k")

    pts = dg_family.recursive_points(ref_el.vertices, order)
    x = numpy.array(pts)
    for r in range(1, 3):
        th = r * (2*numpy.pi)/3
        ct = numpy.cos(th)
        st = numpy.sin(th)
        Q = numpy.array([[ct, -st], [st, ct]])
        x = numpy.concatenate([x, numpy.dot(x-x0, Q)+x0])
    plt.scatter(x[:, 0], x[:, 1])

    plt.gca().set_aspect('equal', 'box')
    plt.show()
