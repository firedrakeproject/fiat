# Copyright (C) 2020 Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
import numpy as np
from FIAT.functional import Functional
from FIAT.dual_set import DualSet
from collections import defaultdict
from itertools import zip_longest


def compute_pointwise_dual(el, pts):
    """Constructs a dual basis to the basis for el as a linear combination
    of a set of pointwise evaluations.  This is useful when the
    prescribed finite element isn't Ciarlet (e.g. the basis functions
    are provided explicitly as formulae).  Alternately, the element's
    given dual basis may involve differentiation, making run-time
    interpolation difficult in FIAT clients.  The pointwise dual,
    consisting only of pointwise evaluations, will effectively replace
    these derivatives with (automatically determined) finite
    differences.  This is exact on the polynomial space, but is an
    approximation if applied to functions outside the space.

    :param el: a :class:`FiniteElement`.
    :param pts: an iterable of points with the same length as el's
                dimension.  These points must be unisolvent for the
                polynomial space
    :returns: a :class `DualSet`
    """
    nbf = el.space_dimension()

    T = el.ref_el
    sd = T.get_spatial_dimension()

    assert np.asarray(pts).shape == (int(nbf / np.prod(el.value_shape())), sd)

    z = tuple([0] * sd)

    nds = []

    V = el.tabulate(0, pts)[z]

    # Make a square system, invert, and then put it back in the right
    # shape so we have (nbf, ..., npts) with more dimensions
    # for vector or tensor-valued elements.
    alphas = np.linalg.inv(V.reshape((nbf, -1)).T).reshape(V.shape)

    # Each row of alphas gives the coefficients of a functional,
    # represented, as elsewhere in FIAT, as a summation of
    # components of the input at particular points.

    # This logic picks out the points and components for which the
    # weights are actually nonzero to construct the functional.

    pts = np.asarray(pts)
    for coeffs in alphas:
        pt_dict = defaultdict(list)
        nonzero = np.where(np.abs(coeffs) > 1.e-12)
        *comp, pt_index = nonzero

        for pt, coeff_comp in zip(pts[pt_index],
                                  zip_longest(coeffs[nonzero],
                                              zip(*comp), fillvalue=())):
            pt_dict[tuple(pt)].append(coeff_comp)

        nds.append(Functional(T, el.value_shape(), dict(pt_dict), {}, "node"))

    return DualSet(nds, T, el.entity_dofs())


def make_entity_ids(ref_el, points, tol=1e-12):
    """The topological association in a dictionary"""
    top = ref_el.topology
    invtop = {top[d][e]: (d, e) for d in top for e in top[d]}
    bary = ref_el.compute_barycentric_coordinates(points)

    entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
    for i in np.lexsort(bary.T):
        verts = tuple(np.flatnonzero(abs(bary[i]) > tol))
        dim, entity = invtop[verts]
        entity_ids[dim][entity].append(i)
    return entity_ids
