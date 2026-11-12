# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified 2017 by RCK
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import TRIANGLE


class HermiteDualSet(dual_set.DualSet):
    """The Hermite dual set is defined in 1D for any degree, with degrees of
    freedom given by the first-order jet at the vertices and point
    evaluations at interior points of the interval.  In higher dimensions,
    it is defined only for degree 3, with the first-order jet at the vertices
    and point evaluations at the barycenters of the 2D entities.  The reduced
    dual set omits the barycenter evaluations."""

    def __init__(self, ref_el, degree, variant=None, reduced=False):
        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # get first order jet at each vertex
        for v in sorted(top[0]):
            pt, = ref_el.make_points(0, v, degree, variant=variant)
            cur = len(nodes)
            nodes.append(functional.PointEvaluation(ref_el, pt))
            nodes.extend(functional.PointDerivative(ref_el, pt, alpha)
                         for alpha in polynomial_set.mis(sd, 1))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        if sd == 1:
            # edge dofs: point evaluations to support higher order in 1D
            for e in sorted(top[1]):
                cur = len(nodes)
                pts = ref_el.make_points(1, e, degree-2, variant=variant)
                nodes.extend(functional.PointEvaluation(ref_el, pt) for pt in pts)
                entity_ids[1][e].extend(range(cur, len(nodes)))
        elif not reduced:
            # no edge dof
            # face dof: point evaluation at barycenter
            for f in sorted(top[2]):
                cur = len(nodes)
                pt, = ref_el.make_points(2, f, degree, variant=variant)
                nodes.append(functional.PointEvaluation(ref_el, pt))
                entity_ids[2][f].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Hermite(finite_element.CiarletElement):
    """The Hermite finite element.  It has any degree of at least three
    on intervals, and degree three on higher dimensional simplices.  The
    reduced element on triangles determines the barycenter value from the
    vertex jets, so that it still contains the quadratics."""

    def __init__(self, ref_el, degree=3, variant=None, reduced=False):
        if ref_el.get_spatial_dimension() > 1 and degree != 3:
            raise ValueError("Hermite elements in more than one dimension must have degree 3.")
        if reduced and ref_el.get_shape() != TRIANGLE:
            raise ValueError("Reduced Hermite elements only defined on triangles.")
        if variant is None:
            variant = "gll"
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        if reduced:
            poly_set = polynomial_set.TaylorConstrainedPolynomialSet(poly_set, (1, 1/2))
        dual = HermiteDualSet(ref_el, degree, variant=variant, reduced=reduced)

        super().__init__(poly_set, dual, degree)
