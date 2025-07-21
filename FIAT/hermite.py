# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified 2017 by RCK
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional


class CubicHermiteDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

    def __init__(self, ref_el):
        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()

        # get jet at each vertex
        nodes = [functional.PointEvaluationBlock(ref_el, 0, v, order=order)
                 for v in sorted(top[0]) for order in (0, 1)]

        # now only have dofs at the barycenter, which is the
        # maximal dimension
        # no edge dof
        if sd > 1:
            # face dof
            # point evaluation at barycenter
            nodes.extend(functional.PointEvaluationBlock(ref_el, 2, f, degree=3)
                         for f in sorted(top[2]))

        super().__init__(nodes, ref_el)


class CubicHermite(finite_element.CiarletElement):
    """The cubic Hermite finite element.  It is what it is."""

    def __init__(self, ref_el, deg=3):
        assert deg == 3
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 3)
        dual = CubicHermiteDualSet(ref_el)

        super().__init__(poly_set, dual, 3)
