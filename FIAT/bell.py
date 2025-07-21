# Copyright (C) 2018 Robert C. Kirby
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This is not quite Bell, but is 21-dofs and includes 3 extra constraint
# functionals.  The first 18 basis functions are the reference element
# bfs, but the extra three are used in the transformation theory.

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import TRIANGLE, ufc_simplex
from FIAT.quadrature_schemes import create_quadrature
from FIAT.jacobi import eval_jacobi


class BellDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        top = ref_el.get_topology()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in top[dim]} for dim in top}
        nodes = []

        # get jet at each vertex
        for v in sorted(top[0]):
            cur = len(nodes)
            x, = ref_el.make_points(0, v, degree)
            nodes.append(functional.PointEvaluation(ref_el, x))

            # first and second derivatives
            nodes.extend(functional.PointDerivative(ref_el, x, alpha)
                         for i in (1, 2) for alpha in polynomial_set.mis(sd, i))
            entity_ids[0][v].extend(range(cur, len(nodes)))

        # we need an edge quadrature rule for the moment
        facet = ufc_simplex(1)
        Q_ref = create_quadrature(facet, 8)
        qpts = Q_ref.get_points()[:, 0]
        leg4_at_qpts = eval_jacobi(0, 0, 4, 2.0*qpts - 1)

        for e in sorted(top[1]):
            cur = len(nodes)
            nodes.append(functional.IntegralMomentOfNormalDerivative(ref_el, e, Q_ref, leg4_at_qpts))
            entity_ids[1][e].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class Bell(finite_element.CiarletElement):
    """The Bell finite element."""

    def __init__(self, ref_el, degree=5):
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError(f"{type(self).__name__} only defined on triangles")
        if degree != 5:
            raise ValueError(f"{type(self).__name__} only defined for degree=5.")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = BellDualSet(ref_el, degree)
        super().__init__(poly_set, dual, degree)
