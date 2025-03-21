# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

# This is not quite Bernardi-Raugel, but it has 2*dim*(dim+1) dofs and includes
# dim**2-1 extra constraint functionals.  The first (dim+1)**2 basis functions
# are the reference element bfs, but the extra dim**2-1 are used in the
# transformation theory.

from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.functional import ComponentPointEvaluation, FrobeniusIntegralMoment
from FIAT.hierarchical import make_dual_bubbles
from FIAT.quadrature import FacetQuadratureRule

import numpy
import math


def BernardiRaugelSpace(ref_el, order):
    """Return a basis for the extended Bernardi-Raugel space: (Pk + FacetBubble)^d."""
    sd = ref_el.get_spatial_dimension()
    if order > sd:
        raise ValueError("The Bernardi-Raugel space is only defined for order <= dim")
    Pd = polynomial_set.ONPolynomialSet(ref_el, sd, shape=(sd,), scale=1, variant="bubble")
    dimPd = expansions.polynomial_dimension(ref_el, sd, continuity="C0")
    entity_ids = expansions.polynomial_entity_ids(ref_el, sd, continuity="C0")

    slices = {dim: slice(math.comb(order-1, dim)) for dim in range(order)}
    slices.pop(sd-1, None)
    ids = [i + j * dimPd
           for dim in slices
           for f in sorted(entity_ids[dim])
           for i in entity_ids[dim][f][slices[dim]]
           for j in range(sd)]

    interior_facets = ref_el.get_interior_facets(sd-1) or ()
    facets = list(set(entity_ids[sd-1]) - set(interior_facets))
    ids.extend(i + j*dimPd
               for f in sorted(facets)
               for i in entity_ids[sd-1][f]
               for j in range(sd))
    return Pd.take(ids)


class BernardiRaugelDualSet(dual_set.DualSet):
    """The Bernardi-Raugel dual set."""
    def __init__(self, ref_el, order=1, degree=None, reduced=False, ref_complex=None):
        if ref_complex is None:
            ref_complex = ref_el
        sd = ref_el.get_spatial_dimension()
        if degree is None:
            degree = sd
        if order > sd:
            raise ValueError(f"{type(self).__name__} is only defined for order <= dim")
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        nodes = []
        if order > 0:
            # Point evaluation at lattice points
            for dim in sorted(top):
                for entity in sorted(top[dim]):
                    cur = len(nodes)
                    pts = ref_el.make_points(dim, entity, order)
                    nodes.extend(ComponentPointEvaluation(ref_el, comp, (sd,), pt)
                                 for pt in pts for comp in range(sd))
                    entity_ids[dim][entity].extend(range(cur, len(nodes)))

        if order < sd:
            # Face moments of normal/tangential components against dual bubbles
            ref_facet = ref_complex.construct_subcomplex(sd-1)
            codim = sd-1 if degree == 1 and ref_facet.is_macrocell() else 0
            Q, phis = make_dual_bubbles(ref_facet, degree, codim=codim)
            f_at_qpts = phis[-1]
            if codim != 0:
                f_at_qpts -= numpy.dot(f_at_qpts, Q.get_weights()) / ref_facet.volume()

            interior_facets = ref_el.get_interior_facets(sd-1) or ()
            facets = list(set(top[sd-1]) - set(interior_facets))

            Qs = {f: FacetQuadratureRule(ref_el, sd-1, f, Q) for f in facets}
            thats = {f: ref_el.compute_tangents(sd-1, f) for f in facets}

            R = numpy.array([[0, 1], [-1, 0]])
            ndir = 1 if reduced else sd
            for i in range(ndir):
                for f in sorted(facets):
                    cur = len(nodes)
                    if i == 0:
                        udir = numpy.dot(R, *thats[f]) if sd == 2 else numpy.cross(*thats[f])
                    else:
                        udir = thats[f][i-1]
                    detJ = Qs[f].jacobian_determinant()
                    phi_at_qpts = udir[:, None] * f_at_qpts[None, :] / detJ
                    nodes.append(FrobeniusIntegralMoment(ref_el, Qs[f], phi_at_qpts))
                    entity_ids[sd-1][f].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class BernardiRaugel(finite_element.CiarletElement):
    """The Bernardi-Raugel (extended) element.

    This element does not belong to a Stokes complex, but can be paired with
    DG_{k-1}. This pair is inf-sup stable, but only weakly divergence-free.
    """
    def __init__(self, ref_el, order=1):
        degree = ref_el.get_spatial_dimension()
        if order >= degree:
            raise ValueError(f"{type(self).__name__} only defined for order < dim")
        poly_set = BernardiRaugelSpace(ref_el, order)
        dual = BernardiRaugelDualSet(ref_el, order, degree=degree)
        formdegree = 0
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
