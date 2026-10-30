# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2024

# The extended element includes dim-1 tangential constraints on each facet.
# These are numbered last and used in the basis transformation.

from FIAT import finite_element, dual_set, polynomial_set, expansions
from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.functional import ComponentPointEvaluation, FrobeniusIntegralMoment
from FIAT.hierarchical import make_dual_bubbles, make_projected_bubble_moments
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature

import numpy
import math


def BernardiRaugelSpace(ref_el, order):
    """Return a basis for the extended Bernardi-Raugel space.

    The space is (Pk + FacetBubble + InteriorBubble P_{k-2})^d.  The interior
    bubbles are needed for k >= 2 when Pk has no interior degrees of freedom.

    Parameters
    ----------
    ref_el : FIAT.reference_element.Cell
        The reference cell or macrocell.
    order : int
        The polynomial degree k.

    Returns
    -------
    FIAT.polynomial_set.PolynomialSet
        A basis for the extended Bernardi-Raugel space, with the Pk basis
        numbered first, followed by facet and interior bubbles.
    """
    sd = ref_el.get_spatial_dimension()
    if order > sd:
        raise ValueError("The Bernardi-Raugel space is only defined for order <= dim")
    # An interior entity of dimension j contributes Pk bubbles when j < k.
    interior = order > 1 and not any(ref_el.get_interior_facets(dim) for dim in range(order))
    degree = sd + order - 1 if interior else sd
    Pd = polynomial_set.ONPolynomialSet(ref_el, degree, shape=(sd,), scale=1, variant="bubble")
    dimPd = expansions.polynomial_dimension(ref_el, degree, continuity="C0")
    entity_ids = expansions.polynomial_entity_ids(ref_el, degree, continuity="C0")

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
               for i in entity_ids[sd-1][f][:1]
               for j in range(sd))
    if interior:
        # The cell bubbles of degree d+k-1 span b_K P_{k-2}.
        ids.extend(i + j*dimPd
                   for cell in sorted(entity_ids[sd])
                   for i in entity_ids[sd][cell]
                   for j in range(sd))
    return Pd.take(ids)


class BernardiRaugelDualSet(dual_set.DualSet):
    """The Bernardi-Raugel dual set."""
    def __init__(self, ref_el, order=1, degree=None, reduced=False, ref_complex=None, hierarchical=False, quad_scheme=None):
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
            # Face moments of normal component against constant
            # Face moments of tangential components against dual bubbles
            ref_facet = ref_complex.construct_subcomplex(sd-1)
            ref_area = ref_facet.volume()
            # Quadrature and weight function for tangential constraints
            codim = sd-1 if degree == 1 and ref_facet.is_macrocell() else 0
            if codim == 0:
                Qt_ref, phis = make_projected_bubble_moments(ref_facet, degree)
                ft_at_qpts = phis[-1]
                bf_at_qpts = ref_facet.compute_bubble(Qt_ref.get_points())
                bf_wts = numpy.multiply(bf_at_qpts, Qt_ref.get_weights())
                scale = numpy.sum(bf_wts) / numpy.dot(ft_at_qpts, bf_wts)
                scale /= ref_area
            else:
                Qt_ref, phis = make_dual_bubbles(ref_facet, degree, codim=codim, scale=1)
                ft_at_qpts = phis[-1]
                scale = ref_area / numpy.dot(ft_at_qpts, Qt_ref.get_weights())

            ft_at_qpts *= scale
            if codim != 0:
                ft_at_qpts -= numpy.dot(ft_at_qpts, Qt_ref.get_weights()) / ref_area

            interior_facets = ref_el.get_interior_facets(sd-1) or ()
            facets = list(set(top[sd-1]) - set(interior_facets))
            Qt = {f: FacetQuadratureRule(ref_el, sd-1, f, Qt_ref, avg=True) for f in facets}

            # Quadrature and weight function for normal degrees of freedom
            if hierarchical:
                Qn = Qt
                fn_at_qpts = ft_at_qpts
            else:
                Qn_ref = parse_quadrature_scheme(ref_facet, degree, quad_scheme=quad_scheme)
                Qn = {f: FacetQuadratureRule(ref_el, sd-1, f, Qn_ref, avg=True) for f in facets}
                fn_at_qpts = numpy.full(Qn_ref.get_weights().shape, 1 / ref_area)

            thats = {f: ref_el.compute_tangents(sd-1, f) for f in facets}
            perp = lambda *v: numpy.array([v[0][1], -v[0][0]]) if len(v) == 1 else numpy.cross(*v)
            ndir = 1 if reduced else sd
            for i in range(ndir):
                for f in sorted(facets):
                    cur = len(nodes)
                    nhat = perp(*thats[f])
                    if i == 0:
                        # Normal DoF: face moment against n \phi_n
                        Q = Qn[f]
                        phi = fn_at_qpts
                        comp = nhat
                    else:
                        # Tangential constraint
                        Q = Qt[f]
                        phi = ft_at_qpts
                        if sd == 2:
                            # Face moment against t \phi_t
                            comp = thats[f][i-1]
                        else:
                            # Face moment against (n x t_j) \phi_t
                            comp = perp(nhat, thats[f][i-1])

                    nodes.append(FrobeniusIntegralMoment(ref_el, Q, numpy.outer(comp, phi)))
                    entity_ids[sd-1][f].extend(range(cur, len(nodes)))

                if i == 0 and order > 1 and not any(ref_el.get_interior_facets(dim) for dim in range(order)):
                    # Interior DoFs: cell averages against the scaled facet normals,
                    # these are preserved by the contravariant Piola pullback
                    cur = len(nodes)
                    Q = create_quadrature(ref_complex, sd + order - 1)
                    phi = numpy.full(Q.get_weights().shape, 1 / ref_el.volume())
                    nodes.extend(FrobeniusIntegralMoment(ref_el, Q, numpy.outer(perp(*thats[f]), phi))
                                 for f in sorted(facets)[:sd])
                    entity_ids[sd][0].extend(range(cur, len(nodes)))
        super().__init__(nodes, ref_el, entity_ids)


class BernardiRaugel(finite_element.CiarletElement):
    """The Bernardi-Raugel (extended) element.

    This element does not belong to a Stokes complex, but can be paired with
    DG_{k-1}. This pair is inf-sup stable, but only weakly divergence-free.
    """
    def __init__(self, ref_el, order=1, hierarchical=False, quad_scheme=None):
        sd = ref_el.get_spatial_dimension()
        if order >= sd:
            raise ValueError(f"{type(self).__name__} only defined for order < dim")
        poly_set = BernardiRaugelSpace(ref_el, order)
        dual = BernardiRaugelDualSet(ref_el, order, degree=sd, hierarchical=hierarchical, quad_scheme=quad_scheme)
        formdegree = 0
        super().__init__(poly_set, dual, poly_set.degree, formdegree, mapping="contravariant piola")
