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
from FIAT.quadrature_schemes import create_quadrature
from FIAT.quadrature import FacetQuadratureRule
import numpy


def ExtendedBernardiRaugelSpace(ref_el, subdegree):
    r"""Return a basis for the extended Bernardi-Raugel space.
    P_k^d + (P_{dim} \ P_{dim-1})^d"""
    sd = ref_el.get_spatial_dimension()
    if subdegree >= sd:
        raise ValueError("The Bernardi-Raugel space is only defined for subdegree < dim")
    Pd = polynomial_set.ONPolynomialSet(ref_el, sd, shape=(sd,), scale=1, variant="bubble")
    dimPd = expansions.polynomial_dimension(ref_el, sd, continuity="C0")
    entity_ids = expansions.polynomial_entity_ids(ref_el, sd, continuity="C0")
    ids = [i + j * dimPd
           for dim in (*tuple(range(subdegree)), sd-1)
           for f in sorted(entity_ids[dim])
           for i in entity_ids[dim][f]
           for j in range(sd)]
    return Pd.take(ids)


class BernardiRaugelDualSet(dual_set.DualSet):
    """The Bernardi-Raugel dual set."""
    def __init__(self, ref_complex, degree, subdegree=1, reduced=False):
        ref_el = ref_complex.get_parent() or ref_complex
        sd = ref_el.get_spatial_dimension()
        if subdegree > sd:
            raise ValueError("The Bernardi-Raugel dual is only defined for subdegree <= dim")
        top = ref_el.get_topology()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # Point evaluation at lattice points
        nodes = []
        for dim in range(subdegree):
            for entity in sorted(top[dim]):
                cur = len(nodes)
                pts = ref_el.make_points(dim, entity, subdegree)
                nodes.extend(ComponentPointEvaluation(ref_el, comp, (sd,), pt)
                             for pt in pts for comp in range(sd))
                entity_ids[dim][entity].extend(range(cur, len(nodes)))

        if subdegree < sd:
            # Face moments of normal/tangential components against mean-free bubbles
            facet = ref_complex.construct_subcomplex(sd-1)
            Q = create_quadrature(facet, 2*degree)
            if degree == 1 and facet.is_macrocell():
                P = polynomial_set.ONPolynomialSet(facet, degree, scale=1, variant="bubble")
                f_at_qpts = P.tabulate(Q.get_points())[(0,)*(sd-1)][-1]
            else:
                ref_facet = facet.get_parent() or facet
                f_at_qpts = ref_facet.compute_bubble(Q.get_points())
            f_at_qpts -= numpy.dot(f_at_qpts, Q.get_weights()) / facet.volume()

            Qs = {f: FacetQuadratureRule(ref_el, sd-1, f, Q)
                  for f in sorted(top[sd-1])}

            thats = {f: ref_el.compute_tangents(sd-1, f)
                     for f in sorted(top[sd-1])}

            R = numpy.array([[0, 1], [-1, 0]])
            ndir = 1 if reduced else sd
            for i in range(ndir):
                for f, Q_mapped in Qs.items():
                    cur = len(nodes)
                    if i == 0:
                        udir = numpy.dot(R, *thats[f]) if sd == 2 else numpy.cross(*thats[f])
                    else:
                        udir = thats[f][i-1]
                    detJ = Q_mapped.jacobian_determinant()
                    phi_at_qpts = udir[:, None] * f_at_qpts[None, :] / detJ
                    nodes.append(FrobeniusIntegralMoment(ref_el, Q_mapped, phi_at_qpts))
                    entity_ids[sd-1][f].extend(range(cur, len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class BernardiRaugel(finite_element.CiarletElement):
    """The Bernardi-Raugel extended element."""
    def __init__(self, ref_el, degree=None, subdegree=1):
        sd = ref_el.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError("Bernardi-Raugel only defined for degree = dim")
        poly_set = ExtendedBernardiRaugelSpace(ref_el, subdegree)
        dual = BernardiRaugelDualSet(ref_el, degree, subdegree=subdegree)
        formdegree = sd - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree, mapping="contravariant piola")
