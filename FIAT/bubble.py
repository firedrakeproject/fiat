# Copyright (C) 2013 Andrew T. T. McRae (Imperial College London)
# Copyright (C) 2015 Jan Blechta
# Copyright (C) 2018 Patrick E. Farrell
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy

from FIAT.check_format_variant import parse_quadrature_scheme
from FIAT.expansions import polynomial_dimension
from FIAT.lagrange import Lagrange
from FIAT.hierarchical import IntegratedLegendre
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature import QuadratureRule
from FIAT.reference_element import SimplicialComplex
from FIAT.restricted import RestrictedElement
from itertools import chain


def make_projected_bubble_moment(
    ref_el: SimplicialComplex, degree: int, quad_scheme: str = None
) -> tuple[QuadratureRule, numpy.ndarray]:
    """Tabulate the leading-degree part of the bubble on the given cell.

    The bubble is projected onto polynomials of the given degree, and the
    components of lower degree are dropped.  What remains is orthogonal to
    polynomials of degree ``degree - 1`` and is invariant under the symmetry
    group of the cell, which is what a constraint functional annihilating that
    lower-degree space needs of its weight.  The result is normalized to unit
    L2 norm, which keeps the moments of comparable size across cells.

    Parameters
    ----------
    ref_el : SimplicialComplex
        The cell on which to tabulate the weight.
    degree : int
        The degree of the weight.
    quad_scheme : str, optional
        The quadrature scheme, see ``parse_quadrature_scheme``.

    Returns
    -------
    tuple
        The quadrature rule and the weight tabulated on its points.
    """
    sd = ref_el.get_spatial_dimension()
    Q = parse_quadrature_scheme(ref_el, degree + sd + 1, quad_scheme)
    Pk = ONPolynomialSet(ref_el, degree, scale="orthonormal")
    phis = Pk.tabulate(Q.get_points())[(0,) * sd]
    duals = numpy.multiply(phis, Q.get_weights())
    coeffs = numpy.dot(duals, ref_el.compute_bubble(Q.get_points()))
    bubble_norm = numpy.linalg.norm(coeffs)
    coeffs[:polynomial_dimension(ref_el, degree-1)] = 0
    norm = numpy.linalg.norm(coeffs)
    if norm <= 1E-12 * bubble_norm:
        raise ValueError(f"The bubble on {type(ref_el).__name__} "
                         f"has no component of degree {degree}")
    return Q, numpy.dot(coeffs / norm, phis)


class CodimBubble(RestrictedElement):
    """Bubbles of a certain codimension."""

    def __init__(self, ref_el, degree, codim, variant=None, quad_scheme=None):
        if variant and variant.startswith("integral"):
            element = IntegratedLegendre(ref_el, degree, variant=variant, quad_scheme=quad_scheme)
        else:
            element = Lagrange(ref_el, degree, variant=variant)

        cell_dim = ref_el.get_dimension()
        assert cell_dim == max(element.entity_dofs().keys())
        dofs = list(sorted(chain(*element.entity_dofs()[cell_dim - codim].values())))
        if len(dofs) == 0:
            raise RuntimeError('Bubble element of degree %d and codimension %d has no dofs' % (degree, codim))

        super().__init__(element, indices=dofs)


class Bubble(CodimBubble):
    """The bubble finite element: the dofs of the Lagrange FE in the interior of the cell"""

    def __init__(self, ref_el, degree, variant=None, quad_scheme=None):
        super().__init__(ref_el, degree, codim=0, variant=variant, quad_scheme=quad_scheme)


class FacetBubble(CodimBubble):
    """The facet bubble finite element: the dofs of the Lagrange FE in the interior of the facets"""

    def __init__(self, ref_el, degree, variant=None, quad_scheme=None):
        super().__init__(ref_el, degree, codim=1, variant=variant, quad_scheme=quad_scheme)
