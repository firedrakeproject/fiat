# Copyright (C) 2013 Andrew T. T. McRae (Imperial College London)
# Copyright (C) 2015 Jan Blechta
# Copyright (C) 2018 Patrick E. Farrell
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT.lagrange import Lagrange
from FIAT.hierarchical import IntegratedLegendre
from FIAT.restricted import RestrictedElement
from itertools import chain


class CodimBubble(RestrictedElement):
    """Bubbles of a certain codimension."""

    def __init__(self, ref_el, degree, codim, variant=None):
        if variant and variant.startswith("integral"):
            CG = IntegratedLegendre
        else:
            CG = Lagrange
        element = CG(ref_el, degree, variant=variant)

        cell_dim = ref_el.get_dimension()
        assert cell_dim == max(element.entity_dofs().keys())
        dofs = list(sorted(chain(*element.entity_dofs()[cell_dim - codim].values())))
        if len(dofs) == 0:
            raise RuntimeError('Bubble element of degree %d and codimension %d has no dofs' % (degree, codim))

        super().__init__(element, indices=dofs)


class Bubble(CodimBubble):
    """The bubble finite element: the dofs of the Lagrange FE in the interior of the cell"""

    def __init__(self, ref_el, degree, variant=None):
        super().__init__(ref_el, degree, codim=0, variant=variant)


class FacetBubble(CodimBubble):
    """The facet bubble finite element: the dofs of the Lagrange FE in the interior of the facets"""

    def __init__(self, ref_el, degree, variant=None):
        super().__init__(ref_el, degree, codim=1, variant=variant)
