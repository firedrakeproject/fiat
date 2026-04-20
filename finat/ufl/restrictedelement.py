"""This module defines the UFL finite element classes."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file was originally part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

from finat.ufl.finiteelementbase import FiniteElementBase
from finat.ufl.mixedelement import MixedElement, VectorElement, TensorElement

valid_restriction_domains = ("interior", "facet", "ridge", "face", "edge", "vertex", "reduced")


class RestrictedElement(FiniteElementBase):
    """Represents the restriction of a finite element to a type of cell entity."""
    def __new__(cls, element, restriction_domain):
        """
        Restricted qualifier must be below Mixed/Vector/Tensor so we
        overload __new__ to return:

        RestrictedElement(MixedElement(elem0, elem1), dom) -> MixedElement(RestrictedElement(elem0, dom), RestrictedElement(elem1, dom))

        and similarly for VectorElement and TensorElement.
        """
        if isinstance(element, (VectorElement, TensorElement)):
            return element.reconstruct(sub_element=RestrictedElement(element.sub_elements[0], restriction_domain))

        elif isinstance(element, MixedElement):
            return MixedElement([RestrictedElement(e, restriction_domain) for e in element.sub_elements])

        else:  # hopefully no special casing needed
            return super().__new__(cls)

    def __init__(self, element, restriction_domain):
        """Doc."""
        if not isinstance(element, FiniteElementBase):
            raise ValueError("Expecting a finite element instance.")
        if restriction_domain not in valid_restriction_domains:
            raise ValueError(f"Expecting one of the strings: {valid_restriction_domains}")

        FiniteElementBase.__init__(self, "RestrictedElement", element.cell,
                                   element.degree(),
                                   element.quadrature_scheme(),
                                   element.reference_value_shape)

        self._element = element

        self._restriction_domain = restriction_domain

    def __repr__(self):
        """Doc."""
        return f"RestrictedElement({repr(self._element)}, {repr(self._restriction_domain)})"

    @property
    def sobolev_space(self):
        """Doc."""
        return self._element.sobolev_space

    def is_cellwise_constant(self):
        """Return whether the basis functions of this element is spatially constant over each cell."""
        return self._element.is_cellwise_constant()

    def _is_linear(self):
        """Doc."""
        return self._element._is_linear()

    def sub_element(self):
        """Return the element which is restricted."""
        return self._element

    def mapping(self):
        """Doc."""
        return self._element.mapping()

    def restriction_domain(self):
        """Return the domain onto which the element is restricted."""
        return self._restriction_domain

    def reconstruct(self, element=None, **kwargs):
        """Doc."""
        if element is None:
            element = self._element.reconstruct(**kwargs)
        return RestrictedElement(element, self._restriction_domain)

    def __str__(self):
        """Format as string for pretty printing."""
        return "<%s>|_{%s}" % (self._element, self._restriction_domain)

    def shortstr(self):
        """Format as string for pretty printing."""
        return "<%s>|_{%s}" % (self._element.shortstr(),
                               self._restriction_domain)

    def symmetry(self):
        r"""Return the symmetry dict, which is a mapping :math:`c_0 \\to c_1`.

        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.  A component is a tuple of one or more ints.
        """
        return self._element.symmetry()

    @property
    def num_sub_elements(self):
        """Return number of sub elements."""
        return self._element.num_sub_elements

    @property
    def sub_elements(self):
        """Return list of sub elements."""
        return self._element.sub_elements

    def num_restricted_sub_elements(self):
        """Return number of restricted sub elements."""
        return 1

    def restricted_sub_elements(self):
        """Return list of restricted sub elements."""
        return (self._element,)

    def variant(self):
        """Doc."""
        return self._element.variant()
