"""Element."""
# -*- coding: utf-8 -*-
# Copyright (C) 2014 Andrew T. T. McRae
#
# This file was originally part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

from finat.ufl.finiteelementbase import FiniteElementBase
from finat.ufl.mixedelement import MixedElement, VectorElement, TensorElement
from ufl.sobolevspace import L2


class BrokenElement(FiniteElementBase):
    """The discontinuous version of an existing Finite Element space."""
    def __new__(cls, element):
        """
        Broken qualifier must be below Mixed/Vector/Tensor so we
        overload __new__ to return:

        BrokenElement(MixedElement(elem0, elem1)) -> MixedElement(BrokenElement(elem0), BrokenElement(elem1))

        and similarly for VectorElement and TensorElement.
        """
        if isinstance(element, VectorElement):
            return VectorElement(
                BrokenElement(element.sub_elements[0]),
                dim=len(element.sub_elements))

        elif isinstance(element, TensorElement):
            return TensorElement(
                BrokenElement(element.sub_elements[0]),
                symmetry=element._symmetry,
                shape=element._shape)

        elif isinstance(element, MixedElement):
            return MixedElement(list(map(BrokenElement, element.sub_elements)))

        else:  # hopefully no special casing needed
            return super().__new__(cls)

    def __init__(self, element):
        """Init."""
        self._element = element

        family = "BrokenElement"
        cell = element.cell
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        reference_value_shape = element.reference_value_shape
        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, reference_value_shape)

    def __repr__(self):
        """Doc."""
        return f"BrokenElement({repr(self._element)})"

    def mapping(self):
        """Doc."""
        return self._element.mapping()

    @property
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return L2

    def reconstruct(self, **kwargs):
        """Doc."""
        return BrokenElement(self._element.reconstruct(**kwargs))

    def __str__(self):
        """Doc."""
        return f"BrokenElement({repr(self._element)})"

    def shortstr(self):
        """Format as string for pretty printing."""
        return f"BrokenElement({repr(self._element)})"

    @property
    def embedded_subdegree(self):
        """Return embedded subdegree."""
        return self._element.embedded_subdegree

    @property
    def embedded_superdegree(self):
        """Return embedded superdegree."""
        return self._element.embedded_superdegree
