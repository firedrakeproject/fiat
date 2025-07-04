"""Doc."""
# Copyright (C) 2008-2016 Andrew T. T. McRae
#
# This file was originally part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

from finat.ufl.finiteelementbase import FiniteElementBase
from ufl.sobolevspace import L2, SobolevSpace
from ufl.sobolevspace import HCurl as HCurlSobolevSpace
from ufl.sobolevspace import HDiv as HDivSobolevSpace


class CallableSobolevSpace(SobolevSpace):
    """A Sobolev space that can be called to create HDiv and HCurl elements."""

    def __init__(self, name, parents=None):
        super().__init__(name, parents)

    def __call__(self, element):
        """Syntax shortcut to create a HDivElement or HCurlElement."""
        if self.name == "HDiv":
            return HDivElement(element)
        elif self.name == "HCurl":
            return HCurlElement(element)
        raise NotImplementedError(
            "SobolevSpace has no call operator (only the specific HDiv and HCurl instances)."
        )


HCurl = CallableSobolevSpace(HCurlSobolevSpace.name, HCurlSobolevSpace.parents)
HDiv = CallableSobolevSpace(HDivSobolevSpace.name, HDivSobolevSpace.parents)


class WrapperElementBase(FiniteElementBase):
    """A modified version of a tensor product element."""
    __slots__ = ("_element", )

    def __init__(self, element, reference_value_shape, sobolev_space, mapping):
        self._element = element
        self._sobolev_space = sobolev_space
        self._mapping = mapping

        family = element.family()
        cell = element.cell
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()

        super().__init__(family, cell, degree, quad_scheme, reference_value_shape)

    def __repr__(self):
        """Doc."""
        return f"{type(self).__name__}({repr(self._element)})"

    def mapping(self):
        """Doc."""
        return self._mapping

    @property
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return self._sobolev_space

    def reconstruct(self, **kwargs):
        """Doc."""
        return type(self)(self._element.reconstruct(**kwargs))

    def variant(self):
        """Doc."""
        return self._element.variant()

    def __str__(self):
        """Doc."""
        return f"{type(self).__name__}({repr(self._element)})"

    def shortstr(self):
        """Format as string for pretty printing."""
        return f"HDivTraceElement({self._element.shortstr()})"

    @property
    def embedded_subdegree(self):
        """Return embedded subdegree."""
        return self._element.embedded_subdegree

    @property
    def embedded_superdegree(self):
        """Return embedded superdegree."""
        return self._element.embedded_superdegree


class HDivTraceElement(WrapperElementBase):
    """A HDiv Trace version of a tensor product element, assuming this makes mathematical sense."""
    __slots__ = ("_element", )

    def __init__(self, element):
        reference_value_shape = ()
        super().__init__(element, reference_value_shape, L2, "identity")


class HDivElement(WrapperElementBase):
    """A div-conforming version of a tensor product element, assuming this makes mathematical sense."""
    __slots__ = ("_element", )

    def __init__(self, element):
        reference_value_shape = (element.cell.topological_dimension(),)
        super().__init__(element, reference_value_shape, HDiv, "contravariant Piola")


class HCurlElement(WrapperElementBase):
    """A curl-conforming version of a tensor product element, assuming this makes mathematical sense."""
    __slots__ = ("_element",)

    def __init__(self, element):
        """Doc."""
        reference_value_shape = (element.cell.topological_dimension(),)
        super().__init__(element, reference_value_shape, HCurl, "covariant Piola")


class WithMapping(FiniteElementBase):
    """Specify an alternative mapping for the wrappee.

    For example,
    to use identity mapping instead of Piola map with an element E,
    write
    remapped = WithMapping(E, "identity")
    """

    def __init__(self, wrapee, mapping):
        """Doc."""
        if mapping == "symmetries":
            raise ValueError("Can't change mapping to 'symmetries'")
        self._mapping = mapping
        self.wrapee = wrapee

    def __getattr__(self, attr):
        """Doc."""
        try:
            return getattr(self.wrapee, attr)
        except AttributeError:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (type(self).__name__, attr))

    def __repr__(self):
        """Doc."""
        return f"WithMapping({repr(self.wrapee)}, '{self._mapping}')"

    def value_shape(self, domain):
        """Doc."""
        gdim = domain.geometric_dimension()
        mapping = self.mapping()
        if mapping in {"covariant Piola", "contravariant Piola"}:
            return (gdim,)
        elif mapping in {"double covariant Piola", "double contravariant Piola"}:
            return (gdim, gdim)
        else:
            return self.wrapee.value_shape(domain)

    @property
    def reference_value_shape(self):
        """Doc."""
        tdim = self.cell.topological_dimension()
        mapping = self.mapping()
        if mapping in {"covariant Piola", "contravariant Piola"}:
            return (tdim,)
        elif mapping in {"double covariant Piola", "double contravariant Piola"}:
            return (tdim, tdim)
        else:
            return self.wrapee.reference_value_shape

    def mapping(self):
        """Doc."""
        return self._mapping

    @property
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        if self.wrapee.mapping() == self.mapping():
            return self.wrapee.sobolev_space
        else:
            return L2

    def reconstruct(self, **kwargs):
        """Doc."""
        mapping = kwargs.pop("mapping", self._mapping)
        wrapee = self.wrapee.reconstruct(**kwargs)
        return type(self)(wrapee, mapping)

    def variant(self):
        """Doc."""
        return self.wrapee.variant()

    def __str__(self):
        """Doc."""
        return f"WithMapping({repr(self.wrapee)}, {self._mapping})"

    def shortstr(self):
        """Doc."""
        return f"WithMapping({self.wrapee.shortstr()}, {self._mapping})"

    @property
    def embedded_subdegree(self):
        """Return embedded subdegree."""
        return self._element.embedded_subdegree

    @property
    def embedded_superdegree(self):
        """Return embedded superdegree."""
        return self._element.embedded_superdegree
