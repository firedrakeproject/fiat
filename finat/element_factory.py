# This file was modified from FFC
# (http://bitbucket.org/fenics-project/ffc), copyright notice
# reproduced below.
#
# Copyright (C) 2009-2013 Kristian B. Oelgaard and Anders Logg
#
# This file is part of FFC.
#
# FFC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFC. If not, see <http://www.gnu.org/licenses/>.

import weakref
from functools import singledispatch, cache

import finat
import finat.ufl
import ufl

from FIAT import ufc_cell

__all__ = ("as_fiat_cell", "create_base_element",
           "create_element", "supported_elements")


# List of supported elements and mapping to element classes
supported_elements = {"Argyris": finat.Argyris,
                      "Bell": finat.Bell,
                      "Bernardi-Raugel": finat.BernardiRaugel,
                      "Bernardi-Raugel Bubble": finat.BernardiRaugelBubble,
                      "Bernstein": finat.Bernstein,
                      "Brezzi-Douglas-Fortin-Marini": finat.BrezziDouglasFortinMarini,
                      "Brezzi-Douglas-Marini Cube Face": finat.BrezziDouglasMariniCubeFace,
                      "Brezzi-Douglas-Marini": finat.BrezziDouglasMarini,
                      "Brezzi-Douglas-Marini Cube Edge": finat.BrezziDouglasMariniCubeEdge,
                      "Bubble": finat.Bubble,
                      "FacetBubble": finat.FacetBubble,
                      "Crouzeix-Raviart": finat.CrouzeixRaviart,
                      "Direct Serendipity": finat.DirectSerendipity,
                      "Discontinuous Lagrange": finat.DiscontinuousLagrange,
                      "Discontinuous Lagrange L2": finat.DiscontinuousLagrange,
                      "Discontinuous Taylor": finat.DiscontinuousTaylor,
                      "Discontinuous Raviart-Thomas": lambda *args, **kwargs: finat.DiscontinuousElement(finat.RaviartThomas(*args, **kwargs)),
                      "DPC": finat.DPC,
                      "DPC L2": finat.DPC,
                      "Hermite": finat.Hermite,
                      "Hsieh-Clough-Tocher": finat.HsiehCloughTocher,
                      "Reduced-Hsieh-Clough-Tocher": finat.ReducedHsiehCloughTocher,
                      "QuadraticPowellSabin6": finat.QuadraticPowellSabin6,
                      "QuadraticPowellSabin12": finat.QuadraticPowellSabin12,
                      "Alfeld-Sorokina": finat.AlfeldSorokina,
                      "Arnold-Qin": finat.ArnoldQin,
                      "Reduced-Arnold-Qin": finat.ReducedArnoldQin,
                      "Christiansen-Hu": finat.ChristiansenHu,
                      "Guzman-Neilan 1st kind H1": finat.GuzmanNeilanFirstKindH1,
                      "Guzman-Neilan 2nd kind H1": finat.GuzmanNeilanSecondKindH1,
                      "Guzman-Neilan H1(div)": finat.GuzmanNeilanH1div,
                      "Guzman-Neilan Bubble": finat.GuzmanNeilanBubble,
                      "Johnson-Mercier": finat.JohnsonMercier,
                      "Lagrange": finat.Lagrange,
                      "Kong-Mulder-Veldhuizen": finat.KongMulderVeldhuizen,
                      "Gauss-Lobatto-Legendre": finat.GaussLobattoLegendre,
                      "Gauss-Legendre": finat.GaussLegendre,
                      "Gauss-Legendre L2": finat.GaussLegendre,
                      "Morley": finat.Morley,
                      "Nedelec 1st kind H(curl)": finat.Nedelec,
                      "Nedelec 2nd kind H(curl)": finat.NedelecSecondKind,
                      "Raviart-Thomas": finat.RaviartThomas,
                      "Real": finat.Real,
                      "S": finat.Serendipity,
                      "SminusF": finat.TrimmedSerendipityFace,
                      "SminusDiv": finat.TrimmedSerendipityDiv,
                      "SminusE": finat.TrimmedSerendipityEdge,
                      "SminusCurl": finat.TrimmedSerendipityCurl,
                      "Regge": finat.Regge,
                      "HDiv Trace": finat.HDivTrace,
                      "Hellan-Herrmann-Johnson": finat.HellanHerrmannJohnson,
                      "Gopalakrishnan-Lederer-Schoberl 1st kind": finat.GopalakrishnanLedererSchoberlFirstKind,
                      "Gopalakrishnan-Lederer-Schoberl 2nd kind": finat.GopalakrishnanLedererSchoberlSecondKind,
                      "Conforming Arnold-Winther": finat.ArnoldWinther,
                      "Nonconforming Arnold-Winther": finat.ArnoldWintherNC,
                      "Hu-Zhang": finat.HuZhang,
                      "Mardal-Tai-Winther": finat.MardalTaiWinther,
                      # These require special treatment
                      "Q": None,
                      "DQ": None,
                      "DQ L2": None,
                      "RTCE": None,
                      "RTCF": None,
                      "NCE": None,
                      "NCF": None,
                      }
"""A :class:`.dict` mapping UFL element family names to their
FInAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FInAT equivalent."""


@cache
def as_fiat_cell(cell):
    """Convert a ufl cell to a FIAT cell.

    :arg cell: the :class:`ufl.Cell` to convert."""
    if not isinstance(cell, ufl.AbstractCell):
        raise ValueError("Expecting a UFL Cell")
    return ufc_cell(cell)


@singledispatch
def convert(element, **kwargs):
    """Handler for converting UFL elements to FInAT elements.

    :arg element: The UFL element to convert.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    raise ValueError("Unsupported element type %s" % type(element))


cg_interval_variants = {
    "fdm": finat.FDMLagrange,
    "fdm_ipdg": finat.FDMLagrange,
    "fdm_quadrature": finat.FDMQuadrature,
    "fdm_broken": finat.FDMBrokenH1,
    "fdm_hermite": finat.FDMHermite,
}


dg_interval_variants = {
    "fdm": finat.FDMDiscontinuousLagrange,
    "fdm_quadrature": finat.FDMDiscontinuousLagrange,
    "fdm_ipdg": lambda *args: finat.DiscontinuousElement(finat.FDMLagrange(*args)),
    "fdm_broken": finat.FDMBrokenL2,
}


# Base finite elements first
@convert.register(finat.ufl.FiniteElement)
def convert_finiteelement(element, **kwargs):
    cell = as_fiat_cell(element.cell)
    if element.family() in {"Quadrature", "Boundary Quadrature"}:
        degree = element.degree()
        scheme = element.quadrature_scheme() or "default"
        if degree is None or scheme is None:
            raise ValueError("Quadrature scheme and degree must be specified!")

        codim = 1 if element.family() == "Boundary Quadrature" else 0
        return finat.make_quadrature_element(cell, degree, scheme, codim), set()

    make_finat_element = supported_elements[element.family()]

    if element.cell.cellname() in {"quadrilateral", "hexahedron"}:
        # Reconstruct Real and Bernstein on tensor product cells
        if element.family() == "Real":
            make_finat_element = None
            element = finat.ufl.FiniteElement("DQ", element.cell, 0)
        elif element.family() == "Bernstein":
            make_finat_element = None

    if make_finat_element is None:
        if element.cell.cellname() == "quadrilateral":
            # Handle quadrilateral short names like RTCF and RTCE.
            element = element.reconstruct(cell=quadrilateral_tpc)
        elif element.cell.cellname() == "hexahedron":
            # Handle hexahedron short names like NCF and NCE.
            element = element.reconstruct(cell=hexahedron_tpc)
        else:
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        finat_elem, deps = _create_element(element, **kwargs)
        return finat.FlattenedDimensions(finat_elem), deps

    deps = set()
    finat_kwargs = {}
    kind = element.variant()
    if kind is None:
        kind = 'spectral'  # default variant

    if element.family() == "Lagrange":
        if kind in ['spectral', 'mimetic']:
            make_finat_element = finat.GaussLobattoLegendre
        elif element.cell.cellname() == "interval" and kind in cg_interval_variants:
            make_finat_element = cg_interval_variants[kind]
        elif any(map(kind.startswith, ['integral', 'demkowicz', 'fdm'])):
            make_finat_element = finat.IntegratedLegendre
            finat_kwargs["variant"] = kind
        elif kind in ['mgd', 'feec', 'qb', 'mse']:
            make_finat_element = finat.RuntimeTabulated
            finat_kwargs["variant"] = kind
            finat_kwargs["shift_axes"] = kwargs["shift_axes"]
            finat_kwargs["restriction"] = kwargs["restriction"]
            deps = {"shift_axes", "restriction"}
        else:
            # Let FIAT handle the general case
            make_finat_element = finat.Lagrange
            finat_kwargs["variant"] = kind

    elif element.family() in ["Discontinuous Lagrange", "Discontinuous Lagrange L2"]:
        if kind == 'spectral':
            make_finat_element = finat.GaussLegendre
        elif kind == 'mimetic':
            make_finat_element = finat.Histopolation
        elif element.cell.cellname() == "interval" and kind in dg_interval_variants:
            make_finat_element = dg_interval_variants[kind]
        elif any(map(kind.startswith, ['integral', 'demkowicz', 'fdm'])):
            make_finat_element = finat.Legendre
            finat_kwargs["variant"] = kind
        elif kind in ['mgd', 'feec', 'qb', 'mse']:
            make_finat_element = finat.RuntimeTabulated
            finat_kwargs["variant"] = kind
            finat_kwargs["shift_axes"] = kwargs["shift_axes"]
            finat_kwargs["restriction"] = kwargs["restriction"]
            finat_kwargs["continuous"] = False
            deps = {"shift_axes", "restriction"}
        else:
            # Let FIAT handle the general case
            make_finat_element = finat.DiscontinuousLagrange
            finat_kwargs["variant"] = kind

    elif element.family() in {"HDiv Trace", "Bubble", "FacetBubble"}:
        finat_kwargs["variant"] = kind

    elif element.variant() is not None:
        finat_kwargs["variant"] = element.variant()

    return make_finat_element(cell, element.degree(), **finat_kwargs), deps


# Element modifiers and compound element types
@convert.register(finat.ufl.BrokenElement)
def convert_brokenelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.DiscontinuousElement(finat_elem), deps


@convert.register(finat.ufl.EnrichedElement)
def convert_enrichedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element._elements])
    return finat.EnrichedElement(elements), set.union(*deps)


@convert.register(finat.ufl.NodalEnrichedElement)
def convert_nodalenrichedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element._elements])
    return finat.NodalEnrichedElement(elements), set.union(*deps)


@convert.register(finat.ufl.MixedElement)
def convert_mixedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element.sub_elements])
    return finat.MixedElement(elements), set.union(*deps)


@convert.register(finat.ufl.VectorElement)
@convert.register(finat.ufl.TensorElement)
def convert_tensorelement(element, **kwargs):
    inner_elem, deps = _create_element(element.sub_elements[0], **kwargs)
    shape = element.reference_value_shape
    shape = shape[:len(shape) - len(inner_elem.value_shape)]
    shape_innermost = kwargs["shape_innermost"]
    return (finat.TensorFiniteElement(inner_elem, shape, not shape_innermost),
            deps | {"shape_innermost"})


@convert.register(finat.ufl.TensorProductElement)
def convert_tensorproductelement(element, **kwargs):
    cell = element.cell
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TensorProductElement not on TensorProductCell?")
    shift_axes = kwargs["shift_axes"]
    dim_offset = 0
    elements = []
    deps = set()
    for elem in element.sub_elements:
        kwargs["shift_axes"] = shift_axes + dim_offset
        dim_offset += elem.cell.topological_dimension()
        finat_elem, ds = _create_element(elem, **kwargs)
        elements.append(finat_elem)
        deps.update(ds)
    return finat.TensorProductElement(elements), deps


@convert.register(finat.ufl.HDivElement)
def convert_hdivelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.HDivElement(finat_elem), deps


@convert.register(finat.ufl.HCurlElement)
def convert_hcurlelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.HCurlElement(finat_elem), deps


@convert.register(finat.ufl.WithMapping)
def convert_withmapping(element, **kwargs):
    return _create_element(element.wrapee, **kwargs)


@convert.register(finat.ufl.RestrictedElement)
def convert_restrictedelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.RestrictedElement(finat_elem, element.restriction_domain()), deps


hexahedron_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval, ufl.interval)
quadrilateral_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()


def create_element(ufl_element, shape_innermost=True, shift_axes=0, restriction=None):
    """Create a FInAT element (suitable for tabulating with) given a UFL element.

    :arg ufl_element: The UFL element to create a FInAT element from.
    :arg shape_innermost: Vector/tensor indices come after basis function indices
    :arg restriction: cell restriction in interior facet integrals
                      (only for runtime tabulated elements)
    """
    finat_element, deps = _create_element(ufl_element,
                                          shape_innermost=shape_innermost,
                                          shift_axes=shift_axes,
                                          restriction=restriction)
    return finat_element


def _create_element(ufl_element, **kwargs):
    """A caching wrapper around :py:func:`convert`.

    Takes a UFL element and an unspecified set of parameter options,
    and returns the converted element with the set of keyword names
    that were relevant for conversion.
    """
    # Look up conversion in cache
    try:
        cache = _cache[ufl_element]
    except KeyError:
        _cache[ufl_element] = {}
        cache = _cache[ufl_element]

    for key, finat_element in cache.items():
        # Cache hit if all relevant parameter values match.
        if all(kwargs[param] == value for param, value in key):
            return finat_element, set(param for param, value in key)

    # Convert if cache miss
    if ufl_element.cell is None:
        raise ValueError("Don't know how to build element when cell is not given")

    finat_element, deps = convert(ufl_element, **kwargs)

    # Store conversion in cache
    key = frozenset((param, kwargs[param]) for param in deps)
    cache[key] = finat_element

    # Forward result
    return finat_element, deps


def create_base_element(ufl_element, **kwargs):
    """Create a "scalar" base FInAT element given a UFL element.
    Takes a UFL element and an unspecified set of parameter options,
    and returns the converted element.
    """
    finat_element = create_element(ufl_element, **kwargs)
    if isinstance(finat_element, finat.TensorFiniteElement):
        finat_element = finat_element.base_element
    return finat_element
