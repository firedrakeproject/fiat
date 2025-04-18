"""Element.

This module provides an extensive list of predefined finite element
families. Users or, more likely, form compilers, may register new
elements by calling the function register_element.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file was originally part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Marie E. Rognes <meg@simula.no>, 2010
# Modified by Lizao Li <lzlarryli@gmail.com>, 2015, 2016
# Modified by Massimiliano Leoni, 2016
# Modified by Robert Kloefkorn, 2022
# Modified by Matthew Scroggs, 2023
# Modified by Pablo Brubeck, 2024

import warnings

from numpy import asarray

from ufl.cell import Cell, TensorProductCell
from ufl.sobolevspace import H1, H2, L2, HCurl, HDiv, HDivDiv, HCurlDiv, HEin, HInf
from ufl.utils.formatting import istr

# List of valid elements
ufl_elements = {}

# Aliases: aliases[name] (...) -> (standard_name, ...)
aliases = {}


# Function for registering new elements
def register_element(family, short_name, value_rank, sobolev_space, mapping,
                     degree_range, cellnames):
    """Register new finite element family."""
    if family in ufl_elements:
        raise ValueError(f"Finite element '{family}%s' has already been registered.")
    ufl_elements[family] = (family, short_name, value_rank, sobolev_space,
                            mapping, degree_range, cellnames)
    if short_name is not None:
        ufl_elements[short_name] = (family, short_name, value_rank, sobolev_space,
                                    mapping, degree_range, cellnames)


def register_alias(alias, to):
    """Doc."""
    aliases[alias] = to


def show_elements():
    """Shows all registered elements."""
    print("Showing all registered elements:")
    print("================================")
    shown = set()
    for k in sorted(ufl_elements.keys()):
        data = ufl_elements[k]
        if data in shown:
            continue
        shown.add(data)
        (family, short_name, value_rank, sobolev_space, mapping, degree_range, cellnames) = data
        print(f"Finite element family: '{family}', '{short_name}'")
        print(f"Sobolev space: {sobolev_space}%s")
        print(f"Mapping: {mapping}")
        print(f"Degree range: {degree_range}")
        print(f"Value rank: {value_rank}")
        print(f"Defined on cellnames: {cellnames}")
        print()


# FIXME: Consider cleanup of element names. Use notation from periodic
# table as the main, keep old names as compatibility aliases.

# NOTE: Any element with polynomial degree 0 will be considered L2,
# independent of the space passed to register_element.

# Cell groups
simplices = ("interval", "triangle", "tetrahedron", "pentatope")
cubes = ("interval", "quadrilateral", "hexahedron", "tesseract")
any_cell = (None, "vertex", *simplices, *cubes[1:], "prism", "pyramid")

# Elements in the periodic table # TODO: Register these as aliases of
# periodic table element description instead of the other way around
register_element("Lagrange", "CG", 0, H1, "identity", (1, None), any_cell)  # "P"
register_element("Brezzi-Douglas-Marini", "BDM", 1, HDiv, "contravariant Piola", (1, None), simplices[1:])  # "BDMF" (2d), "N2F" (3d)
register_element("Discontinuous Lagrange", "DG", 0, L2, "identity", (0, None), any_cell)  # "DP"
register_element("Discontinuous Taylor", "TDG", 0, L2, "identity", (0, None), simplices)
register_element("Nedelec 1st kind H(curl)", "N1curl", 1, HCurl, "covariant Piola", (1, None), simplices[1:])  # "RTE"  (2d), "N1E" (3d)
register_element("Nedelec 2nd kind H(curl)", "N2curl", 1, HCurl, "covariant Piola", (1, None), simplices[1:])  # "BDME" (2d), "N2E" (3d)
register_element("Raviart-Thomas", "RT", 1, HDiv, "contravariant Piola", (1, None), simplices[1:])   # "RTF"  (2d), "N1F" (3d)

# Elements not in the periodic table
# TODO: Implement generic Tear operator for elements instead of this:
register_element("Brezzi-Douglas-Fortin-Marini", "BDFM", 1, HDiv, "contravariant Piola", (1, None), simplices[1:])
register_element("Crouzeix-Raviart", "CR", 0, L2, "identity", (1, None), simplices[1:])
register_element("Discontinuous Raviart-Thomas", "DRT", 1, L2, "contravariant Piola", (1, None), simplices[1:])
register_element("Kong-Mulder-Veldhuizen", "KMV", 0, H1, "identity", (1, None), simplices[1:])

# Tensor elements
register_element("Regge", "Regge", 2, HEin, "double covariant Piola", (0, None), simplices)
register_element("Hellan-Herrmann-Johnson", "HHJ", 2, HDivDiv, "double contravariant Piola", (0, None), ("triangle", "tetrahedron"))
register_element("Gopalakrishnan-Lederer-Schoberl 1st kind", "GLS", 2, HCurlDiv, "covariant contravariant Piola", (1, None), simplices[1:])
register_element("Gopalakrishnan-Lederer-Schoberl 2nd kind", "GLS2", 2, HCurlDiv, "covariant contravariant Piola", (0, None), simplices[1:])

register_element("Nonconforming Arnold-Winther", "AWnc", 2, HDiv, "double contravariant Piola", (2, 2), ("triangle",))
register_element("Conforming Arnold-Winther", "AWc", 2, HDiv, "double contravariant Piola", (3, None), ("triangle",))
register_element("Hu-Zhang", "HZ", 2, HDiv, "double contravariant Piola", (3, None), ("triangle"))

# Zany elements
register_element("Bernardi-Raugel", "BR", 1, H1, "contravariant Piola", (1, None), simplices[1:])
register_element("Bernardi-Raugel Bubble", "BRB", 1, H1, "contravariant Piola", (None, None), simplices[1:])
register_element("Mardal-Tai-Winther", "MTW", 1, H1, "contravariant Piola", (3, 3), ("triangle",))
register_element("Hermite", "HER", 0, H1, "custom", (3, 3), simplices)
register_element("Argyris", "ARG", 0, H2, "custom", (5, None), ("triangle",))
register_element("Bell", "BELL", 0, H2, "custom", (5, 5), ("triangle",))
register_element("Morley", "MOR", 0, H2, "custom", (2, 2), ("triangle",))

# Macro elements
register_element("QuadraticPowellSabin6", "PS6", 0, H2, "custom", (2, 2), ("triangle",))
register_element("QuadraticPowellSabin12", "PS12", 0, H2, "custom", (2, 2), ("triangle",))
register_element("Hsieh-Clough-Tocher", "HCT", 0, H2, "custom", (3, None), ("triangle",))
register_element("Reduced-Hsieh-Clough-Tocher", "HCT-red", 0, H2, "custom", (3, 3), ("triangle",))
register_element("Johnson-Mercier", "JM", 2, HDiv, "double contravariant Piola", (1, 1), simplices[1:])

register_element("Arnold-Qin", "AQ", 1, H1, "identity", (2, 2), ("triangle",))
register_element("Reduced-Arnold-Qin", "AQ-red", 1, H1, "contravariant Piola", (2, 2), ("triangle",))
register_element("Christiansen-Hu", "CH", 1, H1, "contravariant Piola", (1, 1), simplices[1:])
register_element("Alfeld-Sorokina", "AS", 1, H1, "contravariant Piola", (2, 2), simplices[1:])

register_element("Guzman-Neilan 1st kind H1", "GN", 1, H1, "contravariant Piola", (1, None), simplices[1:])
register_element("Guzman-Neilan 2nd kind H1", "GN2", 1, H1, "contravariant Piola", (1, None), simplices[1:])
register_element("Guzman-Neilan H1(div)", "GNH1div", 1, H1, "contravariant Piola", (2, None), simplices[1:])
register_element("Guzman-Neilan Bubble", "GNB", 1, H1, "contravariant Piola", (None, None), simplices[1:])

# Special elements
register_element("Boundary Quadrature", "BQ", 0, L2, "identity", (0, None), any_cell)
register_element("Bubble", "B", 0, H1, "identity", (2, None), simplices)
register_element("FacetBubble", "FB", 0, H1, "identity", (2, None), simplices)
register_element("Quadrature", "Quadrature", 0, L2, "identity", (0, None), any_cell)
register_element("Real", "R", 0, HInf, "identity", (0, 0), any_cell + ("TensorProductCell",))
register_element("Undefined", "U", 0, L2, "identity", (0, None), any_cell)
register_element("Radau", "Rad", 0, L2, "identity", (0, None), ("interval",))
register_element("HDiv Trace", "HDivT", 0, L2, "identity", (0, None), any_cell)

# Spectral elements.
register_element("Gauss-Legendre", "GL", 0, L2, "identity", (0, None), ("interval",))
register_element("Gauss-Lobatto-Legendre", "GLL", 0, H1, "identity", (1, None), ("interval",))
register_alias("Lobatto",
               lambda family, dim, order, degree: ("Gauss-Lobatto-Legendre", order))
register_alias("Lob",
               lambda family, dim, order, degree: ("Gauss-Lobatto-Legendre", order))

register_element("Bernstein", None, 0, H1, "identity", (1, None), simplices)


# Let Nedelec H(div) elements be aliases to BDMs/RTs
register_alias("Nedelec 1st kind H(div)",
               lambda family, dim, order, degree: ("Raviart-Thomas", order))
register_alias("N1div",
               lambda family, dim, order, degree: ("Raviart-Thomas", order))

register_alias("Nedelec 2nd kind H(div)",
               lambda family, dim, order, degree: ("Brezzi-Douglas-Marini",
                                                   order))
register_alias("N2div",
               lambda family, dim, order, degree: ("Brezzi-Douglas-Marini",
                                                   order))

# Let Discontinuous Lagrange Trace element be alias to HDiv Trace
register_alias("Discontinuous Lagrange Trace",
               lambda family, dim, order, degree: ("HDiv Trace", order))
register_alias("DGT",
               lambda family, dim, order, degree: ("HDiv Trace", order))

# New elements introduced for the periodic table 2014
register_element("Q", None, 0, H1, "identity", (1, None), cubes)
register_element("DQ", None, 0, L2, "identity", (0, None), cubes)
register_element("RTCE", None, 1, HCurl, "covariant Piola", (1, None), ("quadrilateral",))
register_element("RTCF", None, 1, HDiv, "contravariant Piola", (1, None), ("quadrilateral",))
register_element("NCE", None, 1, HCurl, "covariant Piola", (1, None), ("hexahedron",))
register_element("NCF", None, 1, HDiv, "contravariant Piola", (1, None), ("hexahedron",))

register_element("S", None, 0, H1, "identity", (1, None), cubes)
register_element("DPC", None, 0, L2, "identity", (0, None), cubes)
register_element("Brezzi-Douglas-Marini Cube Edge", "BDMCE", 1, HCurl, "covariant Piola", (1, None), ("quadrilateral",))
register_element("Brezzi-Douglas-Marini Cube Face", "BDMCF", 1, HDiv, "contravariant Piola", (1, None), ("quadrilateral",))
register_element("SminusE", "SminusE", 1, HCurl, "covariant Piola", (1, None), cubes[1:3])
register_element("SminusF", "SminusF", 1, HDiv, "contravariant Piola", (1, None), cubes[1:2])
register_element("SminusDiv", "SminusDiv", 1, HDiv, "contravariant Piola", (1, None), cubes[1:3])
register_element("SminusCurl", "SminusCurl", 1, HCurl, "covariant Piola", (1, None), cubes[1:3])
register_element("AAE", None, 1, HCurl, "covariant Piola", (1, None), ("hexahedron",))
register_element("AAF", None, 1, HDiv, "contravariant Piola", (1, None), ("hexahedron",))

# New aliases introduced for the periodic table 2014
register_alias("P", lambda family, dim, order, degree: ("Lagrange", order))
register_alias("DP", lambda family, dim, order,
               degree: ("Discontinuous Lagrange", order))
register_alias("RTE", lambda family, dim, order,
               degree: ("Nedelec 1st kind H(curl)", order))
register_alias("RTF", lambda family, dim, order,
               degree: ("Raviart-Thomas", order))
register_alias("N1E", lambda family, dim, order,
               degree: ("Nedelec 1st kind H(curl)", order))
register_alias("N1F", lambda family, dim, order, degree: ("Raviart-Thomas",
                                                          order))

register_alias("BDME", lambda family, dim, order,
               degree: ("Nedelec 2nd kind H(curl)", order))
register_alias("BDMF", lambda family, dim, order,
               degree: ("Brezzi-Douglas-Marini", order))
register_alias("N2E", lambda family, dim, order,
               degree: ("Nedelec 2nd kind H(curl)", order))
register_alias("N2F", lambda family, dim, order,
               degree: ("Brezzi-Douglas-Marini", order))

# discontinuous elements using l2 pullbacks
register_element("DPC L2", None, 0, L2, "L2 Piola", (1, None), cubes)
register_element("DQ L2", None, 0, L2, "L2 Piola", (0, None), cubes)
register_element("Gauss-Legendre L2", "GL L2", 0, L2, "L2 Piola", (0, None),
                 ("interval",))
register_element("Discontinuous Lagrange L2", "DG L2", 0, L2, "L2 Piola", (0, None),
                 any_cell)  # "DP"

register_alias("DP L2", lambda family, dim, order,
               degree: ("Discontinuous Lagrange L2", order))

register_alias("P- Lambda L2", lambda family, dim, order,
               degree: feec_element_l2(family, dim, order, degree))
register_alias("P Lambda L2", lambda family, dim, order,
               degree: feec_element_l2(family, dim, order, degree))
register_alias("Q- Lambda L2", lambda family, dim, order,
               degree: feec_element_l2(family, dim, order, degree))
register_alias("S Lambda L2", lambda family, dim, order,
               degree: feec_element_l2(family, dim, order, degree))

register_alias("P- L2", lambda family, dim, order,
               degree: feec_element_l2(family, dim, order, degree))
register_alias("Q- L2", lambda family, dim, order,
               degree: feec_element_l2(family, dim, order, degree))

# mimetic spectral elements - primal and dual complexs
register_element("Extended-Gauss-Legendre", "EGL", 0, H1, "identity", (2, None),
                 ("interval",))
register_element("Extended-Gauss-Legendre Edge", "EGL-Edge", 0, L2, "identity", (1, None),
                 ("interval",))
register_element("Extended-Gauss-Legendre Edge L2", "EGL-Edge L2", 0, L2, "L2 Piola", (1, None),
                 ("interval",))
register_element("Gauss-Lobatto-Legendre Edge", "GLL-Edge", 0, L2, "identity", (0, None),
                 ("interval",))
register_element("Gauss-Lobatto-Legendre Edge L2", "GLL-Edge L2", 0, L2, "L2 Piola", (0, None),
                 ("interval",))

# directly-defined serendipity elements ala Arbogast
# currently the theory is only really worked out for quads.
register_element("Direct Serendipity", "Sdirect", 0, H1, "physical", (1, None),
                 ("quadrilateral",))
register_element("Direct Serendipity Full H(div)", "Sdirect H(div)", 1, HDiv, "physical", (1, None),
                 ("quadrilateral",))
register_element("Direct Serendipity Reduced H(div)", "Sdirect H(div) red", 1, HDiv, "physical", (1, None),
                 ("quadrilateral",))


# NOTE- the edge elements for primal mimetic spectral elements are accessed by using
# variant='mse' in the appropriate places

def feec_element(family, n, r, k):
    """Finite element exterior calculus notation.

    n = topological dimension of domain
    r = polynomial order
    k = form_degree
    """
    # Note: We always map to edge elements in 2D, don't know how to
    # differentiate otherwise?

    # Mapping from (feec name, domain dimension, form degree) to
    # (family name, polynomial order)
    _feec_elements = {
        "P- Lambda": (
            (("P", r), ("DP", r - 1)),
            (("P", r), ("RTE", r), ("DP", r - 1)),
            (("P", r), ("N1E", r), ("N1F", r), ("DP", r - 1)),
        ),
        "P Lambda": (
            (("P", r), ("DP", r)),
            (("P", r), ("BDME", r), ("DP", r)),
            (("P", r), ("N2E", r), ("N2F", r), ("DP", r)),
        ),
        "Q- Lambda": (
            (("Q", r), ("DQ", r - 1)),
            (("Q", r), ("RTCE", r), ("DQ", r - 1)),
            (("Q", r), ("NCE", r), ("NCF", r), ("DQ", r - 1)),
        ),
        "S Lambda": (
            (("S", r), ("DPC", r)),
            (("S", r), ("BDMCE", r), ("DPC", r)),
            (("S", r), ("AAE", r), ("AAF", r), ("DPC", r)),
        ),
    }

    # New notation, old verbose notation (including "Lambda") might be
    # removed
    _feec_elements["P-"] = _feec_elements["P- Lambda"]
    _feec_elements["P"] = _feec_elements["P Lambda"]
    _feec_elements["Q-"] = _feec_elements["Q- Lambda"]
    _feec_elements["S"] = _feec_elements["S Lambda"]

    family, r = _feec_elements[family][n - 1][k]

    return family, r


def feec_element_l2(family, n, r, k):
    """Finite element exterior calculus notation.

    n = topological dimension of domain
    r = polynomial order
    k = form_degree
    """
    # Note: We always map to edge elements in 2D, don't know how to
    # differentiate otherwise?

    # Mapping from (feec name, domain dimension, form degree) to
    # (family name, polynomial order)
    _feec_elements = {
        "P- Lambda L2": (
            (("P", r), ("DP L2", r - 1)),
            (("P", r), ("RTE", r), ("DP L2", r - 1)),
            (("P", r), ("N1E", r), ("N1F", r), ("DP L2", r - 1)),
        ),
        "P Lambda L2": (
            (("P", r), ("DP L2", r)),
            (("P", r), ("BDME", r), ("DP L2", r)),
            (("P", r), ("N2E", r), ("N2F", r), ("DP L2", r)),
        ),
        "Q- Lambda L2": (
            (("Q", r), ("DQ L2", r - 1)),
            (("Q", r), ("RTCE", r), ("DQ L2", r - 1)),
            (("Q", r), ("NCE", r), ("NCF", r), ("DQ L2", r - 1)),
        ),
        "S Lambda L2": (
            (("S", r), ("DPC L2", r)),
            (("S", r), ("BDMCE", r), ("DPC L2", r)),
            (("S", r), ("AAE", r), ("AAF", r), ("DPC L2", r)),
        ),
    }

    # New notation, old verbose notation (including "Lambda") might be
    # removed
    _feec_elements["P- L2"] = _feec_elements["P- Lambda L2"]
    _feec_elements["P L2"] = _feec_elements["P Lambda L2"]
    _feec_elements["Q- L2"] = _feec_elements["Q- Lambda L2"]
    _feec_elements["S L2"] = _feec_elements["S Lambda L2"]

    family, r = _feec_elements[family][n - 1][k]

    return family, r


# General FEEC notation, old verbose (can be removed)
register_alias("P- Lambda", lambda family, dim, order,
               degree: feec_element(family, dim, order, degree))
register_alias("P Lambda", lambda family, dim, order,
               degree: feec_element(family, dim, order, degree))
register_alias("Q- Lambda", lambda family, dim, order,
               degree: feec_element(family, dim, order, degree))
register_alias("S Lambda", lambda family, dim, order,
               degree: feec_element(family, dim, order, degree))

# General FEEC notation, new compact notation
register_alias("P-", lambda family, dim, order,
               degree: feec_element(family, dim, order, degree))
register_alias("Q-", lambda family, dim, order,
               degree: feec_element(family, dim, order, degree))


def canonical_element_description(family, cell, order, form_degree):
    """Given basic element information, return corresponding element information on canonical form.

    Input: family, cell, (polynomial) order, form_degree
    Output: family (canonical), short_name (for printing), order, value shape,
    reference value shape, sobolev_space.

    This is used by the FiniteElement constructor to ved input
    data against the element list and aliases defined in ufl.
    """
    # Get domain dimensions
    if cell is not None:
        tdim = cell.topological_dimension()
        if isinstance(cell, Cell):
            cellname = cell.cellname()
        else:
            cellname = None
    else:
        tdim = None
        cellname = None

    # Catch general FEEC notation "P" and "S"
    if form_degree is not None and family in ("P", "S"):
        family, order = feec_element(family, tdim, order, form_degree)

    if form_degree is not None and family in ("P L2", "S L2"):
        family, order = feec_element_l2(family, tdim, order, form_degree)

    # Check whether this family is an alias for something else
    while family in aliases:
        if tdim is None:
            raise ValueError("Need dimension to handle element aliases.")
        (family, order) = aliases[family](family, tdim, order, form_degree)

    # Check that the element family exists
    if family not in ufl_elements:
        raise ValueError(f"Unknown finite element '{family}'.")

    # Check that element data is valid (and also get common family
    # name)
    (family, short_name, value_rank, sobolev_space, mapping, krange, cellnames) = ufl_elements[family]

    # Accept CG/DG on all kind of cells, but use Q/DQ on "product" cells
    if cellname in set(cubes) - set(simplices) or isinstance(cell, TensorProductCell):
        if family == "Lagrange":
            family = "Q"
        elif family == "Discontinuous Lagrange":
            if order >= 1:
                warnings.warn("Discontinuous Lagrange element requested on %s, creating DQ element." % cell.cellname())
            family = "DQ"
        elif family == "Discontinuous Lagrange L2":
            if order >= 1:
                warnings.warn(f"Discontinuous Lagrange L2 element requested on {cell.cellname()}, "
                              "creating DQ L2 element.")
            family = "DQ L2"

    # Validate cellname if a valid cell is specified
    if not (cellname is None or cellname in cellnames):
        raise ValueError(f"Cellname '{cellname}' invalid for '{family}' finite element.")

    # Validate order if specified
    if order is not None:
        if krange is None:
            raise ValueError(f"Order {order} invalid for '{family}' finite element, should be None.")
        kmin, kmax = krange
        if not (kmin is None or (asarray(order) >= kmin).all()):
            raise ValueError(f"Order {order} invalid for '{family}' finite element.")
        if not (kmax is None or (asarray(order) <= kmax).all()):
            raise ValueError(f"Order {istr(order)} invalid for '{family}' finite element.")

    if value_rank == 2:
        # Tensor valued fundamental elements in HEin have this shape
        if tdim is None:
            raise ValueError("Cannot infer shape of element without topological and geometric dimensions.")
        reference_value_shape = (tdim, tdim)
    elif value_rank == 1:
        # Vector valued fundamental elements in HDiv and HCurl have a shape
        if tdim is None:
            raise ValueError("Cannot infer shape of element without topological and geometric dimensions.")
        reference_value_shape = (tdim,)
    elif value_rank == 0:
        # All other elements are scalar values
        reference_value_shape = ()
    else:
        raise ValueError(f"Invalid value rank {value_rank}.")

    embedded_degree = order
    if any(bubble in family for bubble in ("Guzman-Neilan", "Bernardi-Raugel")):
        embedded_degree = tdim
    return family, short_name, order, reference_value_shape, sobolev_space, mapping, embedded_degree
