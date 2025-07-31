# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014
# Modified by Thomas H. Gibson (t.gibson15@imperial.ac.uk), 2016

import numpy
import scipy
import warnings

from FIAT.dual_set import DualSet
from FIAT.polynomial_set import PolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.macro import MacroPolynomialSet


class FiniteElement(object):
    """Class implementing a basic abstraction template for general
    finite element families. Finite elements which inherit from
    this class are non-nodal unless they are CiarletElement subclasses.
    """

    def __init__(self, ref_el, dual, order, formdegree=None, mapping="affine", ref_complex=None):
        # Relevant attributes that do not necessarily depend on a PolynomialSet object:
        # The order (degree) of the polynomial basis
        self.order = order
        self.formdegree = formdegree

        # The reference element and the appropriate dual
        self.ref_el = ref_el
        self.dual = dual

        # when the finite element is a macro-element, the basis will be defined over
        # a partition of the reference element into some complex.
        self.ref_complex = ref_complex or ref_el

        # The appropriate mapping for the finite element space
        self._mapping = mapping

    def get_reference_element(self):
        """Return the reference element for the finite element."""
        return self.ref_el

    def get_reference_complex(self):
        """Return the reference element complex, which is either the reference element or
        its subdivision in the case of a macro element."""
        return self.ref_complex

    def get_dual_set(self):
        """Return the dual for the finite element."""
        return self.dual

    def get_order(self):
        """Return the order of the element (may be different from the degree)."""
        return self.order

    def dual_basis(self):
        """Return the dual basis (list of functionals) for the finite
        element."""
        return self.dual.get_nodes()

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.dual.get_entity_ids()

    def entity_closure_dofs(self):
        """Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element."""
        return self.dual.get_entity_closure_ids()

    def entity_permutations(self):
        return self.dual.get_entity_permutations()

    entity_permutations.__doc__ = DualSet.get_entity_permutations.__doc__

    def get_formdegree(self):
        """Return the degree of the associated form (FEEC)"""
        return self.formdegree

    def mapping(self):
        """Return a list of appropriate mappings from the reference
        element to a physical element for each basis function of the
        finite element."""
        return [self._mapping] * self.space_dimension()

    def num_sub_elements(self):
        """Return the number of sub-elements."""
        return 1

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return len(self.dual_basis())

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        raise NotImplementedError("Must be specified in the element subclass of FiniteElement.")

    @staticmethod
    def is_nodal():
        """True if primal and dual bases are orthogonal. If false,
        dual basis is not implemented or is undefined.

        Subclasses may not necessarily be nodal, unless it is a CiarletElement.
        """
        return False

    def is_macroelement(self):
        return self.ref_el is not self.ref_complex


class CiarletElement(FiniteElement):
    """Class implementing Ciarlet's abstraction of a finite element
    being a domain, function space, and set of nodes.

    Elements derived from this class are nodal finite elements, with a nodal
    basis generated from polynomials encoded in a `PolynomialSet`.
    """

    def __init__(self, poly_set, dual, order, formdegree=None, mapping="affine", ref_complex=None):
        ref_el = dual.get_reference_element()
        ref_complex = ref_complex or poly_set.get_reference_element()
        super().__init__(ref_el, dual, order, formdegree, mapping, ref_complex)

        # Tile the poly_set
        if ref_complex.is_macrocell() and len(poly_set) > len(dual):
            base_element = type(self)(ref_el, order)
            poly_set = MacroPolynomialSet(ref_complex, base_element)

        if len(poly_set) != len(dual):
            raise ValueError(f"Dimension of function space is {len(poly_set)}, but got {len(dual)} nodes.")

        # build generalized Vandermonde matrix
        old_coeffs = poly_set.get_coeffs()
        dualmat = dual.to_riesz(poly_set)

        shp = dualmat.shape
        A = dualmat.reshape((shp[0], -1))
        B = old_coeffs.reshape((shp[0], -1))
        V = numpy.dot(A, numpy.transpose(B))
        self.V = V

        # new_coeffs_flat = numpy.linalg.solve(V.T, B)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                new_coeffs_flat = scipy.linalg.solve(V, B, transposed=True)
            except (scipy.linalg.LinAlgWarning, scipy.linalg.LinAlgError):
                raise numpy.linalg.LinAlgError("Singular Vandermonde matrix")

        new_shp = new_coeffs_flat.shape[:1] + shp[1:]
        new_coeffs = new_coeffs_flat.reshape(new_shp)

        self.poly_set = PolynomialSet(poly_set.get_reference_element(),
                                      poly_set.get_degree(),
                                      poly_set.get_embedded_degree(),
                                      poly_set.get_expansion_set(),
                                      new_coeffs)

    def degree(self):
        "Return the degree of the (embedding) polynomial space."
        return self.poly_set.get_embedded_degree()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        return self.poly_set

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        return self.poly_set.get_coeffs()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        if entity is None:
            entity = (self.ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        return self.poly_set.tabulate(transform(points), order)

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self.poly_set.get_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        return self.get_nodal_basis().get_dmats()

    def get_num_members(self, arg):
        "Return number of members of the expansion set."
        return self.get_nodal_basis().get_expansion_set().get_num_members(arg)

    @staticmethod
    def is_nodal():
        """True if primal and dual bases are orthogonal. If false,
        dual basis is not implemented or is undefined.

        All implementations/subclasses are nodal including this one.
        """
        return True


def entity_support_dofs(elem, entity_dim):
    """Return the map of entity id to the degrees of freedom for which the
    corresponding basis functions take non-zero values

    :arg elem: FIAT finite element
    :arg entity_dim: Dimension of the cell subentity.
    """
    if not hasattr(elem, "_entity_support_dofs"):
        elem._entity_support_dofs = {}
    cache = elem._entity_support_dofs
    try:
        return cache[entity_dim]
    except KeyError:
        pass

    ref_el = elem.get_reference_element()
    dim = ref_el.get_spatial_dimension()

    entity_cell = ref_el.construct_subelement(entity_dim)
    quad = create_quadrature(entity_cell, max(2*elem.degree(), 1))
    weights = quad.get_weights()

    eps = 1.e-8  # Is this a safe value?

    result = {}
    for f in elem.entity_dofs()[entity_dim].keys():
        entity_transform = ref_el.get_entity_transform(entity_dim, f)
        points = entity_transform(quad.get_points())

        # Integrate the square of the basis functions on the facet.
        vals = elem.tabulate(0, points)[(0,) * dim]
        # Ints contains the square of the basis functions
        # integrated over the facet.
        if elem.value_shape():
            # Vector-valued functions.
            ints = numpy.dot(numpy.einsum("...ij,...ij->...j", vals, vals), weights)
        else:
            ints = numpy.dot(vals**2, weights)

        result[f] = [dof for dof, i in enumerate(ints) if i > eps]

    cache[entity_dim] = result
    return result
