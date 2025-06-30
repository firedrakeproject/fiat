# Copyright (C) 2016 Thomas H. Gibson
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
from collections import defaultdict
from FIAT.barycentric_interpolation import get_lagrange_points
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.hierarchical import Legendre
from FIAT.dual_set import DualSet
from FIAT.finite_element import FiniteElement
from FIAT.functional import IntegralMoment, PointEvaluation
from FIAT.polynomial_set import mis
from FIAT.quadrature import FacetQuadratureRule
from FIAT.reference_element import (ufc_simplex, POINT,
                                    LINE, QUADRILATERAL,
                                    TRIANGLE, TETRAHEDRON,
                                    TENSORPRODUCT)
from FIAT.tensor_product import TensorProductElement

# Numerical tolerance for facet-entity identifications
epsilon = 1e-10


class TraceError(Exception):
    """Exception caused by tabulating a trace element on the interior of a cell,
    or the gradient of a trace element."""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class HDivTrace(FiniteElement):
    """Class implementing the trace of hdiv elements. This class
    is a stand-alone element family that produces a DG-facet field.
    This element is what's produced after performing the trace
    operation on an existing H(Div) element.

    This element is also known as the discontinuous trace field that
    arises in several DG formulations.
    """

    def __init__(self, ref_el, degree, variant=None):
        """Constructor for the HDivTrace element.

        :arg ref_el: A reference element, which may be a tensor product
                     cell.
        :arg degree: The degree of approximation. If on a tensor product
                     cell, then provide a tuple of degrees if you want
                     varying degrees.
        :arg variant: The point distribution variant passed on to recursivenodes.
        """
        sd = ref_el.get_spatial_dimension()
        if sd in (0, 1):
            raise ValueError("Cannot take the trace of a %d-dim cell." % sd)

        # Store the degrees if on a tensor product cell
        if ref_el.get_shape() == TENSORPRODUCT:
            try:
                degree = tuple(degree)
            except TypeError:
                degree = (degree,) * len(ref_el.cells)

            assert len(ref_el.cells) == len(degree), (
                "Number of specified degrees must be equal to the number of cells."
            )
        else:
            if ref_el.get_shape() not in [TRIANGLE, TETRAHEDRON, QUADRILATERAL]:
                raise NotImplementedError(
                    "Trace element on a %s not implemented" % type(ref_el)
                )
            # Cannot have varying degrees for these reference cells
            if isinstance(degree, tuple):
                raise ValueError("Must have a tensor product cell if providing multiple degrees")

        # Initialize entity dofs
        facet_sd = sd - 1
        topology = ref_el.get_topology()
        entity_dofs = {dim: {entity: [] for entity in topology[dim]} for dim in topology}

        # Construct the DG element for the facets
        dg_elements = {}
        for dim in topology:
            fdim = sum(dim) if isinstance(dim, tuple) else dim
            if fdim == facet_sd:
                cell = ref_el.construct_subelement(dim)
                dg_elements[dim] = construct_dg_element(cell, degree, variant)

        # Compute the dof numbering for all facet entities
        # and extract nodes
        nodes = []
        for facet_dim in sorted(dg_elements):
            element = dg_elements[facet_dim]
            facet_nodes = element.dual_basis()
            for i in sorted(topology[facet_dim]):
                cur = len(nodes)
                nodes.extend(transform_nodes(facet_nodes, ref_el, facet_dim, i))
                entity_dofs[facet_dim][i] = list(range(cur, len(nodes)))

        # Setting up dual basis
        dual = DualSet(nodes, ref_el, entity_dofs)

        # Degree of the element
        deg = max(e.degree() for e in dg_elements.values())

        super().__init__(ref_el, dual, order=deg,
                         formdegree=facet_sd,
                         mapping="affine")

        # Set up facet elements
        self.dg_elements = dg_elements

        # Degree for quadrature rule
        self.polydegree = deg

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented for the trace element.")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented for the trace element.")

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to a given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     tabulated values are computed by geometrically
                     approximating which facet the points are on.

        .. note ::

           Performing illegal tabulations on this element will result in either
           a tabulation table of `numpy.nan` arrays (`entity=None` case), or
           insertions of the `TraceError` exception class. This is due to the
           fact that performing cell-wise tabulations, or asking for any order
           of derivative evaluations, are not mathematically well-defined.
        """
        sd = self.ref_el.get_spatial_dimension()
        facet_sd = sd - 1
        evalkey = (0,) * sd

        # Initializing dictionary with zeros
        phivals = {}
        for i in range(order + 1):
            alphas = mis(sd, i)
            for alpha in alphas:
                phivals[alpha] = np.zeros(shape=(self.space_dimension(), len(points)))
                if alpha != evalkey:
                    # If asking for gradient evaluations, insert TraceError in gradient slots
                    phivals[alpha] = TraceError("Gradients on trace elements are not well-defined.")

        # If entity is None, identify facet using numerical tolerance and
        # return the tabulated values
        if entity is None or entity == (sd, 0):
            # NOTE: Numerical approximation of the facet id is currently only
            # implemented for simplex reference cells.
            if self.ref_el.get_shape() not in [TRIANGLE, TETRAHEDRON]:
                raise NotImplementedError(
                    "Tabulating this element on a %s cell without providing "
                    "an entity is not currently supported." % type(self.ref_el)
                )

            # Attempt to identify which facet (if any) the given points are on
            vertices = self.ref_el.vertices
            coordinates = barycentric_coordinates(points, vertices)
            facet_to_pts, success = extract_facets(coordinates)

            # If not successful, return NaNs
            if not success:
                for key in phivals:
                    if entity is None:
                        phivals[key].fill(np.nan)
                    else:
                        msg = "The HDivTrace element can only be tabulated on facets."
                        phivals[key] = TraceError(msg)

            # Otherwise, extract non-zero values and insertion indices
            element = self.dg_elements[facet_sd]
            nf = element.space_dimension()
            for facet, ipts in facet_to_pts.items():
                # Map points to the reference facet
                new_points = map_to_reference_facet(points[ipts], vertices, facet)

                # Retrieve values by tabulating the DG element
                nonzerovals = element.tabulate(order, new_points)[(0,)*facet_sd]
                indices = slice(nf * facet, nf * (facet + 1))

                # Insert non-zero values in appropriate place
                phivals[evalkey][indices, ipts] = nonzerovals

        else:
            entity_dim, _ = entity

            # If the user is directly specifying cell-wise tabulation, return
            # TraceErrors in dict for appropriate handling in the form compiler
            if entity_dim not in self.dg_elements:
                for key in phivals:
                    msg = "The HDivTrace element can only be tabulated on facets."
                    phivals[key] = TraceError(msg)

            else:
                # Retrieve function evaluations (order = 0 case)
                offset = 0
                for facet_dim in sorted(self.dg_elements):
                    # Loop over the number of facets until we find a facet
                    # with matching dimension and id
                    element = self.dg_elements[facet_dim]
                    nf = element.space_dimension()
                    for i in sorted(self.ref_el.get_topology()[facet_dim]):
                        # Found it! Grab insertion indices
                        if (facet_dim, i) == entity:
                            nonzerovals = element.tabulate(0, points)[(0,)*facet_sd]
                            indices = slice(offset, offset + nf)
                        offset += nf

                # Insert non-zero values in appropriate place
                phivals[evalkey][indices] = nonzerovals

        return phivals

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return ()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented for the trace element.")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented for the trace element.")

    @staticmethod
    def is_nodal():
        return True


def construct_dg_element(ref_el, degree, variant):
    """Constructs a discontinuous galerkin element of a given degree
    on a particular reference cell.
    """
    if variant and variant.startswith("integral"):
        DG = Legendre
    else:
        DG = DiscontinuousLagrange
    if ref_el.get_shape() in [LINE, TRIANGLE]:
        dg_element = DG(ref_el, degree, variant)

    # Quadrilateral facets could be on a FiredrakeQuadrilateral.
    # In this case, we treat this as an interval x interval cell:
    elif ref_el.get_shape() == QUADRILATERAL:
        dg_line = DG(ufc_simplex(1), degree, variant)
        dg_element = TensorProductElement(dg_line, dg_line)

    # This handles the more general case for facets:
    elif ref_el.get_shape() == TENSORPRODUCT:
        assert len(degree) == len(ref_el.cells), (
            "Must provide the same number of degrees as the number "
            "of cells that make up the tensor product cell."
        )
        sub_elements = [construct_dg_element(c, d, variant)
                        for c, d in zip(ref_el.cells, degree)
                        if c.get_shape() != POINT]

        if len(sub_elements) > 1:
            dg_element = TensorProductElement(*sub_elements)
        else:
            dg_element, = sub_elements

    else:
        raise NotImplementedError(
            "Reference cells of type %s not currently supported" % type(ref_el)
        )

    return dg_element


def transform_nodes(ells, ref_el, facet_dim, facet_id):
    """Map functionals into a given facet."""
    try:
        facet_pts = get_lagrange_points(ells)
        transform = ref_el.get_entity_transform(facet_dim, facet_id)
        pts = transform(facet_pts)
        for pt in pts:
            yield PointEvaluation(ref_el, pt)
    except ValueError:
        Q_ref, = set(ell.Q for ell in ells)
        Q = FacetQuadratureRule(ref_el, facet_dim, facet_id, Q_ref)
        for ell in ells:
            yield IntegralMoment(ref_el, Q, ell.f_at_qpts)


# The following functions are credited to Marie E. Rognes:
def extract_facets(coordinates, tolerance=epsilon):
    """Determines whether a set of points (described in barycentric coordinates)
    are all on facet sub-entities, and return a dict mapping facets to
    point indices and whether the search has been successful.

    :arg coordinates: A set of points described in barycentric coordinates.
    :arg tolerance: A fixed tolerance for geometric identifications.
    """
    facet_to_pts = defaultdict(list)
    for ipt, c in enumerate(coordinates):
        on_facet = set(i for (i, l) in enumerate(c) if abs(l) < tolerance)
        try:
            f, = on_facet
        except ValueError:
            # Handle coordinates not on facets
            return ({}, False)
        facet_to_pts[f].append(ipt)

    # If all points are on facets, return indices and success
    return (facet_to_pts, True)


def barycentric_coordinates(points, vertices):
    """Computes the barycentric coordinates for a set of points relative to a
    simplex defined by a set of vertices.

    :arg points: A set of points.
    :arg vertices: A set of vertices that define the simplex.
    """

    # Form mapping matrix
    T = (np.asarray(vertices[:-1]) - vertices[-1]).T
    invT = np.linalg.inv(T)

    points = np.asarray(points)
    bary = np.einsum("ij,kj->ki", invT, (points - vertices[-1]))
    last = (1 - bary.sum(axis=1))
    return np.concatenate([bary, last[..., np.newaxis]], axis=1)


def map_from_reference_facet(point, vertices):
    """Evaluates the physical coordinate of a point using barycentric
    coordinates.

    :arg point: The reference points to be mapped to the facet.
    :arg vertices: The vertices defining the physical element.
    """

    # Compute the barycentric coordinates of the point relative to the reference facet
    reference_simplex = ufc_simplex(len(vertices) - 1)
    reference_vertices = reference_simplex.get_vertices()
    coords = barycentric_coordinates([point, ], reference_vertices)[0]

    # Evaluates the physical coordinate of the point using barycentric coordinates
    point = sum(vertices[j] * coords[j] for j in range(len(coords)))
    return tuple(point)


def map_to_reference_facet(points, vertices, facet):
    """Given a set of points and vertices describing a facet of a simplex in n-dimensional
    coordinates (where the points lie on the facet), map the points to the reference simplex
    of dimension (n-1).

    :arg points: A set of points in n-D.
    :arg vertices: A set of vertices describing a facet of a simplex in n-D.
    :arg facet: Integer representing the facet number.
    """

    # Compute the barycentric coordinates of the points with respect to the
    # full physical simplex
    all_coords = barycentric_coordinates(points, vertices)

    # Extract vertices of the reference facet
    reference_facet_simplex = ufc_simplex(len(vertices) - 2)
    reference_vertices = reference_facet_simplex.get_vertices()

    reference_points = []
    for (i, coords) in enumerate(all_coords):
        # Extract the correct subset of barycentric coordinates since we know
        # which facet we are on
        new_coords = [coords[j] for j in range(len(coords)) if j != facet]

        # Evaluate the reference coordinate of a point in barycentric coordinates
        reference_pt = sum(np.asarray(reference_vertices[j]) * new_coords[j]
                           for j in range(len(new_coords)))

        reference_points += [reference_pt]
    return reference_points
