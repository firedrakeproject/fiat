# Copyright (C) 2019 Cyrus Cheng (Imperial College London)
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2019
# Modified by Thomas Bendall (thomas.bendall@metoffice.gov.uk) 2021

from sympy import symbols, legendre, Array, diff, binomial, lambdify
import numpy as np
from FIAT.dual_set import make_entity_closure_ids
from FIAT.finite_element import FiniteElement
from FIAT.polynomial_set import mis
from FIAT.reference_element import compute_unflattening_map, flatten_reference_cube

x, y = symbols('x y')
variables = (x, y)
leg = legendre


def triangular_number(n):
    return int((n+1)*n/2)


class BrezziDouglasMariniCube(FiniteElement):
    """
    The Brezzi-Douglas-Marini element on quadrilateral cells.

    :arg ref_el: The reference element.
    :arg k: The degree.
    :arg mapping: A string giving the Piola mapping.
                  Either 'contravariant Piola' or 'covariant Piola'.
    """

    def __init__(self, ref_el, degree, mapping):

        # Check that ref_el and degree are appropriate
        if degree < 1:
            raise Exception("BDMc_k elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            raise Exception("BDMc_k elements only valid for dimension 2")

        # Collect the IDs of the reference element entities
        flat_topology = flat_el.get_topology()

        entity_ids = {}
        counter = 0

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        for j in sorted(flat_topology[1]):
            entity_ids[1][j] = list(range(counter, counter + degree + 1))
            counter += degree + 1

        entity_ids[2][0] = list(range(counter, counter + 2*triangular_number(degree - 1)))
        counter += 2*triangular_number(degree - 1)

        entity_closure_ids = make_entity_closure_ids(flat_el, entity_ids)

        # Set up FiniteElement
        super().__init__(ref_el=ref_el, dual=None,
                         order=degree, formdegree=1,
                         mapping=mapping)

        # Store unflattened entity ID dictionaries
        topology = ref_el.get_topology()
        unflattening_map = compute_unflattening_map(topology)
        unflattened_entity_ids = {}
        unflattened_entity_closure_ids = {}

        for dim, entities in sorted(topology.items()):
            unflattened_entity_ids[dim] = {}
            unflattened_entity_closure_ids[dim] = {}
        for dim, entities in sorted(flat_topology.items()):
            for entity in entities:
                unflat_dim, unflat_entity = unflattening_map[(dim, entity)]
                unflattened_entity_ids[unflat_dim][unflat_entity] = entity_ids[dim][entity]
                unflattened_entity_closure_ids[unflat_dim][unflat_entity] = entity_closure_ids[dim][entity]
        self.entity_ids = unflattened_entity_ids
        self.entity_closure_ids = unflattened_entity_closure_ids
        self._degree = degree
        self.flat_el = flat_el

    def degree(self):
        """Return the degree of the polynomial space."""
        return self._degree

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for BDMCE/F elements")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for BDMCE/F elements")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for BDMCE/F elements")

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
        """
        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = np.asarray(list(map(transform, points)))
        npoints, pointdim = points.shape

        # Turn analytic basis functions into python functions via lambdify
        basis_callable = {(0, 0): numpy_lambdify(variables, self.basis[(0, 0)],
                                                 modules="numpy")}

        # Dictionary of values of functions
        phivals = {}
        for o in range(order+1):
            alphas = mis(2, o)
            # Collect basis and derivatives of basis
            for alpha in alphas:
                try:
                    callable = basis_callable[alpha]
                except KeyError:
                    diff_basis = diff(self.basis[(0, 0)], *zip(variables, alpha))
                    callable = numpy_lambdify(variables, diff_basis, modules="numpy")
                    self.basis[alpha] = diff_basis

                # tabulate by passing points through all lambdified functions
                # resulting array has shape (len(self.basis), spatial_dim, npoints)
                T = np.array([[[func_component(point) for point in points]
                               for func_component in func] for func in callable])

                phivals[alpha] = T

        return phivals

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.entity_ids

    def entity_closure_dofs(self):
        """Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element."""
        return self.entity_closure_ids

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return np.shape(self.basis[(0, 0)][0])

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return int(len(self.basis[(0, 0)])/2)


def bdmce_edge_basis(deg, dx, dy, x_mid, y_mid):
    """Returns the basis functions associated with DoFs on the
    edges of elements for the HCurl Brezz-Douglas-Marini element on
    quadrilateral cells.

    These were introduced by Brezzi, Douglas, Marini (1985)
    "Two families of mixed finite elements for Second Order Elliptic Problems"

    Following, e.g. Brezzi, Douglas, Fortin, Marini (1987)
    "Efficient rectangular mixed finite elements in two and three space variables"
    For rectangle K and degree j:
    BDM_j(K) = [P_j(K)^2 + Span(curl(xy^{j+1}, x^{j+1}y))] x P_{j-1}(K)

    The resulting basis functions all have a curl whose polynomials are
    of degree (j - 1).

    :arg deg: The element degree.
    :arg dx: A tuple of sympy expressions, expanding the interval in the
             x direction. Probably (1-x, x).
    :arg dy: A tuple of sympy expressions, expanding the interval in the
             y direction. Probably (1-y, y).
    :arg x_mid: A sympy expression, probably 2*x-1.
    :arg y_mid: A sympy expression, probably 2*y-1.
    """

    # For some functions, we need to multiply by a coefficient to ensure that
    # the result curl is of the correct degree. The coefficient of the
    # highest-order term of leg(deg, 2x-1), is binomial(2*deg, deg)
    coeff = binomial(2*deg, deg) / ((deg+1)*binomial(2*deg-2, deg-1))

    basis = tuple([(0, -leg(j, y_mid)*dx[0]) for j in range(deg)] +
                  [(coeff*-leg(deg-1, y_mid)*dy[0]*dy[1], -leg(deg, y_mid)*dx[0])] +
                  [(0, -leg(j, y_mid)*dx[1]) for j in range(deg)] +
                  [(coeff*leg(deg-1, y_mid)*dy[0]*dy[1], -leg(deg, y_mid)*dx[1])] +
                  [(-leg(j, x_mid)*dy[0], 0) for j in range(deg)] +
                  [(-leg(deg, x_mid)*dy[0], coeff*-leg(deg-1, x_mid)*dx[0]*dx[1])] +
                  [(-leg(j, x_mid)*dy[1], 0) for j in range(deg)] +
                  [(-leg(deg, x_mid)*dy[1], coeff*leg(deg-1, x_mid)*dx[0]*dx[1])])

    return basis


def bdmce_face_basis(deg, dx, dy, x_mid, y_mid):
    """Returns the basis functions associated with DoFs on the
    faces of elements for the HCurl Brezz-Douglas-Marini element on
    quadrilateral cells.

    These were introduced by Brezzi, Douglas, Marini (1985)
    "Two families of mixed finite elements for Second Order Elliptic Problems"

    Following, e.g. Brezzi, Douglas, Fortin, Marini (1987)
    "Efficient rectangular mixed finite elements in two and three space variables"
    For rectangle K and degree j:
    BDM_j(K) = [P_j(K)^2 + Span(curl(xy^{j+1}, x^{j+1}y))] x P_{j-1}(K)

    The resulting basis functions all have a curl whose polynomials are
    of degree (j - 1).

    :arg deg: The element degree.
    :arg dx: A tuple of sympy expressions, expanding the interval in the
             x direction. Probably (1-x, x).
    :arg dy: A tuple of sympy expressions, expanding the interval in the
             y direction. Probably (1-y, y).
    :arg x_mid: A sympy expression, probably 2*x-1.
    :arg y_mid: A sympy expression, probably 2*y-1.
    """

    basis = []
    for k in range(2, deg+1):
        for j in range(k-1):
            basis += [(0, leg(j, x_mid)*leg(k-2-j, y_mid)*dx[0]*dx[1])]
            basis += [(leg(k-2-j, x_mid)*leg(j, y_mid)*dy[0]*dy[1], 0)]

    return tuple(basis)


class BrezziDouglasMariniCubeEdge(BrezziDouglasMariniCube):
    """
    The Brezzi-Douglas-Marini HCurl element on quadrilateral cells.

    :arg ref_el: The reference element.
    :arg k: The degree.
    """

    def __init__(self, ref_el, degree):

        bdmce_list = construct_bdmce_basis(ref_el, degree)
        self.basis = {(0, 0): Array(bdmce_list)}

        super().__init__(ref_el=ref_el, degree=degree, mapping="covariant piola")


class BrezziDouglasMariniCubeFace(BrezziDouglasMariniCube):
    """
    The Brezzi-Douglas-Marini HDiv element on quadrilateral cells.

    :arg ref_el: The reference element.
    :arg k: The degree.
    """

    def __init__(self, ref_el, degree):

        bdmce_list = construct_bdmce_basis(ref_el, degree)

        # BDMCF functions are rotations of BDMCE functions
        bdmcf_list = [[-a[1], a[0]] for a in bdmce_list]
        self.basis = {(0, 0): Array(bdmcf_list)}

        super().__init__(ref_el=ref_el, degree=degree, mapping="contravariant piola")


def construct_bdmce_basis(ref_el, degree):
    """
    Return the basis functions for a particular BDMCE space as a list.

    :arg ref_el: The reference element.
    :arg k: The degree.
    """

    # Extract the vertices from the reference element
    flat_el = flatten_reference_cube(ref_el)
    verts = flat_el.get_vertices()

    # dx on reference quad is (1-x, x)
    dx = ((verts[-1][0] - x) / (verts[-1][0] - verts[0][0]),
          (x - verts[0][0]) / (verts[-1][0] - verts[0][0]))
    # dy on reference quad is (1-y, y)
    dy = ((verts[-1][1] - y) / (verts[-1][1] - verts[0][1]),
          (y - verts[0][1]) / (verts[-1][1] - verts[0][1]))

    # x_mid and y_mid are (2x-1) and (2y-1)
    x_mid = 2*x-(verts[-1][0] + verts[0][0])
    y_mid = 2*y-(verts[-1][1] + verts[0][1])

    # Compute basis functions for BDMcE
    edge_basis = bdmce_edge_basis(degree, dx, dy, x_mid, y_mid)
    face_basis = bdmce_face_basis(degree, dx, dy, x_mid, y_mid) if degree > 1 else ()

    return edge_basis + face_basis


def numpy_lambdify(X, F, modules="numpy", dummify=False):
    '''Unfortunately, SymPy's own lambdify() doesn't work well with
    NumPy in that simple functions like
        lambda x: 1.0,
    when evaluated with NumPy arrays, return just "1.0" instead of
    an array of 1s with the same shape as x. This function does that.
    '''
    try:
        lambda_x = [numpy_lambdify(X, f, modules=modules, dummify=dummify) for f in F]
    except TypeError:  # 'function' object is not iterable
        # SymPy's lambdify also works on functions that return arrays.
        # However, use it componentwise here so we can add 0*x to each
        # component individually. This is necessary to maintain shapes
        # if evaluated with NumPy arrays.
        lmbd_tmp = lambdify(X, F, modules=modules, dummify=dummify)
        lambda_x = lambda u: lmbd_tmp(*[v for v in u]) + 0 * u[0]

    return lambda_x
