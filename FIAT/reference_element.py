# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014
# Modified by Lizao Li (lzlarryli@gmail.com), 2016

"""
Abstract class and particular implementations of finite element
reference simplex geometry/topology.

Provides an abstract base class and particular implementations for the
reference simplex geometry and topology.
The rest of FIAT is abstracted over this module so that different
reference element geometry (e.g. a vertex at (0,0) versus at (-1,-1))
and orderings of entities have a single point of entry.

Currently implemented are UFC and Default Line, Triangle and Tetrahedron.
"""
import operator
from collections import defaultdict
from functools import reduce
from itertools import chain, count, product
from math import factorial

import numpy
from gem.utils import safe_repr
from recursivenodes.nodes import _decode_family, _recursive

from FIAT.orientation_utils import (
    Orientation,
    make_cell_orientation_reflection_map_simplex,
    make_cell_orientation_reflection_map_tensorproduct,
    make_entity_permutations_simplex,
)

POINT = 0
LINE = 1
TRIANGLE = 2
TETRAHEDRON = 3
QUADRILATERAL = 11
HEXAHEDRON = 111
TENSORPRODUCT = 99

hypercube_shapes = {2: QUADRILATERAL, 3: HEXAHEDRON}


def multiindex_equal(d, isum, imin=0):
    """A generator for d-tuple multi-indices whose sum is isum and minimum is imin.
    """
    if d <= 0:
        return
    imax = isum - (d - 1) * imin
    if imax < imin:
        return
    for i in range(imin, imax):
        for a in multiindex_equal(d - 1, isum - i, imin=imin):
            yield a + (i,)
    yield (imin,) * (d - 1) + (imax,)


def lattice_iter(start, finish, depth):
    """Generator iterating over the depth-dimensional lattice of
    integers between start and (finish-1).  This works on simplices in
    0d, 1d, 2d, 3d, and beyond"""
    if depth == 0:
        yield tuple()
    elif depth == 1:
        for ii in range(start, finish):
            yield (ii,)
    else:
        for ii in range(start, finish):
            for jj in lattice_iter(start, finish - ii, depth - 1):
                yield jj + (ii,)


def make_lattice(verts, n, interior=0, variant=None):
    """Constructs a lattice of points on the simplex defined by verts.
    For example, the 1:st order lattice will be just the vertices.
    The optional argument interior specifies how many points from
    the boundary to omit.  For example, on a line with n = 2,
    and interior = 0, this function will return the vertices and
    midpoint, but with interior = 1, it will only return the
    midpoint."""
    if variant is None:
        variant = "equispaced"
    recursivenodes_families = {
        "equispaced": "equi",
        "equispaced_interior": "equi_interior",
        "gll": "lgl"}
    family = recursivenodes_families.get(variant, variant)
    family = _decode_family(family)
    D = len(verts)
    X = numpy.array(verts)
    get_point = lambda alpha: tuple(numpy.dot(_recursive(D - 1, n, alpha, family), X))
    return list(map(get_point, multiindex_equal(D, n, interior)))


def linalg_subspace_intersection(A, B):
    """Computes the intersection of the subspaces spanned by the
    columns of 2-dimensional arrays A,B using the algorithm found in
    Golub and van Loan (3rd ed) p. 604.  A should be in
    R^{m,p} and B should be in R^{m,q}.  Returns an orthonormal basis
    for the intersection of the spaces, stored in the columns of
    the result."""

    # check that vectors are in same space
    if A.shape[0] != B.shape[0]:
        raise Exception("Dimension error")

    # A,B are matrices of column vectors
    # compute the intersection of span(A) and span(B)

    # Compute the principal vectors/angles between the subspaces, G&vL
    # p.604
    (qa, _ra) = numpy.linalg.qr(A)
    (qb, _rb) = numpy.linalg.qr(B)

    C = numpy.dot(numpy.transpose(qa), qb)

    (y, c, _zt) = numpy.linalg.svd(C)

    U = numpy.dot(qa, y)

    rank_c = len([s for s in c if numpy.abs(1.0 - s) < 1.e-10])

    return U[:, :rank_c]


class Cell:
    """Abstract class for a reference cell.  Provides accessors for
    geometry (vertex coordinates) as well as topology (orderings of
    vertices that make up edges, faces, etc."""
    def __init__(self, shape, vertices, topology):
        """The constructor takes a shape code, the physical vertices expressed
        as a list of tuples of numbers, and the topology of a cell.

        The topology is stored as a dictionary of dictionaries t[i][j]
        where i is the dimension and j is the index of the facet of
        that dimension.  The result is a list of the vertices
        comprising the facet."""
        self.shape = shape
        self.vertices = vertices
        self.topology = topology

        # Given the topology, work out for each entity in the cell,
        # which other entities it contains.
        self.sub_entities = {}
        for dim, entities in topology.items():
            self.sub_entities[dim] = {}

            for e, v in entities.items():
                vertices = frozenset(v)
                sub_entities = []

                for dim_, entities_ in topology.items():
                    for e_, vertices_ in entities_.items():
                        if vertices.issuperset(vertices_):
                            sub_entities.append((dim_, e_))

                # Sort for the sake of determinism and by UFC conventions
                self.sub_entities[dim][e] = sorted(sub_entities)

        # Build super-entity dictionary by inverting the sub-entity dictionary
        self.super_entities = {dim: {entity: [] for entity in topology[dim]} for dim in topology}
        for dim0 in topology:
            for e0 in topology[dim0]:
                for dim1, e1 in self.sub_entities[dim0][e0]:
                    self.super_entities[dim1][e1].append((dim0, e0))

        # Build connectivity dictionary for easier queries
        self.connectivity = {}
        for dim0 in sorted(topology):
            for dim1 in sorted(topology):
                self.connectivity[(dim0, dim1)] = []

            for entity in sorted(topology[dim0]):
                children = self.sub_entities[dim0][entity]
                parents = self.super_entities[dim0][entity]
                for dim1 in sorted(topology):
                    neighbors = children if dim1 < dim0 else parents
                    d01_entities = tuple(e for d, e in neighbors if d == dim1)
                    self.connectivity[(dim0, dim1)].append(d01_entities)

        # Dictionary with derived cells
        self._split_cache = {}

    def __repr__(self):
        return f"{type(self).__name__}({self.shape!r}, {safe_repr(self.vertices)}, {self.topology!r})"

    def _key(self):
        """Hashable object key data (excluding type)."""
        # Default: only type matters
        return None

    def __hash__(self):
        return hash((type(self), self._key()))

    def get_shape(self):
        """Returns the code for the element's shape."""
        return self.shape

    def get_vertices(self):
        """Returns an iterable of the element's vertices, each stored as a
        tuple."""
        return self.vertices

    def get_spatial_dimension(self):
        """Returns the spatial dimension in which the element lives."""
        return len(self.vertices[0])

    def get_topology(self):
        """Returns a dictionary encoding the topology of the element.

        The dictionary's keys are the spatial dimensions (0, 1, ...)
        and each value is a dictionary mapping."""
        return self.topology

    def get_connectivity(self):
        """Returns a dictionary encoding the connectivity of the element.

        The dictionary's keys are the spatial dimensions pairs ((1, 0),
        (2, 0), (2, 1), ...) and each value is a list with entities
        of second dimension ordered by local dim0-dim1 numbering."""
        return self.connectivity

    def get_vertices_of_subcomplex(self, t):
        """Returns the tuple of vertex coordinates associated with the labels
        contained in the iterable t."""
        return tuple(self.vertices[ti] for ti in t)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  For tensor
        product cells, this a tuple of dimensions for each cell in the
        product.  For all other cells, this is the same as the spatial
        dimension."""
        raise NotImplementedError("Should be implemented in a subclass.")

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: `tuple` for tensor product cells, `int` otherwise
        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def construct_subcomplex(self, dimension):
        """Constructs the reference subcomplex of the parent cell subentity
        specified by subcomplex dimension.

        :arg dimension: `tuple` for tensor product cells, `int` otherwise
        """
        if self.get_parent() is None:
            return self.construct_subelement(dimension)
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: `tuple` for tensor product cells, `int` otherwise
        :arg entity_i: entity number (integer)
        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def symmetry_group_size(self, dim):
        """Returns the size of the symmetry group of an entity of
        dimension `dim`."""
        raise NotImplementedError("Should be implemented in a subclass.")

    def cell_orientation_reflection_map(self):
        """Return the map indicating whether each possible cell orientation causes reflection (``1``) or not (``0``)."""
        raise NotImplementedError("Should be implemented in a subclass.")

    def extract_extrinsic_orientation(self, o):
        """Extract extrinsic orientation.

        Parameters
        ----------
        o : Orientation
            Total orientation.

        Returns
        -------
        Orientation
            Extrinsic orientation.

        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def extract_intrinsic_orientation(self, o, axis):
        """Extract intrinsic orientation.

        Parameters
        ----------
        o : Orientation
            Total orientation.
        axis : int
            Reference cell axis for which intrinsic orientation is computed.

        Returns
        -------
        Orientation
            Intrinsic orientation.

        """
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def extrinsic_orientation_permutation_map(self):
        """A map from extrinsic orientations to corresponding axis permutation matrices.

        Notes
        -----
        result[eo] gives the physical axis-reference axis permutation matrix corresponding to
        eo (extrinsic orientation).

        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def is_simplex(self):
        return False

    def is_macrocell(self):
        return False

    def get_interior_facets(self, dim):
        """Return the interior facets this cell is a split and () otherwise."""
        return ()

    def get_parent(self):
        """Return the parent cell if this cell is a split and None otherwise."""
        return None

    def get_parent_complex(self):
        """Return the parent complex if this cell is a split and None otherwise."""
        return None

    def is_parent(self, other, strict=False):
        """Return whether this cell is the parent of the other cell."""
        parent = other
        if strict:
            parent = parent.get_parent_complex()
        while parent is not None:
            if self == parent:
                return True
            parent = parent.get_parent_complex()
        return False

    def __eq__(self, other):
        if self is other:
            return True
        A, B = self.get_vertices(), other.get_vertices()
        if not (len(A) == len(B) and numpy.allclose(A, B)):
            return False
        atop = self.get_topology()
        btop = other.get_topology()
        for dim in atop:
            if set(atop[dim].values()) != set(btop[dim].values()):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return other.is_parent(self, strict=True)

    def __lt__(self, other):
        return self.is_parent(other, strict=True)

    def __ge__(self, other):
        return other.is_parent(self, strict=False)

    def __le__(self, other):
        return self.is_parent(other, strict=False)


class SimplicialComplex(Cell):
    r"""Abstract class for a simplicial complex.

    This consists of list of vertex locations and a topology map defining facets.
    """
    def __init__(self, shape, vertices, topology):
        # Make sure that every facet has the right number of vertices to be
        # a simplex.
        for dim in topology:
            for entity in topology[dim]:
                assert len(topology[dim][entity]) == dim + 1

        super().__init__(shape, vertices, topology)

    def compute_normal(self, facet_i, cell=None):
        """Returns the unit normal vector to facet i of codimension 1."""

        t = self.get_topology()
        sd = self.get_spatial_dimension()

        # To handle simplicial complex case:
        # Find a subcell of which facet_i is on the boundary
        # Note: this is trivial and vastly overengineered for the single-cell
        # case.
        if cell is None:
            cell = next(k for k, facets in enumerate(self.connectivity[(sd, sd-1)])
                        if facet_i in facets)
        verts = numpy.asarray(self.get_vertices_of_subcomplex(t[sd][cell]))
        # Interval case
        if self.get_shape() == LINE:
            v_i = t[1][cell].index(t[0][facet_i][0])
            n = verts[v_i] - verts[[1, 0][v_i]]
            return n / numpy.linalg.norm(n)

        # vectors from vertex 0 to each other vertex.
        vert_vecs_from_v0 = verts[1:, :] - verts[:1, :]

        (u, s, _) = numpy.linalg.svd(vert_vecs_from_v0)
        rank = len([si for si in s if si > 1.e-10])

        # this is the set of vectors that span the simplex
        spanu = u[:, :rank]

        vert_coords_of_facet = \
            self.get_vertices_of_subcomplex(t[sd-1][facet_i])

        # now I find everything normal to the facet.
        vcf = numpy.asarray(vert_coords_of_facet)
        facet_span = vcf[1:, :] - vcf[:1, :]
        (_, sf, vft) = numpy.linalg.svd(facet_span)

        # now get the null space from vft
        rankfacet = len([si for si in sf if si > 1.e-10])
        facet_normal_space = numpy.transpose(vft[rankfacet:, :])

        # now, I have to compute the intersection of
        # facet_span with facet_normal_space
        foo = linalg_subspace_intersection(facet_normal_space, spanu)

        num_cols = foo.shape[1]

        if num_cols != 1:
            raise Exception("barf in normal computation")

        # now need to get the correct sign
        # get a vector in the direction
        nfoo = foo[:, 0]

        # what is the vertex not in the facet?
        verts_set = set(t[sd][cell])
        verts_facet = set(t[sd - 1][facet_i])
        verts_diff = verts_set.difference(verts_facet)
        if len(verts_diff) != 1:
            raise Exception("barf in normal computation: getting sign")
        vert_off = verts_diff.pop()
        vert_on = verts_facet.pop()

        # get a vector from the off vertex to the facet
        v_to_facet = numpy.array(self.vertices[vert_on]) \
            - numpy.array(self.vertices[vert_off])

        if numpy.dot(v_to_facet, nfoo) > 0.0:
            return nfoo
        else:
            return -nfoo

    def compute_tangents(self, dim, i):
        """Computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are *NOT* normalized to have unit length."""
        t = self.get_topology()
        vs = numpy.array(self.get_vertices_of_subcomplex(t[dim][i]))
        return vs[1:] - vs[:1]

    def compute_normalized_tangents(self, dim, i):
        """Computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are normalized to have unit length."""
        ts = self.compute_tangents(dim, i)
        ts /= numpy.linalg.norm(ts, axis=1)[:, None]
        return ts

    def compute_edge_tangent(self, edge_i):
        """Computes the nonnormalized tangent to a 1-dimensional facet.
        returns a single vector."""
        t = self.get_topology()
        vs = numpy.asarray(self.get_vertices_of_subcomplex(t[1][edge_i]))
        return vs[1] - vs[0]

    def compute_normalized_edge_tangent(self, edge_i):
        """Computes the unit tangent vector to a 1-dimensional facet"""
        v = self.compute_edge_tangent(edge_i)
        v /= numpy.linalg.norm(v)
        return v

    def compute_face_tangents(self, face_i):
        """Computes the two tangents to a face.  Only implemented
        for a tetrahedron."""
        if self.get_spatial_dimension() != 3:
            raise Exception("can't get face tangents yet")
        t = self.get_topology()
        vs = numpy.asarray(self.get_vertices_of_subcomplex(t[2][face_i]))
        return vs[1:] - vs[:1]

    def compute_face_edge_tangents(self, dim, entity_id):
        """Computes all the edge tangents of any k-face with k>=1.
        The result is a array of binom(dim+1,2) vectors.
        This agrees with `compute_edge_tangent` when dim=1.
        """
        vert_ids = self.get_topology()[dim][entity_id]
        vert_coords = numpy.asarray(self.get_vertices_of_subcomplex(vert_ids))
        v0 = []
        v1 = []
        for source in range(dim):
            for dest in range(source + 1, dim + 1):
                v0.append(source)
                v1.append(dest)
        return vert_coords[v1] - vert_coords[v0]

    def make_points(self, dim, entity_id, order, variant=None, interior=1):
        """Constructs a lattice of points on the entity_id:th
        facet of dimension dim.  Order indicates how many points to
        include in each direction."""
        if dim == 0:
            return (self.get_vertices()[entity_id], )
        elif 0 < dim <= self.get_spatial_dimension():
            entity_verts = \
                self.get_vertices_of_subcomplex(
                    self.get_topology()[dim][entity_id])
            return make_lattice(entity_verts, order, interior=interior, variant=variant)
        else:
            raise ValueError("illegal dimension")

    def volume(self):
        """Computes the volume of the simplicial complex in the appropriate
        dimensional measure."""
        sd = self.get_spatial_dimension()
        return sum(self.volume_of_subcomplex(sd, k)
                   for k in self.topology[sd])

    def volume_of_subcomplex(self, dim, facet_no):
        vids = self.topology[dim][facet_no]
        return volume(self.get_vertices_of_subcomplex(vids))

    def compute_scaled_normal(self, facet_i):
        """Returns the unit normal to facet_i of scaled by the
        volume of that facet."""
        dim = self.get_spatial_dimension()
        if dim == 2:
            n, = self.compute_tangents(dim-1, facet_i)
            n[0], n[1] = n[1], -n[0]
            return n
        elif dim == 3:
            return -numpy.cross(*self.compute_tangents(dim-1, facet_i))
        v = self.volume_of_subcomplex(dim - 1, facet_i)
        return self.compute_normal(facet_i) * v

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == self.get_spatial_dimension() - 1
        n = SimplicialComplex.compute_normal(self, facet_i)  # skip UFC overrides
        return n / numpy.linalg.norm(n, numpy.inf)

    def get_entity_transform(self, dim, entity):
        """Returns a mapping of point coordinates from the
        `entity`-th subentity of dimension `dim` to the cell.

        :arg dim: subentity dimension (integer)
        :arg entity: entity number (integer)
        """
        topology = self.get_topology()
        celldim = self.get_spatial_dimension()
        codim = celldim - dim
        if dim == 0:
            # Special case vertices.
            i, = topology[dim][entity]
            offset = numpy.asarray(self.get_vertices()[i])
            C = numpy.zeros((dim, ) + offset.shape)
        elif dim == celldim and len(self.topology[celldim]) == 1:
            assert entity == 0
            return lambda x: x
        else:
            subcell = self.construct_subelement(dim)
            subdim = subcell.get_spatial_dimension()
            assert subdim == celldim - codim

            # Entity vertices in entity space.
            v_e = numpy.asarray(subcell.get_vertices())
            A = v_e[1:] - v_e[:1]

            # Entity vertices in cell space.
            v_c = numpy.asarray(self.get_vertices_of_subcomplex(topology[dim][entity]))
            B = v_c[1:] - v_c[:1]

            C = numpy.linalg.solve(A, B)

            offset = v_c[0] - numpy.dot(v_e[0], C)

        def transform(point):
            out = numpy.dot(point, C)
            return numpy.add(out, offset, out=out)

        return transform

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()

    def compute_barycentric_coordinates(self, points, entity=None, rescale=False):
        """Returns the barycentric coordinates of a list of points on the complex."""
        if len(points) == 0:
            return points
        if entity is None:
            entity = (self.get_spatial_dimension(), 0)
        entity_dim, entity_id = entity
        top = self.get_topology()
        sd = self.get_spatial_dimension()

        # get a subcell containing the entity and the restriction indices of the entity
        indices = slice(None)
        subcomplex = top[entity_dim][entity_id]
        if entity_dim != sd:
            cell_id = self.connectivity[(entity_dim, sd)][entity_id][0]
            indices = [i for i, v in enumerate(top[sd][cell_id]) if v in subcomplex]
            subcomplex = top[sd][cell_id]

        cell_verts = self.get_vertices_of_subcomplex(subcomplex)
        ref_verts = numpy.eye(sd + 1)
        A, b = make_affine_mapping(cell_verts, ref_verts)
        A, b = A[indices], b[indices]
        if rescale:
            # rescale barycentric coordinates by the height wrt. to the facet
            h = 1 / numpy.linalg.norm(A, axis=1)
            b *= h
            A *= h[:, None]
        out = numpy.dot(points, A.T)
        return numpy.add(out, b, out=out)

    def compute_bubble(self, points, entity=None):
        """Returns the lowest-order bubble on an entity evaluated at the given
        points on the cell."""
        return numpy.prod(self.compute_barycentric_coordinates(points, entity), axis=1)

    def distance_to_point_l1(self, points, entity=None, rescale=False):
        # noqa: D301
        """Get the L1 distance (aka 'manhatten', 'taxicab' or rectilinear
        distance) from an entity to a point with 0.0 if the point is inside the entity.

        Parameters
        ----------
        points : numpy.ndarray or list
            The coordinates of the points.
        entity : tuple or None
            A tuple of entity dimension and entity id.
        rescale : bool
            If true, the L1 distance is measured with respect to rescaled
            barycentric coordinates, such that the L1 and L2 distances agree
            for points opposite to a single facet.

        Returns
        -------
        numpy.float64 or numpy.ndarray
            The L1 distance, also known as taxicab, manhatten or rectilinear
            distance, of the cell to the point. If 0.0 the point is inside the
            cell.

        Notes
        -----

        This is done with the help of barycentric coordinates where the general
        algorithm is to compute the most negative (i.e. minimum) barycentric
        coordinate then return its negative. For implementation reasons we
        return the sum of all the negative barycentric coordinates. In each of
        the below examples the point coordinate is `X` with appropriate
        dimensions.

        Consider, for example, a UFCInterval. We have two vertices which make
        the interval,
            `P0 = [0]` and
            `P1 = [1]`.
        Our point is
            `X = [x]`.
        Barycentric coordinates are defined as
            `X = alpha * P0 + beta * P1` where
            `alpha + beta = 1.0`.
        The solution is
            `alpha = 1 - X[0] = 1 - x` and
            `beta = X[0] = x`.
        If both `alpha` and `beta` are positive, the point is inside the
        reference interval.

        `---regionA---P0=0------P1=1---regionB---`

        If we are in `regionA`, `alpha` is negative and
        `-alpha = X[0] - 1.0` is the (positive) distance from `P0`.
        If we are in `regionB`, `beta` is negative and `-beta = -X[0]` is
        the exact (positive) distance from `P1`. Since we are in 1D the L1
        distance is the same as the L2 distance. If we are in the interval we
        can just return 0.0.

        Things get more complicated when we consider higher dimensions.
        Consider a UFCTriangle. We have three vertices which make the
        reference triangle,
            `P0 = (0, 0)`,
            `P1 = (1, 0)` and
            `P2 = (0, 1)`.
        Our point is
            `X = [x, y]`.
        Below is a diagram of the cell (which may not render correctly in
        sphinx):

        .. code-block:: text
        ```
                y-axis
                |
                |
          (0,1) P2
                | \\
                |  \\
                |   \\
                |    \\
                |  T  \\
                |      \\
                |       \\
                |        \\
            ---P0--------P1--- x-axis
          (0,0) |         (1,0)
        ```

        Barycentric coordinates are defined as
            `X = alpha * P0 + beta * P1 + gamma * P2` where
            `alpha + beta + gamma = 1.0`.
        The solution is
            `alpha = 1 - X[0] - X[1] = 1 - x - y`,
            `beta = X[0] = x` and
            `gamma = X[1] = y`.
        If all three are positive, the point is inside the reference cell.
        If any are negative, we are outside it. The absolute sum of any
        negative barycentric coordinates usefully gives the L1 distance from
        the cell to the point. For example the point (1,1) has L1 distance
        1 from the cell: on this case alpha = -1, beta = 1 and gamma = 1.
        -alpha = 1 is the L1 distance. For comparison the L2 distance (the
        length of the vector from the nearest point on the cell to the point)
        is sqrt(0.5^2 + 0.5^2) = 0.707. Similarly the point (-1.0, -1.0) has
        alpha = 3, beta = -1 and gamma = -1. The absolute sum of beta and gamma
        2 which is again the L1 distance. The L2 distance in this case is
        sqrt(1^2 + 1^2) = 1.414.

        For a UFCTetrahedron we have four vertices
            `P0 = (0,0,0)`,
            `P1 = (1,0,0)`,
            `P2 = (0,1,0)` and
            `P3 = (0,0,1)`.
        Our point is
            `X = [x, y, z]`.
        The barycentric coordinates are defined as
            `X = alpha * P0 + beta * P1 + gamma * P2 + delta * P3`
            where
            `alpha + beta + gamma + delta = 1.0`.
        The solution is
            `alpha = 1 - X[0] - X[1] - X[2] = 1 - x - y - z`,
            `beta = X[0] = x`,
            `gamma = X[1] = y` and
            `delta = X[2] = z`.
        The rules are the same as for the tetrahedron but with one extra
        barycentric coordinate. Our approximate distance, the absolute sum of
        the negative barycentric coordinates, is at worse around 4 times the
        actual distance to the tetrahedron.

        """
        # sum the negative part of each barycentric coordinate
        bary = self.compute_barycentric_coordinates(points, entity=entity, rescale=rescale)
        return 0.5 * abs(numpy.sum(abs(bary) - bary, axis=-1))

    def contains_point(self, point, epsilon=0.0, entity=None):
        """Checks if reference cell contains given point
        (with numerical tolerance as given by the L1 distance (aka 'manhatten',
        'taxicab' or rectilinear distance) to the cell.

        Parameters
        ----------
        point : numpy.ndarray, list or symbolic expression
            The coordinates of the point.
        epsilon : float
            The tolerance for the check.
        entity : tuple or None
            A tuple of entity dimension and entity id.

        Returns
        -------
        bool : True if the point is inside the cell, False otherwise.

        """
        return self.distance_to_point_l1(point, entity=entity) <= epsilon

    def extract_extrinsic_orientation(self, o):
        """Extract extrinsic orientation.

        Parameters
        ----------
        o : Orientation
            Total orientation.

        Returns
        -------
        Orientation
            Extrinsic orientation.

        """
        if not isinstance(o, Orientation):
            raise TypeError(f"Expecting an instance of Orientation : got {o}")
        return 0

    def extract_intrinsic_orientation(self, o, axis):
        """Extract intrinsic orientation.

        Parameters
        ----------
        o : Orientation
            Total orientation.
        axis : int
            Reference cell axis for which intrinsic orientation is computed.

        Returns
        -------
        Orientation
            Intrinsic orientation.

        """
        if not isinstance(o, Orientation):
            raise TypeError(f"Expecting an instance of Orientation : got {o}")
        if axis != 0:
            raise ValueError(f"axis ({axis}) != 0")
        return o

    @property
    def extrinsic_orientation_permutation_map(self):
        """A map from extrinsic orientations to corresponding axis permutation matrices.

        Notes
        -----
        result[eo] gives the physical axis-reference axis permutation matrix corresponding to
        eo (extrinsic orientation).

        """
        return numpy.diag((1, )).astype(int).reshape((1, 1, 1))


class Simplex(SimplicialComplex):
    r"""Abstract class for a reference simplex.

    Orientation of a physical cell is computed systematically
    by comparing the canonical orderings of its facets and
    the facets in the FIAT reference cell.

    As an example, we compute the orientation of a
    triangular cell:

       +                    +
       | \                  | \
       1   0               47   42
       |     \              |     \
       +--2---+             +--43--+
    FIAT canonical     Mapped example physical cell

    Suppose that the facets of the physical cell
    are canonically ordered as:

    C = [43, 42, 47]

    FIAT facet to Physical facet map is given by:

    M = [42, 47, 43]

    Then the orientation of the cell is computed as:

    C.index(M[0]) = 1; C.remove(M[0])
    C.index(M[1]) = 1; C.remove(M[1])
    C.index(M[2]) = 0; C.remove(M[2])

    o = (1 * 2!) + (1 * 1!) + (0 * 0!) = 3
    """
    def is_simplex(self):
        return True

    def symmetry_group_size(self, dim):
        return factorial(dim + 1)

    def cell_orientation_reflection_map(self):
        """Return the map indicating whether each possible cell orientation causes reflection (``1``) or not (``0``)."""
        return make_cell_orientation_reflection_map_simplex(self.get_dimension())

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)


# Backwards compatible name
ReferenceElement = Simplex


class UFCSimplex(Simplex):

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return ufc_simplex(dimension)


class DefaultSimplex(Simplex):

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return default_simplex(dimension)


class SymmetricSimplex(Simplex):

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return symmetric_simplex(dimension)


class Point(Simplex):
    """This is the reference point."""

    def __init__(self):
        verts = ((),)
        topology = {0: {0: (0,)}}
        super().__init__(POINT, verts, topology)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer). Must be zero.
        """
        assert dimension == 0
        return self


class DefaultLine(DefaultSimplex):
    """This is the reference line with vertices (-1.0,) and (1.0,)."""

    def __init__(self):
        verts = ((-1.0,), (1.0,))
        edges = {0: (0, 1)}
        topology = {0: {0: (0,), 1: (1,)},
                    1: edges}
        super().__init__(LINE, verts, topology)


class UFCInterval(UFCSimplex):
    """This is the reference interval with vertices (0.0,) and (1.0,)."""

    def __init__(self):
        verts = ((0.0,), (1.0,))
        edges = {0: (0, 1)}
        topology = {0: {0: (0,), 1: (1,)},
                    1: edges}
        super().__init__(LINE, verts, topology)


class DefaultTriangle(DefaultSimplex):
    """This is the reference triangle with vertices (-1.0,-1.0),
    (1.0,-1.0), and (-1.0,1.0)."""

    def __init__(self):
        verts = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
        edges = {0: (1, 2),
                 1: (2, 0),
                 2: (0, 1)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super().__init__(TRIANGLE, verts, topology)


class UFCTriangle(UFCSimplex):
    """This is the reference triangle with vertices (0.0,0.0),
    (1.0,0.0), and (0.0,1.0)."""

    def __init__(self):
        verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        edges = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super().__init__(TRIANGLE, verts, topology)

    def compute_normal(self, i):
        "UFC consistent normal"
        t = self.compute_tangents(1, i)[0]
        n = numpy.array((t[1], -t[0]))
        return n / numpy.linalg.norm(n)


class IntrepidTriangle(Simplex):
    """This is the Intrepid triangle with vertices (0,0),(1,0),(0,1)"""

    def __init__(self):
        verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        edges = {0: (0, 1),
                 1: (1, 2),
                 2: (2, 0)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super().__init__(TRIANGLE, verts, topology)

    def get_facet_element(self):
        # I think the UFC interval is equivalent to what the
        # IntrepidInterval would be.
        return UFCInterval()


class DefaultTetrahedron(DefaultSimplex):
    """This is the reference tetrahedron with vertices (-1,-1,-1),
    (1,-1,-1),(-1,1,-1), and (-1,-1,1)."""

    def __init__(self):
        verts = ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
                 (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (1, 2),
                 1: (2, 0),
                 2: (0, 1),
                 3: (0, 3),
                 4: (1, 3),
                 5: (2, 3)}
        faces = {0: (1, 3, 2),
                 1: (2, 3, 0),
                 2: (3, 1, 0),
                 3: (0, 1, 2)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super().__init__(TETRAHEDRON, verts, topology)


class IntrepidTetrahedron(Simplex):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1) used in the Intrepid project."""

    def __init__(self):
        verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (0, 1),
                 1: (1, 2),
                 2: (2, 0),
                 3: (0, 3),
                 4: (1, 3),
                 5: (2, 3)}
        faces = {0: (0, 1, 3),
                 1: (1, 2, 3),
                 2: (0, 3, 2),
                 3: (0, 2, 1)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super().__init__(TETRAHEDRON, verts, topology)

    def get_facet_element(self):
        return IntrepidTriangle()


class UFCTetrahedron(UFCSimplex):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1)."""

    def __init__(self):
        verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (2, 3),
                 1: (1, 3),
                 2: (1, 2),
                 3: (0, 3),
                 4: (0, 2),
                 5: (0, 1)}
        faces = {0: (1, 2, 3),
                 1: (0, 2, 3),
                 2: (0, 1, 3),
                 3: (0, 1, 2)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super().__init__(TETRAHEDRON, verts, topology)

    def compute_normal(self, i):
        "UFC consistent normals."
        t = self.compute_tangents(2, i)
        n = numpy.cross(t[0], t[1])
        return -2.0 * n / numpy.linalg.norm(n)


class TensorProductCell(Cell):
    """A cell that is the product of FIAT cells."""

    def __init__(self, *cells):
        # Vertices
        vertices = tuple(tuple(chain(*coords))
                         for coords in product(*[cell.get_vertices()
                                                 for cell in cells]))

        # Topology
        shape = tuple(len(c.get_vertices()) for c in cells)
        topology = {}
        for dim in product(*[cell.get_topology().keys()
                             for cell in cells]):
            topology[dim] = {}
            topds = [cell.get_topology()[d]
                     for cell, d in zip(cells, dim)]
            for tuple_ei in product(*[sorted(topd)for topd in topds]):
                tuple_vs = list(product(*[topd[ei]
                                          for topd, ei in zip(topds, tuple_ei)]))
                vs = tuple(numpy.ravel_multi_index(numpy.transpose(tuple_vs), shape))
                topology[dim][tuple_ei] = vs
            # flatten entity numbers
            topology[dim] = dict(enumerate(topology[dim][key]
                                           for key in sorted(topology[dim])))

        super().__init__(TENSORPRODUCT, vertices, topology)
        self.cells = tuple(cells)

    def __repr__(self):
        return f"{type(self).__name__}({self.cells!r})"

    def _key(self):
        return self.cells

    @staticmethod
    def _split_slices(lengths):
        n = len(lengths)
        delimiter = [0] * (n + 1)
        for i in range(n):
            delimiter[i + 1] = delimiter[i] + lengths[i]
        return [slice(delimiter[i], delimiter[i+1])
                for i in range(n)]

    def get_dimension(self):
        """Returns the subelement dimension of the cell, a tuple of
        dimensions for each cell in the product."""
        return tuple(c.get_dimension() for c in self.cells)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: dimension in each "direction" (tuple)
        """
        return TensorProductCell(*[c.construct_subelement(d)
                                   for c, d in zip(self.cells, dimension)])

    def construct_subcomplex(self, dimension):
        """Constructs the reference subcomplex of the parent cell subentity
        specified by subcomplex dimension.

        :arg dimension: dimension in each "direction" (tuple)
        """
        return TensorProductCell(*[c.construct_subcomplex(d)
                                   for c, d in zip(self.cells, dimension)])

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: subelement dimension (tuple)
        :arg entity_i: entity number (integer)
        """
        # unravel entity_i
        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.cells, dim))
        alpha = numpy.unravel_index(entity_i, shape)

        # entity transform on each subcell
        sct = [c.get_entity_transform(d, i)
               for c, d, i in zip(self.cells, dim, alpha)]

        slices = TensorProductCell._split_slices(dim)

        def transform(point):
            point = numpy.asarray(point)
            return numpy.concatenate(tuple(t(point[..., s])
                                     for t, s in zip(sct, slices)), axis=-1)

        return transform

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return numpy.prod([c.volume() for c in self.cells])

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i of
        subelement dimension facet_dim."""
        assert len(facet_dim) == len(self.get_dimension())
        indicator = numpy.array(self.get_dimension()) - numpy.array(facet_dim)
        (cell_i,), = numpy.nonzero(indicator)

        n = []
        for i, c in enumerate(self.cells):
            if cell_i == i:
                n.extend(c.compute_reference_normal(facet_dim[i], facet_i))
            else:
                n.extend([0] * c.get_spatial_dimension())
        return numpy.asarray(n)

    def contains_point(self, point, epsilon=0.0):
        """Checks if reference cell contains given point
        (with numerical tolerance as given by the L1 distance (aka 'manhattan',
        'taxicab' or rectilinear distance) to the cell.

        Parameters
        ----------
        point : numpy.ndarray, list or symbolic expression
            The coordinates of the point.
        epsilon : float
            The tolerance for the check.

        Returns
        -------
        bool : True if the point is inside the cell, False otherwise.

        """
        subcell_dimensions = self.get_dimension()
        assert len(point) == sum(subcell_dimensions)
        point_slices = TensorProductCell._split_slices(subcell_dimensions)
        return reduce(operator.and_,
                      (c.contains_point(point[s], epsilon=epsilon)
                       for c, s in zip(self.cells, point_slices)),
                      True)

    def distance_to_point_l1(self, point, rescale=False):
        """Get the L1 distance (aka 'manhatten', 'taxicab' or rectilinear
        distance) to a point with 0.0 if the point is inside the cell.

        For more information see the docstring for the UFCSimplex method."""
        subcell_dimensions = self.get_dimension()
        assert len(point) == sum(subcell_dimensions)
        point_slices = TensorProductCell._split_slices(subcell_dimensions)
        point = numpy.asarray(point)
        return sum(c.distance_to_point_l1(point[..., s], rescale=rescale)
                   for c, s in zip(self.cells, point_slices))

    def symmetry_group_size(self, dim):
        return tuple(c.symmetry_group_size(d) for d, c in zip(dim, self.cells))

    def cell_orientation_reflection_map(self):
        """Return the map indicating whether each possible cell orientation causes reflection (``1``) or not (``0``)."""
        return make_cell_orientation_reflection_map_tensorproduct(self.cells)

    def compare(self, op, other):
        """Parent-based comparison between simplicial complexes.
        This is done dimension by dimension."""
        if hasattr(other, "product"):
            other = other.product
        if isinstance(other, type(self)):
            return all(op(a, b) for a, b in zip(self.cells, other.cells))
        else:
            return op(self, other)

    def __gt__(self, other):
        return self.compare(operator.gt, other)

    def __lt__(self, other):
        return self.compare(operator.lt, other)

    def __ge__(self, other):
        return self.compare(operator.ge, other)

    def __le__(self, other):
        return self.compare(operator.le, other)

    def extract_extrinsic_orientation(self, o):
        """Extract extrinsic orientation.

        Parameters
        ----------
        o : Orientation
            Total orientation.

        Returns
        -------
        Orientation
            Extrinsic orientation.

        Notes
        -----
        The difinition of orientations used here must be consistent with
        that used in make_entity_permutations_tensorproduct.

        """
        if not isinstance(o, Orientation):
            raise TypeError(f"Expecting an instance of Orientation : got {o}")
        dim = len(self.cells)
        size_io = 2  # Number of possible intrinsic orientations along each axis.
        return o // size_io**dim

    def extract_intrinsic_orientation(self, o, axis):
        """Extract intrinsic orientation.

        Parameters
        ----------
        o : Orientation
            Total orientation. ``//`` and ``%`` must be overloaded in type(o).
        axis : int
            Reference cell axis for which intrinsic orientation is computed.

        Returns
        -------
        Orientation
            Intrinsic orientation.

        Notes
        -----
        Must be consistent with make_entity_permutations_tensorproduct.

        """
        if not isinstance(o, Orientation):
            raise TypeError(f"Expecting an instance of Orientation : got {o}")
        dim = len(self.cells)
        if axis >= dim:
            raise ValueError(f"Must give 0 <= axis < {dim} : got {axis}")
        size_io = 2  # Number of possible intrinsic orientations along each axis.
        return o % size_io**dim // size_io**(dim - 1 - axis) % size_io

    @property
    def extrinsic_orientation_permutation_map(self):
        """A map from extrinsic orientations to corresponding axis permutation matrices.

        Notes
        -----
        result[eo] gives the physical axis-reference axis permutation matrix corresponding to
        eo (extrinsic orientation).

        """
        dim = len(self.cells)
        a = numpy.zeros((factorial(dim), dim, dim), dtype=int)
        ai = numpy.array(list(make_entity_permutations_simplex(dim - 1, 2).values()), dtype=int).reshape((factorial(dim), dim, 1))
        numpy.put_along_axis(a, ai, 1, axis=2)
        return a


class Hypercube(Cell):
    """Abstract class for a reference hypercube"""

    def __init__(self, dimension, product):
        self.dimension = dimension
        self.shape = hypercube_shapes[dimension]

        pt = product.get_topology()
        verts = product.get_vertices()
        topology = flatten_entities(pt)

        super().__init__(self.shape, verts, topology)

        self.product = product
        self.unflattening_map = compute_unflattening_map(pt)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        sd = self.get_spatial_dimension()
        if dimension > sd:
            raise ValueError(f"Invalid dimension: {(dimension,)}")
        elif dimension == sd:
            return self
        else:
            sub_element = self.product.construct_subelement((dimension,) + (0,)*(len(self.product.cells) - 1))
            return flatten_reference_cube(sub_element)

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: entity dimension (integer)
        :arg entity_i: entity number (integer)
        """
        d, e = self.unflattening_map[(dim, entity_i)]
        return self.product.get_entity_transform(d, e)

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return self.product.volume()

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        sd = self.get_spatial_dimension()
        assert facet_dim == sd - 1
        d, i = self.unflattening_map[(facet_dim, facet_i)]
        return self.product.compute_reference_normal(d, i)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance as given by the L1 distance (aka 'manhattan',
        'taxicab' or rectilinear distance) to the cell.

        Parameters
        ----------
        point : numpy.ndarray, list or symbolic expression
            The coordinates of the point.
        epsilon : float
            The tolerance for the check.

        Returns
        -------
        bool : True if the point is inside the cell, False otherwise.

        """
        return self.product.contains_point(point, epsilon=epsilon)

    def distance_to_point_l1(self, point, rescale=False):
        """Get the L1 distance (aka 'manhattan', 'taxicab' or rectilinear
        distance) to a point with 0.0 if the point is inside the cell.

        For more information see the docstring for the UFCSimplex method."""
        return self.product.distance_to_point_l1(point, rescale=rescale)

    def symmetry_group_size(self, dim):
        """Size of hypercube symmetry group is d! * 2**d"""
        return factorial(dim) * (2**dim)

    def cell_orientation_reflection_map(self):
        """Return the map indicating whether each possible cell orientation causes reflection (``1``) or not (``0``)."""
        return self.product.cell_orientation_reflection_map()

    def __gt__(self, other):
        return self.product > other

    def __lt__(self, other):
        return self.product < other

    def __ge__(self, other):
        return self.product >= other

    def __le__(self, other):
        return self.product <= other


class UFCHypercube(Hypercube):
    """Reference UFC Hypercube

    UFCHypercube: [0, 1]^d with vertices in
    lexicographical order."""

    def __init__(self, dim):
        cells = [UFCInterval()] * dim
        product = TensorProductCell(*cells)
        super().__init__(dim, product)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        sd = self.get_spatial_dimension()
        if dimension > sd:
            raise ValueError(f"Invalid dimension: {dimension}")
        elif dimension == sd:
            return self
        else:
            return ufc_hypercube(dimension)


class UFCQuadrilateral(UFCHypercube):
    r"""This is the reference quadrilateral with vertices
    (0.0, 0.0), (0.0, 1.0), (1.0, 0.0) and (1.0, 1.0).

    Orientation of a physical cell is computed systematically
    by comparing the canonical orderings of its facets and
    the facets in the FIAT reference cell.

    As an example, we compute the orientation of a
    quadrilateral cell:

       +---3---+           +--57---+
       |       |           |       |
       0       1          43       55
       |       |           |       |
       +---2---+           +--42---+
    FIAT canonical     Mapped example physical cell

    Suppose that the facets of the physical cell
    are canonically ordered as:

    C = [55, 42, 43, 57]

    FIAT index to Physical index map must be such that
    C[0] = 55 is mapped to a vertical facet; in this
    example it is:

    M = [43, 55, 42, 57]

    C and M are decomposed into "vertical" and "horizontal"
    parts, keeping the relative orders of numbers:

    C -> C0 = [55, 43], C1 = [42, 57]
    M -> M0 = [43, 55], M1 = [42, 57]

    Then the orientation of the cell is computed as the
    following:

    C0.index(M0[0]) = 1; C0.remove(M0[0])
    C0.index(M0[1]) = 0; C0.remove(M0[1])
    C1.index(M1[0]) = 0; C1.remove(M1[0])
    C1.index(M1[1]) = 0; C1.remove(M1[1])

    o = 2 * 1 + 0 = 2
    """

    def __init__(self):
        super().__init__(2)


class UFCHexahedron(UFCHypercube):
    """This is the reference hexahedron with vertices
    (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0) and (1.0, 1.0, 1.0)."""

    def __init__(self):
        super().__init__(3)


def make_affine_mapping(xs, ys):
    """Constructs (A,b) such that x --> A * x + b is the affine
    mapping from the simplex defined by xs to the simplex defined by ys."""

    dim_x = len(xs[0])
    dim_y = len(ys[0])

    if len(xs) != len(ys):
        raise Exception("")

    # find A in R^{dim_y,dim_x}, b in R^{dim_y} such that
    # A xs[i] + b = ys[i] for all i

    mat = numpy.zeros((dim_x * dim_y + dim_y, dim_x * dim_y + dim_y), "d")
    rhs = numpy.zeros((dim_x * dim_y + dim_y,), "d")

    # loop over points
    for i in range(len(xs)):
        # loop over components of each A * point + b
        for j in range(dim_y):
            row_cur = i * dim_y + j
            col_start = dim_x * j
            col_finish = col_start + dim_x
            mat[row_cur, col_start:col_finish] = numpy.array(xs[i])
            rhs[row_cur] = ys[i][j]
            # need to get terms related to b
            mat[row_cur, dim_y * dim_x + j] = 1.0

    sol = numpy.linalg.solve(mat, rhs)

    A = numpy.reshape(sol[:dim_x * dim_y], (dim_y, dim_x))
    b = sol[dim_x * dim_y:]

    return A, b


def ufc_hypercube(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the UFC reference hypercube of that dimension."""
    if spatial_dim == 0:
        return Point()
    elif spatial_dim == 1:
        return UFCInterval()
    elif spatial_dim == 2:
        return UFCQuadrilateral()
    elif spatial_dim == 3:
        return UFCHexahedron()
    else:
        raise RuntimeError(f"Can't create UFC hypercube of dimension {spatial_dim}.")


def default_simplex(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the default reference simplex of that dimension."""
    if spatial_dim == 0:
        return Point()
    elif spatial_dim == 1:
        return DefaultLine()
    elif spatial_dim == 2:
        return DefaultTriangle()
    elif spatial_dim == 3:
        return DefaultTetrahedron()
    else:
        raise RuntimeError(f"Can't create default simplex of dimension {spatial_dim}.")


def ufc_simplex(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the UFC reference simplex of that dimension."""
    if spatial_dim == 0:
        return Point()
    elif spatial_dim == 1:
        return UFCInterval()
    elif spatial_dim == 2:
        return UFCTriangle()
    elif spatial_dim == 3:
        return UFCTetrahedron()
    else:
        raise RuntimeError(f"Can't create UFC simplex of dimension {spatial_dim}.")


def symmetric_simplex(spatial_dim):
    A = numpy.array([[2, 1, 1],
                     [0, numpy.sqrt(3), numpy.sqrt(3)/3],
                     [0, 0, numpy.sqrt(6)*(2/3)]])
    A = A[:spatial_dim, :][:, :spatial_dim]
    b = A.sum(axis=1) * (-1 / (1 + spatial_dim))
    Ref1 = ufc_simplex(spatial_dim)
    v = numpy.dot(Ref1.get_vertices(), A.T) + b[None, :]
    vertices = tuple(map(tuple, v))
    return SymmetricSimplex(Ref1.get_shape(), vertices, Ref1.get_topology())


def ufc_cell(cell):
    """Handle incoming calls from FFC."""

    # celltype could be a string or a cell.
    if isinstance(cell, str):
        celltype = cell
    else:
        celltype = cell.cellname()

    if " * " in celltype:
        # Tensor product cell
        return TensorProductCell(*map(ufc_cell, celltype.split(" * ")))
    elif celltype == "quadrilateral":
        return UFCQuadrilateral()
    elif celltype == "hexahedron":
        return UFCHexahedron()
    elif celltype == "vertex":
        return ufc_simplex(0)
    elif celltype == "interval":
        return ufc_simplex(1)
    elif celltype == "triangle":
        return ufc_simplex(2)
    elif celltype == "tetrahedron":
        return ufc_simplex(3)
    else:
        raise RuntimeError(f"Don't know how to create UFC cell of type {str(celltype)}")


def volume(verts):
    """Constructs the volume of the simplex spanned by verts"""

    # use fact that volume of UFC reference element is 1/n!
    sd = len(verts) - 1
    ufcel = ufc_simplex(sd)
    ufcverts = ufcel.get_vertices()

    A, b = make_affine_mapping(ufcverts, verts)

    # can't just take determinant since, e.g. the face of
    # a tet being mapped to a 2d triangle doesn't have a
    # square matrix

    (u, s, vt) = numpy.linalg.svd(A)

    # this is the determinant of the "square part" of the matrix
    # (ie the part that maps the restriction of the higher-dimensional
    # stuff to UFC element
    p = numpy.prod([si for si in s if (si) > 1.e-10])

    return p / factorial(sd)


def tuple_sum(tree):
    """
    This function calculates the sum of elements in a tuple, it is needed to handle nested tuples in TensorProductCell.
    Example: tuple_sum(((1, 0), 1)) returns 2
    If input argument is not the tuple, returns input.
    """
    if isinstance(tree, tuple):
        return sum(map(tuple_sum, tree))
    else:
        return tree


def is_ufc(cell):
    if isinstance(cell, (Point, UFCInterval, UFCHypercube, UFCSimplex)):
        return True
    elif isinstance(cell, TensorProductCell):
        return all(is_ufc(c) for c in cell.cells)
    else:
        return False


def is_hypercube(cell):
    if isinstance(cell, (DefaultLine, UFCInterval, Hypercube)):
        return True
    elif isinstance(cell, TensorProductCell):
        return all(is_hypercube(c) for c in cell.cells)
    else:
        return False


def flatten_reference_cube(ref_el):
    """This function flattens a Tensor Product hypercube to the corresponding UFC hypercube"""
    if ref_el.get_spatial_dimension() <= 1:
        # Just return point/interval cell arguments
        return ref_el
    else:
        # Handle cases where cell is a quad/cube constructed from a tensor product or
        # an already flattened element
        if isinstance(ref_el, TensorProductCell):
            if is_ufc(ref_el):
                return ufc_hypercube(ref_el.get_spatial_dimension())
            return Hypercube(ref_el.get_spatial_dimension(), ref_el)
        elif is_hypercube(ref_el):
            return ref_el
        else:
            raise TypeError('Invalid cell type')


def flatten_entities(topology_dict):
    """This function flattens topology dict of TensorProductCell and entity_dofs dict of TensorProductElement"""

    flattened_entities = defaultdict(list)
    for dim in sorted(topology_dict.keys()):
        flat_dim = tuple_sum(dim)
        flattened_entities[flat_dim] += [v for k, v in sorted(topology_dict[dim].items())]

    return {dim: dict(enumerate(entities))
            for dim, entities in flattened_entities.items()}


def flatten_permutations(perm_dict):
    """This function flattens permutation dict of TensorProductElement"""

    flattened_permutations = defaultdict(list)
    for dim in sorted(perm_dict.keys()):
        flat_dim = tuple_sum(dim)
        flattened_permutations[flat_dim] += [{o: v[o_tuple] for o, o_tuple in enumerate(sorted(v))}
                                             for k, v in sorted(perm_dict[dim].items())]
    return {dim: dict(enumerate(perms))
            for dim, perms in flattened_permutations.items()}


def compute_unflattening_map(topology_dict):
    """This function returns unflattening map for the given tensor product topology dict."""

    counter = defaultdict(count)
    unflattening_map = {}

    for dim, entities in sorted(topology_dict.items()):
        flat_dim = tuple_sum(dim)
        for entity in entities:
            flat_entity = next(counter[flat_dim])
            unflattening_map[(flat_dim, flat_entity)] = (dim, entity)

    return unflattening_map


def max_complex(complexes):
    max_cell = max(complexes)
    if all(max_cell >= b for b in complexes):
        return max_cell
    else:
        raise ValueError("Cannot find the maximal complex")
