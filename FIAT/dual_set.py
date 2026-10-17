# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same at Baylor University.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from itertools import chain
from functools import cached_property
from collections import defaultdict

from FIAT import polynomial_set, functional
from FIAT.reference_element import compute_unflattening_map


class DualSet:
    """Represent a set of functionals and their assignment to cell entities.

    Parameters
    ----------
    nodes : sequence of Functional
        Source functionals used to define the dual set. If ``coeffs`` is
        provided, these correspond to its columns; otherwise they are the
        functionals in local degree-of-freedom order.
    ref_el : Cell
        Reference element or simplicial complex on which the functionals are
        defined.
    entity_ids : dict
        Mapping from entity dimension and entity number to the indices of the
        functionals represented by this dual set assigned to that entity.
    entity_permutations : dict, optional
        Mapping from entity dimension, entity number, and orientation to the
        permutation of local DoFs.
    coeffs : array_like, optional
        Matrix defining linear combinations of the functionals. Its rows give
        the coefficients for the recombined functionals, and its columns
        correspond to ``nodes``.

    node_mapping : sequence of str, optional
        Physical convention of each source node: ``cartesian``, ``frame``
        or ``invariant``. Defaults follow point and moment conventions.

    Raises
    ------
    ValueError
        If ``coeffs`` does not have one column per functional.
    """

    def __init__(self, nodes, ref_el, entity_ids, entity_permutations=None,
                 coeffs=None, node_mapping=None):
        if ref_el.get_dimension() != max(entity_ids):
            entity_ids = unflatten_entity_ids(ref_el, entity_ids)
        nodes, ref_el, entity_ids, entity_permutations, macro_entity_ids = merge_entities(
            nodes, ref_el, entity_ids, entity_permutations)
        if coeffs is not None:
            coeffs = numpy.asarray(coeffs)
            if coeffs.ndim != 2 or coeffs.shape[1] != len(nodes):
                raise ValueError("Dual coefficients must have one column per functional")

        self.nodes = nodes
        self.coeffs = coeffs
        self.ref_el = ref_el
        self.entity_ids = entity_ids
        # The entity ids on the simplicial complex, numbered by the merged node ordering
        self._macro_entity_ids = macro_entity_ids
        self.entity_permutations = entity_permutations
        self._node_mapping = node_mapping

        # Compute the nodes on the closure of each sub_entity.
        self.entity_closure_ids = {}
        for dim, entities in ref_el.sub_entities.items():
            self.entity_closure_ids[dim] = {}

            for e, sub_entities in entities.items():
                ids = []

                for d, se in sub_entities:
                    ids += self.entity_ids[d][se]
                ids.sort()
                self.entity_closure_ids[d][e] = ids

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes) if self.coeffs is None else self.coeffs.shape[0]

    def get_nodes(self):
        return self.nodes

    def get_coeffs(self):
        """Return the recombination coefficients of the dual set."""
        return self.coeffs

    def recombine(self, coefficients):
        """Return a dual set with an additional node recombination.

        :arg coefficients: Matrix multiplying the current dual set.
        :returns: A dual set representing the recombined functionals.
        """
        coefficients = numpy.asarray(coefficients)
        if coefficients.shape[-1] != len(self):
            raise ValueError("Recombination coefficients must have one column per dual functional")
        if self.coeffs is not None:
            coefficients = numpy.dot(coefficients, self.coeffs)
        return DualSet(self.nodes, self.ref_el, self.entity_ids,
                       self.entity_permutations, coeffs=coefficients,
                       node_mapping=self._node_mapping)

    def get_entity_closure_ids(self):
        return self.entity_closure_ids

    def get_entity_ids(self):
        return self.entity_ids

    def get_entity_permutations(self):
        r"""This method returns a nested dictionary that gives, for
        each dimension, for each entity, and for each possible entity
        orientation, the DoF permutation array that maps the entity
        local DoF ordering to the canonical global DoF ordering; see
        :class:`~.Simplex` and :class:`~.UFCQuadrilateral` for how we
        define entity orientations for standard cells.

        The entity permutations `dict` for the degree 4 Lagrange finite
        element on the interval, for instance, is given by:

        .. code-block:: python3

            {0: {0: {0: [0]},
                 1: {0: [0]}},
             1: {0: {0: [0, 1, 2],
                     1: [2, 1, 0]}}}

        Note that there are two entities on dimension ``0`` (vertices),
        each of which has only one possible orientation, while there is
        a single entity on dimension ``1`` (interval), which has two
        possible orientations representing non-reflected and reflected
        intervals.
        """
        if self.entity_permutations is None:
            raise NotImplementedError("entity_permutations not yet implemented for %s" % type(self))
        return self.entity_permutations

    def get_reference_element(self):
        return self.ref_el

    def to_riesz(self, poly_set):
        r"""This method gives the action of the entire dual set
        on each member of the expansion set underlying poly_set.
        Then, applying the linear functionals of the dual set to an
        arbitrary polynomial in poly_set is accomplished by (generalized)
        matrix multiplication.

        For scalar-valued spaces, this produces a matrix
        :\math:`R_{i, j}` such that
        :\math:`\ell_i(f) = \sum_{j} a_j \ell_i(\phi_j)`
        for :\math:`f=\sum_{j} a_j \phi_j`.

        More generally, it will have shape concatenating
        the number of functionals in the dual set, the value shape
        of functions it takes, and the number of members of the
        expansion set.
        """

        # This rather technical code queries the low-level information
        # in pt_dict and deriv_dict
        # for each functional to find out where it evaluates its
        # inputs and/or their derivatives.  Then, it tabulates the
        # expansion set one time for all the function values and
        # another for all of the derivatives.  This circumvents
        # needing to call the to_riesz method of each functional and
        # also limits the number of different calls to tabulate.

        tshape = self.nodes[0].target_shape
        num_nodes = len(self.nodes)
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        num_exp = es.get_num_members(poly_set.get_embedded_degree())

        riesz_shape = (num_nodes, *tshape, num_exp)
        mat = numpy.zeros(riesz_shape, "d")

        def map_quadratures_to_points(nodes, deriv=False):
            Qs_to_ells = defaultdict(list)
            for i, ell in enumerate(nodes):
                if deriv and len(ell.deriv_dict) == 0:
                    continue
                elif not deriv and len(ell.pt_dict) == 0:
                    continue
                if isinstance(ell, (functional.IntegralMoment, functional.IntegralMomentOfDerivative)):
                    Q = ell.Q
                else:
                    Q = None
                Qs_to_ells[Q].append(i)
            pts = set()
            Qs_to_pts = {}
            for Q in Qs_to_ells:
                if Q is None:
                    if deriv:
                        cur_pts = chain.from_iterable(nodes[i].deriv_dict.keys() for i in Qs_to_ells[None])
                    else:
                        cur_pts = chain.from_iterable(nodes[i].pt_dict.keys() for i in Qs_to_ells[None])
                    cur_pts = tuple(set(cur_pts))
                else:
                    cur_pts = tuple(map(tuple, Q.pts))
                Qs_to_pts[Q] = cur_pts
                pts.update(cur_pts)
            pts = list(sorted(pts))
            return Qs_to_ells, Qs_to_pts, pts

        # Now tabulate the function values
        Qs_to_ells, Qs_to_pts, pts = map_quadratures_to_points(self.nodes)
        expansion_values = numpy.transpose(es.tabulate(ed, pts))
        for Q in Qs_to_ells:
            ells = Qs_to_ells[Q]
            cur_pts = Qs_to_pts[Q]
            indices = list(map(pts.index, cur_pts))
            wshape = (len(ells), *tshape, len(cur_pts))
            wts = numpy.zeros(wshape, "d")
            if Q is None:
                for i, k in enumerate(ells):
                    ell = self.nodes[k]
                    for pt, wc_list in ell.pt_dict.items():
                        j = cur_pts.index(pt)
                        for (w, c) in wc_list:
                            wts[i][c][j] = w
            else:
                for i, k in enumerate(ells):
                    ell = self.nodes[k]
                    wts[i][ell.comp][:] = ell.f_at_qpts
                qwts = Q.get_weights()
                wts = numpy.multiply(wts, qwts, out=wts)
            mat[ells] += numpy.dot(wts, expansion_values[indices])

        # Tabulate the derivative values that are needed
        max_deriv_order = max(ell.max_deriv_order for ell in self.nodes)
        if max_deriv_order > 0:
            Qs_to_ells, Qs_to_pts, pts = map_quadratures_to_points(self.nodes, deriv=True)
            # It's easiest/most efficient to get derivatives of the
            # expansion set through the polynomial set interface.
            # This is creating a short-lived set to do just this.
            coeffs = numpy.eye(num_exp)
            expansion = polynomial_set.PolynomialSet(self.ref_el, ed, ed, es, coeffs)
            dexpansion_values = expansion.tabulate(pts, max_deriv_order)
            for Q in Qs_to_ells:
                ells = Qs_to_ells[Q]
                cur_pts = Qs_to_pts[Q]
                indices = list(map(pts.index, cur_pts))
                wshape = (len(ells), *tshape, len(cur_pts))
                dwts = {alpha: numpy.zeros(wshape, "d") for alpha in dexpansion_values if sum(alpha) > 0}
                if Q is None:
                    for i, k in enumerate(ells):
                        ell = self.nodes[k]
                        for pt, wac_list in ell.deriv_dict.items():
                            j = cur_pts.index(pt)
                            for (w, alpha, c) in wac_list:
                                dwts[alpha][i][c][j] = w
                else:
                    for i, k in enumerate(ells):
                        ell = self.nodes[k]
                        for alpha in ell.weights:
                            dwts[alpha][i][ell.comp][:] = ell.weights[alpha]
                for alpha in dwts:
                    wts = dwts[alpha]
                    expansion_values = dexpansion_values[alpha].T
                    mat[ells] += numpy.dot(wts, expansion_values[indices])
        coeffs = self.get_coeffs()
        if coeffs is not None:
            mat = numpy.tensordot(coeffs, mat, axes=(1, 0))
        return mat

    @cached_property
    def node_mapping(self):
        """Physical conventions of the source nodes, in source-node order."""
        if self._node_mapping is not None:
            if len(self._node_mapping) != len(self.nodes):
                raise ValueError("Expected one physical convention per source node")
            if set(self._node_mapping) - {"cartesian", "frame", "invariant"}:
                raise ValueError("Unknown physical node convention")
            return tuple(self._node_mapping)
        sd = self.ref_el.get_spatial_dimension()
        mappings = [None] * len(self.nodes)
        for dim, entities in self.entity_ids.items():
            for dofs in entities.values():
                for i in dofs:
                    node = self.nodes[i]
                    order = node.max_deriv_order
                    rank = len(node.target_shape)
                    if order == rank == 0 or (dim == sd and _is_integral(node) and order == 0):
                        mappings[i] = "invariant"
                    elif dim == sd - 1 and (_is_integral(node) or order > 0 or rank > 1):
                        mappings[i] = "frame"
                    else:
                        mappings[i] = "cartesian"
        return tuple(mappings)

    @cached_property
    def entity_support_ids(self):
        """Partition each entity_ids[dim][entity] by supporting entity."""
        support_ids = {}
        for dim, entities in self.entity_ids.items():
            support_ids[dim] = {}
            for entity, dofs in entities.items():
                supports = defaultdict(list)
                for i in dofs:
                    node = self.nodes[i]
                    points = set(node.pt_dict)
                    points.update(node.deriv_dict)
                    support = _support_entity(self.ref_el, sorted(points))
                    supports[support].append(i)
                support_ids[dim][entity] = dict(supports)
        return support_ids

    def is_integral_moment(self, i):
        """Whether source functional i is an integral moment."""
        return _is_integral(self.nodes[i])

    def as_tensor(self, mappings, tol=1e-10):
        r"""Write source functionals as :math:`n_i(u)=\sum_q W_{i,q}:D_i u(x_{i,q})`.

        Return evaluation points, coefficient tensors W, and axis pullback laws,
        in source-node order. Component axes precede derivative axes; divergence
        axes are contracted. Recombination coefficients are kept separate.
        """
        points = [None] * len(self.nodes)
        coefficients = [None] * len(self.nodes)
        axis_mappings = [None] * len(self.nodes)
        sd = self.ref_el.get_spatial_dimension()
        for dim, entities in self.entity_ids.items():
            for entity, dofs in entities.items():
                if 0 < dim < sd - 1:
                    t = self.ref_el.compute_edge_tangent(entity)
                    normal = numpy.eye(sd) - numpy.outer(t, t) / (t @ t)
                for i in dofs:
                    pts, W, axes = self.nodes[i].as_tensor(mappings[i], tol)
                    if 0 < dim < sd - 1:
                        if any(axis == "derivative" and _has_direction(W, k, normal, tol)
                               for k, axis in enumerate(axes)):
                            raise NotImplementedError("No physical convention for derivatives normal to an edge")
                    points[i] = pts
                    coefficients[i] = W
                    axis_mappings[i] = axes
        return tuple(points), tuple(coefficients), tuple(axis_mappings)

    def get_scaling_exponents(self, coefficients, axis_mappings, avg=True, tol=1e-10):
        r"""Return :math:`p_i` for physical source nodes that dilate as :math:`h^{-p_i}`.

        Use the coefficient tensors and axis mappings from as_tensor.
        The avg flag selects physical averages for framed integral moments."""
        sd = self.ref_el.get_spatial_dimension()
        exponents = numpy.zeros(len(self.nodes), dtype=int)
        for dim, entities in self.entity_ids.items():
            for entity, dofs in entities.items():
                for i in dofs:
                    W = coefficients[i]
                    axes = axis_mappings[i]
                    mapping = self.node_mapping[i]
                    if mapping == "invariant":
                        continue
                    exponent = axes.count("derivative")
                    if mapping == "cartesian":
                        weights = {"covariant piola": 1, "contravariant piola": sd - 1, "divergence": sd}
                        exponent += sum(weights.get(axis, 0) for axis in axes)
                    else:
                        nu = self.ref_el.compute_scaled_normal(entity)
                        tangent = numpy.eye(sd) - numpy.outer(nu, nu) / (nu @ nu)
                        for k, axis in enumerate(axes):
                            if axis == "contravariant piola" and _has_direction(W, k, tangent, tol):
                                exponent += 2 - sd
                        if self.is_integral_moment(i) and not avg:
                            exponent -= dim
                    exponents[i] = exponent
        return exponents

    def get_indices(self, restriction_domain, take_closure=True):
        """Returns the list of dofs with support on a given restriction domain.

        :arg restriction_domain: can be 'interior', 'vertex', 'edge', 'face' or 'facet'
        :kwarg take_closure: Are we taking the closure of the restriction domain?
        """
        entity_dofs = self.get_entity_ids()
        if restriction_domain == "interior":
            # Return dofs from interior, never taking the closure
            indices = []
            entities = entity_dofs[max(entity_dofs.keys())]
            for (entity, ids) in sorted_by_key(entities):
                indices.extend(ids)
            return indices

        # otherwise return dofs with d <= dim
        if restriction_domain == "vertex":
            dim = 0
        elif restriction_domain == "edge":
            dim = 1
        elif restriction_domain == "face":
            dim = 2
        elif restriction_domain == "facet":
            dim = self.get_reference_element().get_spatial_dimension() - 1
        elif restriction_domain == "ridge":
            dim = self.get_reference_element().get_spatial_dimension() - 2
        else:
            raise RuntimeError("Invalid restriction domain")

        is_prodcell = isinstance(max(entity_dofs.keys()), tuple)

        ldim = 0 if take_closure else dim
        indices = []
        for d in range(ldim, dim + 1):
            if is_prodcell:
                for edim in entity_dofs:
                    if sum(edim) == d:
                        entities = entity_dofs[edim]
                        for (entity, ids) in sorted_by_key(entities):
                            indices.extend(ids)
            else:
                entities = entity_dofs[d]
                for (entity, ids) in sorted_by_key(entities):
                    indices.extend(ids)
        return indices


def sorted_by_key(mapping):
    "Sort dict items by key, allowing different key types."
    # Python3 doesn't allow comparing builtins of different type, therefore the typename trick here
    def _key(x):
        return (type(x[0]).__name__, x[0])
    return sorted(mapping.items(), key=_key)


def make_entity_closure_ids(ref_el, entity_ids):
    entity_closure_ids = {}
    for dim, entities in ref_el.sub_entities.items():
        entity_closure_ids[dim] = {}

        for e, sub_entities in entities.items():
            ids = []

            for d, se in sub_entities:
                ids += entity_ids[d][se]
            ids.sort()
            entity_closure_ids[d][e] = ids

    return entity_closure_ids


def unflatten_entity_ids(ref_el, entity_ids):
    """Reconstruct entity_ids to match the entities of ref_el."""
    topology = ref_el.get_topology()
    unflattening_map = compute_unflattening_map(topology)
    unflattened_entity_ids = {dim: {} for dim in sorted(topology)}
    for dim in sorted(entity_ids):
        for entity in sorted(entity_ids[dim]):
            unflat_dim, unflat_entity = unflattening_map[(dim, entity)]
            unflattened_entity_ids[unflat_dim][unflat_entity] = entity_ids[dim][entity]
    return unflattened_entity_ids


def lexsort_nodes(ref_el, nodes, entity=None, offset=0):
    """Sort PointEvaluation nodes in lexicographical ordering."""
    if len(nodes) > 1:
        pts = []
        for node in nodes:
            pt, = node.get_point_dict()
            pts.append(pt)
        bary = ref_el.compute_barycentric_coordinates(pts)
        order = list(offset + numpy.lexsort(bary.T))
    else:
        order = list(range(offset, offset + len(nodes)))
    return order


def merge_entities(nodes, ref_el, entity_ids, entity_permutations):
    """Collect DOFs from simplicial complex onto facets of parent cell.

    Also returns the entity ids on the simplicial complex, numbered by the
    merged node ordering.
    """
    parent_cell = ref_el.get_parent()
    if parent_cell is None:
        return nodes, ref_el, entity_ids, entity_permutations, entity_ids
    parent_ids = {}
    parent_permutations = None
    parent_to_children = ref_el.get_parent_to_children()

    if all(isinstance(node, functional.PointEvaluation) for node in nodes):
        # Merge Lagrange dual with lexicographical reordering
        parent_nodes = []
        child_ids = {dim: {} for dim in entity_ids}
        for dim in sorted(parent_to_children):
            parent_ids[dim] = {}
            for entity in sorted(parent_to_children[dim]):
                cur = len(parent_nodes)
                for child_dim, child_entity in parent_to_children[dim][entity]:
                    start = len(parent_nodes)
                    parent_nodes.extend(nodes[i] for i in entity_ids[child_dim][child_entity])
                    child_ids[child_dim][child_entity] = list(range(start, len(parent_nodes)))
                ids = lexsort_nodes(parent_cell, parent_nodes[cur:], entity=(dim, entity), offset=cur)
                parent_ids[dim][entity] = ids
    else:
        # Merge everything else with the same node ordering
        parent_nodes = nodes
        child_ids = entity_ids
        for dim in sorted(parent_to_children):
            parent_ids[dim] = {}
            for entity in sorted(parent_to_children[dim]):
                parent_ids[dim][entity] = []
                for child_dim, child_entity in parent_to_children[dim][entity]:
                    parent_ids[dim][entity].extend(entity_ids[child_dim][child_entity])

    return parent_nodes, parent_cell, parent_ids, parent_permutations, child_ids


def _support_entity(ref_el, points, tol=1e-12):
    """Find the smallest simplex entity containing all evaluation points.

    :arg ref_el: Reference simplex.
    :arg points: Points in reference Cartesian coordinates.
    :arg tol: Absolute tolerance for a zero barycentric coordinate.
    :returns: Entity dimension and number.
    """
    bary = ref_el.compute_barycentric_coordinates(numpy.asarray(points))
    # A nonzero barycentric coordinate requires its vertex in the entity.
    verts = set(numpy.flatnonzero(numpy.abs(bary).max(axis=0) > tol))
    top = ref_el.get_topology()
    for dim, entities in sorted(top.items()):
        for entity, vertices in sorted(entities.items()):
            if verts <= set(vertices):
                return dim, entity
    raise ValueError("The evaluation points have no supporting entity.")


def _has_direction(W, axis, projector, tol):
    residual = numpy.tensordot(W, projector, axes=(1 + axis, 1))
    return numpy.abs(residual).max() > tol * numpy.abs(W).max()


def _is_integral(node):
    return isinstance(node, (functional.IntegralMoment, functional.IntegralMomentOfDerivative,
                             functional.IntegralMomentOfDivergence, functional.IntegralMomentOfTensorDivergence,
                             functional.IntegralMomentOfEdgeTangentEvaluation,
                             functional.IntegralMomentOfFaceTangentEvaluation,
                             functional.IntegralMomentOfScaledNormalEvaluation,
                             functional.IntegralMomentOfNormalEvaluation,
                             functional.IntegralMomentOfTangentialEvaluation))
