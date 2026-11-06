# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same at Baylor University.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from itertools import chain
from collections import defaultdict

from FIAT import functional
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
    mapping : sequence of str or None, optional
        The mapping of the test function of each functional in ``nodes``,
        with None where it is not declared.

    Raises
    ------
    ValueError
        If ``coeffs`` does not have one column per functional, or
        ``mapping`` does not have one entry per functional.
    """

    def __init__(self, nodes, ref_el, entity_ids, entity_permutations=None,
                 coeffs=None, mapping=None):
        if ref_el.get_dimension() != max(entity_ids):
            entity_ids = unflatten_entity_ids(ref_el, entity_ids)
        if mapping is not None and len(mapping) != len(nodes):
            raise ValueError("The mapping must have one entry per functional")
        # The merge may reorder the nodes, so follow each node with its mapping
        node_mappings = dict(zip(nodes, mapping or ()))
        nodes, ref_el, entity_ids, entity_permutations, macro_entity_ids = merge_entities(
            nodes, ref_el, entity_ids, entity_permutations)
        if mapping is not None:
            mapping = [node_mappings[node] for node in nodes]
        if coeffs is not None:
            coeffs = numpy.asarray(coeffs)
            if coeffs.ndim != 2 or coeffs.shape[1] != len(nodes):
                raise ValueError("Dual coefficients must have one column per functional")

        self.nodes = nodes
        self.coeffs = coeffs
        self.mapping = mapping
        self.ref_el = ref_el
        self.entity_ids = entity_ids
        # The entity ids on the simplicial complex, numbered by the merged node ordering
        self._macro_entity_ids = macro_entity_ids
        self.entity_permutations = entity_permutations

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

    def get_mapping(self):
        """Return the mapping of the test function of each node, or None if it is not declared."""
        return self.mapping

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
                       self.entity_permutations, coeffs=coefficients, mapping=self.mapping)

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
        tshape = self.nodes[0].target_shape
        sd = self.ref_el.get_spatial_dimension()
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        num_exp = es.get_num_members(ed)
        points = defaultdict(dict)
        weights = defaultdict(dict)
        for node_id, node in enumerate(self.nodes):
            for order in node.orders:
                pts, wts = node.get_order_tensors(order)
                points[order][node_id] = pts
                weights[order][node_id] = wts

        # Tabulate the expansion set one time for all the values and another
        # for all the derivatives, and contract nodes with the same point set
        # in a single product.
        mat = numpy.zeros((len(self.nodes), *tshape, num_exp))
        all_orders = set(points)
        for orders in (all_orders & {0}, all_orders - {0}):
            if not orders:
                continue
            pts = sorted(set(chain.from_iterable(
                node_points for order in orders for node_points in points[order].values())))
            point_index = {pt: j for j, pt in enumerate(pts)}
            tabulation = es._tabulate(ed, pts, max(orders))
            for order in sorted(orders):
                # D^order phi with axes (j, gamma, point)
                dshape = (sd,) * order
                dphi = numpy.zeros((num_exp, *dshape, len(pts)))
                for index in numpy.ndindex(dshape):
                    dphi[:, *index] = tabulation[tuple(map(index.count, range(sd)))]

                point_groups = defaultdict(list)
                for node_id, node_points in points[order].items():
                    point_groups[node_points].append(node_id)
                for node_points, node_ids in point_groups.items():
                    wts = numpy.stack([weights[order][node_id] for node_id in node_ids])
                    indices = [point_index[pt] for pt in node_points]
                    # Contract the derivative axes and the points
                    axes = range(-order - 1, 0)
                    mat[node_ids] += numpy.tensordot(wts, dphi[..., indices], axes=(axes, axes))
        coeffs = self.get_coeffs()
        if coeffs is not None:
            mat = numpy.tensordot(coeffs, mat, axes=(1, 0))
        return mat

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
