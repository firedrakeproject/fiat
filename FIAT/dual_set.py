# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same at Baylor University.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy

from FIAT import polynomial_set, functional


class DualSet(object):
    def __init__(self, nodes, ref_el, entity_ids, entity_permutations=None):
        nodes, ref_el, entity_ids, entity_permutations = merge_entities(nodes, ref_el, entity_ids, entity_permutations)
        self.nodes = nodes
        self.ref_el = ref_el
        self.entity_ids = entity_ids
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
        return len(self.nodes)

    def get_nodes(self):
        return self.nodes

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

        pts = set()
        dpts = set()
        Qs_to_ells = dict()
        for i, ell in enumerate(self.nodes):
            if len(ell.deriv_dict) > 0:
                dpts.update(ell.deriv_dict.keys())
                continue
            if isinstance(ell, functional.IntegralMoment):
                Q = ell.Q
            else:
                Q = None
                pts.update(ell.pt_dict.keys())
            if Q in Qs_to_ells:
                Qs_to_ells[Q].append(i)
            else:
                Qs_to_ells[Q] = [i]

        Qs_to_pts = {}
        if len(pts) > 0:
            Qs_to_pts[None] = tuple(sorted(pts))
        for Q in Qs_to_ells:
            if Q is not None:
                cur_pts = tuple(map(tuple, Q.pts))
                Qs_to_pts[Q] = cur_pts
                pts.update(cur_pts)

        # Now tabulate the function values
        pts = list(sorted(pts))
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
            dpts = list(sorted(dpts))
            # It's easiest/most efficient to get derivatives of the
            # expansion set through the polynomial set interface.
            # This is creating a short-lived set to do just this.
            coeffs = numpy.eye(num_exp)
            expansion = polynomial_set.PolynomialSet(self.ref_el, ed, ed, es, coeffs)
            dexpansion_values = expansion.tabulate(dpts, max_deriv_order)

            ells = [k for k, ell in enumerate(self.nodes) if len(ell.deriv_dict) > 0]
            wshape = (len(ells), *tshape, len(dpts))
            dwts = {alpha: numpy.zeros(wshape, "d") for alpha in dexpansion_values if sum(alpha) > 0}
            for i, k in enumerate(ells):
                ell = self.nodes[k]
                for pt, wac_list in ell.deriv_dict.items():
                    j = dpts.index(pt)
                    for (w, alpha, c) in wac_list:
                        dwts[alpha][i][c][j] = w
            for alpha in dwts:
                mat[ells] += numpy.dot(dwts[alpha], dexpansion_values[alpha].T)
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
    """Collect DOFs from simplicial complex onto facets of parent cell."""
    parent_cell = ref_el.get_parent()
    if parent_cell is None:
        return nodes, ref_el, entity_ids, entity_permutations
    parent_ids = {}
    parent_permutations = None
    parent_to_children = ref_el.get_parent_to_children()

    if all(isinstance(node, functional.PointEvaluation) for node in nodes):
        # Merge Lagrange dual with lexicographical reordering
        parent_nodes = []
        for dim in sorted(parent_to_children):
            parent_ids[dim] = {}
            for entity in sorted(parent_to_children[dim]):
                cur = len(parent_nodes)
                for child_dim, child_entity in parent_to_children[dim][entity]:
                    parent_nodes.extend(nodes[i] for i in entity_ids[child_dim][child_entity])
                ids = lexsort_nodes(parent_cell, parent_nodes[cur:], entity=(dim, entity), offset=cur)
                parent_ids[dim][entity] = ids
    else:
        # Merge everything else with the same node ordering
        parent_nodes = nodes
        for dim in sorted(parent_to_children):
            parent_ids[dim] = {}
            for entity in sorted(parent_to_children[dim]):
                parent_ids[dim][entity] = []
                for child_dim, child_entity in parent_to_children[dim][entity]:
                    parent_ids[dim][entity].extend(entity_ids[child_dim][child_entity])

    return parent_nodes, parent_cell, parent_ids, parent_permutations
