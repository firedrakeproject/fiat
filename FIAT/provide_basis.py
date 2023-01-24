# Copyright (C) 2019 Cyrus Cheng (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2019

import numbers
import sympy
from sympy import symbols, legendre, Array, diff, lambdify
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT.lagrange import Lagrange
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis
from FIAT.reference_element import (compute_unflattening_map,
                                    flatten_reference_cube)
from FIAT.reference_element import make_lattice

from FIAT.pointwise_dual import compute_pointwise_dual
from FIAT.serendipity import tr, _replace_numbers_with_symbols

x, y, z = symbols('x y z')
variables = (x, y, z)
leg = legendre


class Provide_Basis(FiniteElement):

    # basis is a dictionary from entity to basis functions, with a set of unisolvent points provided under the key pts
    def __init__(self, ref_el, degree, basis):

        flat_el = flatten_reference_cube(ref_el) #something needed here
        dim = flat_el.get_spatial_dimension()
        flat_topology = flat_el.get_topology()

        verts = flat_el.get_vertices()

        for i in range(4):
            try:
                x = basis[i]
            except KeyError:
                basis[i] = []

        VL = basis[0]
        EL = basis[1]
        FL = basis[2]
        IL = basis[3]
        s_list = []
        entity_ids = {}
        cur = 0

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        for j in sorted(flat_topology[0]):
            entity_ids[0][j] = [cur]
            cur = cur + 1

        for j in sorted(flat_topology[1]):
            entity_ids[1][j] = list(range(cur, cur + degree - 1))
            cur = cur + degree - 1

        for j in sorted(flat_topology[2]):
            entity_ids[2][j] = list(range(cur, cur + tr(degree)))
            cur = cur + tr(degree)

        if dim == 3:
            entity_ids[3] = {}
            entity_ids[3][0] = list(range(cur, cur + len(IL)))
            cur = cur + len(IL)

        s_list = VL + EL + FL + IL
        assert len(s_list) == cur
        formdegree = 0

        super(Provide_Basis, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree)

        self.basis = {(0,)*dim: Array(s_list)}
        polynomials, extra_vars = _replace_numbers_with_symbols(Array(s_list))
        self.basis_callable = {(0,)*dim: [lambdify(variables[:dim], polynomials,
                                                   modules="numpy", dummify=True),
                                          extra_vars]}
        topology = ref_el.get_topology()
        unflattening_map = compute_unflattening_map(topology)
        unflattened_entity_ids = {}
        unflattened_entity_closure_ids = {}

        entity_closure_ids = make_entity_closure_ids(flat_el, entity_ids)

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

        self.dual = compute_pointwise_dual(self, basis["pts"])

    def degree(self):
        return self._degree + 1

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for serendipity")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for serendipity")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for serendipity")

    def tabulate(self, order, points, entity=None):

        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        phivals = {}
        dim = self.flat_el.get_spatial_dimension()
        if dim <= 1:
            raise NotImplementedError('no tabulate method for basis provided elements of dimension 1 or less.')
        if dim >= 4:
            raise NotImplementedError('tabulate does not support higher dimensions than 3.')
        points = np.asarray(points)
        npoints, pointdim = points.shape
        for o in range(order + 1):
            alphas = mis(dim, o)
            for alpha in alphas:
                try:
                    callable, extra_vars = self.basis_callable[alpha]
                except KeyError:
                    polynomials = diff(self.basis[(0,)*dim], *zip(variables, alpha))
                    polynomials, extra_vars = _replace_numbers_with_symbols(polynomials)
                    callable = lambdify(variables[:dim] + tuple(extra_vars.values()), polynomials, modules="numpy", dummify=True)
                    self.basis[alpha] = polynomials
                    self.basis_callable[alpha] = [callable, extra_vars]
                # Can no longer make a numpy array from objects of inhomogeneous shape
                # (unless we specify `dtype==object`);
                # see https://github.com/firedrakeproject/fiat/pull/32.
                #
                # Casting `key`s to float() is needed, otherwise we somehow get the following error:
                #
                # E           TypeError: unsupported type for persistent hash keying: <class 'complex'>
                #
                # ../../lib/python3.8/site-packages/pytools/persistent_dict.py:243: TypeError
                #
                # `key`s have been checked to be numbers.Real.
                extra_arrays = [np.ones((npoints, ), dtype=points.dtype) * float(key) for key in extra_vars]
                phivals[alpha] = callable(*([points[:, i] for i in range(pointdim)] + extra_arrays))
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
        return ()

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError

    def space_dimension(self):
        return len(self.basis[(0,)*self.flat_el.get_spatial_dimension()])