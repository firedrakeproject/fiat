# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same from Baylor University
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# functionals require:
# - a degree of accuracy (-1 indicates that it works for all functions
#   such as point evaluation)
# - a reference element domain
# - type information

from itertools import chain, permutations, repeat
from collections import defaultdict
from functools import cached_property
import numpy

from FIAT import polynomial_set, jacobi, quadrature, quadrature_schemes


class Functional:
    r"""Abstract class representing a linear functional.
    All FIAT functionals are discrete in the sense that
    they are written as a weighted sum of (derivatives of components of) their
    argument evaluated at particular points.

    :arg ref_el: a :class:`Cell`
    :arg target_shape: a tuple indicating the value shape of functions on
         the functional operates (e.g. if the function eats 2-vectors
         then target_shape is (2,) and if it eats scalars then
         target_shape is ()
    :arg pt_dict: A dict mapping points to lists of information about
         how the functional is evaluated.  Each entry in the list takes
         the form of a tuple (wt, comp) so that (at least if the
         deriv_dict argument is empty), the functional takes the form
         :math:`\ell(f) = \sum_{q=1}^{N_q} \sum_{k=1}^{K_q} w^q_k f_{c_k}(x_q)`
         where :math:`f_{c_k}` indicates a particular vector or tensor component
    :arg deriv_dict: A dict that is similar to `pt_dict`, although the entries
         of each list are tuples (wt, alpha, comp) with alpha a tuple
         of nonnegative integers corresponding to the order of partial
         differentiation in each spatial direction.
    :arg functional_type: a string labeling the kind of functional
         this is.
    """
    def __init__(self, ref_el, target_shape, pt_dict, deriv_dict,
                 functional_type):
        self.ref_el = ref_el
        self.target_shape = target_shape
        self.pt_dict = pt_dict
        self.deriv_dict = deriv_dict
        self.functional_type = functional_type
        if len(deriv_dict) > 0:
            self.max_deriv_order = max(sum(wac[1]) for wac in chain(*deriv_dict.values()))
        else:
            self.max_deriv_order = 0

    @cached_property
    def _order_tensors(self):
        r"""Represent the functional using points and weights for each order.

        For derivative order :math:`k`, the weights :math:`W^k` satisfy
        :math:`\ell(f) = \sum_k \sum_q W^k_q : D^k f(x_q)`. The axes of
        :math:`W^k` are the value axes of ``target_shape``, then :math:`k`
        symmetric derivative axes, then the point axis.
        """
        sd = self.ref_el.get_spatial_dimension()
        order_dict = defaultdict(lambda: defaultdict(list))
        for pt, wc_list in self.pt_dict.items():
            order_dict[0][pt].extend((w, (0,) * sd, c) for w, c in wc_list)
        for pt, wac_list in self.deriv_dict.items():
            for w, alpha, c in wac_list:
                order_dict[sum(alpha)][pt].append((w, alpha, c))

        order_tensors = {}
        for order, pt_dict in sorted(order_dict.items()):
            W = numpy.zeros((*self.target_shape, *(sd,) * order, len(pt_dict)))
            for q, wac_list in enumerate(pt_dict.values()):
                for w, alpha, c in wac_list:
                    c = c if isinstance(c, tuple) else (c,)
                    indices = set(permutations(sum(([k] * a for k, a in enumerate(alpha)), [])))
                    for index in indices:
                        W[(*c, *index, q)] += w / len(indices)
            pts = tuple(pt_dict)
            order_tensors[order] = (pts, W)
        return order_tensors

    @cached_property
    def orders(self):
        """Derivative orders encoded by this functional.

        Returns
        -------
        tuple of int
            Orders present in the functional's point and derivative data.
        """
        return tuple(sorted(self._order_tensors))

    def get_order_tensors(self, order):
        """Return the points and weights for one derivative order.

        Parameters
        ----------
        order : int
            Derivative order of the requested tensors.

        Returns
        -------
        points : tuple of tuple of float
            Points at which the functional samples derivatives of ``order``.
        weights : numpy.ndarray
            Weights are indexed first by component, then by spatial
            direction for each derivative, and finally by sampling point.

        Raises
        ------
        KeyError
            If this functional has no terms of the requested order.
        """
        return self._order_tensors[order]

    def evaluate(self, f):
        """Obsolete and broken functional evaluation.

        To evaluate the functional, call it on the target function:

          functional(function)
        """
        raise AttributeError("To evaluate the functional just call it on a function.")

    def __call__(self, fn):
        raise NotImplementedError("Evaluation is not yet implemented for %s" % type(self))

    def get_point_dict(self):
        """Returns the functional information, which is a dictionary
        mapping each point in the support of the functional to a list
        of pairs containing the weight and component."""
        return self.pt_dict

    def get_reference_element(self):
        """Returns the reference element."""
        return self.ref_el

    def get_type_tag(self):
        """Returns the type of function (e.g. point evaluation or
        normal component, which is probably handy for clients of FIAT"""
        return self.functional_type

    def to_riesz(self, poly_set):
        r"""Constructs an array representation of the functional so
        that the functional may be applied to a function expressed in
        in terms of the expansion set underlying  `poly_set` by means
        of contracting coefficients.

        That is, `poly_set` will have members all expressed in the
        form :math:`p = \sum_{i} \alpha^i \phi_i`
        where :math:`\{\phi_i\}_{i}` is some orthonormal expansion set
        and :math:`\alpha^i` are coefficients.  Note: the orthonormal
        expansion set is always scalar-valued but if the members of
        `poly_set` are vector or tensor valued the :math:`\alpha^i`
        will be scalars or vectors.

        This function constructs a tensor :math:`R` such that the
        contraction of :math:`R` with the array of coefficients
        :math:`\alpha` produces the effect of :math:`\ell(f)`

        In the case of scalar-value functions, :math:`R` is just a
        vector of the same length as the expansion set, and
        :math:`R_i = \ell(\phi_i)`.  For vector-valued spaces,
        :math:`R_{ij}` will be :math:`\ell(e^i \phi_j)` where
        :math:`e^i` is the canonical unit vector nonzero only in one
        entry :math:`i`.
        """
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        nexp = es.get_num_members(ed)

        pt_dict = self.get_point_dict()

        pts = list(pt_dict.keys())
        npts = len(pts)

        bfs = es.tabulate(ed, pts)
        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")

        # loop over points
        for j in range(npts):
            pt_cur = pts[j]
            wc_list = pt_dict[pt_cur]

            # loop over expansion functions
            for i in range(nexp):
                for (w, c) in wc_list:
                    c = c if isinstance(c, tuple) else (c,)
                    result[(*c, i)] += w * bfs[i, j][c]

        if self.deriv_dict:
            dpt_dict = self.deriv_dict

            # this makes things quicker since it uses dmats after
            # instantiation
            es_foo = polynomial_set.ONPolynomialSet(self.ref_el, ed)
            dpts = list(dpt_dict.keys())

            dbfs = es_foo.tabulate(dpts, self.max_deriv_order)

            ndpts = len(dpts)
            for j in range(ndpts):
                dpt_cur = dpts[j]
                wac_list = dpt_dict[dpt_cur]
                for i in range(nexp):
                    for (w, alpha, c) in wac_list:
                        result[c][i] += w * dbfs[tuple(alpha)][i, j]

        return result

    def tostr(self):
        return self.functional_type


class PointEvaluation(Functional):
    """Class representing point evaluation of scalar functions at a
    particular point x."""

    def __init__(self, ref_el, x):
        pt_dict = {tuple(x): [(1.0, tuple())]}
        super().__init__(ref_el, tuple(), pt_dict, {}, "PointEval")

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        return fn(tuple(self.pt_dict.keys())[0])

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "u(%s)" % (','.join(x),)


class ComponentPointEvaluation(Functional):
    """Class representing point evaluation of a particular component
    of a vector/tensor function at a particular point x."""

    def __init__(self, ref_el, comp, shp, x):
        if not isinstance(comp, tuple):
            comp = (comp,)
        if len(shp) != len(comp):
            raise ValueError("Component and shape are incompatible")
        if any(i < 0 or i >= n for i, n in zip(comp, shp)):
            raise ValueError("Illegal component")
        self.comp = comp
        pt_dict = {tuple(x): [(1.0, comp)]}
        super().__init__(ref_el, shp, pt_dict, {}, "ComponentPointEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u[%d](%s)" % (self.comp, ','.join(x))


class PointDerivative(Functional):
    """Class representing point partial differentiation of scalar
    functions at a particular point x."""

    def __init__(self, ref_el, x, alpha):
        dpt_dict = {x: [(1.0, tuple(alpha), tuple())]}
        self.alpha = tuple(alpha)
        self.order = sum(self.alpha)

        super().__init__(ref_el, tuple(), {}, dpt_dict, "PointDeriv")

    def __call__(self, fn):
        """Evaluate the functional on the function fn. Note that this depends
        on sympy being able to differentiate fn."""
        import sympy
        x, = self.deriv_dict

        X = tuple(sympy.Symbol(f"X[{i}]") for i in range(len(x)))

        dvars = tuple(d for d, a in zip(X, self.alpha)
                      for count in range(a))

        df = sympy.lambdify(X, sympy.diff(fn(X), *dvars))
        return df(*x)


class PointDirectionalDerivative(Functional):
    """Represents d/ds at a point."""
    def __init__(self, ref_el, s, pt, comp=(), shp=(), nm=None):
        sd = ref_el.get_spatial_dimension()
        alphas = tuple(map(tuple, numpy.eye(sd, dtype=int)))
        dpt_dict = {pt: [(s[i], tuple(alphas[i]), comp) for i in range(sd)]}

        super().__init__(ref_el, shp, {}, dpt_dict, nm or "PointDirectionalDeriv")


class PointNormalDerivative(PointDirectionalDerivative):
    """Represents d/dn at a point on a facet."""
    def __init__(self, ref_el, facet_no, pt, comp=(), shp=()):
        n = ref_el.compute_normal(facet_no)
        super().__init__(ref_el, n, pt, comp=comp, shp=shp, nm="PointNormalDeriv")


class PointTangentialDerivative(PointDirectionalDerivative):
    """Represents d/dt at a point on an edge."""
    def __init__(self, ref_el, edge_no, pt, comp=(), shp=()):
        t = ref_el.compute_edge_tangent(edge_no)
        super().__init__(ref_el, t, pt, comp=comp, shp=shp, nm="PointTangentialDeriv")


class PointSecondDerivative(Functional):
    """Represents d/ds1 d/ds2 at a point."""
    def __init__(self, ref_el, s1, s2, pt, comp=(), shp=(), nm=None):
        S = numpy.outer(s1, s2)
        sd = ref_el.get_spatial_dimension()
        tau = defaultdict(float)
        for index in numpy.ndindex(S.shape):
            alpha = [0] * sd
            for i in index:
                alpha[i] += 1
            tau[tuple(alpha)] += S[index]

        dpt_dict = {tuple(pt): [(tau[alpha], alpha, comp) for alpha in tau]}

        super().__init__(ref_el, shp, {}, dpt_dict, nm or "PointSecondDeriv")


class PointNormalSecondDerivative(PointSecondDerivative):
    """Represents d^/dn^2 at a point on a facet."""
    def __init__(self, ref_el, facet_no, pt, comp=(), shp=()):
        n = ref_el.compute_normal(facet_no)
        super().__init__(ref_el, n, n, pt, comp=comp, shp=shp, nm="PointNormalSecondDeriv")


class PointTangentialSecondDerivative(PointSecondDerivative):
    """Represents d^/dt^2 at a point on an edge."""
    def __init__(self, ref_el, edge_no, pt, comp=(), shp=()):
        t = ref_el.compute_edge_tangent(edge_no)
        super().__init__(ref_el, t, t, pt, comp=comp, shp=shp, nm="PointTangentialSecondDeriv")


class PointDivergence(Functional):
    """Class representing point divergence of vector
    functions at a particular point x."""

    def __init__(self, ref_el, x):
        sd = ref_el.get_spatial_dimension()
        alphas = tuple(map(tuple, numpy.eye(sd, dtype=int)))
        dpt_dict = {x: [(1.0, alpha, (alpha.index(1),)) for alpha in alphas]}

        super().__init__(ref_el, (len(x),), {}, dpt_dict, "PointDiv")


class IntegralMoment(Functional):
    """Functional representing integral of the input against some tabulated function f.

    :arg ref_el: a :class:`Cell`.
    :arg Q: a :class:`QuadratureRule`.
    :arg f_at_qpts: values of the coefficient at the quadrature points. A
         one-dimensional array gives a scalar coefficient; higher-dimensional
         arrays use their leading axes as the value shape and are paired with
         all components of the input.
    :arg comp: Optional argument indicating that only a particular
         component of the input function should be integrated against f
    :arg shp: Optional argument giving the value shape of input functions.
    """

    def __init__(self, ref_el, Q, f_at_qpts, comp=(), shp=(), nm=""):
        f_at_qpts = numpy.asarray(f_at_qpts)
        points = Q.get_points()
        if f_at_qpts.ndim == 0 or f_at_qpts.shape[-1] != len(points):
            raise ValueError("Mismatch in number of quadrature points and values")

        coefficient_shape = tuple(f_at_qpts.shape[:-1])
        target_shape = tuple(shp) or coefficient_shape
        if coefficient_shape and coefficient_shape != target_shape:
            raise ValueError("Coefficient shape does not match the input value shape")

        if coefficient_shape or comp == ():
            comp = (slice(None),) * len(target_shape)
        weights = numpy.moveaxis(numpy.multiply(f_at_qpts, Q.get_weights()), -1, 0)
        pt_dict = {tuple(pt): [(wt, comp)] for pt, wt in zip(points, weights)}

        self.f_at_qpts = f_at_qpts
        self.comp = comp
        self.Q = Q
        super().__init__(ref_el, target_shape, pt_dict, {}, nm or "IntegralMoment")

    @cached_property
    def _order_tensors(self):
        """Represent this moment using points and weights for each order."""
        sd = self.ref_el.get_spatial_dimension()
        order = self.max_deriv_order
        W = numpy.zeros((*self.target_shape, *(sd,) * order, len(self.Q.pts)))
        W[self.comp] = numpy.multiply(self.f_at_qpts, self.Q.get_weights())
        if order > 1:
            # Only the symmetric part pairs with D^k f.
            rank = len(self.target_shape)
            W = numpy.mean([W.transpose((*range(rank), *axes, W.ndim - 1))
                            for axes in permutations(range(rank, rank + order))], axis=0)
        pts = tuple(self.deriv_dict or self.pt_dict)
        return {order: (pts, W)}

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        if self.deriv_dict:
            return Functional.__call__(self, fn)
        points = tuple(self.pt_dict)
        values = numpy.asarray([fn(pt) for pt in points])[self.comp]
        weights = numpy.multiply(self.f_at_qpts, self.Q.get_weights())
        ndim = weights.ndim
        return numpy.tensordot(weights, values, axes=(range(ndim), (*range(1, ndim), 0)))


class IntegralMomentOfDerivative(IntegralMoment):
    """Functional integrating derivatives against component-shaped coefficients.

    :arg ref_el: a :class:`Cell`.
    :arg Q: a :class:`QuadratureRule`.
    :arg f_at_qpts: Values of f at the quadrature points. Without directions,
         component axes precede derivative axes; with directions, the values
         are treated as scalar coefficients.
    :arg *directions: Optional vectors of directions of differentiation.
    :arg comp: Optional argument indicating that only a particular
         component of the input function should be integrated against f
    :arg shp: Optional argument giving the value shape of input functions.
    """

    def __init__(self, ref_el, Q, f_at_qpts, *directions, comp=(), shp=(), nm=""):
        sd = ref_el.get_spatial_dimension()
        f_at_qpts = numpy.asarray(f_at_qpts)
        for direction in reversed(directions):
            f_at_qpts = numpy.multiply.outer(direction, f_at_qpts)
        coefficient_shape = () if directions else shp

        if comp == ():
            comp = (slice(None),) * len(shp)
        elif not isinstance(comp, tuple):
            comp = (comp,)

        order = f_at_qpts.ndim - len(coefficient_shape) - 1
        expected_shape = coefficient_shape + (sd,) * order + (len(Q.pts),)
        if order < 1 or f_at_qpts.shape != expected_shape:
            raise ValueError("Expected component axes, derivative axes, and quadrature points")

        alphas = polynomial_set.mis(sd, order)
        alpha_rows = {alpha: row for row, alpha in enumerate(alphas)}
        sym_tensor = numpy.zeros((len(alphas), sd**order))
        for column, index in enumerate(numpy.ndindex((sd,) * order)):
            alpha = tuple(map(index.count, range(sd)))
            sym_tensor[alpha_rows[alpha], column] = 1

        weights = numpy.multiply(f_at_qpts, Q.get_weights())
        weights = weights.reshape((*coefficient_shape, -1, len(Q.pts)))
        weights = numpy.tensordot(sym_tensor, weights, axes=(1, len(coefficient_shape)))
        if coefficient_shape and comp != (slice(None),) * len(coefficient_shape):
            weights = weights[(slice(None),) + comp + (slice(None),)]
        self.weights = weights
        dpt_dict = {
            tuple(pt): list(zip(wt, alphas, repeat(comp)))
            for pt, wt in zip(Q.get_points(), numpy.moveaxis(self.weights, -1, 0))
        }

        self.Q = Q
        self.f_at_qpts = f_at_qpts
        self.comp = comp
        Functional.__init__(self, ref_el, shp, {}, dpt_dict, nm or "IntegralMomentOfDerivative")


class IntegralMomentOfNormalDerivative(IntegralMomentOfDerivative):
    """Functional giving normal derivative integrated against some function on a facet."""

    def __init__(self, ref_el, facet_no, Q_face, f_at_qpts):
        n = ref_el.compute_normal(facet_no)
        # map points onto facet
        sd = ref_el.get_spatial_dimension()
        Q = quadrature.FacetQuadratureRule(ref_el, sd-1, facet_no, Q_face, avg=True)
        super().__init__(ref_el, Q, f_at_qpts, n, nm="IntegralMomentOfNormalDerivative")


class FrobeniusIntegralMoment(IntegralMoment):

    def __init__(self, ref_el, Q, f_at_qpts, nm=None):
        shp = tuple(f_at_qpts.shape[:-1])
        super().__init__(ref_el, Q, f_at_qpts, comp=(slice(None),) * len(shp),
                         shp=shp, nm=nm or "FrobeniusIntegralMoment")


class IntegralLegendreDirectionalMoment(FrobeniusIntegralMoment):
    """Moment of v.s against a Legendre polynomial over an edge"""
    def __init__(self, cell, s, entity, mom_deg, quad_deg, nm=""):
        # mom_deg is degree of moment, quad_deg is the total degree of
        # polynomial you might need to integrate (or something like that)
        assert cell.get_spatial_dimension() == 2
        entity = (1, entity)

        Q = quadrature_schemes.create_quadrature(cell, quad_deg, entity=entity)
        x = cell.compute_barycentric_coordinates(Q.get_points(), entity=entity)

        f_at_qpts = jacobi.eval_jacobi(0, 0, mom_deg, x[:, 1] - x[:, 0])
        f_at_qpts /= Q.jacobian_determinant()

        f_at_qpts = numpy.multiply(s[..., None], f_at_qpts)
        super().__init__(cell, Q, f_at_qpts, nm=nm)


class IntegralLegendreNormalMoment(IntegralLegendreDirectionalMoment):
    """Moment of v.n against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_scaled_normal(entity)
        super().__init__(cell, n, entity, mom_deg, comp_deg,
                         "IntegralLegendreNormalMoment")


class IntegralLegendreTangentialMoment(IntegralLegendreDirectionalMoment):
    """Moment of v.t against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        t = cell.compute_edge_tangent(entity)
        super().__init__(cell, t, entity, mom_deg, comp_deg,
                         "IntegralLegendreTangentialMoment")


class IntegralLegendreBidirectionalMoment(IntegralLegendreDirectionalMoment):
    """Moment of dot(s1, dot(tau, s2)) against Legendre on entity, multiplied by the size of the reference facet"""
    def __init__(self, cell, s1, s2, entity, mom_deg, comp_deg, nm=""):
        s1s2T = numpy.outer(s1, s2)
        super().__init__(cell, s1s2T, entity, mom_deg, comp_deg, nm=nm)


class IntegralLegendreNormalNormalMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(n, dot(tau, n)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_scaled_normal(entity)
        super().__init__(cell, n, n, entity, mom_deg, comp_deg,
                         "IntegralNormalNormalLegendreMoment")


class IntegralLegendreNormalTangentialMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(n, dot(tau, t)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_scaled_normal(entity)
        t = cell.compute_edge_tangent(entity)
        super().__init__(cell, n, t, entity, mom_deg, comp_deg,
                         "IntegralNormalTangentialLegendreMoment")


class IntegralLegendreTangentialTangentialMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(t, dot(tau, t)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        t = cell.compute_edge_tangent(entity)
        super().__init__(cell, t, t, entity, mom_deg, comp_deg,
                         "IntegralTangentialTangentialLegendreMoment")


class IntegralMomentOfDivergence(IntegralMomentOfDerivative):
    """Functional representing integral of the divergence of the input
    against some tabulated function f."""
    def __init__(self, ref_el, Q, f_at_qpts):
        sd = ref_el.get_spatial_dimension()
        coefficients = numpy.eye(sd)[..., None] * f_at_qpts
        super().__init__(ref_el, Q, coefficients, shp=(sd,), nm="IntegralMomentOfDivergence")


class IntegralMomentOfTensorDivergence(IntegralMomentOfDerivative):
    """Like IntegralMomentOfDivergence, but on symmetric tensors."""

    def __init__(self, ref_el, Q, f_at_qpts):
        sd = ref_el.get_spatial_dimension()
        coefficients = numpy.einsum("iq,jk->ijkq", f_at_qpts, numpy.eye(sd))
        super().__init__(ref_el, Q, coefficients, shp=(sd, sd), nm="IntegralMomentOfTensorDivergence")


class PointNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1."""

    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        shp = n.shape
        pt_dict = {pt: [(n[i], (i,)) for i in range(shp[0])]}
        super().__init__(ref_el, shp, pt_dict, {}, "PointNormalEval")


class PointEdgeTangentEvaluation(Functional):
    """Implements the evaluation of the tangential component of a
    vector at a point on a facet of dimension 1."""

    def __init__(self, ref_el, edge_no, pt):
        t = ref_el.compute_edge_tangent(edge_no)
        self.t = t
        shp = t.shape
        pt_dict = {pt: [(t[i], (i,)) for i in range(shp[0])]}
        super().__init__(ref_el, shp, pt_dict, {}, "PointEdgeTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t)(%s)" % (','.join(x),)


class IntegralMomentOfEdgeTangentEvaluation(FrobeniusIntegralMoment):
    r"""
    \int_e v\cdot t p ds

    p \in Polynomials

    :arg ref_el: reference element for which e is a dim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg edge: which edge.
    """
    def __init__(self, ref_el, Q, P_at_qpts, edge):
        t = ref_el.compute_edge_tangent(edge)
        Q = quadrature.FacetQuadratureRule(ref_el, 1, edge, Q, avg=True)
        f_at_qpts = t[:, None] * P_at_qpts
        super().__init__(ref_el, Q, f_at_qpts, "IntegralMomentOfEdgeTangentEvaluation")


class PointFaceTangentEvaluation(Functional):
    """Implements the evaluation of a tangential component of a
    vector at a point on a facet of codimension 1."""

    def __init__(self, ref_el, face_no, tno, pt):
        t = ref_el.compute_face_tangents(face_no)[tno]
        self.t = t
        self.tno = tno
        sd = ref_el.get_spatial_dimension()
        pt_dict = {pt: [(t[i], (i,)) for i in range(sd)]}
        shp = (sd,)
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "PointFaceTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t%d)(%s)" % (self.tno, ','.join(x),)


class IntegralMomentOfFaceTangentEvaluation(FrobeniusIntegralMoment):
    r"""
    \int_F v \times n \cdot p ds

    p \in Polynomials

    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        Q = quadrature.FacetQuadratureRule(ref_el, sd - 1, facet, Q, avg=True)
        f_at_qpts = numpy.cross(n, numpy.cross(P_at_qpts.T, n)).T
        super().__init__(ref_el, Q, f_at_qpts, "IntegralMomentOfFaceTangentEvaluation")


class PointScaledNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1, where the normal is scaled by
    the volume of that facet."""

    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_scaled_normal(facet_no)
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        pt_dict = {pt: [(n[i], (i,)) for i in range(sd)]}
        super().__init__(ref_el, shp, pt_dict, {}, "PointScaledNormalEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.n)(%s)" % (','.join(x),)


class IntegralMomentOfScaledNormalEvaluation(FrobeniusIntegralMoment):
    r"""
    \int_F v\cdot n p ds

    p \in Polynomials

    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        Q = quadrature.FacetQuadratureRule(ref_el, sd - 1, facet, Q, avg=True)
        f_at_qpts = n[:, None] * P_at_qpts
        super().__init__(ref_el, Q, f_at_qpts, "IntegralMomentOfScaledNormalEvaluation")


class PointwiseInnerProductEvaluation(Functional):
    """
    This is a functional on symmetric 2-tensor fields. Let u be such a
    field, p be a point, and v,w be vectors. This implements the evaluation
    v^T u(p) w.

    Clearly v^iu_{ij}w^j = u_{ij}v^iw^j. Thus the value can be computed
    from the Frobenius inner product of u with wv^T. This gives the
    correct weights.
    """

    def __init__(self, ref_el, v, w, pt):
        wvT = numpy.outer(w, v)
        shp = wvT.shape

        pt_dict = {tuple(pt): [(wvT[idx], idx) for idx in numpy.ndindex(shp)]}

        super().__init__(ref_el, shp, pt_dict, {}, "PointwiseInnerProductEval")


class TensorBidirectionalIntegralMoment(FrobeniusIntegralMoment):
    r"""
    This is a functional on symmetric 2-tensor fields. Let u be such a
    field, f a function tabulated at points, and v,w be vectors. This implements the evaluation
    \int v^T u(x) w f(x).
    Clearly v^iu_{ij}w^j = u_{ij}v^iw^j. Thus the value can be computed
    from the Frobenius inner product of u with vw^T. This gives the
    correct weights.
    """

    def __init__(self, ref_el, v, w, Q, f_at_qpts):
        vwT = numpy.outer(v, w)
        F_at_qpts = numpy.multiply(vwT[..., None], f_at_qpts)
        super().__init__(ref_el, Q, F_at_qpts, "TensorBidirectionalMomentInnerProductEvaluation")


class IntegralMomentOfNormalEvaluation(FrobeniusIntegralMoment):
    r"""
    \int_F v\cdot n p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        Q = quadrature.FacetQuadratureRule(ref_el, sd - 1, facet, Q, avg=True)
        f_at_qpts = n[:, None] * P_at_qpts
        super().__init__(ref_el, Q, f_at_qpts, "IntegralMomentOfNormalEvaluation")


class IntegralMomentOfTangentialEvaluation(FrobeniusIntegralMoment):
    r"""
    \int_F v\cdot n p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        sd = ref_el.get_spatial_dimension()
        assert sd == 2
        t = ref_el.compute_edge_tangent(facet)
        Q = quadrature.FacetQuadratureRule(ref_el, sd - 1, facet, Q, avg=True)
        f_at_qpts = t[:, None] * P_at_qpts
        super().__init__(ref_el, Q, f_at_qpts, "IntegralMomentOfScaledTangentialEvaluation")
