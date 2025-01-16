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

from itertools import chain
import numpy

from FIAT import polynomial_set, jacobi, quadrature_schemes


def index_iterator(shp):
    """Constructs a generator iterating over all indices in
    shp in generalized column-major order  So if shp = (2,2), then we
    construct the sequence (0,0),(0,1),(1,0),(1,1)"""
    return numpy.ndindex(shp)


class Functional(object):
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
                    result[c][i] += w * bfs[i, j]

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
        sd = ref_el.get_spatial_dimension()
        tau = numpy.zeros((sd*(sd+1)//2,))

        alphas = []
        cur = 0
        for i in range(sd):
            for j in range(i, sd):
                alpha = [0] * sd
                alpha[i] += 1
                alpha[j] += 1
                alphas.append(tuple(alpha))
                tau[cur] = s1[i] * s2[j] + (i != j) * s2[i] * s1[j]
                cur += 1

        dpt_dict = {tuple(pt): [(tau[i], alphas[i], comp) for i in range(len(alphas))]}

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
    :arg f_at_qpts: an array tabulating the function f at the quadrature
         points.
    :arg comp: Optional argument indicating that only a particular
         component of the input function should be integrated against f
    :arg shp: Optional argument giving the value shape of input functions.
    """

    def __init__(self, ref_el, Q, f_at_qpts, comp=tuple(), shp=tuple()):
        self.Q = Q
        self.f_at_qpts = f_at_qpts
        self.comp = comp
        points = Q.get_points()
        weights = numpy.multiply(f_at_qpts, Q.get_weights())
        pt_dict = {tuple(pt): [(wt, comp)] for pt, wt in zip(points, weights)}
        super().__init__(ref_el, shp, pt_dict, {}, "IntegralMoment")

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        pts = list(self.pt_dict.keys())
        wts = numpy.array([foo[0][0] for foo in list(self.pt_dict.values())])
        result = numpy.dot([fn(p) for p in pts], wts)

        if self.comp:
            result = result[self.comp]
        return result


class IntegralMomentOfDerivative(Functional):
    """Functional giving directional derivative integrated against some function on a facet."""

    def __init__(self, ref_el, s, Q, f_at_qpts, comp=(), shp=()):
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        points = Q.get_points()
        weights = numpy.multiply(f_at_qpts, Q.get_weights())

        alphas = tuple(map(tuple, numpy.eye(sd, dtype=int)))
        dpt_dict = {tuple(pt): [(wt*s[i], alphas[i], comp) for i in range(sd)]
                    for pt, wt in zip(points, weights)}

        super().__init__(ref_el, shp,
                         {}, dpt_dict, "IntegralMomentOfDerivative")


class IntegralMomentOfNormalDerivative(Functional):
    """Functional giving normal derivative integrated against some function on a facet."""

    def __init__(self, ref_el, facet_no, Q, f_at_qpts):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        # map points onto facet
        transform = ref_el.get_entity_transform(sd-1, facet_no)
        points = transform(Q.get_points())
        self.dpts = points
        weights = numpy.multiply(f_at_qpts, Q.get_weights())

        alphas = tuple(map(tuple, numpy.eye(sd, dtype=int)))
        dpt_dict = {tuple(pt): [(wt*n[i], alphas[i], tuple()) for i in range(sd)]
                    for pt, wt in zip(points, weights)}

        super().__init__(ref_el, tuple(),
                         {}, dpt_dict, "IntegralMomentOfNormalDerivative")


class FrobeniusIntegralMoment(IntegralMoment):

    def __init__(self, ref_el, Q, f_at_qpts, nm=None):
        # f_at_qpts is (some shape) x num_qpts
        shp = tuple(f_at_qpts.shape[:-1])
        if len(Q.pts) != f_at_qpts.shape[-1]:
            raise Exception("Mismatch in number of quadrature points and values")

        self.Q = Q
        self.comp = slice(None, None)
        self.f_at_qpts = f_at_qpts
        qpts, qwts = Q.get_points(), Q.get_weights()
        weights = numpy.transpose(numpy.multiply(f_at_qpts, qwts), (-1,) + tuple(range(len(shp))))
        alphas = list(index_iterator(shp))

        pt_dict = {tuple(pt): [(wt[alpha], alpha) for alpha in alphas] for pt, wt in zip(qpts, weights)}
        Functional.__init__(self, ref_el, shp, pt_dict, {}, nm or "FrobeniusIntegralMoment")


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


class IntegralMomentOfDivergence(Functional):
    """Functional representing integral of the divergence of the input
    against some tabulated function f."""
    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        points = Q.get_points()
        self.dpts = points
        weights = numpy.multiply(f_at_qpts, Q.get_weights())

        alphas = tuple(map(tuple, numpy.eye(sd, dtype=int)))
        dpt_dict = {tuple(pt): [(wt, alphas[i], (i,)) for i in range(sd)]
                    for pt, wt in zip(points, weights)}

        super().__init__(ref_el, tuple(), {}, dpt_dict,
                         "IntegralMomentOfDivergence")


class IntegralMomentOfTensorDivergence(Functional):
    """Like IntegralMomentOfDivergence, but on symmetric tensors."""

    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q
        points = Q.get_points()
        self.dpts = points
        sd = ref_el.get_spatial_dimension()
        shp = (sd, sd)

        assert len(f_at_qpts.shape) == 2
        assert f_at_qpts.shape[0] == sd
        assert f_at_qpts.shape[1] == len(points)
        weights = numpy.multiply(f_at_qpts, Q.get_weights()).T

        alphas = tuple(map(tuple, numpy.eye(sd, dtype=int)))
        dpt_dict = {tuple(pt): [(wt[i], alphas[j], (i, j)) for i, j in index_iterator(shp)]
                    for pt, wt in zip(points, weights)}

        super().__init__(ref_el, tuple(), {}, dpt_dict, "IntegralMomentOfDivergence")


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


class IntegralMomentOfEdgeTangentEvaluation(Functional):
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
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(1, edge)
        points = transform(Q.get_points())
        weights = numpy.multiply(P_at_qpts, Q.get_weights())
        pt_dict = {tuple(pt): [(wt*t[i], (i,)) for i in range(sd)]
                   for pt, wt in zip(points, weights)}
        super().__init__(ref_el, (sd, ), pt_dict, {},
                         "IntegralMomentOfEdgeTangentEvaluation")


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


class IntegralMomentOfFaceTangentEvaluation(Functional):
    r"""
    \int_F v \times n \cdot p ds

    p \in Polynomials

    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        P_at_qpts = [[P_at_qpts[0][i], P_at_qpts[1][i], P_at_qpts[2][i]]
                     for i in range(P_at_qpts.shape[1])]
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(sd-1, facet)
        pts = tuple(map(tuple, transform(Q.get_points())))
        weights = Q.get_weights()
        pt_dict = {}
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            phixn = [phi[1]*n[2] - phi[2]*n[1],
                     phi[2]*n[0] - phi[0]*n[2],
                     phi[0]*n[1] - phi[1]*n[0]]
            pt_dict[pt] = [(wgt*(-n[2]*phixn[1]+n[1]*phixn[2]), (0, )),
                           (wgt*(n[2]*phixn[0]-n[0]*phixn[2]), (1, )),
                           (wgt*(-n[1]*phixn[0]+n[0]*phixn[1]), (2, ))]
        super().__init__(ref_el, (sd, ), pt_dict, {},
                         "IntegralMomentOfFaceTangentEvaluation")


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


class IntegralMomentOfScaledNormalEvaluation(Functional):
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
        transform = ref_el.get_entity_transform(sd - 1, facet)
        pts = transform(Q.get_points())
        weights = Q.get_weights() * P_at_qpts
        pt_dict = {tuple(pt): [(wt*n[i], (i, )) for i in range(sd)]
                   for pt, wt in zip(pts, weights)}
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfScaledNormalEvaluation")


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

        pt_dict = {tuple(pt): [(wvT[idx], idx) for idx in index_iterator(shp)]}

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


class IntegralMomentOfNormalEvaluation(Functional):
    r"""
    \int_F v\cdot n p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        # scaling on the normal is ok because edge length then weights
        # the reference element quadrature appropriately
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(sd - 1, facet)
        pts = transform(Q.get_points())
        weights = numpy.multiply(P_at_qpts, Q.get_weights())
        pt_dict = {tuple(pt): [(wt*n[i], (i, )) for i in range(sd)]
                   for pt, wt in zip(pts, weights)}
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfNormalEvaluation")


class IntegralMomentOfTangentialEvaluation(Functional):
    r"""
    \int_F v\cdot n p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        # scaling on the tangent is ok because edge length then weights
        # the reference element quadrature appropriately
        sd = ref_el.get_spatial_dimension()
        assert sd == 2
        t = ref_el.compute_edge_tangent(facet)
        transform = ref_el.get_entity_transform(sd - 1, facet)
        points = transform(Q.get_points())
        weights = numpy.multiply(P_at_qpts, Q.get_weights())
        pt_dict = {tuple(pt): [(wt*t[i], (i, )) for i in range(sd)]
                   for pt, wt in zip(points, weights)}
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfScaledTangentialEvaluation")
