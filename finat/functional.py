r"""Degrees of freedom as coefficient tensors.

A FIAT functional built from point and derivative dictionaries is
:math:`\ell(f) = \sum_q \langle W_q, \nabla^m f(x_q) \rangle`, with one
axis of :math:`W_q` per value component and one per derivative.
"""

from functools import reduce
from itertools import permutations, product
from math import prod
from operator import mul

import gem
import numpy

#: The mapping of a derivative axis of the coefficient tensor.
DERIVATIVE = "derivative"

#: The mapping of a divergence axis of the coefficient tensor.
DIVERGENCE = "divergence"

zero = gem.Zero()
one = gem.Literal(1.0)


def split_axis_mappings(mapping, rank):
    """Splits a FIAT mapping into the mapping of each component axis.

    :arg mapping: The FIAT mapping of the basis functions the functional
        acts on, e.g. ``"affine"`` or ``"double contravariant piola"``.
    :arg rank: The number of component axes of the functional.
    :returns: A tuple with the mapping of each component axis, e.g.
        ``("contravariant piola", "contravariant piola")``.
    """
    words = mapping.split()
    if words == ["affine"]:
        return (mapping,) * rank
    if words[0] == "double":
        kinds = (f"{words[1]} piola",) * 2
    else:
        kinds = tuple(f"{word} piola" for word in words[:-1])
    if len(kinds) != rank:
        raise ValueError(f"A {mapping} functional must have rank {len(kinds)}, not {rank}.")
    return kinds


def as_gem(v):
    """Converts a number to a GEM constant, preserving structural zeros."""
    if v == 0:
        return zero
    if v == 1:
        return one
    return gem.Literal(v)


def as_gem_array(A):
    """Converts a numeric array to an object array of GEM constants."""
    A = numpy.asarray(A)
    out = numpy.full(A.shape, zero, dtype=object)
    for index in numpy.ndindex(A.shape):
        out[index] = as_gem(float(A[index]))
    return out


def support_entity(ref_el, points, tol=1e-12):
    """The smallest entity whose closure contains all the points.

    :arg ref_el: The reference cell.
    :arg points: The points.
    :arg tol: Tolerance below which a barycentric coordinate vanishes.
    :returns: The pair ``(dim, entity)``.
    """
    bary = ref_el.compute_barycentric_coordinates(numpy.asarray(points))
    verts = set(numpy.flatnonzero(numpy.abs(bary).max(axis=0) > tol))
    top = ref_el.get_topology()
    for dim in sorted(top):
        for entity in sorted(top[dim]):
            if verts <= set(top[dim][entity]):
                return dim, entity


class DirectionPullback:
    r"""The pullback to the reference cell of one axis of a physical node.

    A physical node differs from its reference node only in the
    directions along each axis of its coefficient tensor, e.g. the
    physical facet normal replaces the reference one, and the pullback
    of the basis functions acts on the same axis by the chain rule or a
    Piola map.  Together they contract that axis with the matrix
    :math:`G = P + \sum_t g_t C_t / d`, where the numeric projector
    :math:`P` keeps the directions that pull back to themselves, and the
    :math:`C_t` are numeric matrices with GEM coefficients :math:`g_t`.

    :arg invariant: The projector :math:`P`, a numeric square array.
    :arg terms: A list of pairs ``(g, C)``.
    :arg denominator: :math:`d`, a GEM scalar.
    """
    def __init__(self, invariant, terms, denominator):
        self.invariant = numpy.asarray(invariant)
        self.terms = list(terms)
        self.denominator = denominator

    def numerator_terms(self):
        """The terms of :math:`Gd`, the invariant projector included."""
        terms = list(self.terms)
        if self.invariant.any():
            terms.insert(0, (self.denominator, self.invariant))
        return terms


def identity_pullback(extent):
    """The pullback of an axis whose physical directions pull back to themselves."""
    return DirectionPullback(numpy.eye(extent), [], one)


def cartesian_pullback(mapping, jacobian):
    """The pullback of an axis keeping its Cartesian directions.

    :arg mapping: The mapping of the axis.
    :arg jacobian: The :class:`Jacobian`.
    :returns: The :class:`DirectionPullback`.
    """
    sd = jacobian.J.shape[0]
    if mapping == "affine":
        return identity_pullback(sd)
    elif mapping == DIVERGENCE:
        return DirectionPullback(numpy.zeros((1, 1)), [(one, numpy.ones((1, 1)))], jacobian.detJ)
    elif mapping == "contravariant piola":
        Q = jacobian.J.T
    elif mapping in {"covariant piola", DERIVATIVE}:
        Q = jacobian.adjJ
    else:
        raise NotImplementedError(f"No pullback for an axis with {mapping} mapping.")
    terms = []
    for a, b in numpy.ndindex(sd, sd):
        E = numpy.zeros((sd, sd))
        E[a, b] = 1
        terms.append((Q[a, b], E))
    return DirectionPullback(numpy.zeros((sd, sd)), terms, jacobian.detJ)


class PhysicalEntityFrame:
    r"""The physical directions of a node on a facet.

    Tangents of the support entity of the node pull back to themselves.
    Each unit normal, to the facet and to an edge within a face, pulls
    back to a GEM vector expanded in an orthonormal reference frame, so
    that the tabulation is only ever contracted with numeric directions.

    :arg ref_el: The reference cell.
    :arg facet: The facet number.
    :arg support: The ``(dim, entity)`` of the smallest entity whose
        closure contains the points of the node.
    :arg jacobian: The :class:`Jacobian`.
    """
    def __init__(self, ref_el, facet, support, jacobian):
        self.ref_el = ref_el
        self.facet = facet
        self.jacobian = jacobian
        sd = ref_el.get_spatial_dimension()
        J, adjJ, detJ, K = jacobian.J, jacobian.adjJ, jacobian.detJ, jacobian.K
        # orthonormal tangents of the facet and of the support entity
        tangents = ref_el.compute_tangents(sd - 1, facet)
        self.facet_tangents = list(numpy.linalg.qr(tangents.T)[0].T)
        n = ref_el.compute_normal(facet)
        n = n / numpy.linalg.norm(n)
        Kn = K @ n
        # unit normals as (nhat, u, d, frame), with J^{-1} nu = u / d
        # expanded in the reference directions frame
        self.normals = []
        if support[0] == sd - 2 >= 1:
            t = ref_el.compute_edge_tangent(support[1])
            m = numpy.cross(t, n)
            m = m / numpy.linalg.norm(m)
            self.tangents = [t / numpy.linalg.norm(t)]
            # the normal to the edge within the face is Jw/|Jw| with
            # w = (J^T J t) x n, since Ja x Kb = J((J^T J)a x b)
            w = numpy.cross(J.T @ (J @ t), n)
            Jw = J @ w
            self.normals.append((m, w, (Jw @ Jw)**0.5, self.tangents + [m]))
        else:
            self.tangents = self.facet_tangents
        frame = self.tangents + [m for m, *_ in self.normals] + [n]
        # the facet normal is Kn/|Kn|, and J^{-1} = adj(J) / detJ
        self.normals.append((n, adjJ @ Kn, detJ * (Kn @ Kn)**0.5, frame))

    def pullback(self, mapping):
        """The :class:`DirectionPullback` of an axis with the given mapping."""
        if mapping == "affine":
            return identity_pullback(self.ref_el.get_spatial_dimension())
        elif mapping == "contravariant piola":
            return self.contravariant_pullback()
        elif mapping in {"covariant piola", DERIVATIVE}:
            return self.covariant_pullback()
        elif mapping == DIVERGENCE:
            return cartesian_pullback(mapping, self.jacobian)
        else:
            raise NotImplementedError(f"No pullback for an axis with {mapping} mapping.")

    def covariant_pullback(self):
        r"""The pullback of a derivative or covariant component axis.

        Tangents are invariant and each unit normal :math:`\hat{n}_k`
        maps to :math:`J^{-1}\nu_k = u_k/d_k`.
        """
        sd = self.ref_el.get_spatial_dimension()
        P = numpy.eye(sd) - sum(numpy.outer(n, n) for n, *_ in self.normals)
        dens = [d for _, _, d, _ in self.normals]
        terms = []
        for k, (n, u, _, frame) in enumerate(self.normals):
            scale = reduce(mul, dens[:k] + dens[k+1:], one)
            for f in frame:
                terms.append(((u @ as_gem_array(f)) * scale, numpy.outer(f, n)))
        return DirectionPullback(P, terms, reduce(mul, dens))

    def contravariant_pullback(self):
        r"""The pullback of a contravariant component axis.

        The scaled normal :math:`\hat\nu` is invariant, since
        :math:`K\hat\nu` is the physical scaled normal, while a
        tangential direction :math:`\hat\tau` maps to :math:`s\hat\tau
        + r\hat\nu` with :math:`s = |K\hat\nu|^2/(\det J\,|\hat\nu|^2)`
        and :math:`r = -(K\hat\tau)\cdot(K\hat\nu)/(\det J\,|\hat\nu|^2)`:
        FIAT takes the physical tangential components against the
        cofactor image of the reference tangents projected onto the
        physical facet.
        """
        K, detJ = self.jacobian.K, self.jacobian.detJ
        sd = self.ref_el.get_spatial_dimension()
        nu = self.ref_el.compute_scaled_normal(self.facet)
        nn = nu @ nu
        Knu = K @ nu
        KtKnu = K.T @ Knu
        P = numpy.outer(nu, nu) / nn
        terms = [(Knu @ Knu, numpy.eye(sd) - P)]
        for tau in self.facet_tangents:
            terms.append((KtKnu @ as_gem_array(-tau), numpy.outer(nu, tau)))
        return DirectionPullback(P, terms, detJ * nn)

    def measure(self):
        """The measure of the physical facet."""
        sd = self.ref_el.get_spatial_dimension()
        nu = self.ref_el.compute_scaled_normal(self.facet)
        Knu = self.jacobian.K @ nu
        vol = self.ref_el.volume_of_subcomplex(sd - 1, self.facet)
        return (vol / numpy.linalg.norm(nu)) * (Knu @ Knu)**0.5


class Functional:
    """A reference degree of freedom as a numeric coefficient tensor at each point.

    The coefficient tensors are read off the point and derivative
    dictionaries of the FIAT functional.  A derivative multi-index is
    spread evenly over the positions of the symmetric derivative tensor
    with that multi-index, so that the full contraction with the
    derivative tensor of :math:`f` reproduces the multi-index pairing.  A
    first derivative whose coefficients contract the last component axis
    with the derivative axis is stored with a divergence axis instead.

    :arg node: The FIAT :class:`~FIAT.functional.Functional`.
    :arg entity: The ``(dim, entity)`` FIAT lists the node under.
    :arg mapping: The FIAT mapping of the basis functions.
    :arg tol: Relative tolerance for recognizing a divergence.

    The attributes are the ``node`` and ``entity``, the ``points``, a
    tuple of reference coordinates, the ``coefficients``, a numeric array
    of shape ``(len(points), sd, ..., sd)`` with one trailing axis per
    component and per derivative, the components first (a divergence
    axis has length one), and the ``mappings`` of each axis: a FIAT
    mapping for a component axis, :data:`DERIVATIVE` for a derivative
    axis, or :data:`DIVERGENCE` for the last axis.
    """
    def __init__(self, node, entity, mapping="affine", tol=1e-12):
        if node.pt_dict and node.deriv_dict:
            raise NotImplementedError(f"{type(node).__name__} mixes values and derivatives.")
        sd = node.ref_el.get_spatial_dimension()
        if node.deriv_dict:
            entries = {pt: [(w, tuple(alpha), tuple(comp)) for w, alpha, comp in wac]
                       for pt, wac in node.deriv_dict.items()}
        else:
            alpha = (0,) * sd
            entries = {pt: [(w, alpha, tuple(comp)) for w, comp in wc]
                       for pt, wc in node.pt_dict.items()}
        orders = {sum(alpha) for wac in entries.values() for _, alpha, _ in wac}
        ranks = {len(comp) for wac in entries.values() for _, _, comp in wac}
        if len(orders) > 1 or len(ranks) > 1:
            raise NotImplementedError(f"{type(node).__name__} mixes derivative orders or ranks.")
        order, = orders
        rank, = ranks
        mappings = split_axis_mappings(mapping, rank) + (DERIVATIVE,) * order

        points = tuple(entries)
        coefficients = numpy.zeros((len(points),) + (sd,) * (rank + order))
        for q, pt in enumerate(points):
            for w, alpha, comp in entries[pt]:
                indices = set(permutations(sum(([k] * a for k, a in enumerate(alpha)), [])))
                for index in indices:
                    coefficients[(q, *comp, *index)] += w / len(indices)

        # A divergence has W[q, ..., i, k] = c[q, ...] delta_ik on the last
        # component axis i and the derivative axis k: detect it by comparing
        # W with its trace times the identity, and keep c on one axis.
        if order == 1 and rank > 0 and mappings[rank - 1] == "contravariant piola":
            trace = numpy.trace(coefficients, axis1=-2, axis2=-1) / sd
            if numpy.allclose(coefficients, trace[..., None, None] * numpy.eye(sd),
                              rtol=0, atol=tol * numpy.abs(coefficients).max()):
                coefficients = trace[..., None]
                mappings = mappings[:rank - 1] + (DIVERGENCE,)

        self.node = node
        #: The cell whose entities number the dofs, the parent of a macro cell
        self.ref_el = node.ref_el.get_parent() or node.ref_el
        self.entity = entity
        self.points = tuple(map(tuple, points))
        self.coefficients = coefficients
        self.mappings = mappings
        #: How FIAT builds the physical counterpart of the node
        self.transformation = self._transformation(tol)

    @property
    def rank(self):
        """The number of component axes."""
        return sum(mapping not in (DERIVATIVE, DIVERGENCE) for mapping in self.mappings)

    @property
    def order(self):
        """The number of derivatives taken."""
        return sum(mapping in (DERIVATIVE, DIVERGENCE) for mapping in self.mappings)

    def _transformation(self, tol):
        """How FIAT builds the physical counterpart of a node.

        Instantiated on a physical cell, FIAT dual sets follow one of three
        conventions, which cannot be read off the reference functional
        alone:

        * ``"cartesian"``: point evaluations keep their Cartesian
          components, values and derivatives alike (vertex jets, the vertex
          divergences of Alfeld-Sorokina, the vertex and interior point
          values of Guzman-Neilan), except that derivatives along an edge
          of a three-dimensional cell are taken along the difference of the
          physical vertices (the vertex-edge dofs of the Stokes element);
        * ``"frame"``: nodes on a facet refer to the physical facet normal
          and tangents (normal derivatives and their moments, normal and
          tangential moments of Piola-mapped fields, the facet point
          values of Hu-Zhang);
        * ``"invariant"``: interior moments are taken against pulled-back
          test functions, so the physical node is the push-forward of the
          reference one; point values of scalar fields are invariant too.

        Vector point values on a facet are Cartesian, not framed: they are
        the C0 data of vector-valued H1 elements (the edge midpoint values
        of Alfeld-Sorokina).

        :arg tol: Relative tolerance for the normal part of a derivative
            along an edge.
        :returns: One of ``"cartesian"``, ``"frame"`` or ``"invariant"``.
        :raises NotImplementedError: For a derivative normal to an edge of a
            three-dimensional cell, which has no physical convention.
        """
        node = self.node
        sd = self.ref_el.get_spatial_dimension()
        dim, entity = self.entity
        order = node.max_deriv_order
        rank = len(node.target_shape)
        single = len(node.deriv_dict or node.pt_dict) == 1
        if order == 0 and rank == 0:
            return "invariant"
        if dim == sd - 1 and not (single and order == 0 and rank == 1):
            return "frame"
        if dim == sd and not single and order == 0:
            return "invariant"
        if 0 < dim < sd - 1 and order > 0:
            t = self.ref_el.compute_edge_tangent(entity)
            normal = numpy.eye(sd) - numpy.outer(t, t) / (t @ t)
            W = self.coefficients
            for axis, mapping in enumerate(self.mappings, start=1):
                residual = numpy.tensordot(W, normal, axes=(axis, 1))
                if mapping == DERIVATIVE and numpy.abs(residual).max() > tol * numpy.abs(W).max():
                    raise NotImplementedError("No physical convention for derivatives normal to an edge.")
        return "cartesian"

    def arrange_tabulation(self, tabulation, columns):
        """Arranges a FIAT tabulation with the axes of a coefficient tensor.

        :arg tabulation: A FIAT tabulation dict, of sufficient derivative order.
        :arg columns: A dict mapping each point to its column in the tabulation.
        :arg functional: The :class:`~finat.functional.Functional`.
        :returns: An array of shape ``(nbf, *shape, len(points))`` with one
            axis per axis of the coefficient tensor: the value components,
            then the derivatives, then the divergence (an axis of length one).
        """
        sd = len(next(iter(tabulation)))
        cols = [columns[pt] for pt in self.points]
        values = tabulation[(0,) * sd][..., cols]
        T = numpy.zeros(values.shape[:-1] + (sd,) * self.order + values.shape[-1:])
        prefix = (slice(None),) * (values.ndim - 1)
        for index in numpy.ndindex((sd,) * self.order):
            alpha = [0] * sd
            for k in index:
                alpha[k] += 1
            T[prefix + index] = tabulation[tuple(alpha)][..., cols]
        if DIVERGENCE in self.mappings:
            # contract the last value axis with the derivative axis
            T = numpy.trace(T, axis1=values.ndim - 2, axis2=-2)[..., None, :]
        return T

    def frame(self, jacobian):
        """The :class:`PhysicalEntityFrame` of a node on a facet.

        :arg jacobian: The :class:`~finat.physically_mapped.Jacobian`.
        """
        ref_el = self.ref_el
        support = support_entity(ref_el, self.points)
        return PhysicalEntityFrame(ref_el, self.entity[1], support, jacobian)

    def pullbacks(self, jacobian):
        """The :class:`DirectionPullback` of each axis of the coefficient tensor.

        :arg jacobian: The :class:`~finat.physically_mapped.Jacobian`.
        """
        sd = self.ref_el.get_spatial_dimension()
        dim, _ = self.entity
        if self.transformation == "invariant":
            return [identity_pullback(sd) for mapping in self.mappings]
        elif self.transformation == "cartesian":
            on_edge = 0 < dim < sd - 1
            return [identity_pullback(sd) if on_edge and mapping == DERIVATIVE
                    else cartesian_pullback(mapping, jacobian)
                    for mapping in self.mappings]
        frame = self.frame(jacobian)
        return [frame.pullback(mapping) for mapping in self.mappings]

    def is_invariant(self, jacobian, tol):
        """Whether the physical node is the push-forward of the reference node.

        :arg jacobian: The :class:`~finat.physically_mapped.Jacobian`.
        :arg tol: Relative tolerance on the coefficients.
        """
        W = self.coefficients
        for k, pullback in enumerate(self.pullbacks(jacobian)):
            P = pullback.invariant
            R = numpy.tensordot(W, numpy.eye(P.shape[0]) - P, axes=(1 + k, 1))
            if numpy.abs(R).max() > tol * numpy.abs(W).max():
                return False
        return True

    def action(self, tabulation, jacobian, tol, avg=True):
        """The values of the physical node on the pulled-back nodal basis.

        The physical node differs from the reference node only in the
        directions along each axis of its coefficient tensor, and its
        row is a sum over products of one term per pullback.  The
        numeric matrices of the terms are contracted with the reference
        tabulation, quadrature sum included, before the GEM coefficient
        multiplies the result, so that whether a coupling is present is
        decided numerically in the frame of the node and never by
        inspecting a GEM expression.

        :arg tabulation: The reference nodal basis at the points of the
            node, as returned by :meth:`arrange_tabulation`.
        :arg jacobian: The :class:`~finat.physically_mapped.Jacobian`.
        :arg tol: Relative tolerance below which numeric entries of the
            tabulation are dropped.
        :arg avg: Whether physical facet moments are integral averages, as
            the reference nodes are; if not, they carry the physical facet
            measure.
        :returns: The pair ``(numerator, denominator)`` of an object
            array of GEM scalars and a GEM scalar, the row of the
            physical Vandermonde matrix being their ratio.
        """
        pullbacks = self.pullbacks(jacobian)
        nbf = tabulation.shape[0]
        # N[I', I, j] pairs the reference coefficients with indices I'
        # against the tabulated basis with indices I
        N = numpy.tensordot(self.coefficients, tabulation, axes=(0, -1))
        size = prod(self.coefficients.shape[1:])
        N = N.reshape(size, nbf, size).transpose(0, 2, 1).reshape(size * size, nbf)
        scale = tol * numpy.abs(N).max()
        numerator = numpy.full(nbf, zero, dtype=object)
        for choice in product(*(pullback.numerator_terms() for pullback in pullbacks)):
            coefficient = reduce(mul, (g for g, _ in choice))
            C = reduce(numpy.kron, [C for _, C in choice])
            row = C.T.ravel() @ N
            for j in numpy.flatnonzero(numpy.abs(row) > scale * numpy.abs(C).max()):
                numerator[j] = numerator[j] + coefficient * row[j]
        if not avg and self.transformation == "frame" and len(self.points) > 1:
            numerator = numerator * self.frame(jacobian).measure()
        denominator = reduce(mul, (pullback.denominator for pullback in pullbacks))
        return numerator, denominator
