r"""Degrees of freedom as coefficient tensors.

A FIAT functional built from point and derivative dictionaries is
:math:`\ell(f) = \sum_q \langle W_q, \nabla^m f(x_q) \rangle`, with one
axis of :math:`W_q` per value component and one per derivative.
"""

from functools import cached_property, reduce
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


class DirectionPullback:
    r"""The pullback to the reference cell of one axis of a physical node.

    The matrix :math:`G = P + \sum_t g_t C_t / d` combines the physical
    direction with the chain rule or Piola map.  Keeping the matrices
    numeric allows cancellations before symbolic row assembly.

    :arg invariant: Numeric projector :math:`P` onto invariant directions.
    :arg terms: Pairs :math:`(g_t, C_t)` of GEM scalars and numeric matrices.
    :arg denominator: Common denominator :math:`d`.
    """
    def __init__(self, invariant, terms, denominator):
        self.invariant = numpy.asarray(invariant)
        self.terms = list(terms)
        self.denominator = denominator

    def numerator_terms(self):
        """Return the terms of :math:`Gd`, including the invariant projector."""
        terms = list(self.terms)
        if self.invariant.any():
            terms.insert(0, (self.denominator, self.invariant))
        return terms


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
    :arg J: The Jacobian as an object array of GEM scalars.
    :arg adjJ: The determinant-scaled inverse of the Jacobian.
    :arg detJ: The Jacobian determinant.
    """
    def __init__(self, ref_el, facet, support, J, adjJ, detJ):
        self.ref_el = ref_el
        self.facet = facet
        self.K = adjJ.T
        self.detJ = detJ
        sd = ref_el.get_spatial_dimension()
        K = self.K
        # orthonormal tangents of the facet and of the support entity
        tangents = ref_el.compute_tangents(sd - 1, facet)
        self.facet_tangents = list(numpy.linalg.qr(tangents.T)[0].T)
        n = ref_el.compute_normal(facet)
        n = n / numpy.linalg.norm(n)
        Kn = K @ n
        # Each normal is (nhat, u, d, frame), with J^{-1} nu = u / d.
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
            return self.contravariant_pullback
        elif mapping in {"covariant piola", DERIVATIVE}:
            return self.covariant_pullback
        elif mapping == DIVERGENCE:
            return DirectionPullback(numpy.zeros((1, 1)), [(one, numpy.ones((1, 1)))], self.detJ)
        else:
            raise NotImplementedError(f"No pullback for an axis with {mapping} mapping.")

    @cached_property
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
                terms.append(((u @ f) * scale, numpy.outer(f, n)))
        return DirectionPullback(P, terms, reduce(mul, dens))

    @cached_property
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
        K, detJ = self.K, self.detJ
        sd = self.ref_el.get_spatial_dimension()
        nu = self.ref_el.compute_scaled_normal(self.facet)
        nn = nu @ nu
        Knu = K @ nu
        KtKnu = K.T @ Knu
        P = numpy.outer(nu, nu) / nn
        terms = [(Knu @ Knu, numpy.eye(sd) - P)]
        for tau in self.facet_tangents:
            terms.append((KtKnu @ -tau, numpy.outer(nu, tau)))
        return DirectionPullback(P, terms, detJ * nn)

    @cached_property
    def measure(self):
        """The measure of the physical facet."""
        sd = self.ref_el.get_spatial_dimension()
        nu = self.ref_el.compute_scaled_normal(self.facet)
        Knu = self.K @ nu
        vol = self.ref_el.volume_of_subcomplex(sd - 1, self.facet)
        return (vol / numpy.linalg.norm(nu)) * (Knu @ Knu)**0.5


class Functional:
    """A reference degree of freedom as a numeric coefficient tensor at each point.

    ``coefficients`` has shape ``(npoints, sd, ..., sd)``, with component
    axes followed by derivative axes.  Multi-index weights are distributed
    over the symmetric derivative tensor so contraction reproduces the
    FIAT pairing.  A divergence is stored on one axis of length one.
    ``mappings`` records the pullback law for each axis.

    :arg node: The FIAT :class:`~FIAT.functional.Functional`.
    :arg entity: The ``(dim, entity)`` FIAT lists the node under.
    :arg mapping: The FIAT mapping of the basis functions.
    :arg tol: Relative tolerance for recognizing a divergence.
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

        # W[..., i, k] = c delta_ik represents a divergence.
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

    @cached_property
    def support(self):
        """Smallest entity containing the evaluation points, irrespective of dof ownership."""
        return support_entity(self.ref_el, self.points)

    def _transformation(self, tol):
        """Classify the physical-node conventions used by the supported FIAT elements.

        Cartesian point data keeps its coordinate directions.  Facet data
        uses physical normals and tangents, except for vector point values,
        which supply C0 data.  Interior moments use pulled-back test functions;
        scalar values are invariant too.  Edge derivatives in 3D follow the
        edge tangent.

        These conventions are assumptions about the element family.  A single
        quadrature point need not denote a point evaluation in general.

        :arg tol: Relative tolerance for the normal part of a derivative
            along an edge.
        :returns: One of ``"cartesian"``, ``"frame"`` or ``"invariant"``.
        :raises NotImplementedError: For derivatives normal to an edge in 3D.
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

    def is_invariant(self, pullbacks, tol):
        """Whether the physical node is the push-forward of the reference node.

        :arg pullbacks: A :class:`DirectionPullback` per coefficient axis.
        :arg tol: Relative tolerance on the coefficients.
        """
        W = self.coefficients
        for k, pullback in enumerate(pullbacks):
            P = pullback.invariant
            R = numpy.tensordot(W, numpy.eye(P.shape[0]) - P, axes=(1 + k, 1))
            if numpy.abs(R).max() > tol * numpy.abs(W).max():
                return False
        return True

    def action(self, riesz, pullbacks, tol):
        """The values of the physical node on the pulled-back nodal basis.

        Contract each tensor-product term with the numeric Riesz representation.
        Drop small couplings before multiplying by GEM coefficients, so
        reference identities determine symbolic sparsity.

        :arg riesz: Numeric reference pairing, as returned by :func:`to_riesz`.
        :arg pullbacks: A :class:`DirectionPullback` per coefficient axis.
        :arg tol: Relative tolerance below which numeric couplings are dropped.
        :returns: The pair ``(numerator, denominator)`` of an object
            array of GEM scalars and a GEM scalar, the row of the
            physical Vandermonde matrix being their ratio.
        """
        N = riesz
        nbf = N.shape[1]
        scale = tol * numpy.abs(N).max()
        numerator = numpy.full(nbf, zero, dtype=object)
        for choice in product(*(pullback.numerator_terms() for pullback in pullbacks)):
            coefficient = reduce(mul, (g for g, _ in choice), one)
            C = reduce(numpy.kron, [C for _, C in choice]) if choice else numpy.ones((1, 1))
            row = C.T.ravel() @ N
            for j in numpy.flatnonzero(numpy.abs(row) > scale * numpy.abs(C).max()):
                numerator[j] = numerator[j] + coefficient * row[j]
        denominator = reduce(mul, (pullback.denominator for pullback in pullbacks), one)
        return numerator, denominator


def split_axis_mappings(mapping, rank):
    """Split a FIAT mapping into one mapping per component axis.

    :arg mapping: FIAT mapping, such as ``"double contravariant piola"``.
    :arg rank: Number of component axes.
    :returns: Mapping of each component axis.
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


def support_entity(ref_el, points, tol=1e-12):
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
    for dim in sorted(top):
        for entity in sorted(top[dim]):
            if verts <= set(top[dim][entity]):
                return dim, entity
    raise ValueError("The evaluation points have no supporting entity.")


def identity_pullback(extent):
    """Return the identity pullback.

    :arg extent: Axis extent.
    :returns: Pullback with every direction invariant.
    """
    return DirectionPullback(numpy.eye(extent), [], one)


def cartesian_pullback(mapping, J, adjJ, detJ):
    """The pullback of an axis keeping its Cartesian directions.

    :arg mapping: Mapping of the axis.
    :arg J: Jacobian as an object array of GEM scalars.
    :arg adjJ: Determinant-scaled inverse of the Jacobian.
    :arg detJ: Jacobian determinant.
    :returns: Pullback in Cartesian coordinates.
    """
    sd = J.shape[0]
    if mapping == "affine":
        return identity_pullback(sd)
    elif mapping == DIVERGENCE:
        return DirectionPullback(numpy.zeros((1, 1)), [(one, numpy.ones((1, 1)))], detJ)
    elif mapping == "contravariant piola":
        Q = J.T
    elif mapping in {"covariant piola", DERIVATIVE}:
        Q = adjJ
    else:
        raise NotImplementedError(f"No pullback for an axis with {mapping} mapping.")
    terms = []
    for a, b in numpy.ndindex(sd, sd):
        E = numpy.zeros((sd, sd))
        E[a, b] = 1
        terms.append((Q[a, b], E))
    return DirectionPullback(numpy.zeros((sd, sd)), terms, detJ)


def to_riesz(functionals, fiat_element):
    r"""Pair the dual coefficient tensors with a reference nodal basis.

    As in FIAT's ``DualSet.to_riesz``, tabulate once at the union of node
    points and contract the point weights.  Keep the coefficient and basis
    direction indices separate: :math:`R_{I',I,j} = \sum_q W_{q,I'} T_{j,I,q}`.
    A pullback then acts by contracting its numeric matrices against these
    indices, without revisiting the quadrature points.

    :arg functionals: Reference functionals in dof order, with None for
        unparsed constraints.  Invariant nodes need no representation.
    :arg fiat_element: The FIAT element providing the reference nodal basis.
    :returns: A dict mapping dofs to arrays with shape ``(size**2, nbf)``,
        where ``size`` is the product of coefficient-axis extents.
    """
    nodes = {i: node for i, node in enumerate(functionals)
             if node is not None and node.transformation != "invariant"}
    if not nodes:
        return {}
    points = sorted({pt for node in nodes.values() for pt in node.points})
    order = max(node.order for node in nodes.values())
    tabulation = fiat_element.tabulate(order, points)
    columns = {pt: q for q, pt in enumerate(points)}
    sd = fiat_element.get_reference_element().get_spatial_dimension()
    riesz = {}
    for i, node in nodes.items():
        cols = [columns[pt] for pt in node.points]
        values = tabulation[(0,) * sd][..., cols]
        T = numpy.zeros(values.shape[:-1] + (sd,) * node.order + values.shape[-1:])
        prefix = (slice(None),) * (values.ndim - 1)
        for index in numpy.ndindex((sd,) * node.order):
            alpha = tuple(numpy.bincount(index, minlength=sd))
            T[prefix + index] = tabulation[alpha][..., cols]
        if DIVERGENCE in node.mappings:
            T = numpy.trace(T, axis1=values.ndim - 2, axis2=-2)[..., None, :]
        N = numpy.tensordot(node.coefficients, T, axes=(0, -1))
        size = prod(node.coefficients.shape[1:])
        nbf = T.shape[0]
        riesz[i] = N.reshape(size, nbf, size).transpose(0, 2, 1).reshape(size * size, nbf)
    return riesz
