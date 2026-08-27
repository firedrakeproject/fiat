r"""Numeric prototype of the unified transformation theory, scalar and Piola.

The transformation matrix is obtained by duality alone:

.. math:: B_{ij} = n_i(\hat\psi_j \circ F^{-1}), \qquad V = B^{-1},

where :math:`\hat\psi_j` is the reference nodal basis, :math:`F` the cell
map, and :math:`n_i` the physical node.  Two ingredients make the rows of
:math:`B` computable without any frame algebra:

* **Physical nodes by per-slot maps.**  The physical node shares the
  points and weights of its reference partner; only the directional data
  changes, one tensor slot at a time.  A derivative slot of a facet node
  maps its unit-normal component to the unit physical normal
  :math:`K\hat{n}/|K\hat{n}|` (:math:`K = \operatorname{adj}(J)^T` the
  cofactor matrix, which maps normals to normals) and its tangential
  complement by :math:`J` (mapped tangents); away from facets derivative
  slots are Cartesian and keep their reference directions.  A
  contravariant value slot of a facet moment maps its scaled-normal
  component by :math:`K` (the cofactor lemma :math:`K\hat\nu^s = \nu^s`
  is exact) and its tangential complement by :math:`J`.  Cartesian point
  data keeps its weights, interior moments are invariant by convention
  (physical test functions are Piola-mapped), and divergence nodes
  contract to :math:`\det J` times the identity.

* **The adjoint acts on the tabulation.**  The push-forward of a
  reference node contracts each derivative slot of its direction with
  :math:`J` (:math:`d = J^{\otimes m}\hat{d}`); dually, instead of
  transforming directions, each derivative slot of the *numeric*
  reference tabulation is contracted with :math:`J^{-T}` (the physical
  derivatives of :math:`\hat\psi_j\circ F^{-1}`) and each value slot
  with :math:`J/\det J` (its physical Piola values), once per node
  group.  A row of :math:`B` is then the plain numeric pairing of the
  physical node data with the transformed tabulation; push-forward
  invariance (mapped-tangential moments, point values) needs no special
  cases -- it falls out as exact Kronecker rows.

Crucially, :math:`B` is never inverted densely.  Its rows obey the support
law :math:`B_{ij} \ne 0` only if dof :math:`j` lives on the closure of dof
:math:`i`'s entity: interior-moment, divergence, and single-point rows are
*exactly* block-diagonal (pointwise identities, independent of the
polynomial space); a vertex-jet row is an exact combination of the
reference jet dofs at the same vertex (duality of the nodal basis); and a
facet row couples to its own facet block plus, possibly, dofs on the
boundary of that facet -- the residual left by the tangential components
is a functional of the trace on the facet, and the trace is determined by
the closure dofs (the same property that makes the element conforming; in
the scalar theory this role is played by the fundamental-theorem-of-
calculus exactness identities).  Hence :math:`B` is block lower triangular
in the entity partial order and :func:`composition_transformation`
computes :math:`V = B^{-1}` by block back-substitution -- one small solve
per entity, fill-in confined to the closure.  Both the support law and the
agreement with ``basis_transformation`` are asserted to machine precision
for the scalar and Piola element zoos, in 2D and 3D, on cells of both
orientations.  This module is the independent verification of the
automated transformations: it shares no code path with
``basis_transformation``.  See ``zany_claude.md`` (Stages 4-5) for the
derivation.
"""

from math import factorial, prod

import numpy as np
import pytest

import finat
from FIAT.polynomial_set import mis
from FIAT.reference_element import make_affine_mapping, ufc_simplex
from finat.functional import PhysicallyMappedFunctional
from gem.interpreter import evaluate

from .conftest import MyMapping


def contract_slot(T, A, axis):
    """Contract one tensor slot with a matrix.

    :arg T: The tensor.
    :arg A: The matrix.
    :arg axis: The slot of ``T`` to contract.
    :returns: The tensor with ``T'[..., i, ...] = sum_k A[i, k] T[..., k, ...]``.
    """
    return np.moveaxis(np.tensordot(T, A, axes=(axis, 1)), -1, axis)


def tensorize_direction(direction, sd, order):
    """Distribute multi-index direction coefficients over a symmetric tensor.

    :arg direction: Direction coefficients in the multi-index basis.
    :arg sd: The spatial dimension.
    :arg order: The derivative order.
    :returns: The symmetric direction tensor, normalized so that the full
        contraction with an index-wise derivative tabulation reproduces
        the multi-index pairing.
    """
    alphas = sorted(mis(sd, order), reverse=True)
    lookup = {alpha: k for k, alpha in enumerate(alphas)}
    D = np.zeros((sd,) * order)
    for index in np.ndindex(D.shape):
        alpha = [0] * sd
        for k in index:
            alpha[k] += 1
        scale = prod(map(factorial, alpha)) / factorial(order)
        D[index] = direction[lookup[tuple(alpha)]] * scale
    return D


def derivative_tabulation(fiat_element, order, points, weights):
    """Weighted symmetric-tensor derivative tabulation of the nodal basis.

    :arg fiat_element: The FIAT element.
    :arg order: The derivative order.
    :arg points: The quadrature points.
    :arg weights: The quadrature weights.
    :returns: The tensor ``T[j, i1, ..., im] = sum_q w_q
        (d^m psi_j / dx_i1 ... dx_im)(x_q)``.
    """
    sd = fiat_element.get_reference_element().get_spatial_dimension()
    tab = fiat_element.tabulate(order, points)
    nbf = tab[(0,) * sd].shape[0]
    T = np.zeros((nbf,) + (sd,) * order)
    for index in np.ndindex((sd,) * order):
        alpha = [0] * sd
        for k in index:
            alpha[k] += 1
        T[(slice(None), *index)] = tab[tuple(alpha)] @ weights
    return T


def facet_slot_map(ref_el, entity, J, derivative):
    r"""Per-slot physical direction map of a facet node.

    A derivative slot maps its component along the FIAT normal
    :math:`\hat{n}` to the FIAT physical normal -- the norm-preserving
    rescaling of the cofactor image :math:`K\hat{n}` (:math:`K =
    \operatorname{adj}(J)^T` maps normals to normals, and the norm of
    the FIAT normal depends only on the reference cell) -- and its
    tangential complement by :math:`J` (mapped tangents).

    A contravariant value slot maps its component along the scaled
    normal :math:`\hat\nu^s` by :math:`K` (the cofactor lemma
    :math:`K\hat\nu^s = \nu^s` is exact), and its tangential complement
    to the cofactor image projected onto the physical facet, scaled by
    :math:`|K\hat\nu^s|^2/(\det J\, |\hat\nu^s|^2)` -- the reciprocal
    of the scalar tangential push-forward coefficient, which in 2D
    reduces to the mapped tangent :math:`J\hat{t}`.

    :arg ref_el: The reference cell.
    :arg entity: The facet number.
    :arg J: The (numeric) cell Jacobian.
    :arg derivative: If True, use the derivative-slot convention;
        if False, the contravariant value-slot convention.
    :returns: The map as an ``(sd, sd)`` array acting on one slot.
    """
    sd = ref_el.get_spatial_dimension()
    detJ = np.linalg.det(J)
    K = detJ * np.linalg.inv(J).T
    if derivative:
        n = ref_el.compute_normal(entity)
        Kn = K @ n
        Kn = Kn * (np.linalg.norm(n) / np.linalg.norm(Kn))
        P = np.outer(n, n) / (n @ n)
        return np.outer(Kn, n) / (n @ n) + J @ (np.eye(sd) - P)
    n = ref_el.compute_scaled_normal(entity)
    Kn = K @ n
    P = np.outer(n, n) / (n @ n)
    Q = np.eye(sd) - np.outer(Kn, Kn) / (Kn @ Kn)
    s = (Kn @ Kn) / (detJ * (n @ n))
    return np.outer(Kn, n) / (n @ n) + s * (Q @ K @ (np.eye(sd) - P))


def scalar_derivative_row(fiat_element, ell, Phi, J):
    r"""B row of a scalar derivative node.

    The physical direction is the per-slot map :math:`\Phi^{\otimes m}`
    of the reference direction; the derivative slots of the tabulation
    carry the adjoint of the push-forward, :math:`J^{-T}` per slot.

    :arg fiat_element: The FIAT element.
    :arg ell: The parsed reference node.
    :arg Phi: The per-slot physical direction map.
    :arg J: The (numeric) cell Jacobian.
    :returns: The vector of values of the physical node on the
        pushed-forward nodal basis.
    """
    sd = J.shape[0]
    T = derivative_tabulation(fiat_element, ell.order, ell.points, ell.weights)
    A = np.linalg.inv(J).T
    D = tensorize_direction(ell.direction, sd, ell.order)
    for slot in range(ell.order):
        T = contract_slot(T, A, 1 + slot)
        D = contract_slot(D, Phi, slot)
    return np.tensordot(T, D, axes=(tuple(range(1, ell.order + 1)),
                                    tuple(range(ell.order))))


def piola_value_row(fiat_element, ell, Phi, J):
    r"""B row of a contravariant value node.

    The value slots of the tabulation carry the physical Piola values,
    :math:`J/\det J` per slot; the weights are mapped per slot by
    ``Phi`` (facet moments) or kept Cartesian (point data).

    :arg fiat_element: The FIAT element.
    :arg ell: The parsed reference node.
    :arg Phi: The per-slot physical weight map, or None for Cartesian
        point data.
    :arg J: The (numeric) cell Jacobian.
    :returns: The vector of values of the physical node on the
        pushed-forward nodal basis.
    """
    sd = J.shape[0]
    r = ell.rank
    npts = len(ell.points)
    Theta = J / np.linalg.det(J)
    T = fiat_element.tabulate(0, ell.points)[(0,) * sd]
    T = T.reshape(T.shape[0], *(sd,) * r, npts)
    W = ell.weights.reshape(npts, *(sd,) * r)
    for slot in range(r):
        T = contract_slot(T, Theta, 1 + slot)
        if Phi is not None:
            W = contract_slot(W, Phi, 1 + slot)
    return np.tensordot(T, W, axes=((*range(1, r + 1), r + 1),
                                    (*range(1, r + 1), 0)))


def evaluate_divergence(ell, fiat_element):
    """Apply a divergence functional to the nodal basis of a FIAT element.

    :arg ell: A divergence :class:`PhysicallyMappedFunctional`.
    :arg fiat_element: The FIAT element providing the nodal basis.
    :returns: The vector of values of the functional on the nodal basis.
    """
    sd = fiat_element.get_reference_element().get_spatial_dimension()
    tab = fiat_element.tabulate(1, ell.points)
    alphas = [tuple(int(k == c) for k in range(sd)) for c in range(sd)]
    div = sum(tab[alphas[c]].reshape(tab[alphas[c]].shape[0], sd, -1)[:, c, :]
              for c in range(sd))
    return div @ ell.weights


def physical_node_row(fiat_element, ell, dim, entity, J, avg):
    """B row of one parseable node, or None if push-forward invariant.

    :arg fiat_element: The FIAT element.
    :arg ell: The parsed reference node.
    :arg dim: The dimension of the entity the node sits on.
    :arg entity: The entity number.
    :arg J: The (numeric) cell Jacobian.
    :arg avg: If False, physical scalar facet moments are plain
        integrals rather than the measure-intrinsic integral averages of
        the reference weights, and the row is rescaled by the physical
        facet measure.
    :returns: The row, or None for a row of the identity.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    if ell.divergence:
        return evaluate_divergence(ell, fiat_element) / np.linalg.det(J)
    if ell.rank:
        point_data = len(ell.points) == 1 and (ell.rank == 1 or dim != sd - 1)
        if dim == sd and not point_data:
            return None
        Phi = (facet_slot_map(ref_el, entity, J, derivative=False)
               if dim == sd - 1 and not point_data else None)
        return piola_value_row(fiat_element, ell, Phi, J)
    if ell.order == 0:
        return None
    Phi = (facet_slot_map(ref_el, entity, J, derivative=True)
           if dim == sd - 1 else np.eye(sd))
    row = scalar_derivative_row(fiat_element, ell, Phi, J)
    if not avg and len(ell.points) > 1 and dim == sd - 1:
        # The reference weights are measure-intrinsic (integral averages),
        # so a plain physical integral carries the physical facet measure.
        n = ref_el.compute_scaled_normal(entity)
        K = np.linalg.det(J) * np.linalg.inv(J).T
        measure = (ref_el.volume_of_subcomplex(sd - 1, entity)
                   * np.linalg.norm(K @ n) / np.linalg.norm(n))
        row = row * measure
    return row


def closure_dofs(fiat_element, dim, entity):
    """Indices of the dofs on the strict closure of a cell entity.

    :arg fiat_element: The FIAT element.
    :arg dim: The dimension of the entity.
    :arg entity: The entity number.
    :returns: The indices of the dofs supported on subentities of strictly
        lower dimension contained in the closure of the entity.
    """
    top = fiat_element.get_reference_element().get_topology()
    entity_ids = fiat_element.entity_dofs()
    verts = set(top[dim][entity])
    return [i for d in sorted(entity_ids) if d < dim
            for e in entity_ids[d] if set(top[d][e]) <= verts
            for i in entity_ids[d][e]]


def composition_transformation(fiat_element, J, ndof=None, avg=True, tol=2e-10):
    """Compute V for a scalar or Piola element by blockwise duality.

    The matrix :math:`B` of the module docstring is block lower triangular
    in the entity partial order, so :math:`V = B^{-1}` is computed by block
    back-substitution over the entities in order of increasing dimension:

    .. math:: V_e = B_{ee}^{-1} (I_e - B_{ec} V_c),

    where :math:`e` collects the dofs of one entity and :math:`c` the
    (already processed) dofs on its strict closure.  The support law that
    justifies the triangular structure is asserted, not assumed: every row
    of :math:`B` must vanish outside its own entity block and closure.
    Fill-in is confined to the closure, so the sparsity of :math:`V`
    matches that of the row recursion in the scalar theory.

    :arg fiat_element: The FIAT element on the reference cell.
    :arg J: The (numeric) cell Jacobian.
    :arg ndof: The number of exposed physical dofs; unparseable constraint
        functionals beyond it keep identity rows (their columns are
        truncated by the caller).  Defaults to all dofs.
    :arg avg: Whether physical scalar facet moments are integral averages.
    :arg tol: Relative tolerance for the support-law assertion.
    :returns: The transformation V as a numpy array, with the same row and
        column ordering as ``basis_transformation`` before truncation of
        the trailing constraint columns.
    """
    nodes = fiat_element.dual_basis()
    entity_ids = fiat_element.entity_dofs()
    nbf = len(nodes)
    if ndof is None:
        ndof = nbf
    eye = np.eye(nbf)
    V = np.zeros((nbf, nbf))
    done = set()
    for dim in sorted(entity_ids):
        for entity in entity_ids[dim]:
            # FIAT may list a dof on more than one entity (e.g. the edge
            # moments of Arnold-Winther reappear in its interior list); the
            # lowest-dimensional entity owns the dof.
            block = [i for i in entity_ids[dim][entity] if i not in done]
            if not block:
                continue
            done.update(block)
            rows = []
            for i in block:
                try:
                    ell = PhysicallyMappedFunctional.from_fiat(nodes[i])
                except NotImplementedError:
                    if i < ndof:
                        raise
                    rows.append(eye[i])
                    continue
                row = physical_node_row(fiat_element, ell, dim, entity, J, avg)
                rows.append(eye[i] if row is None else row)
            Bblock = np.asarray(rows)
            prior = closure_dofs(fiat_element, dim, entity)
            outside = np.setdiff1d(np.arange(nbf), block + prior)
            if outside.size:
                assert (np.abs(Bblock[:, outside]).max()
                        <= tol * np.abs(Bblock).max()), \
                    f"support law violated on entity ({dim}, {entity})"
            V[block] = np.linalg.solve(
                Bblock[:, block], eye[block] - Bblock[:, prior] @ V[prior])
    return V


# Scalar elements, including hand-coded macroelement transformations
# (HCT, Powell-Sabin, Alfeld/Bramble-Zlamal C2) never before reproduced by
# the automated theory.  BrambleZlamalC2 needs a looser tolerance: FIAT's
# own dual basis and tabulation only agree to ~4e-10 on its ill-conditioned
# order-4 vertex jets.  Walkington is excluded here only because this
# prototype lacks the flag frame of its edge moments (normal to an edge
# within a face); the production engine handles it (see zany_claude.md),
# and test_zany_mapping checks it against a direct physical-cell fit.
scalar_zoo = {
    2: [(finat.Morley, ()),
        (finat.Hermite, ()),
        (finat.Bell, ()),
        (finat.Argyris, ()),
        (finat.Argyris, (5, "point")),
        (finat.Argyris, (6,)),
        (finat.WuXuH3NC, ()),
        (finat.WuXuRobustH3NC, ()),
        (finat.HsiehCloughTocher, ()),
        (finat.HsiehCloughTocher, (4,)),
        (finat.ReducedHsiehCloughTocher, ()),
        (finat.QuadraticPowellSabin6, ()),
        (finat.QuadraticPowellSabin12, ()),
        (finat.AlfeldC2, ()),
        (finat.BrambleZlamalC2, ()),
        ],
    3: [(finat.Morley, ()),
        (finat.Hermite, ())],
}

piola_zoo = {
    2: [(finat.MardalTaiWinther, ()),
        (finat.JohnsonMercier, ()),
        (finat.ArnoldWintherNC, ()),
        (finat.ArnoldWinther, ()),
        (finat.AlfeldSorokina, ()),
        (finat.BernardiRaugel, ()),
        (finat.BernardiRaugelBubble, ()),
        (finat.GuzmanNeilanFirstKindH1, ()),
        (finat.GuzmanNeilanSecondKindH1, ()),
        (finat.GuzmanNeilanBubble, ()),
        (finat.GuzmanNeilanH1div, ()),
        (finat.ReducedArnoldQin, ()),
        (finat.ChristiansenHu, ()),
        (finat.HuZhang, (3, "integral")),
        (finat.HuZhang, (4, "integral")),
        (finat.HuZhang, (3, "point")),
        (finat.HuZhang, (4, "point"))],
    3: [(finat.MardalTaiWinther, ()),
        (finat.MardalTaiWinther, (2,)),
        (finat.JohnsonMercier, ()),
        (finat.AlfeldSorokina, ()),
        (finat.BernardiRaugel, ()),
        (finat.BernardiRaugelBubble, ()),
        (finat.GuzmanNeilanFirstKindH1, ()),
        (finat.GuzmanNeilanFirstKindH1, (2,)),
        (finat.GuzmanNeilanSecondKindH1, ()),
        (finat.GuzmanNeilanBubble, ()),
        (finat.GuzmanNeilanH1div, ()),
        (finat.ChristiansenHu, ())],
}

orientations = {
    (2, "positive"): ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84)),
    (2, "negative"): ((0.0, 0.1), (0.15, 1.84), (1.17, -0.09)),
    (3, "positive"): ((0, 0, 0), (1., 0.1, -0.37),
                      (0.01, 0.987, -.23), (-0.1, -0.2, 1.38)),
    (3, "negative"): ((0, 0, 0), (0.01, 0.987, -.23),
                      (1., 0.1, -0.37), (-0.1, -0.2, 1.38)),
}


def check_composition(dimension, element, args, orientation):
    """Compare the blockwise duality V against basis_transformation."""
    ref_cell = ufc_simplex(dimension)
    phys_cell = ufc_simplex(dimension)
    phys_cell.vertices = orientations[(dimension, orientation)]
    J, b = make_affine_mapping(ref_cell.vertices, phys_cell.vertices)

    finat_element = element(ref_cell, *args)
    mapping = MyMapping(ref_cell, phys_cell)
    M = evaluate([finat_element.basis_transformation(mapping)])[0].arr

    ndof = finat_element.space_dimension()
    avg = getattr(finat_element, "avg", True)
    V = composition_transformation(finat_element._element, J, ndof=ndof, avg=avg)
    assert np.allclose(V[:, :ndof], M.T, atol=2e-10)


@pytest.mark.parametrize("orientation", ["positive", "negative"])
@pytest.mark.parametrize("dimension, element, args", [
    (dim, *case) for dim in scalar_zoo for case in scalar_zoo[dim]])
def test_scalar_composition(dimension, element, args, orientation):
    check_composition(dimension, element, args, orientation)


@pytest.mark.parametrize("orientation", ["positive", "negative"])
@pytest.mark.parametrize("dimension, element, args", [
    (dim, *case) for dim in piola_zoo for case in piola_zoo[dim]])
def test_piola_composition(dimension, element, args, orientation):
    check_composition(dimension, element, args, orientation)
