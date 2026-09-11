r"""Numeric prototype of the unified transformation theory, scalar and Piola.

The transformation matrix is obtained by duality alone:

.. math:: B_{ij} = n_i(\hat\psi_j \circ F^{-1}), \qquad V = B^{-1},

where :math:`\hat\psi_j` is the reference nodal basis, :math:`F` the cell
map, and :math:`n_i` the physical node.  Two ingredients make the rows of
:math:`B` computable without any frame algebra:

* **Physical nodes by per-mapping maps.**  The physical node shares the
  points and weights of its reference partner; only the directional data
  changes, one tensor mapping at a time.  A derivative mapping of a facet node
  maps its unit-normal component to the unit physical normal
  :math:`K\hat{n}/|K\hat{n}|` (:math:`K = \operatorname{adj}(J)^T` the
  cofactor matrix, which maps normals to normals) and its tangential
  complement by :math:`J` (mapped tangents); away from facets derivative
  mappings are Cartesian and keep their reference directions.  A
  contravariant value mapping of a facet moment maps its scaled-normal
  component by :math:`K` (the cofactor lemma :math:`K\hat\nu^s = \nu^s`
  is exact) and its tangential complement by :math:`J`.  Cartesian point
  data keeps its weights, interior moments are invariant by convention
  (physical test functions are Piola-mapped), and divergence nodes
  contract to :math:`\det J` times the identity.

* **The adjoint acts on the tabulation.**  The push-forward of a
  reference node contracts each derivative mapping of its direction with
  :math:`J` (:math:`d = J^{\otimes m}\hat{d}`); dually, instead of
  transforming directions, each derivative mapping of the *numeric*
  reference tabulation is contracted with :math:`J^{-T}` (the physical
  derivatives of :math:`\hat\psi_j\circ F^{-1}`) and each value mapping
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

import numpy as np
import pytest

import finat
from FIAT.reference_element import make_affine_mapping, ufc_simplex
from finat.functional import DERIVATIVE, DIVERGENCE, FunctionalData
from gem.interpreter import evaluate

from .conftest import MyMapping


def contract(T, A, axis):
    """Contract one tensor mapping with a matrix.

    :arg T: The tensor.
    :arg A: The matrix.
    :arg axis: The mapping of ``T`` to contract.
    :returns: The tensor with ``T'[..., i, ...] = sum_k A[i, k] T[..., k, ...]``.
    """
    return np.moveaxis(np.tensordot(T, A, axes=(axis, 1)), -1, axis)


def tabulate(fiat_element, mappings, points):
    """Tabulate the nodal basis as tensors with one axis per mapping.

    :arg fiat_element: The FIAT element.
    :arg mappings: The mapping kinds of a :class:`FunctionalData`.
    :arg points: The points.
    :returns: An array of shape ``(nbf, *shape, len(points))``: the value
        components, then the derivatives, then the divergence (an axis
        of length one).
    """
    sd = fiat_element.get_reference_element().get_spatial_dimension()
    order = sum(mapping in (DERIVATIVE, DIVERGENCE) for mapping in mappings)
    tab = fiat_element.tabulate(order, points)
    values = tab[(0,) * sd]
    T = np.zeros(values.shape[:-1] + (sd,) * order + values.shape[-1:])
    prefix = (slice(None),) * (values.ndim - 1)
    for index in np.ndindex((sd,) * order):
        alpha = [0] * sd
        for k in index:
            alpha[k] += 1
        T[prefix + index] = tab[tuple(alpha)]
    if DIVERGENCE in mappings:
        T = np.trace(T, axis1=values.ndim - 2, axis2=-2)[..., None, :]
    return T


def pullback_map(mapping, J):
    """The matrix by which the pullback of a basis function acts on one mapping.

    :arg mapping: The mapping kind.
    :arg J: The (numeric) cell Jacobian.
    :returns: The matrix.
    """
    if mapping == "contravariant piola":
        return J / np.linalg.det(J)
    if mapping == DIVERGENCE:
        return np.ones((1, 1)) / np.linalg.det(J)
    if mapping == "affine":
        return np.eye(J.shape[0])
    return np.linalg.inv(J).T


def facet_direction_map(ref_el, entity, J, mapping):
    r"""Per-mapping physical direction map of a facet node.

    A derivative mapping maps its component along the FIAT normal
    :math:`\hat{n}` to the FIAT physical normal -- the norm-preserving
    rescaling of the cofactor image :math:`K\hat{n}` (:math:`K =
    \operatorname{adj}(J)^T` maps normals to normals, and the norm of
    the FIAT normal depends only on the reference cell) -- and its
    tangential complement by :math:`J` (mapped tangents).

    A contravariant value mapping maps its component along the scaled
    normal :math:`\hat\nu^s` by :math:`K` (the cofactor lemma
    :math:`K\hat\nu^s = \nu^s` is exact), and its tangential complement
    to the cofactor image projected onto the physical facet, scaled by
    :math:`|K\hat\nu^s|^2/(\det J\, |\hat\nu^s|^2)` -- the reciprocal
    of the scalar tangential push-forward coefficient, which in 2D
    reduces to the mapped tangent :math:`J\hat{t}`.

    :arg ref_el: The reference cell.
    :arg entity: The facet number.
    :arg J: The (numeric) cell Jacobian.
    :arg mapping: The mapping kind.
    :returns: The map as an ``(sd, sd)`` array acting on one mapping.
    """
    sd = ref_el.get_spatial_dimension()
    detJ = np.linalg.det(J)
    K = detJ * np.linalg.inv(J).T
    if mapping == DIVERGENCE:
        return np.ones((1, 1))
    if mapping == "affine":
        return np.eye(sd)
    if mapping != "contravariant piola":
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


def physical_node_row(fiat_element, ell, dim, entity, J, avg):
    """B row of one parseable node, or None if push-forward invariant.

    The physical node keeps the points and coefficients of the reference
    node with each mapping mapped by the FIAT convention for its entity:
    point data keeps Cartesian directions, facet nodes are framed on the
    physical facet, and interior moments are invariant.  The mappings of
    the tabulation carry the pullback of the basis functions.

    :arg fiat_element: The FIAT element.
    :arg ell: The :class:`FunctionalData` of the reference node.
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
    single = len(ell.points) == 1
    if ell.order == 0 and ell.rank == 0:
        return None
    if dim == sd - 1 and not (single and ell.order == 0 and ell.rank == 1):
        maps = [facet_direction_map(ref_el, entity, J, mapping) for mapping in ell.mappings]
    elif dim == sd and not single and ell.order == 0:
        return None
    else:
        maps = [np.eye(1 if mapping == DIVERGENCE else sd) for mapping in ell.mappings]
    T = tabulate(fiat_element, ell.mappings, ell.points)
    for k, mapping in enumerate(ell.mappings):
        ell = ell.contract(maps[k], k)
        T = contract(T, pullback_map(mapping, J), 1 + k)
    row = ell.evaluate(T)
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
    mappings = fiat_element.mapping()
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
                    ell = FunctionalData.from_fiat(nodes[i], mappings[i])
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
