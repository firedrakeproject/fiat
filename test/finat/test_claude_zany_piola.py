r"""Numeric prototype of the unified transformation theory for Piola elements.

On an affine cell, every Piola pullback factors through the componentwise
affine pullback: :math:`F^*_{piola} = \theta_A \circ F^*_{affine}`, with
:math:`\theta_A` pointwise multiplication of the value components by a
constant matrix.  Dually, the push-forward of a degree of freedom acts
slot-locally on its weight tensor: each contravariant value slot is
contracted with :math:`\Theta^T = J^T/\det J = K^{-1}` (:math:`K` the
cofactor matrix of :math:`J`), and each derivative slot with :math:`J^{-1}`,
exactly as in the scalar theory.  The matrix :math:`V` relating reference
nodes to push-forwards of physical nodes is then obtained by duality alone:

.. math:: B_{ij} = F_*(n_i)(\hat\psi_j), \qquad V = B^{-1},

where the physical node :math:`n_i` is defined from the reference node by
the FIAT frame conventions:

* facet moments: normal profiles against the physical scaled normal
  :math:`K\hat\nu^s` (invariant, by the exact covariance of
  ``compute_scaled_normal``), tangential profiles against the mapped
  tangents :math:`J\hat{t}` in 2D, and against cross products
  :math:`\nu^s \times J\hat{b}` of the scaled normal with mapped in-plane
  vectors in 3D, whose push-forward has the closed form implemented in
  :func:`facet_frame_net` (a *scalar* tangential block -- the ``Y`` mixing
  matrix and ``Sinv`` reciprocal-basis correction of
  :class:`finat.zany.PiolaFacetFrame` are consequences of this formula);
* single-point evaluations (vertex, edge, or interior point data): Cartesian
  components kept, so the slot map is plain :math:`\Theta^T`;
* interior moments: invariant by convention (physical test functions are
  Piola-mapped);
* divergence nodes: the value and derivative slot maps contract to
  :math:`\delta/\det J`, so the row is :math:`\det J` times the identity.

This module asserts that :math:`B^{-1}` agrees with
``basis_transformation`` to machine precision for the Piola element zoo,
in 2D and 3D, on cells of both orientations -- including hand-coded
elements not yet handled by :class:`finat.zany.PiolaPhysicallyMappedElement`
(Guzman-Neilan second kind, Hu-Zhang).  See ``zany_claude.md`` (Stage 4)
for the derivation.
"""

import numpy as np
import pytest

import FIAT
import finat
from FIAT.reference_element import make_affine_mapping, ufc_simplex
from finat.functional import PhysicallyMappedFunctional
from gem.interpreter import evaluate

from .conftest import MyMapping


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


def facet_frame_net(ref_el, entity, J):
    r"""Net Cartesian map taking a facet node's reference weights to the
    weights of its pushed-forward physical partner.

    In frame coordinates :math:`[\hat\nu^s | \hat{t}_l]` the map is

    .. math::
        N = \begin{pmatrix} 1 & -\det J\, \hat{t}_l \cdot M^{-1}\hat\nu^s
                                / |\hat\nu^s|^2 \\
                            0 & s I \end{pmatrix},
        \qquad
        s = \det J\, \frac{\hat\nu^s \cdot M^{-1}\hat\nu^s}{|\hat\nu^s|^2},
        \qquad M = J^T J,

    for the 3D cross-product tangential convention; in 2D the tangential
    directions are the plain mapped tangents.  The top-right entries are the
    completion residuals: a pulled-back tangential profile leaves behind a
    normal-direction functional, eliminated through the generalized
    Vandermonde row exactly as in the scalar theory.

    :arg ref_el: The reference cell.
    :arg entity: The facet number.
    :arg J: The (numeric) cell Jacobian.
    :returns: The net map as an (sd, sd) array acting on Cartesian weights.
    """
    sd = ref_el.get_spatial_dimension()
    detJ = np.linalg.det(J)
    nu = ref_el.compute_scaled_normal(entity)
    tangents = ref_el.compute_tangents(sd - 1, entity)
    Ghat = np.column_stack([nu, *tangents])
    if sd == 2:
        K = detJ * np.linalg.inv(J).T
        P = np.column_stack([K @ nu, J @ tangents[0]])
        return (J.T / detJ) @ P @ np.linalg.inv(Ghat)
    M = J.T @ J
    Minv_nu = np.linalg.solve(M, nu)
    nn = nu @ nu
    N = np.zeros((sd, sd))
    N[0, 0] = 1.0
    for l, t in enumerate(tangents):
        N[l + 1, l + 1] = detJ * (nu @ Minv_nu) / nn
        N[0, l + 1] = -detJ * (t @ Minv_nu) / nn
    return Ghat @ N @ np.linalg.inv(Ghat)


def composition_transformation(fiat_element, J):
    """Compute V for a contravariant Piola element by duality alone.

    :arg fiat_element: The FIAT element on the reference cell.
    :arg J: The (numeric) cell Jacobian.
    :returns: The transformation V as a numpy array, with the same row and
        column ordering as ``basis_transformation`` before truncation of
        the trailing constraint columns.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    detJ = np.linalg.det(J)
    Theta_T = J.T / detJ
    nodes = fiat_element.dual_basis()
    entity_ids = fiat_element.entity_dofs()
    B = np.eye(len(nodes))
    for dim in entity_ids:
        for entity in entity_ids[dim]:
            for i in entity_ids[dim][entity]:
                ell = PhysicallyMappedFunctional.from_fiat(nodes[i])
                if ell.divergence:
                    B[i] = evaluate_divergence(ell, fiat_element) / detJ
                    continue
                point_data = (len(ell.points) == 1
                              and (ell.rank == 1 or dim != sd - 1))
                if dim == sd and not point_data:
                    continue
                if dim == sd - 1 and not point_data:
                    net = facet_frame_net(ref_el, entity, J)
                else:
                    net = Theta_T
                W = ell.weights.reshape(-1, *(sd,) * ell.rank)
                for _ in range(ell.rank):
                    W = np.tensordot(W, net, axes=(1, 1))
                W = W.reshape(len(ell.points), -1)
                B[i] = PhysicallyMappedFunctional(
                    ell.points, W, rank=ell.rank).evaluate(fiat_element)
    return np.linalg.inv(B)


piola_zoo = {
    2: [(finat.MardalTaiWinther, ()),
        (finat.JohnsonMercier, ()),
        (finat.ArnoldWintherNC, ()),
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


@pytest.mark.parametrize("orientation", ["positive", "negative"])
@pytest.mark.parametrize("dimension, element, args", [
    (dim, *case) for dim in piola_zoo for case in piola_zoo[dim]])
def test_piola_composition(dimension, element, args, orientation):
    ref_cell = ufc_simplex(dimension)
    phys_cell = ufc_simplex(dimension)
    phys_cell.vertices = orientations[(dimension, orientation)]
    J, b = make_affine_mapping(ref_cell.vertices, phys_cell.vertices)

    finat_element = element(ref_cell, *args)
    mapping = MyMapping(ref_cell, phys_cell)
    M = evaluate([finat_element.basis_transformation(mapping)])[0].arr

    V = composition_transformation(finat_element._element, J)
    V = V[:, :finat_element.space_dimension()]
    assert np.allclose(V, M.T, atol=1e-10)
