"""Automatic basis transformations for physically mapped elements.

This module automates the transformation theory of Kirby (2017) and
Brubeck & Kirby (2025).  Given a FIAT element whose degrees of freedom
are not preserved under push-forward, it constructs the matrix
:math:`V` relating the reference nodes to the push-forwards of the
physical nodes, so that the physical basis functions are obtained as
:math:`M F^*(\\hat\\Psi)` with :math:`M = V^T`.

The construction follows the factorization :math:`V = E V^c D`:

* each reference node is pulled back to the physical cell and expanded
  by the chain rule in a frame adapted to its entity: the physical
  direction appearing in the corresponding physical node (e.g. the
  physical facet normal) completed with the push-forwards of the
  reference tangents (the :math:`V^c` factor and the extraction
  :math:`E`);
* the completion functionals are derivatives along *mapped* reference
  tangents, so they coincide with reference functionals whose expansion
  in the element's own nodes is a purely numeric generalized Vandermonde
  row (the :math:`D` factor), computed by dual evaluation instead of
  hand-derived univariate exactness rules.
"""

from functools import reduce
from operator import add

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.functional import (Functional, IntegralMoment,
                             IntegralMomentOfDerivative,
                             IntegralMomentOfNormalDerivative,
                             PointEvaluation)
from gem import Literal, ListTensor, Node, Power
from finat.physically_mapped import (PhysicalGeometry, adjugate,
                                     determinant, identity)


def dual_evaluation_matrix(fiat_element: FiniteElement,
                           functionals: list[Functional]) -> numpy.ndarray:
    """Evaluate functionals against the nodal basis of a FIAT element.

    This is the generalized Vandermonde computation providing the
    numeric coefficients of the :math:`D` factor: the restriction of a
    functional :math:`\\ell` to the polynomial space :math:`P` satisfies
    :math:`\\pi \\ell = \\sum_j \\ell(\\psi_j)\\, \\pi n_j`.

    Parameters
    ----------
    fiat_element :
        The FIAT element providing the nodal basis.
    functionals :
        Functionals to evaluate.

    Returns
    -------
    numpy.ndarray
        Matrix with entry (k, j) equal to the k-th functional applied
        to the j-th nodal basis function.
    """
    poly_set = fiat_element.get_nodal_basis()
    coeffs = poly_set.get_coeffs()
    riesz = numpy.array([f.to_riesz(poly_set).flatten() for f in functionals])
    return riesz @ coeffs.reshape(coeffs.shape[0], -1).T


def is_invariant(node: Functional) -> bool:
    """Return whether a functional is preserved under push-forward.

    Point evaluations and integral moments of the function value against
    an intrinsically defined weight (constructed from the same
    reference-facet quadrature rule on both cells) satisfy
    :math:`F_*(n) = \\hat{n}`, so their rows of :math:`V` are identity.
    """
    return type(node) in {PointEvaluation, IntegralMoment}


def _normal_tangential_frame(fiat_element: FiniteElement, entity: int,
                             J: Node, detJ: Node) -> tuple:
    """Chain-rule data for the facet normal/tangential frame.

    The pullback of the reference normal-derivative direction expands in
    the frame of the physical facet normal and the mapped reference
    tangents,

    .. math:: J\\hat{n} = a\\, n + \\sum_k b_k\\, J\\hat{t}_k.

    Orthogonality of the physical normal to the mapped tangents and the
    identical (tangent-based) normal convention on both cells reduce the
    coefficients to Gram-matrix algebra:
    :math:`a = \\det J \\sqrt{\\det\\hat{G}/\\det G}` and
    :math:`b = G^{-1} T^T J\\hat{n}` with :math:`T = [J\\hat{t}_k]` and
    Gram matrices :math:`G = T^T T`, :math:`\\hat{G}_{kl} = \\hat{t}_k
    \\cdot \\hat{t}_l`.

    Parameters
    ----------
    fiat_element :
        The FIAT element, providing the reference cell.
    entity :
        The facet number.
    J :
        GEM expression for the cell Jacobian.
    detJ :
        GEM expression for the Jacobian determinant.

    Returns
    -------
    tuple
        ``(a, b)`` with ``a`` the GEM coefficient of the physical
        normal node and ``b`` the list of GEM coefficients of the
        mapped tangential functionals.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    that = ref_el.compute_tangents(sd - 1, entity)
    nhat = ref_el.compute_normal(entity)

    Jn = J @ Literal(nhat)
    Jt = [J @ Literal(t) for t in that]
    G = numpy.array([[Jt[k] @ Jt[l] for l in range(sd - 1)]
                     for k in range(sd - 1)], dtype=object)
    detG = determinant(G)
    adjG = adjugate(G)
    Tn = [Jt[k] @ Jn for k in range(sd - 1)]
    b = [reduce(add, (adjG[k, l] * Tn[l] for l in range(sd - 1))) / detG
         for k in range(sd - 1)]

    Ghat = numpy.dot(that, that.T)
    a = detJ * Literal(numpy.linalg.det(Ghat) ** 0.5) / Power(detG, Literal(0.5))
    return a, b


def _facet_normal_moment_rows(V: numpy.ndarray, dofs: list,
                              fiat_element: FiniteElement, entity: int,
                              J: Node, detJ: Node, invariant: set,
                              tol: float) -> None:
    """Fill the rows of V for normal-derivative moments on a facet.

    Parameters
    ----------
    V :
        Object array being assembled, with entry (i, j) relating
        reference node i to the push-forward of physical node j.
    dofs :
        Indices of the normal-derivative moment nodes on this facet.
    fiat_element :
        The FIAT element.
    entity :
        The facet number.
    J, detJ :
        GEM expressions for the cell Jacobian and its determinant.
    invariant :
        Indices of the push-forward invariant nodes.
    tol :
        Tolerance for detecting zeros in the numeric completion
        coefficients.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    that = ref_el.compute_tangents(sd - 1, entity)
    nodes = fiat_element.dual_basis()

    a, b = _normal_tangential_frame(fiat_element, entity, J, detJ)

    # Nodal completion: same integral moment rule, tangential directions.
    completion = [IntegralMomentOfDerivative(ref_el, nodes[i].Q,
                                             nodes[i].f_at_qpts, t)
                  for i in dofs for t in that]
    C = dual_evaluation_matrix(fiat_element, completion)
    C[abs(C) < tol] = 0

    for row, (i, bk) in zip(C, ((i, bk) for i in dofs for bk in b)):
        V[i, i] = a
        for j in numpy.flatnonzero(row):
            if j not in invariant:
                raise NotImplementedError(
                    f"Completion of node {i} couples to node {j} of type "
                    f"{type(nodes[j]).__name__}, which is not yet handled.")
            V[i, j] = V[i, j] + Literal(row[j]) * bk


def _conditioning_scaling(V: numpy.ndarray, fiat_element: FiniteElement,
                          coordinate_mapping: PhysicalGeometry) -> None:
    """Rescale derivative degrees of freedom by the cell size.

    Each physical node of derivative order :math:`m` is redefined with a
    factor :math:`h^{-m}`, where :math:`h` averages the cell size over
    the vertices of its entity.  This is the FInAT convention keeping
    the mass matrix well-conditioned; it is consistent across cells
    because the scaling only depends on shared entities.

    Parameters
    ----------
    V :
        Object array being assembled; columns are rescaled in place.
    fiat_element :
        The FIAT element.
    coordinate_mapping :
        Object providing the physical geometry as GEM expressions.
    """
    # cell_size may be a GEM expression or a numpy array of numbers
    h = coordinate_mapping.cell_size()
    top = fiat_element.get_reference_element().get_topology()
    nodes = fiat_element.dual_basis()
    entity_ids = fiat_element.entity_dofs()
    for dim in entity_ids:
        for entity in entity_ids[dim]:
            verts = top[dim][entity]
            havg = reduce(add, (h[v] for v in verts)) / len(verts)
            for i in entity_ids[dim][entity]:
                order = nodes[i].max_deriv_order
                if order > 0:
                    V[:, i] = V[:, i] * havg**(-order)


def zany_basis_transformation(fiat_element: FiniteElement,
                              coordinate_mapping: PhysicalGeometry,
                              tol: float = 1e-12) -> ListTensor:
    """Compute the basis transformation matrix of a FIAT element.

    Parameters
    ----------
    fiat_element :
        The FIAT element defined on the reference cell.
    coordinate_mapping :
        Object providing the physical geometry as GEM expressions.
    tol :
        Tolerance for detecting zeros in the numeric completion
        coefficients.

    Returns
    -------
    gem.ListTensor
        The transformation :math:`M = V^T` mapping pulled-back reference
        basis functions to physical nodal basis functions.
    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    bary, = ref_el.make_points(sd, 0, sd + 1)
    J = coordinate_mapping.jacobian_at(bary)
    detJ = coordinate_mapping.detJ_at(bary)

    nodes = fiat_element.dual_basis()
    invariant = {i for i, node in enumerate(nodes) if is_invariant(node)}

    V = identity(fiat_element.space_dimension())
    entity_ids = fiat_element.entity_dofs()
    for dim in entity_ids:
        for entity in entity_ids[dim]:
            dofs = [i for i in entity_ids[dim][entity] if i not in invariant]
            if not dofs:
                continue
            if all(isinstance(nodes[i], IntegralMomentOfNormalDerivative)
                   for i in dofs):
                _facet_normal_moment_rows(V, dofs, fiat_element, entity,
                                          J, detJ, invariant, tol)
            else:
                unhandled = {type(nodes[i]).__name__ for i in dofs}
                raise NotImplementedError(
                    f"Cannot yet transform nodes of type {unhandled}.")

    _conditioning_scaling(V, fiat_element, coordinate_mapping)
    return ListTensor(V.T)
