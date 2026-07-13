r"""Automatic basis transformations for physically mapped elements.

This module automates the transformation theory of Kirby (2017) and
Brubeck & Kirby (2025).  Given a FIAT element whose degrees of freedom
are not preserved under push-forward, it constructs the matrix
:math:`V` relating the reference nodes to the push-forwards of the
physical nodes, so that the physical basis functions are obtained as
:math:`M F^*(\\hat\\Psi)` with :math:`M = V^T`.

Degrees of freedom are represented symbolically by
:class:`finat.functional.Functional` and processed generically, without
dispatching over FIAT functional types.  Each reference node is pulled
back to the physical cell by the chain rule and expanded in the frame
of the physical facet normal and the mapped reference tangents; the
tangential components are derivatives along *mapped* reference tangents
and therefore coincide with reference functionals, whose expansion in
the element's own nodes is a purely numeric generalized Vandermonde
row.  In the language of the theory, the frame expansion realizes
:math:`E V^c` and the numeric elimination of the tangential completion
realizes :math:`D`.
"""

from functools import reduce
from operator import add

import numpy

from FIAT.finite_element import FiniteElement
from gem import Literal, ListTensor, Node, Power, Zero
from finat.functional import Functional
from finat.physically_mapped import (PhysicalGeometry, adjugate,
                                     determinant, identity)


def generalized_cross(tangents) -> numpy.ndarray:
    r"""Generalized cross product of d-1 vectors in d dimensions.

    Parameters
    ----------
    tangents :
        A (d-1, d) array of vectors, with numeric or GEM entries.

    Returns
    -------
    numpy.ndarray
        The vector :math:`C` such that :math:`C \\cdot w =
        \\det([t_1; \\dots; t_{d-1}; w])` for all :math:`w`; it is
        orthogonal to every :math:`t_k`.

    """
    A = numpy.asarray(tangents)
    d = A.shape[1]
    cols = numpy.ones(d, dtype=bool)
    C = []
    for i in range(d):
        cols[i] = False
        C.append((-1) ** (d - 1 + i) * determinant(A[:, cols]))
        cols[i] = True
    return numpy.asarray(C)


class FacetFrame:
    r"""Normal/tangential frame of a facet and its push-forward.

    The reference frame consists of the FIAT facet normal
    :math:`\\hat{n}` and the scaled facet tangents :math:`\\hat{t}_k`;
    the physical frame consists of the physical facet normal and the
    mapped tangents :math:`J\\hat{t}_k`.  Because FIAT normals are
    computed from the tangents by the same formula on the reference and
    physical cells, the physical normal is :math:`\\kappa\\, C / \\|C\\|`
    with :math:`C` the generalized cross product of the mapped tangents
    and :math:`\\kappa` a cell-independent constant recovered from the
    reference data.

    Parameters
    ----------
    fiat_element :
        The FIAT element, providing the reference cell.
    entity :
        The facet number.
    J :
        GEM expression for the cell Jacobian.

    """

    def __init__(self, fiat_element: FiniteElement, entity: int, J: Node):
        ref_el = fiat_element.get_reference_element()
        sd = ref_el.get_spatial_dimension()
        self.tangents = ref_el.compute_tangents(sd - 1, entity)
        self.normal = ref_el.compute_normal(entity)

        Chat = generalized_cross(self.tangents)
        kappa = self.normal @ Chat / numpy.linalg.norm(Chat)

        self.mapped_tangents = [J @ Literal(t) for t in self.tangents]
        C = generalized_cross([[Jt[i] for i in range(sd)]
                               for Jt in self.mapped_tangents])
        A = numpy.empty((sd, sd), dtype=object)
        A[:, 0] = C
        for k, Jt in enumerate(self.mapped_tangents):
            A[:, k + 1] = [Jt[i] for i in range(sd)]
        self._adjA = adjugate(A)
        self._detA = determinant(A)

        normC = Power(reduce(add, (C[i] * C[i] for i in range(sd))),
                      Literal(0.5))
        self.normal_scale = normC / kappa
        vol = ref_el.volume_of_subcomplex(sd - 1, entity)
        self.measure = normC * (vol / numpy.linalg.norm(Chat))

    def reference_coefficients(self, direction: numpy.ndarray) -> numpy.ndarray:
        r"""Expand a numeric direction in the reference frame.

        Parameters
        ----------
        direction :
            A numeric direction vector.

        Returns
        -------
        numpy.ndarray
            Coefficients ``(a, b_1, ..., b_{d-1})`` such that the
            direction equals :math:`a\\hat{n} + \\sum_k b_k \\hat{t}_k`.

        """
        A = numpy.column_stack([self.normal, *self.tangents])
        return numpy.linalg.solve(A, direction)

    def decompose(self, direction: Node) -> list:
        r"""Expand a GEM direction in the un-normalized physical frame.

        Parameters
        ----------
        direction :
            A GEM direction vector.

        Returns
        -------
        list
            GEM coefficients ``(x_0, x_1, ..., x_{d-1})`` such that the
            direction equals :math:`x_0 C + \\sum_k x_k J\\hat{t}_k`.

        """
        sd = self._adjA.shape[0]
        return [reduce(add, (self._adjA[m, i] * direction[i]
                             for i in range(sd))) / self._detA
                for m in range(sd)]


def _weight_ratio(wi: numpy.ndarray, wj: numpy.ndarray, tol: float) -> float:
    """Return the scalar s with wi == s * wj, if it exists."""
    s = wi @ wj / (wj @ wj)
    if not numpy.allclose(wi, s * wj, atol=tol * numpy.linalg.norm(wi)):
        raise NotImplementedError("Weights are not parallel.")
    return s


def _conditioning_scaling(V: numpy.ndarray, fiat_element: FiniteElement,
                          coordinate_mapping: PhysicalGeometry) -> None:
    r"""Rescale derivative degrees of freedom by the cell size.

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
                              tol: float = 1e-12, avg: bool = True,
                              ndof: int = None) -> ListTensor:
    r"""Compute the basis transformation matrix of a FIAT element.

    Parameters
    ----------
    fiat_element :
        The FIAT element defined on the reference cell.
    coordinate_mapping :
        Object providing the physical geometry as GEM expressions.
    tol :
        Tolerance for detecting zeros in the numeric coefficients.
    avg :
        If False, physical facet moments are plain integrals rather than
        the measure-intrinsic integral averages of the reference nodes,
        and their columns are rescaled by the physical facet measure.
    ndof :
        Optional number of physical degrees of freedom; trailing columns
        are discarded, so that constrained elements can drop the basis
        functions of their extended element.

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

    nodes = fiat_element.dual_basis()
    V = identity(fiat_element.space_dimension())

    mappings = set(fiat_element.mapping())
    piola = mappings in ({"contravariant piola"},
                         {"double contravariant piola"})
    if not piola and mappings != {"affine"}:
        raise NotImplementedError(f"Cannot transform mappings {mappings}.")

    processed = set()
    entity_ids = fiat_element.entity_dofs()
    for dim in sorted(entity_ids):
        for entity in sorted(entity_ids[dim]):
            ells = {i: Functional.from_fiat(nodes[i])
                    for i in entity_ids[dim][entity]}
            if piola:
                # Interior moments are Piola invariant by construction
                processed.update(i for i, ell in ells.items()
                                 if ell.order == 0 and dim == sd)
                group = {i: ell for i, ell in ells.items()
                         if i not in processed}
                if not group:
                    continue
                if any(ell.rank == 0 or ell.order > 0
                       for ell in group.values()):
                    raise NotImplementedError(
                        "Cannot yet Piola-transform this node group.")
                if dim == sd - 1:
                    _piola_facet_rows(V, group, fiat_element, entity, J,
                                      processed, tol)
                else:
                    _piola_point_rows(V, group, J, processed, tol)
                continue
            # Value functionals are push-forward invariant
            processed.update(i for i, ell in ells.items() if ell.order == 0)
            group = {i: ell for i, ell in ells.items() if ell.order > 0}
            if not group:
                continue
            if dim == sd - 1:
                _facet_rows(V, group, fiat_element, entity, J, processed,
                            tol, avg)
            else:
                _point_jet_rows(V, group, J, processed, tol)

    _conditioning_scaling(V, fiat_element, coordinate_mapping)
    return ListTensor(V[:, :ndof].T)


def _facet_rows(V: numpy.ndarray, group: dict, fiat_element: FiniteElement,
                entity: int, J: Node, processed: set, tol: float,
                avg: bool = True) -> None:
    r"""Assemble the rows of V for derivative nodes on a facet.

    Physical facet nodes take their normal component along the physical
    facet normal and their tangential components along the mapped
    reference tangents.  The pulled-back reference node is expanded in
    this frame, and the tangential remainders, being derivatives along
    mapped reference tangents, coincide with reference functionals that
    are eliminated numerically through already assembled rows of V.

    Parameters
    ----------
    V :
        Object array being assembled.
    group :
        Mapping from node index to symbolic Functional for the
        derivative nodes on this facet.
    fiat_element :
        The FIAT element.
    entity :
        The facet number.
    J :
        GEM expression for the cell Jacobian.
    processed :
        Indices of the already assembled rows; updated in place.
    tol :
        Tolerance for detecting zeros in the numeric coefficients.
    avg :
        If False, physical facet moments are plain integrals rather than
        integral averages, and their columns are rescaled by the
        physical facet measure.

    """
    frame = FacetFrame(fiat_element, entity, J)
    for i, ell in group.items():
        # Split the direction into normal and tangential parts
        a, *beta = frame.reference_coefficients(ell.direction)
        if abs(a) < tol:
            # Mapped tangential derivatives are invariant
            processed.add(i)
            continue

        # Expand the pulled-back node in the physical frame
        x = frame.decompose(ell.pullback(J).direction)
        c = x[0] * frame.normal_scale / a
        if not avg and len(ell.points) > 1:
            # the physical moment is a plain integral, not an average
            c = c / frame.measure
        row = numpy.full(V.shape[1], Zero(), dtype=object)
        row[i] = c
        for k, that in enumerate(frame.tangents):
            r = x[k + 1] - c * beta[k]
            coefficients = ell.with_direction(that).evaluate(fiat_element)
            coefficients[abs(coefficients) < tol] = 0
            for j in numpy.flatnonzero(coefficients):
                if j not in processed:
                    raise NotImplementedError(
                        f"Completion of node {i} couples to node {j}, "
                        "which has not been transformed yet.")
                row = row + V[j, :] * (r * coefficients[j])
        V[i, :] = row
        processed.add(i)


def _point_jet_rows(V: numpy.ndarray, group: dict, J: Node,
                    processed: set, tol: float) -> None:
    r"""Assemble the rows of V for derivative nodes away from facets.

    Away from facets there is no geometric frame, and physical nodes
    keep the reference (Cartesian) directions, so the group must span
    all directions and acts as its own completion: this is the
    affine-interpolation equivalent case, and each pulled-back node is
    expanded within the group.

    Parameters
    ----------
    V :
        Object array being assembled.
    group :
        Mapping from node index to symbolic Functional for the
        derivative nodes on this entity.
    J :
        GEM expression for the cell Jacobian.
    processed :
        Indices of the already assembled rows; updated in place.
    tol :
        Tolerance for detecting zeros in the numeric coefficients.

    """
    suborders = {}
    for i, ell in group.items():
        suborders.setdefault(ell.order, {})[i] = ell

    for sub in suborders.values():
        directions = numpy.array([ell.direction for ell in sub.values()])
        if len(set(ell.points for ell in sub.values())) > 1:
            raise NotImplementedError("Group nodes at different points.")
        if directions.shape[0] != directions.shape[1]:
            raise NotImplementedError(
                "Directions do not span the derivative jet.")

        # coefficients of the direction basis expansion of each multi-index
        Dinv = numpy.linalg.inv(directions.T)
        for i, ell in sub.items():
            Jd = ell.pullback(J).direction
            for col, (j, ellj) in enumerate(sub.items()):
                s = _weight_ratio(ell.weights, ellj.weights, tol)
                x = s * Dinv[col]
                nz = numpy.flatnonzero(abs(x) > tol)
                V[i, j] = reduce(add, (Jd[m] * x[m] for m in nz)) if len(nz) else Zero()
            processed.add(i)


def _piola_facet_rows(V: numpy.ndarray, group: dict,
                      fiat_element: FiniteElement, entity: int, J: Node,
                      processed: set, tol: float) -> None:
    r"""Assemble the rows of V for facet moments of Piola-mapped values.

    This mirrors :func:`_facet_rows` with the roles of the normal and
    tangential directions exchanged: under the contravariant Piola map
    the scaled facet normal is the image of the reference one under the
    cofactor matrix :math:`K = \operatorname{adj}(J)^T`, so pure normal
    moments are invariant, while the scaled tangents map by :math:`J`.
    Each node carries a per-point profile of frame coordinates, shared
    by its physical counterpart; the pulled-back reference node mixes
    the coordinates through the frame expansion of :math:`K\hat{t}_k`,
    the tangential profiles are matched within the group, and the
    residual normal profile is eliminated numerically through already
    assembled rows of V.

    Parameters
    ----------
    V :
        Object array being assembled.
    group :
        Mapping from node index to symbolic Functional for the value
        moments on this facet.
    fiat_element :
        The FIAT element.
    entity :
        The facet number.
    J :
        GEM expression for the cell Jacobian.
    processed :
        Indices of the already assembled rows; updated in place.
    tol :
        Tolerance for detecting zeros in the numeric coefficients.

    """
    ref_el = fiat_element.get_reference_element()
    sd = ref_el.get_spatial_dimension()
    that = ref_el.compute_tangents(sd - 1, entity)
    nhat = ref_el.compute_scaled_normal(entity)
    Ghat = numpy.column_stack([nhat, *that])
    Ghatinv = numpy.linalg.inv(Ghat)

    Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                      dtype=object)
    K = adjugate(Jnp).T

    # Reference frame coordinate profiles, shared with the physical nodes
    coords = {}
    for i, ell in group.items():
        C = ell.weights.reshape(-1, *(sd,) * ell.rank)
        for _ in range(ell.rank):
            C = numpy.tensordot(C, Ghatinv, axes=(1, 1))
        coords[i] = C.reshape(len(ell.points), -1)
        # Pure normal moments are Piola invariant
        if numpy.allclose(coords[i][:, 1:], 0, atol=tol):
            processed.add(i)

    group = {i: ell for i, ell in group.items() if i not in processed}
    if not group:
        return
    rank, = {ell.rank for ell in group.values()}
    points, = {ell.points for ell in group.values()}

    # Frame coordinates of the mapped frame image of the reference frame:
    # the normal is invariant and the pulled tangents are expanded by a
    # symbolic solve in the mapped frame [K nhat | J that_k]
    A = numpy.column_stack([K @ nhat, *(Jnp @ t for t in that)])
    adjA = adjugate(A)
    detA = determinant(A)
    Y = numpy.full((sd, sd), Zero(), dtype=object)
    Y[0, 0] = Literal(1.0)
    for k, t in enumerate(that):
        Y[:, k + 1] = (adjA @ (K @ t)) / detA

    # Physical tangential components are built on the reciprocal basis
    # (cross products of the frame), so they carry the in-plane
    # contravariant transformation S = adj(G Ghat^{-1})^T of the change
    # of tangent Gram matrices.  Absorb S^{-1} into the coordinate
    # mixing so that the physical profiles keep the reference
    # coordinates; in 2D the tangent plane is one-dimensional and S = 1.
    G = numpy.array([[Jnp @ t1 @ (Jnp @ t2) for t2 in that] for t1 in that])
    Ghat_t = that @ that.T
    Sinv = (numpy.linalg.inv(Ghat_t) @ G) * \
        (numpy.linalg.det(Ghat_t) / determinant(G))
    Y[1:, :] = Sinv @ Y[1:, :]

    # Numeric matching of the tangential coordinate profiles in the group
    B = numpy.array([coords[j][:, 1:].ravel() for j in group])
    Bpinv = numpy.linalg.pinv(B)
    Bpinv[abs(Bpinv) < tol] = 0

    # Numeric elimination of the normal profile: one pure normal moment
    # per quadrature point, evaluated on the nodal basis
    ndir = numpy.ones(())
    for _ in range(rank):
        ndir = numpy.multiply.outer(ndir, nhat)
    T = fiat_element.tabulate(0, points)[(0,) * sd]
    L = numpy.einsum("jcq,c->jq", T.reshape(T.shape[0], -1, len(points)),
                     ndir.ravel())
    L[abs(L) < tol] = 0

    for i, ell in group.items():
        # Pull back the coordinate profile, contracting each slot with Y
        P = coords[i].reshape(-1, *(sd,) * rank)
        for _ in range(rank):
            P = numpy.tensordot(P, Y, axes=(1, 1))
        P = P.reshape(len(points), -1)

        row = numpy.full(V.shape[1], Zero(), dtype=object)
        c = Bpinv.T @ P[:, 1:].ravel()
        for cj, j in zip(c, group):
            row[j] = cj
        # Residual normal profile after removing the group contribution
        residual = P[:, 0] - c @ numpy.array([coords[j][:, 0] for j in group])
        for q in range(len(points)):
            for m in numpy.flatnonzero(L[:, q]):
                if m not in processed:
                    raise NotImplementedError(
                        f"Completion of node {i} couples to node {m}, "
                        "which has not been transformed yet.")
                row = row + V[m, :] * (residual[q] * L[m, q])
        V[i, :] = row
        processed.add(i)


def _piola_point_rows(V: numpy.ndarray, group: dict, J: Node,
                      processed: set, tol: float) -> None:
    r"""Assemble the rows of V for point values of Piola-mapped fields.

    This mirrors :func:`_point_jet_rows`: away from facets, physical
    point evaluations keep the reference (Cartesian) components, which
    pull back through the cofactor matrix :math:`K = \mathrm{adj}(J)^T`
    of the contravariant Piola map, so the group of components at each
    point acts as its own completion.

    Parameters
    ----------
    V :
        Object array being assembled.
    group :
        Mapping from node index to symbolic Functional for the value
        nodes on this entity.
    J :
        GEM expression for the cell Jacobian.
    processed :
        Indices of the already assembled rows; updated in place.
    tol :
        Tolerance for detecting zeros in the numeric coefficients.

    """
    sd = J.shape[0]
    Jnp = numpy.array([[J[i, k] for k in range(sd)] for i in range(sd)],
                      dtype=object)
    K = adjugate(Jnp).T

    subgroups = {}
    for i, ell in group.items():
        if len(ell.points) != 1 or ell.rank != 1:
            raise NotImplementedError(
                "Only single-point vector evaluations are handled.")
        subgroups.setdefault(ell.points, {})[i] = ell

    for sub in subgroups.values():
        directions = numpy.array([ell.weights[0] for ell in sub.values()])
        if directions.shape[0] != directions.shape[1]:
            raise NotImplementedError(
                "Directions do not span the vector components.")
        Dinv = numpy.linalg.inv(directions.T)
        for i, ell in sub.items():
            Kd = K @ ell.weights[0]
            for col, j in enumerate(sub):
                x = Dinv[col]
                nz = numpy.flatnonzero(abs(x) > tol)
                V[i, j] = reduce(add, (Kd[m] * x[m] for m in nz)) if len(nz) else Zero()
            processed.add(i)
