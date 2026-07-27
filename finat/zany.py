r"""Family mixins for automatically transformed elements.

The basis transformation itself is derived generically by
:meth:`finat.physically_mapped.PhysicallyMappedElement.basis_transformation`,
which assembles the generalized Vandermonde matrix :math:`B_{ij} =
n_i(\hat\psi_j\circ F^{-1})` from the FIAT dual basis and inverts it by
sparse block back-substitution over its numerically inferred sparsity
pattern.  The classes here only carry the family-specific conventions:

* :class:`ScalarPhysicallyMappedElement`, for scalar elements with an
  affine (identity) pullback -- Morley, Hermite, Argyris, Bell.  The
  physical facet nodes take their normal component along the unit
  physical normal and their tangential components along the mapped
  reference tangents; the ``avg`` attribute records whether physical
  facet moments are integral averages or plain integrals.

* :class:`PiolaPhysicallyMappedElement`, for vector- or tensor-valued
  elements under the (double) contravariant Piola pullback --
  Mardal-Tai-Winther, Johnson-Mercier, Arnold-Winther, Guzman-Neilan.
  The roles of the normal and tangential directions are mirrored: the
  scaled facet normal is the cofactor image of the reference one, so
  pure normal-component moments are invariant, while scaled tangents
  map by the Jacobian.  Vertex values of tensor-valued fields are
  rescaled by :math:`h^{-2}` following the hand-coded Arnold-Winther
  and Hu-Zhang conventions.
"""

import numpy as np

from FIAT.finite_element import FiniteElement
from gem.utils import cached_property

from finat.fiat_elements import FuseElement
from finat.functional import PhysicallyMappedFunctional, multiindices
from finat.physically_mapped import PhysicallyMappedElement


class ScalarPhysicallyMappedElement(PhysicallyMappedElement):
    """Mixin for scalar elements with an affine (identity) pullback."""

    def _check_mapping(self, fiat_element: FiniteElement) -> None:
        """Reject FIAT elements that are not affinely mapped.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not affine.
        """
        mappings = set(fiat_element.mapping())
        if mappings != {"affine"}:
            raise NotImplementedError(
                f"{type(self).__name__} expects an affine pullback, not {mappings}.")


class PiolaPhysicallyMappedElement(PhysicallyMappedElement):
    """Mixin for vector- or tensor-valued elements under the (double)
    contravariant Piola pullback."""

    def _check_mapping(self, fiat_element: FiniteElement) -> None:
        """Reject FIAT elements that are not (double) contravariant Piola mapped.

        :arg fiat_element: The FIAT element defined on the reference cell.
        :raises NotImplementedError: If the pullback is not supported.
        """
        mappings = set(fiat_element.mapping())
        if mappings not in ({"contravariant piola"}, {"double contravariant piola"}):
            raise NotImplementedError(
                f"{type(self).__name__} expects a (double) contravariant "
                f"Piola pullback, not {mappings}.")

    def dof_scale(self, node, dim: int, havg):
        r"""Return the conditioning rescaling factor of one physical dof.

        On top of the default derivative-order convention, vertex point
        evaluations of tensor-valued (rank-2) fields are redefined with a
        factor :math:`h^{-2}`, following the hand-coded Arnold-Winther and
        Hu-Zhang transformations; facet moments are already scaled by the
        facet measure in FIAT and need no further rescaling.

        :arg node: The FIAT functional of the dof.
        :arg dim: Topological dimension of the entity the dof sits on.
        :arg havg: GEM scalar for the cell size averaged over the vertices
            of the dof's entity.
        :returns: The GEM scaling factor, or ``None`` for no rescaling.
        """
        if dim == 0 and node.pt_dict:
            comps = {comp for pt in node.pt_dict for w, comp in node.pt_dict[pt]}
            if len(max(comps)) == 2:
                return havg**(-2)
        return super().dof_scale(node, dim, havg)


class ScalarZanyFuseElement(ScalarPhysicallyMappedElement, FuseElement):
    """A FUSE element whose dofs involve derivatives.

    Such an element is not affinely equivalent, so its reference basis must
    be transformed on each physical cell. The transformation is derived
    generically from the dual basis by
    :meth:`~finat.physically_mapped.PhysicallyMappedElement.basis_transformation`.
    """

    @cached_property
    def _fuse_dofs(self):
        """Map from dual basis index to the FUSE dof it was built from."""
        return {self.triple.dof_id_to_fiat_id[dof.id]: dof
                for dof in self.triple.generate()}

    def _functional_from_node(self, node, index, mapping):
        """Take the derivative direction from the FUSE trace that defines it.

        FUSE states the direction of a derivative dof symbolically, so it can
        be used as it stands instead of being recovered from the derivative
        weights by the factorization in
        :meth:`~finat.functional.PhysicallyMappedFunctional.from_fiat`.
        """
        direction = self._fuse_direction(node, index)
        if direction is None:
            return super()._functional_from_node(node, index, mapping)

        sd = node.ref_el.get_spatial_dimension()
        order = node.max_deriv_order
        alphas = multiindices(sd, order)
        lookup = {alpha: k for k, alpha in enumerate(alphas)}
        points = tuple(node.deriv_dict)
        weights = np.zeros((len(points), len(alphas)))
        for q, pt in enumerate(points):
            for w, alpha, comp in node.deriv_dict[pt]:
                weights[q, lookup[tuple(alpha)]] += w

        # the weights are the direction scaled point by point
        scale = weights @ direction / (direction @ direction)
        if not np.allclose(np.outer(scale, direction), weights):
            return super()._functional_from_node(node, index, mapping)
        return PhysicallyMappedFunctional(points, scale, order=order,
                                          direction=direction, mapping=mapping)

    def _fuse_direction(self, node, index):
        """The exact derivative direction of a dof, or None if it has none."""
        if not node.deriv_dict or node.get_point_dict():
            return None
        dof = self._fuse_dofs.get(index)
        if dof is None or dof.target_space is None:
            return None
        terms = dof.target_space.tabulate_derivs(None, dof.cell_defined_on)
        if not terms:
            return None
        sd = node.ref_el.get_spatial_dimension()
        alphas = multiindices(sd, node.max_deriv_order)
        lookup = {alpha: k for k, alpha in enumerate(alphas)}
        direction = np.zeros(len(alphas))
        for coeff, alpha in terms:
            if tuple(alpha) not in lookup:
                return None
            direction[lookup[tuple(alpha)]] += coeff
        if not direction.any():
            return None
        return direction


def is_scalar_zany(fiat_element):
    """Whether a FUSE-generated FIAT element needs an affine zany transformation.

    True for affinely pulled back elements carrying at least one derivative
    dof. Elements whose dofs are all point values or moments (Lagrange,
    Raviart-Thomas, Nedelec) are affinely equivalent and need no transformation.

    :arg fiat_element: The FIAT element defined on the reference cell.
    """
    if set(fiat_element.mapping()) != {"affine"}:
        return False
    return any(node.max_deriv_order > 0 for node in fiat_element.dual_basis())
