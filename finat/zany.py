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

from FIAT.finite_element import FiniteElement

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
