# Copyright (C) 2010-2012 Marie E. Rognes
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.functional import FacetIntegralMomentBlock, PointDirectionalEvaluationBlock
from FIAT.raviart_thomas import RaviartThomas
from FIAT.quadrature_schemes import create_quadrature
from FIAT.check_format_variant import check_format_variant


class NedelecSecondKindDual(DualSet):
    r"""
    This class represents the dual basis for the Nedelec H(curl)
    elements of the second kind. The degrees of freedom (L) for the
    elements of the k'th degree are

    d = 2:

      vertices: None

      edges:    L(f) = f (x_i) * t       for (k+1) points x_i on each edge

      cell:     L(f) = \int f * g * dx   for g in RT_{k-1}


    d = 3:

      vertices: None

      edges:    L(f)  = f(x_i) * t         for (k+1) points x_i on each edge

      faces:    L(f) = \int_F f * g * ds   for g in RT_{k-1}(F) for each face F

      cell:     L(f) = \int f * g * dx     for g in RT_{k-2}

    Higher spatial dimensions are not yet implemented. (For d = 1,
    these elements coincide with the CG_k elements.)
    """
    def __init__(self, cell, degree, variant, interpolant_deg):
        # Extract spatial dimension and topology
        d = cell.get_spatial_dimension()
        assert (d in (2, 3)), "Second kind Nedelecs only implemented in 2/3D."

        # Zero vertex-based degrees of freedom
        nodes = []

        # (degree+1) degrees of freedom per entity of codimension 1 (edges)
        nodes.extend(self._generate_edge_dofs(cell, degree, variant, interpolant_deg))

        # Include face degrees of freedom if 3D
        if d == 3:
            nodes.extend(self._generate_facet_dofs(d-1, cell, degree,
                                                   variant, interpolant_deg))

        # Varying degrees of freedom (possibly zero) per cell
        nodes.extend(self._generate_facet_dofs(d, cell, degree, variant, interpolant_deg))

        # Call init of super-class
        super().__init__(nodes, cell)

    def _generate_edge_dofs(self, cell, degree, variant, interpolant_deg):
        """Generate degrees of freedom (dofs) for entities of
        codimension 1 (edges)."""

        if variant == "integral":
            return self._generate_facet_dofs(1, cell, degree, variant, interpolant_deg)

        # (degree+1) tangential component point evaluation degrees of
        # freedom per entity of codimension 1 (edges)
        top = cell.get_topology()
        nodes = []
        for edge in top[1]:
            # A tangential component evaluation for each point
            nodes.append(PointDirectionalEvaluationBlock(cell, 1, edge, direction="tangential", degree=degree+2))

        return nodes

    def _generate_facet_dofs(self, dim, cell, degree, variant, interpolant_deg):
        """Generate degrees of freedom (dofs) for facets."""

        # Initialize empty dofs
        top = cell.get_topology()
        nodes = []

        # Return empty info if not applicable
        rt_degree = degree - dim + 1
        if rt_degree < 1:
            return nodes

        if interpolant_deg is None:
            interpolant_deg = degree

        # Construct quadrature scheme for the reference facet
        ref_facet = cell.construct_subelement(dim)
        Q_ref = create_quadrature(ref_facet, interpolant_deg + rt_degree)

        # Construct Raviart-Thomas on the reference facet
        if dim == 1:
            Phi = ONPolynomialSet(ref_facet, rt_degree, shape=(dim,))
            mapping = "contravariant piola"
        else:
            RT = RaviartThomas(ref_facet, rt_degree, variant)
            Phi = RT.get_nodal_basis()
            mapping, = set(RT.mapping())

        # Evaluate basis functions at reference quadrature points

        # Iterate over the facets
        for entity in top[dim]:
            # Construct degrees of freedom as integral moments on this cell,
            # using the face quadrature weighted against the values
            # of the (physical) Raviart--Thomas'es on the face
            nodes.append(FacetIntegralMomentBlock(cell, dim, entity, Q_ref, Phi, mapping=mapping))

        return nodes


class NedelecSecondKind(CiarletElement):
    """
    The H(curl) Nedelec elements of the second kind on triangles and
    tetrahedra: the polynomial space described by the full polynomials
    of degree k, with a suitable set of degrees of freedom to ensure
    H(curl) conformity.

    :arg ref_el: The reference element.
    :arg degree: The degree.
    :arg variant: optional variant specifying the types of nodes.

    variant can be chosen from ["point", "integral", "integral(q)"]
    "point" -> dofs are evaluated by point evaluation. Note that this variant
    has suboptimal convergence order in the H(curl)-norm
    "integral" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence.
    "integral(q)" -> dofs are evaluated by quadrature rules with the minimum
    degree required for unisolvence plus q. You might want to choose a high
    quadrature degree to make sure that expressions will be interpolated
    exactly. This is important when you want to have (nearly) curl-preserving
    interpolation.
    """

    def __init__(self, ref_el, degree, variant=None):
        splitting, variant, interpolant_deg = check_format_variant(variant, degree)
        if splitting is not None:
            ref_el = splitting(ref_el)

        # Check degree
        assert degree >= 1, "Second kind Nedelecs start at 1!"

        # Get dimension
        d = ref_el.get_spatial_dimension()
        # Construct polynomial basis for d-vector fields
        Ps = ONPolynomialSet(ref_el, degree, (d, ))

        # Construct dual space
        Ls = NedelecSecondKindDual(ref_el, degree, variant, interpolant_deg)

        # Set form degree
        formdegree = 1  # 1-form

        # Set mapping
        mapping = "covariant piola"

        # Call init of super-class
        super().__init__(Ps, Ls, degree, formdegree, mapping=mapping)
