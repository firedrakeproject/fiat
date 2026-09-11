r"""Degrees of freedom as coefficient tensors.

Every FIAT functional built from point and derivative dictionaries has
the form

.. math:: \ell(f) = \sum_q \langle W_q, \nabla^m f(x_q) \rangle,

with reference points :math:`x_q` and coefficient tensors :math:`W_q`
carrying one index for each value component of :math:`f` and one for
each derivative.  :class:`FunctionalData` stores exactly this, together
with the mapping of each index, which says how that index transforms
under the pullback of the basis functions: a component index carries
the Piola mapping of the element, a derivative index the chain rule,
and a divergence index (a contravariant component contracted with a
derivative) the scalar :math:`1/\det J`.  Point evaluations, integral
moments, normal derivatives, divergences and tensor divergences are
all instances of the same data, so the transformation theory of
:mod:`finat.physically_mapped` never dispatches on FIAT functional
types.
"""

from itertools import permutations

import numpy

#: The mapping of a derivative index of the coefficient tensor.
DERIVATIVE = "derivative"

#: The mapping of a divergence index of the coefficient tensor.
DIVERGENCE = "divergence"


def component_mappings(mapping, rank):
    """The mapping of each component index of a functional.

    :arg mapping: The FIAT mapping of the basis functions the functional
        acts on, e.g. ``"affine"`` or ``"double contravariant piola"``.
    :arg rank: The number of component indices of the functional.
    :returns: A tuple with the Piola mapping of each component index.
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


class FunctionalData:
    """A degree of freedom as a coefficient tensor at each point.

    :arg points: The points, a tuple of reference coordinates.
    :arg coefficients: An array of shape ``(len(points), sd, ..., sd)``
        with one trailing axis per index of the coefficient tensor, the
        component indices first.  The axis of a divergence index has
        length one.
    :arg mappings: The mapping of each index: a Piola mapping for a
        component index, :data:`DERIVATIVE` for a derivative index, or
        :data:`DIVERGENCE` for the last index.
    """
    def __init__(self, points, coefficients, mappings):
        self.points = tuple(map(tuple, points))
        self.coefficients = numpy.asarray(coefficients)
        self.mappings = tuple(mappings)

    @property
    def rank(self):
        """The number of component indices."""
        return sum(mapping not in (DERIVATIVE, DIVERGENCE) for mapping in self.mappings)

    @property
    def order(self):
        """The number of derivatives taken."""
        return sum(mapping in (DERIVATIVE, DIVERGENCE) for mapping in self.mappings)

    @classmethod
    def from_fiat(cls, node, mapping="affine", tol=1e-12):
        """Read the coefficient tensors off a FIAT functional.

        A derivative multi-index is spread evenly over the positions of
        the symmetric derivative tensor with that multi-index, so that the
        full contraction with the derivative tensor of :math:`f`
        reproduces the multi-index pairing.  A first derivative whose
        coefficients contract the last component index with the
        derivative index is stored with a divergence index instead.

        :arg node: The FIAT :class:`~FIAT.functional.Functional`.
        :arg mapping: The FIAT mapping of the basis functions.
        :arg tol: Relative tolerance for recognizing a divergence.
        :returns: The :class:`FunctionalData`.
        """
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
        mappings = component_mappings(mapping, rank) + (DERIVATIVE,) * order

        points = tuple(entries)
        coefficients = numpy.zeros((len(points),) + (sd,) * (rank + order))
        for q, pt in enumerate(points):
            for w, alpha, comp in entries[pt]:
                indices = set(permutations(sum(([k] * a for k, a in enumerate(alpha)), [])))
                for index in indices:
                    coefficients[(q, *comp, *index)] += w / len(indices)

        if order == 1 and rank > 0 and mappings[rank - 1] == "contravariant piola":
            trace = numpy.trace(coefficients, axis1=-2, axis2=-1) / sd
            if numpy.allclose(coefficients, trace[..., None, None] * numpy.eye(sd),
                              atol=tol * numpy.abs(coefficients).max()):
                coefficients = trace[..., None]
                mappings = mappings[:rank - 1] + (DIVERGENCE,)
        return cls(points, coefficients, mappings)

    def contract(self, A, index):
        """Contract one index of the coefficients with a matrix.

        :arg A: The matrix, numeric or an object array of GEM scalars.
        :arg index: The position of the index in the coefficient tensor.
        :returns: The :class:`FunctionalData` with ``W'[q, ..., i, ...]
            = sum_k A[i, k] W[q, ..., k, ...]``.
        """
        coefficients = numpy.tensordot(self.coefficients, A, axes=(1 + index, 1))
        coefficients = numpy.moveaxis(coefficients, -1, 1 + index)
        return type(self)(self.points, coefficients, self.mappings)

    def evaluate(self, tabulation):
        """Apply the functional to tabulated basis functions.

        :arg tabulation: The tabulation of the basis functions at
            ``points``, shape ``(nbf, sd, ..., sd, len(points))`` with
            the axes of the coefficient tensor, as returned by
            :meth:`finat.physically_mapped.ReferenceNodalBasis.tabulate`.
        :returns: The vector of values on the basis, shape ``(nbf,)``.
        """
        n = len(self.mappings)
        return numpy.tensordot(tabulation, self.coefficients,
                               axes=(tuple(range(1, n + 2)), tuple(range(1, n + 1)) + (0,)))
