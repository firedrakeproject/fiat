r"""Degrees of freedom as coefficient tensors.

A FIAT functional built from point and derivative dictionaries is
:math:`\ell(f) = \sum_q \langle W_q, \nabla^m f(x_q) \rangle`, with one
axis of :math:`W_q` per value component and one per derivative.
"""

from itertools import permutations

import numpy

#: The mapping of a derivative axis of the coefficient tensor.
DERIVATIVE = "derivative"

#: The mapping of a divergence axis of the coefficient tensor.
DIVERGENCE = "divergence"


def split_axis_mappings(mapping, rank):
    """Splits a FIAT mapping into the mapping of each component axis.

    :arg mapping: The FIAT mapping of the basis functions the functional
        acts on, e.g. ``"affine"`` or ``"double contravariant piola"``.
    :arg rank: The number of component axes of the functional.
    :returns: A tuple with the mapping of each component axis, e.g.
        ``("contravariant piola", "contravariant piola")``.
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
    """A reference degree of freedom as a numeric coefficient tensor at each point.

    :arg points: The points, a tuple of reference coordinates.
    :arg coefficients: A numeric array of shape ``(len(points), sd, ...,
        sd)`` with one trailing axis per component and per derivative,
        the components first.  A divergence axis has length one.
    :arg mappings: The mapping of each axis: a FIAT mapping for a
        component axis, :data:`DERIVATIVE` for a derivative axis, or
        :data:`DIVERGENCE` for the last axis.
    """
    def __init__(self, points, coefficients, mappings):
        self.points = tuple(map(tuple, points))
        self.coefficients = numpy.asarray(coefficients)
        self.mappings = tuple(mappings)

    @property
    def rank(self):
        """The number of component axes."""
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
        coefficients contract the last component axis with the
        derivative axis is stored with a divergence axis instead.

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
        mappings = split_axis_mappings(mapping, rank) + (DERIVATIVE,) * order

        points = tuple(entries)
        coefficients = numpy.zeros((len(points),) + (sd,) * (rank + order))
        for q, pt in enumerate(points):
            for w, alpha, comp in entries[pt]:
                indices = set(permutations(sum(([k] * a for k, a in enumerate(alpha)), [])))
                for index in indices:
                    coefficients[(q, *comp, *index)] += w / len(indices)

        # A divergence has W[q, ..., i, k] = c[q, ...] delta_ik on the last
        # component axis i and the derivative axis k: detect it by comparing
        # W with its trace times the identity, and keep c on one axis.
        if order == 1 and rank > 0 and mappings[rank - 1] == "contravariant piola":
            trace = numpy.trace(coefficients, axis1=-2, axis2=-1) / sd
            if numpy.allclose(coefficients, trace[..., None, None] * numpy.eye(sd),
                              atol=tol * numpy.abs(coefficients).max()):
                coefficients = trace[..., None]
                mappings = mappings[:rank - 1] + (DIVERGENCE,)
        return cls(points, coefficients, mappings)
