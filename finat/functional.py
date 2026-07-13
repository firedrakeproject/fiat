"""Symbolic representation of degrees of freedom.

A :class:`Functional` represents a degree of freedom in the form

.. math:: \\ell(f) = \\sum_q w_q \\langle D, \\nabla^m f(x_q) \\rangle,

where the points :math:`x_q` and quadrature/moment weights :math:`w_q`
are numeric, and the direction tensor :math:`D` (of rank equal to the
derivative order :math:`m`) may be numeric, for functionals defined on
the reference cell, or a GEM expression, for functionals carrying
physical geometry.

This representation is the foundation for automating the transformation
theory of Kirby (2017): degrees of freedom of any FIAT element are
converted to this common form directly from their point and derivative
dictionaries, with the derivative direction recovered numerically, so
that no dispatch over FIAT functional types is required.

The physical counterpart of a reference functional is assumed to share
its points and weights: integral moments must be measure-intrinsic
(e.g. integral averages), following the reference node convention of
Brubeck & Kirby (2025).
"""

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.functional import Functional as FIATFunctional
from gem import Literal, Node


class Functional:
    """Symbolic degree of freedom with a single derivative direction.

    Parameters
    ----------
    points :
        Tuple of reference-cell points.
    weights :
        Numeric weight for each point.
    order :
        The derivative order :math:`m`.
    direction :
        For ``order > 0``, the direction tensor of rank ``order``,
        either numeric or a GEM expression; ``None`` for ``order == 0``.
    """

    def __init__(self, points: tuple, weights: numpy.ndarray,
                 order: int = 0, direction=None):
        self.points = points
        self.weights = weights
        self.order = order
        self.direction = direction

    @classmethod
    def from_fiat(cls, node: FIATFunctional, tol: float = 1e-12) -> "Functional":
        """Construct a symbolic Functional from a FIAT functional.

        The construction only inspects the point and derivative
        dictionaries: the derivative order and the (common) direction of
        differentiation are recovered numerically by factorizing the
        matrix of derivative weights.

        Parameters
        ----------
        node :
            The FIAT functional.
        tol :
            Relative tolerance for the rank-one factorization of the
            derivative weights.

        Returns
        -------
        Functional
            The symbolic representation of the FIAT functional.
        """
        if node.pt_dict and node.deriv_dict:
            raise NotImplementedError(
                f"{type(node).__name__} mixes value and derivative weights.")

        if not node.deriv_dict:
            points = tuple(node.pt_dict)
            weights = []
            for pt in points:
                (w, comp), = node.pt_dict[pt]
                if comp != tuple():
                    raise NotImplementedError(
                        f"{type(node).__name__} has vector components.")
                weights.append(w)
            return cls(points, numpy.asarray(weights))

        sd = node.ref_el.get_spatial_dimension()
        order = node.max_deriv_order
        if order != 1:
            raise NotImplementedError(
                f"{type(node).__name__} has derivative order {order}.")

        points = tuple(node.deriv_dict)
        W = numpy.zeros((len(points), sd))
        for q, pt in enumerate(points):
            for w, alpha, comp in node.deriv_dict[pt]:
                if comp != tuple():
                    raise NotImplementedError(
                        f"{type(node).__name__} has vector components.")
                k, = numpy.flatnonzero(alpha)
                W[q, k] += w

        # Factor the weights as a common direction times scalar weights
        u, s, vt = numpy.linalg.svd(W)
        if any(s[1:] > tol * s[0]):
            raise NotImplementedError(
                f"{type(node).__name__} has no common derivative direction.")
        direction = vt[0]
        weights = u[:, 0] * s[0]
        return cls(points, weights, order=1, direction=direction)

    def with_direction(self, direction) -> "Functional":
        """Return the same functional with another direction tensor."""
        return type(self)(self.points, self.weights,
                          order=self.order, direction=direction)

    def pullback(self, J: Node) -> "Functional":
        """View this reference functional as acting on physical functions.

        By the chain rule, reference derivatives of a pullback are
        physical derivatives contracted with the Jacobian, so the
        direction tensor maps covariantly: each slot is contracted
        with :math:`J`.

        Parameters
        ----------
        J :
            GEM expression for the cell Jacobian.

        Returns
        -------
        Functional
            The functional with direction :math:`J \\otimes \\dots
            \\otimes J : D`, acting on physical derivatives at the
            images of the reference points.
        """
        if self.order == 0:
            return self
        elif self.order == 1:
            return self.with_direction(J @ Literal(self.direction))
        else:
            raise NotImplementedError(
                f"Pullback of derivative order {self.order} not implemented.")

    def evaluate(self, fiat_element: FiniteElement) -> numpy.ndarray:
        """Apply this functional to the nodal basis of a FIAT element.

        This is the generalized Vandermonde computation: the restriction
        of a functional :math:`\\ell` to the polynomial space satisfies
        :math:`\\pi \\ell = \\sum_j \\ell(\\psi_j)\\, \\pi n_j`.  Only
        valid for functionals with numeric direction.

        Parameters
        ----------
        fiat_element :
            The FIAT element providing the nodal basis.

        Returns
        -------
        numpy.ndarray
            The vector of values of this functional on the nodal basis.
        """
        sd = fiat_element.get_reference_element().get_spatial_dimension()
        tab = fiat_element.tabulate(self.order, self.points)
        row = 0
        for index in numpy.ndindex((sd,) * self.order):
            alpha = [0] * sd
            for k in index:
                alpha[k] += 1
            coef = self.direction[index] if self.order else 1
            row = row + coef * (tab[tuple(alpha)] @ self.weights)
        return row
