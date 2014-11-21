from finiteelementbase import FiniteElementBase
from points import StroudPointSet
from ast import ForAll, Recipe, Wave, Let, IndexSum
import pymbolic.primitives as p
from indices import BasisFunctionIndex, PointIndex
import numpy as np


class Bernstein(FiniteElementBase):
    """Scalar-valued Bernstein element. Note: need to work out the
    correct heirarchy for different Bernstein elements."""

    def __init__(self, cell, degree):
        super(Bernstein, self).__init__()

        self._cell = cell
        self._degree = degree

    def _points_variable(self, points, kernel_data):
        """Return the symbolic variables for the static data
        corresponding to the :class:`PointSet` ``points``."""

        static_key = (id(points),)
        if static_key in kernel_data.static:
            xi = kernel_data.static[static_key][0]
        else:
            xi = p.Variable(kernel_data.point_variable_name(points))
            kernel_data.static[static_key] = (xi, lambda: points.points)

        return xi

    @property
    def dofs_shape(self):

        degree = self.degree
        dim = self.cell.get_spatial_dimension()
        return (int(np.prod(xrange(degree + 1, degree + 1 + dim))
                    / np.prod(xrange(1, dim + 1))),)

    def field_evaluation(self, field_var, q, kernel_data, derivative=None):
        if not isinstance(q.points, StroudPointSet):
            raise ValueError("Only Stroud points may be employed with Bernstein polynomials")

        if derivative is not None:
            raise NotImplementedError

        def pd(sd, d):
            if sd == 3:
                return (d + 1) * (d + 2) * (d + 3) / 6
            elif sd == 2:
                return (d + 1) * (d + 2) / 2
            elif sd == 1:
                return d + 1
            else:
                raise NotImplementedError

        # Get the symbolic names for the points.
        xi = [self._points_variable(f.points, kernel_data)
              for f in q.factors]

        qs = q.factors

        deg = self.degree

        sd = self.cell.get_spatial_dimension()

        # reimplement sum using reduce to avoid problem with infinite loop
        # into pymbolic
        def mysum(vals):
            return reduce(lambda a, b: a + b, vals, 0)

        r = kernel_data.new_variable("r")
        w = kernel_data.new_variable("w")
        tmps = [kernel_data.new_variable("tmp") for d in range(sd - 1)]

        # Create basis function indices that run over
        # the possible multiindex space.  These have
        # to be jagged
        alphas = [BasisFunctionIndex(deg + 1)]
        for d in range(1, sd):
            asum = mysum(alphas)
            alpha_cur = BasisFunctionIndex(deg + 1 - asum)
            alphas.append(alpha_cur)

        # temporary quadrature indices so I don't clobber the ones that
        # have to be free for the entire recipe
        qs_internal = [PointIndex(qf) for qf in q.points.factor_sets]

        # For each phase I need to figure out the free variables of
        # that phase
        free_vars_per_phase = []
        for d in range(sd - 1):
            alphas_free_cur = tuple(alphas[:(-1 - d)])
            qs_free_cur = tuple(qs_internal[(-1 - d):])
            free_vars_per_phase.append(((), alphas_free_cur, qs_free_cur))
        # last phase: the free variables are the free quadrature point indices
        free_vars_per_phase.append(((), (), (q,)))

        # the first phase reads from the field_var storage
        # the rest of the phases will read from a tmp variable
        # This code computes the offset into the field_var storage
        # for the internal sum variable loop.
        offset = 0
        for d in range(sd - 1):
            deg_begin = deg - mysum(alphas[:d])
            deg_end = deg - alphas[d]
            offset += pd(sd - d, deg_begin) - pd(sd - d, deg_end)

        # each phase of the sum-factored algorithm reads from a particular
        # location.  The first of these is field_var, the rest are the
        # temporaries.
        read_locs = [field_var[alphas[-1] + offset]]
        if sd > 1:
            # intermediate phases will read from the alphas and
            # internal quadrature points
            for d in range(1, sd - 1):
                tmp_cur = tmps[d - 1]
                read_alphas = alphas[:(-d)]
                read_qs = qs_internal[(-d):]
                read_locs.append(tmp_cur[tuple(read_alphas + read_qs)])

            # last phase reads from the last alpha and the incoming quadrature points
            read_locs.append(tmps[-1][tuple(alphas[:1] + qs[1:])])

        # Figure out the "xi" for each phase being used in the recurrence.
        # In the last phase, it has to refer to one of the free incoming
        # quadrature points, and it refers to the internal ones in previous phases.
        xi_per_phase = [xi[-(d + 1)][qs_internal[-(d + 1)]]
                        for d in range(sd - 1)] + [xi[0][qs[0]]]

        # first phase: no previous phase to bind
        xi_cur = xi_per_phase[0]
        s = 1 - xi_cur
        expr = Let(((r, xi_cur / s),),
                   IndexSum((alphas[-1],),
                            Wave(w,
                                 alphas[-1],
                                 s ** (deg - mysum(alphas[:(sd - 1)])),
                                 w * r * (deg - mysum(alphas) + 1) / (alphas[-1]),
                                 w * read_locs[0]
                                 )
                            )
                   )
        recipe_cur = Recipe(free_vars_per_phase[0], expr)

        for d in range(1, sd):
            # Need to bind the free variables that came before in Let
            # then do what I think is right.
            xi_cur = xi_per_phase[d]
            s = 1 - xi_cur
            alpha_cur = alphas[-(d + 1)]
            asum0 = mysum(alphas[:(sd - d - 1)])
            asum1 = mysum(alphas[:(sd - d)])

            expr = Let(((tmps[d - 1], recipe_cur),
                        (r, xi_cur / s)),
                       IndexSum((alpha_cur,),
                                Wave(w,
                                     alpha_cur,
                                     s ** (deg - asum0),
                                     w * r * (deg - asum1 + 1) / alpha_cur,
                                     w * read_locs[d]
                                     )
                                )
                       )
            recipe_cur = Recipe(free_vars_per_phase[d], expr)

        return recipe_cur
