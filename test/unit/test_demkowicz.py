# Copyright (C) 2016 Imperial College London and others
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# Authors:
#
# Pablo Brubeck

import pytest
import numpy
from FIAT.hierarchical import IntegratedLegendre as CG
from FIAT.nedelec_second_kind import NedelecSecondKind as N2Curl
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini as N2Div


@pytest.mark.parametrize("family, dim, degree, variant",
                         [(f, d, p, v)
                          for f in (CG, N2Curl, N2Div)
                          for v in ("demkowicz", "fdm")
                          for d in (2, 3)
                          for p in range(1, 7)])
def test_galerkin_symmetry(dim, family, degree, variant):
    from FIAT.quadrature_schemes import create_quadrature
    from FIAT.reference_element import symmetric_simplex
    from FIAT.demkowicz import grad, curl, div, inner

    s = symmetric_simplex(dim)
    fe = family(s, degree, variant=variant)
    exterior_derivative = {CG: grad, N2Curl: curl, N2Div: div}[family]

    Q = create_quadrature(s, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()
    galerkin = lambda V: inner(V, V, Qwts)

    tab = fe.tabulate(1, Qpts)
    phi = tab[(0,) * dim]
    dphi = exterior_derivative(tab)

    entity_dofs = fe.entity_dofs()
    for dim in sorted(entity_dofs):
        for V in (phi, dphi):
            A = [galerkin(V[entity_dofs[dim][entity]]) for entity in sorted(entity_dofs[dim])]
            Aref = numpy.diag(A[0].diagonal()) if variant == "fdm" else A[0]
            for A1 in A:
                assert numpy.allclose(Aref, A1, rtol=1E-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
