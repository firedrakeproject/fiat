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
from FIAT.nedelec import Nedelec as N1Curl
from FIAT.raviart_thomas import RaviartThomas as N1Div
from FIAT.nedelec_second_kind import NedelecSecondKind as N2Curl
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini as N2Div
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import symmetric_simplex, ufc_simplex
from FIAT.demkowicz import grad, curl, div, inner
from FIAT.restricted import RestrictedElement


@pytest.fixture(params=(2, 3), ids=("T", "S"))
def ref_el(request):
    return symmetric_simplex(request.param)


@pytest.fixture(params=(2, 3), ids=("T", "S"))
def cell(request):
    return ufc_simplex(request.param)


@pytest.mark.parametrize("degree", (2, 3, 5))
@pytest.mark.parametrize("variant", ("demkowicz", "fdm"))
@pytest.mark.parametrize("family", (CG, N1Curl, N1Div, N2Curl, N2Div))
def test_galerkin_symmetry(ref_el, family, degree, variant):
    sd = ref_el.get_spatial_dimension()
    fe = family(ref_el, degree, variant=variant)
    exterior_derivative = {CG: grad, N1Curl: curl, N2Curl: curl, N1Div: div, N2Div: div}[family]

    Q = create_quadrature(ref_el, 2 * degree)
    Qpts, Qwts = Q.get_points(), Q.get_weights()
    galerkin = lambda V: inner(V, V, Qwts)

    tab = fe.tabulate(1, Qpts)
    phi = tab[(0,) * sd]
    dphi = exterior_derivative(tab)

    entity_dofs = fe.entity_dofs()
    for dim in sorted(entity_dofs):
        for V in (phi, dphi):
            A = [galerkin(V[entity_dofs[dim][entity]]) for entity in sorted(entity_dofs[dim])]
            Aref = numpy.diag(A[0].diagonal()) if variant == "fdm" or dim == sd else A[0]
            for A1 in A:
                assert numpy.allclose(Aref, A1, rtol=1E-14)


@pytest.mark.parametrize("degree", (2, 3, 7))
@pytest.mark.parametrize("variant", ("demkowicz",))
@pytest.mark.parametrize("family", (CG, N1Curl, N1Div))
def test_hierarchical_interpolation(cell, family, degree, variant):
    V1 = family(cell, 1, variant=variant)
    Vp = family(cell, degree, variant=variant)
    Vf = RestrictedElement(Vp, restriction_domain="facet")
    primal = V1.get_nodal_basis()

    for V in (Vp, Vf):
        dual = V.get_dual_set()
        A = dual.to_riesz(primal)
        B = primal.get_coeffs()
        D = numpy.tensordot(A, B, axes=(range(1, A.ndim), range(1, B.ndim)))

        dim1 = V1.space_dimension()
        dimp = V.space_dimension()
        dofs_per_entity = len(V1.entity_dofs()[V1.formdegree][0])
        dofs = V.entity_dofs()[V.formdegree]
        dof1 = sum((dofs[entity][:dofs_per_entity] for entity in sorted(dofs)), [])
        dofp = numpy.setdiff1d(numpy.arange(dimp), dof1)
        assert numpy.allclose(D[dofp], 0.0)
        assert numpy.allclose(D[dof1], numpy.eye(dim1))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
