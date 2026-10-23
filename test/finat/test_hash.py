import os
import pickle
import sys
import subprocess
import textwrap

import pytest
import ufl
import finat
import finat.ufl
from finat.quadrature import make_quadrature
from FIAT.reference_element import UFCInterval, UFCTriangle


def test_same_hash():
    """ The same element created twice should have the same hash.
    """
    cg = finat.ufl.finiteelement.FiniteElement("Lagrange", ufl.cell.Cell("triangle"), 1)
    same_cg = finat.ufl.finiteelement.FiniteElement("Lagrange", ufl.cell.Cell("triangle"), 1)
    assert hash(cg) == hash(same_cg)


def test_different_hash():
    """ Two different elements should have different hashes.
    """
    cg = finat.ufl.finiteelement.FiniteElement("Lagrange", ufl.cell.Cell("triangle"), 1)
    dg = finat.ufl.finiteelement.FiniteElement("DG", ufl.cell.Cell("triangle"), 2)
    assert hash(cg) != hash(dg)


def test_variant_hashes_different():
    """ Different variants of the same element should have different hashes.
    """
    dg = finat.ufl.finiteelement.FiniteElement("DG", ufl.cell.Cell("triangle"), 2)
    dg_gll = finat.ufl.finiteelement.FiniteElement("DG", ufl.cell.Cell("triangle"), 2, variant="gll")
    assert hash(dg) != hash(dg_gll)


def test_persistent_hash(tmp_path):
    """ Hashes should be the same across Python invocations.
    """
    filename = "print_hash.py"
    code = textwrap.dedent("""\
        import ufl
        import finat.ufl

        dg = finat.ufl.finiteelement.FiniteElement("RT", ufl.cell.Cell("triangle"), 1)
        print(hash(dg))
        """)
    filepath = tmp_path.joinpath(filename)
    with open(filepath, "w") as fh:
        fh.write(code)

    output1 = subprocess.run([sys.executable, filepath], capture_output=True)
    assert output1.returncode == os.EX_OK
    output2 = subprocess.run([sys.executable, filepath], capture_output=True)
    assert output2.returncode == os.EX_OK
    assert output1.stdout == output2.stdout


@pytest.mark.parametrize("construct", [
    lambda: finat.Lagrange(UFCTriangle(), 2),
    lambda: finat.DiscontinuousLagrange(UFCTriangle(), 2, variant="spectral"),
    lambda: finat.TensorProductElement([finat.Lagrange(UFCInterval(), 1),
                                        finat.DiscontinuousLagrange(UFCInterval(), 0)]),
    lambda: finat.EnrichedElement([finat.Lagrange(UFCTriangle(), 1), finat.Bubble(UFCTriangle(), 3)]),
    lambda: finat.TensorFiniteElement(finat.Lagrange(UFCTriangle(), 2), (2,)),
    lambda: finat.QuadratureElement(UFCTriangle(), make_quadrature(UFCTriangle(), 2)),
], ids=["Lagrange", "DG-spectral", "TensorProduct", "Enriched", "TensorFinite", "Quadrature"])
def test_finat_same_hash(construct):
    """The same FInAT element constructed twice should be equal and have the same hash."""
    element = construct()
    same_element = construct()
    assert element is not same_element
    assert element == same_element
    assert hash(element) == hash(same_element)


def test_finat_keyword_arguments():
    """Passing an argument by keyword should not change the FInAT element."""
    cell = UFCTriangle()
    assert finat.Lagrange(cell, 2) == finat.Lagrange(cell, degree=2)
    assert finat.Hermite(cell) == finat.Hermite(cell, degree=3)


def test_finat_different_hash():
    """FInAT elements that share their entity DoFs but not their basis should differ."""
    cell = UFCTriangle()
    equispaced = finat.DiscontinuousLagrange(cell, 2, variant="equispaced")
    spectral = finat.DiscontinuousLagrange(cell, 2, variant="spectral")
    assert equispaced.entity_dofs() == spectral.entity_dofs()
    assert equispaced != spectral
    assert hash(equispaced) != hash(spectral)
    assert finat.Lagrange(cell, 2) != finat.DiscontinuousLagrange(cell, 2)


def test_finat_pickle():
    """A pickled FInAT element should be equal to the original."""
    element = finat.TensorFiniteElement(finat.Lagrange(UFCTriangle(), 2), (2,))
    assert pickle.loads(pickle.dumps(element)) == element
    assert hash(pickle.loads(pickle.dumps(element))) == hash(element)
