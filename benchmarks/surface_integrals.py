"""Check mapped-element surface assembly on the checked-out Firedrake branch.

Run with one MPI rank and a fresh FIREDRAKE_CACHE_DIR. Each polynomial lies in
its element's approximation space. Its projection therefore provides known
values and derivatives on both exterior and interior facets.

AI assistance: OpenAI Codex.
"""

import argparse
import json

import firedrake as fd
import numpy as np


CASES = [('Argyris', 5, 2), ('Bell', 5, 2), ('Morley', 2, 2),
         ('Johnson-Mercier', 1, 2), ('Johnson-Mercier', 1, 3),
         ('Walkington', 5, 3)]


def check(family, degree, dim):
    """Compare surface energies and matrix actions against exact polynomials."""
    mesh = fd.UnitSquareMesh(1, 1) if dim == 2 else fd.UnitCubeMesh(1, 1, 1)
    V = fd.FunctionSpace(mesh, family, degree)
    x = fd.SpatialCoordinate(mesh)
    power = 1 if family == 'Johnson-Mercier' else 2
    scalar = 1 + sum((i+1)*x[i]**power for i in range(dim))
    exact = scalar if not V.value_shape else scalar*fd.Identity(dim)
    field = fd.project(exact, V, solver_parameters={
        'ksp_type': 'preonly', 'pc_type': 'lu'})
    derivative = fd.div if family == 'Johnson-Mercier' else fd.grad
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    errors = []
    for measure, interior in [(fd.ds, False), (fd.dS, True)]:
        values = field('+') if interior else field
        expected = exact('+') if interior else exact
        energy = lambda a: fd.inner(a, a) + fd.inner(derivative(a), derivative(a))
        actual_energy = fd.assemble(energy(values)*measure)
        exact_energy = fd.assemble(energy(expected)*measure)
        assert np.isclose(actual_energy, exact_energy, rtol=2e-9, atol=2e-9)
        errors.append(abs(actual_energy-exact_energy))

        trial, test = (u('+'), v('-')) if interior else (u, v)
        coefficient = field('+') if interior else field
        form = lambda a: (fd.inner(a, test)
                          + fd.inner(derivative(a), derivative(test)))*measure
        matrix = fd.assemble(form(trial), mat_type='aij')
        rhs = fd.assemble(form(coefficient))
        with field.dat.vec_ro as coefficients, rhs.dat.vec_ro as result:
            difference = result.duplicate()
            matrix.petscmat.mult(coefficients, difference)
            difference.axpy(-1, result)
            assert difference.norm() < 2e-9*max(result.norm(), 1)
            difference.destroy()
    return dict(family=family, dim=dim, surface_energy_errors=errors)


def main():
    """Run the requested element families and print their numerical errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=sorted({case[0] for case in CASES}))
    args = parser.parse_args()
    for family, degree, dim in CASES:
        if args.family is None or args.family == family:
            print(json.dumps(check(family, degree, dim)), flush=True)


if __name__ == '__main__':
    main()
