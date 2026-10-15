"""Measure TSFC kernels with separately selected FIAT and TSFC source trees.

Run each variant in a fresh process with PYTHONPATH pointing to its sources
and FIREDRAKE_CACHE_DIR pointing to an empty directory. For example::

    PYTHONPATH=/tmp/fiat-main:/tmp/tsfc-main python benchmarks/pr_stack.py \
        --output /tmp/main --cases rt2-matrix q3-action

Generated C, shared libraries, numerical outputs and JSON measurements are
saved under --output. Compare outputs across variants before interpreting
speedups. AI assistance: OpenAI Codex.
"""

import argparse
import ctypes
import json
import re
from pathlib import Path
import subprocess
import time
import warnings

import loopy
import numpy as np
import ufl

import FIAT
import gem
from finat.ufl import FiniteElement, VectorElement
import tsfc
from tsfc.kernel_args import (CellOrientationsKernelArg, CellSizesKernelArg,
                              CoefficientKernelArg, CoordinatesKernelArg,
                              ExteriorFacetKernelArg, InteriorFacetKernelArg,
                              OrientationsCellKernelArg,
                              OrientationsExteriorFacetKernelArg,
                              OrientationsInteriorFacetKernelArg,
                              OutputKernelArg)


CASES = {
    'q3-matrix': ('Q', ufl.hexahedron, 3, 'matrix'),
    'q3-action': ('Q', ufl.hexahedron, 3, 'action'),
    'nce3-matrix': ('NCE', ufl.TensorProductCell(ufl.quadrilateral, ufl.interval), 3, 'matrix'),
    'rt2-matrix': ('RT', ufl.triangle, 3, 'matrix'),
    'rt3-matrix': ('RT', ufl.tetrahedron, 3, 'matrix'),
    'rt3-action': ('RT', ufl.tetrahedron, 3, 'action'),
    'argyris-action': ('Argyris', ufl.triangle, 5, 'action'),
    'argyris-matrix': ('Argyris', ufl.triangle, 5, 'matrix'),
    'jm3-action': ('Johnson-Mercier', ufl.tetrahedron, 1, 'action'),
    'walkington-action': ('Walkington', ufl.tetrahedron, 5, 'action'),
    'walkington-matrix': ('Walkington', ufl.tetrahedron, 5, 'matrix'),
    'q3-interpolate': ('Q', ufl.hexahedron, 4, 'interpolate'),
    'q3-square': ('Q', ufl.hexahedron, 4, 'square'),
    'q3-vector-dot': ('Q', ufl.hexahedron, 4, 'dot'),
    'q3-vector-norm': ('Q', ufl.hexahedron, 4, 'norm'),
    'bell-ds': ('Bell', ufl.triangle, 5, 'ds'),
    'walkington-ds': ('Walkington', ufl.tetrahedron, 5, 'ds'),
    'walkington-dS': ('Walkington', ufl.tetrahedron, 5, 'dS'),
    'argyris-ds': ('Argyris', ufl.triangle, 5, 'ds'),
    'argyris-dS': ('Argyris', ufl.triangle, 5, 'dS'),
    'morley-ds': ('Morley', ufl.triangle, 2, 'ds'),
    'jm3-ds': ('Johnson-Mercier', ufl.tetrahedron, 1, 'ds'),
    'jm3-dS': ('Johnson-Mercier', ufl.tetrahedron, 1, 'dS'),
}


def compile_case(case):
    """Compile a Helmholtz form or a tensor product dual evaluation."""
    family, cell, degree, kind = CASES[case]
    mesh = ufl.Mesh(VectorElement('P', cell, 1))
    element = FiniteElement(family, cell, degree)
    V = ufl.FunctionSpace(mesh, element)
    if kind in ('interpolate', 'square', 'dot', 'norm'):
        if kind in ('dot', 'norm'):
            V = ufl.FunctionSpace(mesh, VectorElement(family, cell, degree))
        f = ufl.Coefficient(V)
        if kind in ('dot', 'norm'):
            g = ufl.Coefficient(V) if kind == 'dot' else f
            expression = ufl.inner(f, g)
        else:
            expression = f if kind == 'interpolate' else f*f
        return tsfc.compile_expression_dual_evaluation(expression, element)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    derivative = (ufl.div if family in ('RT', 'Johnson-Mercier')
                  else ufl.curl if family == 'NCE' else ufl.grad)
    form = (ufl.inner(u, v) + ufl.inner(derivative(u), derivative(v)))*ufl.dx
    if kind == 'ds':
        form = ufl.inner(u, v)*ufl.ds
    elif kind == 'dS':
        form = ufl.inner(u('+'), v('+'))*ufl.dS
    if kind == 'action':
        form = ufl.action(form)
    kernel, = tsfc.compile_form(form)
    return kernel


def input_arrays(kernel, cell):
    """Use deterministic data on a nondegenerate affine physical cell."""
    dim = cell.topological_dimension
    cellname = cell.cellname
    if cellname == 'quadrilateral * interval':
        cellname = 'hexahedron'
    vertices = {
        'triangle': [[0, 0], [1, 0], [0, 1]],
        'tetrahedron': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'hexahedron': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                       [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
    }[cellname]
    transform = np.eye(dim) + 0.13*np.ones((dim, dim))
    coords = np.asarray(vertices) @ transform.T + np.arange(dim)*0.17
    arrays = []
    coefficient_number = 0
    for argument in kernel.arguments:
        arg = argument.loopy_arg
        shape = tuple(map(int, arg.shape))
        data = np.zeros(shape, dtype=arg.dtype.numpy_dtype)
        if isinstance(argument, CoordinatesKernelArg):
            data.flat[:] = np.tile(coords.ravel(), data.size // coords.size)
        elif isinstance(argument, CoefficientKernelArg):
            data.flat[:] = np.random.default_rng(71 + coefficient_number).normal(size=data.size)
            coefficient_number += 1
        elif isinstance(argument, CellSizesKernelArg):
            data.fill(1.1)
        elif isinstance(argument, CellOrientationsKernelArg):
            data.fill(1)
        elif isinstance(argument, (ExteriorFacetKernelArg, InteriorFacetKernelArg,
                                   OrientationsCellKernelArg,
                                   OrientationsExteriorFacetKernelArg,
                                   OrientationsInteriorFacetKernelArg)):
            data.fill(0)
        elif not isinstance(argument, OutputKernelArg):
            raise ValueError(f'Unsupported benchmark argument: {type(argument)}')
        arrays.append(data)
    return arrays


def measure(case, directory, batches):
    """Count generated operations and time calls in a native C loop."""
    started = time.perf_counter()
    kernel = compile_case(case)
    compile_seconds = time.perf_counter() - started
    program = kernel.ast.with_entrypoints(kernel.name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        operations = loopy.get_op_map(program, subgroup_size=1).filter_by(
            name=['add', 'sub', 'mul', 'div', 'func:abs'], dtype=[float])
        flops = int(operations.eval_and_sum({}))
    code = loopy.generate_code_v2(program).device_code()
    arrays = input_arrays(kernel, CASES[case][1])
    declarations = ', '.join(f'void *p{i}' for i in range(len(arrays)))
    arguments = ', '.join(f'p{i}' for i in range(len(arrays)))
    wrapper = f'''
#include <time.h>
double benchmark(long n, {declarations}) {{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < n; ++i) {kernel.name}({arguments});
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a.tv_sec) + 1e-9*(b.tv_nsec - a.tv_nsec);
}}
'''
    source, library = directory / f'{case}.c', directory / f'{case}.so'
    source.write_text(code + wrapper)
    started = time.perf_counter()
    subprocess.run(['cc', '-O3', '-fPIC', '-shared', str(source), '-lm',
                    '-o', str(library)], check=True, capture_output=True)
    c_compile_seconds = time.perf_counter() - started
    loaded = ctypes.CDLL(str(library))
    native = getattr(loaded, kernel.name)
    native.argtypes = [ctypes.c_void_p]*len(arrays)
    pointers = [ctypes.c_void_p(array.ctypes.data) for array in arrays]
    native(*pointers)
    if not np.isfinite(arrays[0]).all():
        raise ValueError('Kernel produced nonfinite values')
    np.save(directory / f'{case}.npy', arrays[0])
    timer = loaded.benchmark
    timer.argtypes = [ctypes.c_long] + native.argtypes
    timer.restype = ctypes.c_double
    n = 1
    while timer(n, *pointers) < 0.01:
        n *= 2
    samples = [timer(n, *pointers)/n for _ in range(batches)]
    temporaries = kernel.ast.default_entrypoint.temporary_variables
    storage = sum(int(np.prod(t.shape)) for t in temporaries.values()
                  if not (t.read_only and t.initializer is not None))
    loops = [line.strip() for line in code.splitlines()
             if re.match(r'\s*for\s*\(', line)]
    shapes = {name: list(map(int, t.shape)) for name, t in temporaries.items()
              if not (t.read_only and t.initializer is not None)}
    return dict(case=case, loops=loops, temporary_shapes=shapes,
                loop_count=len(loops), iname_count=len(
                    kernel.ast.default_entrypoint.all_inames()), flops=flops, impero_flops=float(kernel.flop_count),
                c_lines=len(code.splitlines()), c_bytes=len(code.encode()),
                temporary_scalars=storage, tsfc_seconds=compile_seconds,
                cc_seconds=c_compile_seconds, native_seconds=min(samples),
                batch_seconds=samples, repetitions=n)


def main():
    """Write one result per case, preserving completed work if a case fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--batches', type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(fiat=FIAT.__file__, gem=gem.__file__, tsfc=tsfc.__file__)
    (args.output / 'sources.json').write_text(json.dumps(metadata, indent=2))
    print(metadata, flush=True)
    for case in args.cases:
        try:
            result = measure(case, args.output, args.batches)
        except Exception as error:
            result = dict(case=case, error=f'{type(error).__name__}: {error}')
        (args.output / f'{case}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
