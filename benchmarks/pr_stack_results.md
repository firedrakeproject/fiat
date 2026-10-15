# FIAT #282 → #284 → #281: generated-kernel measurements

AI assistance: OpenAI Codex. These are local measurements and local branch changes; the contribution is human led.

## What the measurements establish

#284 reduces arithmetic and improves the RT matrix kernel in 3D. Its 2D arithmetic reduction does not translate into faster execution. #281 reduces compiled instruction text, while padding and indirect reads slow the mapped kernels measured here. The revisions fix a severe contraction-planning regression and the affected surface integrals, but do not turn padding into a general runtime improvement.

## Method

- Original FIAT revisions: `main` 46e3eeec, #282 `88fc73dd`, #284 `5ae398a9`, #281 `fb1e1d4c`. Revised source hashes and full refs are in the JSON.
- Each cumulative variant uses archived sources and an independent cache. #282 uses TSFC main. #284 uses the companion with the #281-only indirect cancellation call removed. #281 and the revised stack use the companion.
- GCC 11.4, `-O3 -fPIC -shared`, double precision, Intel Core i7-10875H. Native C wrappers time repeated cell-kernel calls; Python, mesh traversal and solver overhead are excluded.
- Timings are minima from repeated passes of 25 alternating batches on one pinned CPU. CPU frequency/load causes noticeable pass-to-pass variation; tiny differences are not evidence of improvement. Byte-identical controls are identified below.
- Operations come from Loopy floating add/subtract/multiply/divide/abs counts, consistently across all variants. Counts omit transcendental costs. Temporary sizes include writable scalar slots and exclude literal tables and output arrays.
- Every available variant output is compared with main at `rtol=atol=2e-10`. Actual Firedrake surface tests independently check values, derivatives and cross-facet matrix actions.
- Direct compile observations are recorded in the JSON but are not repeated compilation-time speedup claims.

Forms: scalar/mapped Helmholtz is `(inner(u,v)+inner(grad(u),grad(v)))*dx`; RT and Johnson–Mercier use `div`. Surface timing forms are mass matrices on `ds`, or the `(+, +)` block on `dS`. Q matrix/action cases use degree 3; Q interpolation, vector dot and vector norm cases use degree 4. RT cases use degree 3. NCE uses the existing quadrilateral × interval tensor-product representation, corresponding to an extruded mesh. No unstructured-hexahedron support was added.

## Per-PR observations

### #282: contraction planning and shared evaluations

The measured interpolation and vector-product kernels gain no arithmetic reduction from the original PR. Its product-tree tie choices increase live storage and split streaming loops. Retaining the supplied axis order on equal arithmetic restores the original structure. The bounded DAG walks and construction of GEM nodes only for the winning tree address compilation work; correctness tests cover joined contractions and rectangular-domain multiplicities.

| Kernel | Main operations | Original #282 loops / temporary slots | Revised #282 loops / temporary slots | Main / revised µs |
|---|---:|---:|---:|---:|
| q3-matrix | 249,871 | 38 / 4,027 | 38 / 4,003 | 27.072 / 27.550 |
| q3-action | 26,491 | 24 / 353 | 21 / 281 | 3.769 / 3.837 |
| q3-interpolate | 3,750 | 12 / 50 | 9 / 30 | 0.421 / 0.423 |
| q3-vector-dot | 23,250 | 12 / 306 | 9 / 186 | 2.496 / 2.508 |
| q3-vector-norm | 12,000 | 12 / 153 | 9 / 93 | 1.164 / 1.169 |
| nce3-matrix | 1,645,759 | 211 / 21,004 | 211 / 20,980 | 245.115 / 246.754 |

The revised Q interpolation C is byte-identical to main. The NCE control retains the previously reported 33 extra operations (about 0.002%); its revised writable storage is 20,980 slots versus 20,971 on main.

### #284: argument linear maps

The existing monomial classifier now recognises both additive sums and contractions as linear maps of one form argument. This replaces a separate distribution/discovery pass and object-identity matching. Complete basis contractions stay atomic through quadrature factorisation, and equivalent argument maps can share a tabulation. Shaped tabulations now contribute their full storage to the cost estimate.

| Kernel | Main operations → revised #284 | C lines | Temporary slots | Main / revised µs |
|---|---:|---:|---:|---:|
| rt2-matrix | 23,318 → 18,038 | 77 → 69 | 69 → 64 | 1.713 / 2.442 |
| rt3-matrix | 374,589 → 255,933 | 123 → 97 | 189 → 175 | 24.204 / 18.178 |
| rt3-action | 20,803 → 20,803 | 106 → 106 | 38 → 38 | 3.099 / 3.089 |

Both RT matrix kernels have four emitted loops. Inspection shows the retained physical-basis vectors being evaluated together before the argument-pair loop. The 3D matrix benefits; the 2D case demonstrates that the arithmetic-first cost estimate is not a runtime oracle. The RT action is unchanged.

### #281: sparse basis maps and surface integrals

| Kernel | Main operations | Original #281 operations | Revised stack operations | Main / revised µs |
|---|---:|---:|---:|---:|
| argyris-action | 12,654 | 13,308 | 13,308 | 1.795 / 2.512 |
| argyris-matrix | 75,916 | 78,879 | 78,879 | 5.977 / 7.437 |
| jm3-action | 62,228 | 85,592 | 85,592 | 13.140 / 19.591 |
| walkington-action | 2,037,310 | 2,412,096 | 2,412,096 | 355.898 / 602.092 |
| walkington-matrix | 6,825,155 | 537,153,099 | 7,291,014 | 707.098 / 897.496 |

The original Walkington matrix integrates reference-basis products over both padded-entry axes, then applies the geometry coefficients. Its largest temporary is **45×45×21×21 = 893,025 doubles**, in addition to other temporaries. Keeping the complete mapped-basis contraction together removes those padded axes from the element tensor:

```text
for quadrature point:
    mapped[row] = sum_entry coefficient[row, entry] * reference[column[row, entry]]
    form the mapped physical gradients
    for test row, trial row:
        accumulate mass and gradient products
```

| Walkington matrix structure | Main | Original #281 | Revised stack |
|---|---:|---:|---:|
| Operations | 6,825,155 | 537,153,099 | 7,291,014 |
| Emitted for loops | 4 | 18 | 9 |
| Writable temporary slots | 1,900 | 896,941 | 1,960 |
| C lines | 3,641 | 2,490 | 2,471 |
| C source bytes | 1,883,696 | 1,817,557 | 1,816,676 |
| Compiled .text bytes | 148,521 | 72,025 | 72,137 |

Against the original #281 kernel, the revision reduces this matrix’s arithmetic by 73.7× and its minimum native execution time by about 85×. The revised instruction text is roughly half of main, while source bytes shrink much less than line count suggests. Padding still adds zero work and indirect gathers. The revised matrix remains slower than main; the original pathological temporary is a separate, much larger regression that has been removed.

## Surface-integral checks

Original #281 fails compilation for Argyris (`ds`, `dS`), Morley (`ds`, `dS`), and Johnson–Mercier in 2D/3D (`ds`, `dS`). The reproductions fail in index binding, facet expression selection or surviving indirect Deltas. The revised stack compiles the cases below and agrees numerically with main.

| Surface mass kernel | Main → revised operations | Main → revised loops | Main → revised temporary slots | Main / revised µs |
|---|---:|---:|---:|---:|
| argyris-ds | 5,999 → 6,161 | 3 → 4 | 102 → 107 | 0.579 / 0.758 |
| argyris-dS | 5,999 → 6,161 | 3 → 4 | 102 → 107 | 0.577 / 0.733 |
| morley-ds | 362 → 407 | 3 → 4 | 48 → 53 | 0.059 / 0.072 |
| jm3-ds | 74,222 → 111,788 | 4 → 5 | 731 → 874 | 6.221 / 9.057 |
| jm3-dS | 74,222 → 111,788 | 4 → 5 | 731 → 874 | 6.328 / 9.045 |
| bell-ds | 4,973 → 6,323 | 3 → 6 | 147 → 153 | 0.488 / 1.118 |
| walkington-ds | 145,296 → 201,596 | 3 → 6 | 1603 → 1634 | 19.095 / 53.470 |
| walkington-dS | 145,296 → 201,596 | 3 → 6 | 1603 → 1634 | 20.244 / 55.460 |

`benchmarks/surface_integrals.py` runs actual Firedrake user code on UnitSquareMesh / UnitCubeMesh, projects exact polynomials, checks mass-plus-derivative energies on both measures, and compares assembled matrices with their `(+, -)` cross-facet actions. All six family/dimension cases pass; the largest absolute energy error is 8.71e-11.

## Placement and validation

- **#282:** bounded shared-product traversal; construction of only the selected product tree; streaming axis order on equal arithmetic; constant-summand/empty-domain multiplicities; general facet selection and memoization. Minimal tests live with these abstractions.
- **#284:** classify linear maps during ordinary monomial collection, retain complete contractions, remove the extra distribution/discovery machinery, and account for shaped storage. Tests expose reused bound indices, atomic contraction preservation and tabulation storage.
- **#281:** fix indirect tensor-axis binding; retain shared source axes and repeated lookup-index multiplicities; compare dense and gathered transpose plans; expose bound columns before `cancel_indirect_deltas`; use consistent Delta terminology and internal basis-dimension names. Tests reproduce the binding/domain failures.
- **Checked-out companion:** update the indirect-Delta API call and comments/docstrings.
- 168 focused FIAT tests and 391 companion TSFC tests pass. Source and docstring lint pass; no new type hints or numpydoc sections were added.

## Reproduction and raw artifacts

Activate the Firedrake environment and use the runtime settings from `firedrake-test`. For each variant, point `PYTHONPATH` to its FIAT and TSFC snapshots and choose a fresh `FIREDRAKE_CACHE_DIR`:

```bash
python benchmarks/pr_stack.py --output /tmp/variant-results
mpiexec -n 1 python benchmarks/surface_integrals.py
make lint
```

Full per-case loop bounds, temporary shapes, arithmetic, source/native sizes, timing passes and numerical errors are retained in [pr_stack_results.json](/tmp/fiat-pr-study/pr_stack_results.json). The local [alternating timing driver](/tmp/fiat-pr-study/retime.py) replays the saved variants. Generated C, native libraries and NumPy outputs are under `/tmp/fiat-pr-study/results-*`. For example: [original Walkington C](/tmp/fiat-pr-study/results-281/walkington-matrix.c), [revised Walkington C](/tmp/fiat-pr-study/results-final/walkington-matrix.c).

For further design work, the remaining decision is the representation tradeoff: this evidence supports smaller code and correct contractions, but does not support making padded maps a universal runtime optimisation.
