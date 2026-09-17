# Mathematical code review

## Scope

This review covers `origin/main...HEAD` for the zany automatic physical-basis
transformation work. The PR replaces a collection of element-specific
transformation matrices with a generic construction, and adds the FIAT support
needed by that construction.

## Summary

The central construction is mathematically sound for affine,
full-dimensional cells. If `g_i` denotes the pulled-back reference basis and
`n_i` the physical dual functionals, the PR forms

\[
B_{ij}=n_i(g_j), \qquad V=B^{-1},
\]

and uses the columns of `V` to express the physical basis in the pulled-back
reference basis. This is the right duality formulation: `B V = I` gives the
physical nodal property directly, while the entity-closure sparsity makes the
inverse computable blockwise.

## Mathematical changes

### 1. Functionals become coefficient tensors

`finat/functional.py` translates FIAT point and derivative functionals into
numeric tensors `W_q` such that

\[
\ell(f)=\sum_q \langle W_q,\nabla^m f(x_q)\rangle.
\]

Repeated derivative indices are distributed over all permutations, so the
contraction with the symmetric derivative tensor has the same value as the
original FIAT multi-index functional. First-order contravariant traces are
compressed to a dedicated divergence axis, giving the expected Piola law
`div(J v / det J) = div(v) / det J`.

### 2. Physical transformations are inferred from the dual basis

`PhysicallyMappedElement` now:

- extracts physical nodes from the FIAT dual basis;
- classifies them as invariant, Cartesian, or facet-frame functionals;
- builds the physical Vandermonde matrix symbolically;
- preserves exact structural zeros and checks that each row stays within its
  entity closure; and
- solves the resulting lower block-triangular system using adjugate/determinant
  formulas, with caching for repeated blocks.

This correctly captures couplings that hand-written diagonal transforms cannot:
vertex jets, normal/tangential facet data, mixed derivative directions, and
constraint completions of extended elements.

### 3. Frame geometry and Piola maps are unified

`PhysicalEntityFrame` constructs orthonormal tangential directions and the
cofactor image of facet normals. Covariant derivative axes use

\[
\nabla_x f = J^{-T}\nabla_{\hat x} f,
\]

while contravariant axes use the appropriate `J/det(J)` Piola map. The code
keeps numerators and denominators separate, which avoids introducing symbolic
division into every matrix entry. The 3D edge-in-face case is handled by a
second normal and the corresponding cofactor geometry.

### 4. Normalization and conditioning are made explicit

The generic path applies the default `h^{-m}` scaling to derivative order `m`,
averaging the cell size over the vertices of the owning entity. Arnold–Winther
and Hu–Zhang retain their established `h^{-2}` vertex convention through
overrides. The Hu–Zhang point variant also applies that convention to its
interior point-value DOFs. Facet-measure scaling is applied only to integral
moments when the physical functional is a plain integral rather than an
average. The dual transformation uses the same Vandermonde and reciprocal
scaling, so basis and dual evaluation remain algebraically consistent.

### 5. FIAT and element-family changes

- Hermite, Morley, Bell, Argyris, HCT, Alfeld–Sorokina, Walkington, Wu–Xu,
  Powell–Sabin 6 and 12, Piola-bubble, Mardal–Tai–Winther, Johnson–Mercier,
  Hu–Zhang, and Arnold–Winther transformations are routed through the generic
  framework.
- Bernstein gains a simplex expansion set with derivatives and a topological
  polynomial ordering. The Bramble–Zlamal C2 element uses that ordering to
  improve the conditioning and entity structure of its basis.
- Arnold–Winther fixes the internal degree-of-freedom offset before assigning
  entity IDs.
- BDM and Mardal–Tai–Winther internal moment construction now inserts the sign
  of `det(J)` into the inverse Jacobian. Since quadrature weights contain
  `|det(J)|`, this makes the moments invariant under a reversed cell
  orientation.

## Findings

### [P2 — resolved] Hu–Zhang point-variant normalization

The PR baseline applied the stress-element `h^{-2}` convention only to
dimension-zero DOFs, although the Hu–Zhang point variant also has interior
point-value DOFs. The follow-up applies the same convention to those interior
DOFs and generalizes the existing mass-conditioning test to tensor-valued
elements using the Frobenius product. This is a conditioning check under cell
dilation; it is separate from the correctness of the Piola transformation
itself.

### [P2 — resolved] The generic Jacobian path did not preserve the shape advertised by `PhysicalGeometry`

The old constructor assumed that `jacobian_at` returned a square matrix and
could fail by indexing a rectangular Jacobian (`finat/physically_mapped.py`).
That is too restrictive for physical cells embedded in a larger coordinate
space, including the one-dimensional Hermite case. The constructor now keeps
the full `(gdim, tdim)` object array and computes the positive metric volume
factor `sqrt(det(J.T @ J))` for a rectangular matrix. Square-only `adjJ`
remains an ordinary unavailable attribute there, while `K` remains an
ordinary attribute and uses the generalized cofactor `detJ * pinv(J).T`. The
new `pseudoinverse` helper handles full-rank rectangular Jacobians. Any
orientation sign is supplied by the caller (as `detJ_at` does in the FInAT
test geometry). A regression test covers these behaviors.

### [P2 — resolved] Divergence recognition ignored the requested relative tolerance

The PR baseline documented `tol` as the relative tolerance for recognizing a
divergence, but called `numpy.allclose` with `atol=tol * max(abs(W))` and left
NumPy's default `rtol=1e-5` (`finat/functional.py:108-116`). Thus diagonal
anisotropy at the `10^{-6}` level could be accepted as a divergence even when
`tol=10^{-12}`.

The comparison now sets `rtol=0`, leaving only the requested scale-based
absolute tolerance, and a near-divergence regression test guards the behavior.

## Verification

The following checks passed after the follow-up fixes:

- `python -m pytest -q`: **2657 passed, 26 skipped, 31 xfailed, 1 xpassed**.
- zany mapping tests: **279 passed** after the fix.
- mass-conditioning tests: **19 passed** after the fix.
- `make srclint` and `make doclint`: **passed**.

The tests cover positive and negative orientations, high-order derivative
elements, Piola-mapped tensor elements, macroelements, duality, scalar mass
conditioning, rectangular Jacobian storage, and tolerance-sensitive divergence
recognition. The tensor-valued mass test now covers Arnold–Winther and both
Hu–Zhang variants.

## AI disclosure

This review and the follow-up changes were prepared with **OpenAI Codex
(GPT-5)**. The specific local inspection and verification tools used were
`git`, `rg`, `nl`, `pytest`, `make srclint`, and `make doclint`. A human
developer must lead the PR and verify and explain every follow-up change.
