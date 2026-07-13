# AGENTS.md for FIAT

This document outlines the guidelines and architectural context for AI agents assisting with the FIAT codebase, functioning as a core component within the broader Firedrake ecosystem. 

---

## AI Contribution Policy

When assisting with contributions to FIAT and the Firedrake project, AI agents and their human counterparts must adhere to the following strict policies:

* The use of AI tools must be explicitly declared alongside the specific tool used.
* A human developer must lead the Pull Request.
* The human contributor must understand every change made to the codebase.
* Reviewer questions must be answered directly by the human, rather than acting as a relay to the AI.
* Any generated code must be executed locally to verify that it functions correctly.
* AI tools must not be used to resolve issues that are labeled as 'good first issue'.

### PR expectations

* CI (`.github/workflows/test.yml`) checks `flake8` and `pydocstyle .` (both
  configured in `setup.cfg`); run them locally and fix all findings before pushing.
* Watch for `pydocstyle` D413 (blank line after the last numpydoc section) and D417
  (every argument, including keyword arguments, needs a description) in new docstrings.

---

## FIAT's Role in the Architecture

FIAT operates within Firedrake's automated system for solving partial differential equations via the finite element method. Its specific architectural responsibilities include:

* FIAT provides compile-time pre-tabulated basis functions.
* These basis functions are utilized when the Two-Stage Form Compiler (TSFC) lowers Unified Form Language (UFL) into the GEM tensor language.
* The resulting GEM expressions represent mathematical operations over quadrature points.
* Mesh-topology bookkeeping routines within the ecosystem rely on FIAT ordering, such as the `create_cell_closure()` loop which builds a FIAT-ordered closure map necessary for subsequent code generation.

---

## Repository Layout

Despite being called "fiat", this repository (Python package `firedrake_fiat`) contains **three** packages plus their tests, all in one tree:

* `FIAT/` — reference-element definitions: cells, polynomial sets, dual bases (`FIAT/functional.py` holds the taxonomy of degree-of-freedom functionals), and element families.
* `finat/` — symbolic layer on top of FIAT. Physically mapped elements (`finat/physically_mapped.py`, `finat/argyris.py`, `finat/morley.py`, `finat/hct.py`, `finat/piola_mapped.py`, …) construct GEM expressions for basis transformations.
* `gem/` — the tensor-algebra intermediate language used to express transformations symbolically (`gem/gem.py` for nodes, `gem/interpreter.py` to evaluate expressions numerically in tests).
* `test/` — note the singular name: tests live at `test/finat/test_zany_mapping.py`, `test/FIAT/...`, `test/gem/...` (not `tests/`).
* `literature/` — untracked folder with LaTeX sources of the two theory papers: `Kirby2017Transformation/paper.tex` (A general approach to transforming finite elements) and `BrubeckKirby2025Macroelements/paper.tex` (transformation theory for macroelements, §"Transformation theory").

## Environment and Setup

Bugs can exist within Firedrake or any of its component packages, explicitly including FIAT. To effectively develop and debug FIAT:

* Developers should use editable installs for subpackages like FIAT so that source code edits take effect without requiring a full reinstallation.
* The active branch or commit of each component must be verified before assuming a bug originates in the top-level Firedrake package.

---

## Core Coding Rules

Agents modifying FIAT code must follow these fundamental development principles:

* Bug fixes must target the underlying mathematical or architectural root cause.
* Developers must avoid merely patching specific failing test cases or edge cases.
* Code complexity should be minimized by favoring the mathematical generality of finite elements over complicated special-case logic.
* Memorized API shapes must not be trusted.
* APIs across the ecosystem evolve, meaning properties can become methods, arguments can be renamed, and signatures can be deprecated.
* Agents must verify current API signatures by reading the installed source code instead of relying on outdated training data.
* Code documentation and comments must explain the present, correct code.
* Comments must not detail what a removed or incorrect approach previously did.

---

## Pattern Matching and Mathematical Reasoning

When designing or debugging FIAT, FInAT, and GEM changes, use the existing codebase as a library of
mathematical patterns rather than starting from ad hoc special cases:

* Match new element constructions against the nearest existing family with the same structural
  decomposition. Tensor-product, restricted, physically mapped, and enriched elements usually share
  a factorization pattern that should be reused explicitly.
* When a feature seems to require a special case, test whether the same mathematics already appears in
  another element family or mapping path. The right answer is often a more general basis
  transformation, not a new branch.
* Separate reference-space reasoning from physical-space reasoning. In FIAT and FInAT, derive basis
  transformations from the element map and continuity requirements first, then encode that structure
  in GEM expressions.
* Treat tensor-product spaces as tensor-product mathematics. Look for Kronecker-style factorization
  in basis matrices, coordinate mappings, and dual evaluations before introducing custom assembly
  logic.
* For extruded or vertically constant factors, identify the dimension that is geometrically active
  and the dimension that is algebraically passive. The passive factor should usually contribute a
  simple constant, identity, or lower-dimensional pullback rather than a new geometric rule.
* Debug by matching the failing object against a known neighboring case: compare the cell, element
  family, mapping type, continuity class, and tensor structure before changing code.
* In GEM, inspect whether an expression should factor, broadcast, or propagate a coordinate mapping.
  If the expression is not matching the expected shape, the bug is often in the way indices or
  subexpressions are assembled, not in the downstream optimizer.
* Use the mathematical continuity target as a design constraint. For finite elements, ask what
  inter-element continuity the space must satisfy, then derive the local basis and transformation
  rules from that requirement.
* Prefer proofs by structure over proofs by example. A construction is correct when the pullback,
  restriction, and tensor-product algebra agree with the element's continuity and approximation
  properties, not when one or two test cases happen to pass.

## Transformation Theory (Kirby 2017; Brubeck & Kirby 2025)

The mathematical framework for mapping non-affinely-equivalent elements. Notation: the
geometric map goes **from physical to reference**, $F: K \to \hat{K}$; pullback
$F^*(\hat{f}) = \hat{f}\circ F$; push-forward of a node $F_*(n) = n \circ F^*$.

* Goal: the matrix $M$ with $\Psi = M\, F^*(\hat\Psi)$, expressing physical nodal basis
  functions as combinations of pulled-back reference basis functions.
* Duality is the whole game: with $B_{ij} = F_*(n_i)(\hat\psi_j) = n_i(F^*(\hat\psi_j))$,
  the Kronecker property gives $M = B^{-T}$ and $V = B^{-1}$, where $V$ relates nodes:
  $\hat{N} = V\, F_*(N)$ (restricted to $P$). Hence **$M = V^T$**, and $V$ is what one
  actually constructs, because push-forwards of nodes are computable by the chain rule.
* Affine equivalence (Lagrange): $F_*(N) = \hat{N}$, so $M = I$.
* Affine-interpolation equivalence (Hermite): spans of nodes at each point are preserved,
  so $V$ is block diagonal, one small block per point-node group (e.g. $J^{-T}$ blocks for
  vertex gradients — in the *paper's* $J$; see the FInAT convention below).
* Neither (Morley, Argyris, HCT): edge normal derivatives alone do not push forward into
  the span of reference nodes. Fix by a **compatible nodal completion** $N^c \supset N$,
  $\hat{N}^c \supset \hat{N}$ with $\mathrm{span}(F_*(N^c)) = \mathrm{span}(\hat{N}^c)$
  (add the tangential-derivative partners so each edge carries a full gradient). Then
  $$V = E\, V^c\, D,$$
  with the three factors having distinct mathematical roles:
  * $D \in \mathbb{R}^{\mu\times\nu}$: expresses the completed physical nodes in terms of
    the actual physical nodes, *restricted to* $P$ ($\pi N^c = D\, \pi N$). Rows for nodes
    already in $N$ are Boolean. Rows for completion nodes come from a univariate exactness
    argument on the edge: a tangential-derivative quantity of a polynomial of known edge
    degree is an exact linear combination of the endpoint values/derivatives (FTC for HCT's
    $\mu^t_e$; the quintic endpoint rule for Argyris/Bell midpoint tangential derivatives).
    This is the only factor that uses the polynomial degree of the space.
  * $V^c \in \mathbb{R}^{\mu\times\mu}$: block diagonal, pure chain rule,
    $\hat{N}^c = V^c F_*(N^c)$. Vertex value $\to 1$; vertex gradient $\to$ Jacobian block;
    vertex Hessian $\to$ symmetric-square block $\Theta$; edge normal/tangential pair $\to$
    the $2\times2$ block $B_i = \hat{G}_i J^{-T} G_i^T$ (paper's $J$), where
    $G = [\mathbf{n}\; \mathbf{t}]^T$. Only the first row of $B_i$ is needed
    ($B_{nn}$, $B_{nt}$) since tangential rows are discarded by $E$.
  * $E \in \mathbb{R}^{\nu\times\mu}$: Boolean extraction of $\hat{N}$ from $\hat{N}^c$.
* Constrained spaces, $F^*(\hat{P}) \neq P$ (Bell, reduced HCT): the space is
  $P = \bigcap_i \mathrm{null}(\lambda_i)$ inside a larger $\tilde{P}$ that *is* preserved
  (e.g. reduced HCT = HCT functions whose edge normal derivative is linear;
  $\lambda_i$ = moment of the normal derivative against the quadratic Legendre polynomial
  on edge $i$). Build the **extended element** $(K, \tilde{P}, [N; L])$ with the
  constraints $\lambda_i$ appended as extra nodes; it is a valid finite element, and its
  first $\nu$ nodal basis functions are exactly the nodal basis of the constrained
  element. Transform the extended element with the $E V^c D$ machinery (completing each
  $\lambda_i$ with its tangential partner $\lambda_i'$, again eliminated through an edge
  exactness rule), then discard the constraint rows.
* Brubeck & Kirby 2025 refinements:
  * Define *reference* edge nodes as integral **averages** ($1/|\hat{e}|$ scaling) while
    physical nodes are plain moments: the edge-length Jacobian of the line integral is then
    absorbed and no reference-edge-length bookkeeping or orientation logic is needed.
  * High-order HCT/Argyris: edge normal-derivative moments are taken against Jacobi
    polynomials $P_i^{(1,1)}$ (weights $(2,2)$ for Argyris, matching the vertex jet order),
    so the edge-based functions are hierarchical. The tangential completion
    $\mu^t_{e,i}$ integrates by parts against the *trace* moments (defined against
    $\frac{d}{ds}P_i^{(1,1)}$): $\mu^t_{e,i}(f) = -\mu_{e,i}(f) + P_i^{(1,1)}(1)\,
    \delta_{v_b}(f) - P_i^{(1,1)}(-1)\,\delta_{v_a}(f)$. This couples normal moments to
    trace moments, not just vertex dofs.

### Active work: automating the transformation (see `plan_zany_auto.md`)

Goal: replace the hand-coded `basis_transformation` methods with a helper that derives
$V = E V^c D$ directly from a FIAT element's dual basis. The implementation must mirror
the theory factor by factor, not merely reproduce matrix entries:

**Status (2026-07-13).** The symbolic dof lives in `finat/functional.py` as
`finat.PhysicallyMappedFunctional`. The OOP structure is a template method: the
entity-by-entity assembly loop is implemented once, as the concrete
`PhysicallyMappedElement.basis_transformation` in `finat/physically_mapped.py`, calling
four hooks (`_check_mapping`, `_invariant_dofs`, `_facet_dof_rows`, `_point_dof_rows`)
that carry ALL mapping-specific knowledge — the loop itself contains no `if piola`
anywhere. `finat/zany.py` supplies the two mixins implementing those hooks,
`ScalarPhysicallyMappedElement` (affine pullback: Morley, Hermite, Argyris, Bell) and
`PiolaPhysicallyMappedElement` ((double) contravariant Piola: MTW, Johnson-Mercier,
Guzman-Neilan), plus the pure math functions they call (`FacetFrame`,
`_scalar_facet_rows`, `_scalar_point_rows`, `_piola_facet_rows`, `_piola_point_rows`) —
these take plain arrays/GEM expressions, no `self`, so the mathematics stays readable
independent of the class plumbing. Concrete elements (`finat.Morley`, etc.) are now
just a citation plus a FIAT constructor call: mixing in the right base class is enough,
`basis_transformation` is inherited. `ndof` truncation is no longer a parameter; the
loop always slices by `self.space_dimension()`, which constrained elements (Bell, GN)
already override. Tests: `test/finat/test_zany_automation.py`; `check_zany_mapping`
lives in the finat conftest and is provided to test modules as a pytest fixture (pytest
runs with `--import-mode=importlib`, so test modules cannot import from each other or
from conftest).

**Framework design.** A dof is a symbolic `finat.PhysicallyMappedFunctional`:
$\ell(f) = \sum_q w_q \langle D, \nabla^m f(x_q)\rangle$ with numeric points/weights
and a direction tensor $D$ that is numeric on the reference cell and GEM otherwise.
There is *no dispatch over FIAT functional types*: `PhysicallyMappedFunctional.from_fiat`
reads only `pt_dict`/`deriv_dict` and recovers the order and common direction numerically
(rank-one SVD of the derivative weights). Operations: covariant `pullback(J)`
(contract direction slots with $J$), `with_direction`, and numeric `evaluate` against
a nodal basis (the generalized Vandermonde realizing the $D$ factor of $V = E V^c D$).
The row of $V$ for a reference node $\hat\ell$ with direction $\hat d = a\hat n +
\sum_k \beta_k \hat t_k$ (numeric split) is assembled generically:

* $a = 0$ (value dofs, or tangential directions): push-forward invariant, identity row.
* otherwise solve $J\hat d = x_0 C + \sum_k x_k J\hat t_k$ symbolically
  (`FacetFrame.decompose`, adjugate/determinant of the polynomial frame matrix), where
  $C$ = generalized cross product of the mapped tangents. The physical node has
  direction $aN + \sum\beta_k J\hat t_k$ with $N = \kappa C/\|C\|$, so its coefficient
  is $c = x_0\|C\|/(\kappa a)$ and the tangential remainders are $r_k = x_k - c\beta_k$
  (careful: *not* $a x_k$ — the SVD direction may be $-\hat n$, and all formulas must be
  invariant under that sign flip; a sign bug here flips exactly the edges where the SVD
  chose the opposite orientation).
* the remainders multiply completion functionals along *mapped reference tangents*,
  which coincide with reference functionals; `PhysicallyMappedFunctional.evaluate` gives their numeric
  expansion in the element's own nodes, and the row combination recurses through the
  already-assembled rows of $V$ (entities processed in increasing dimension), which
  will later let completions couple to vertex jets (Argyris/HCT) for free.

Derivative nodes *away from facets* (`_scalar_point_rows`, covering Hermite vertex
gradients) have no geometric frame: FIAT keeps Cartesian directions on the physical
cell, so the group of derivative nodes on the entity acts as its own completion — this
is precisely affine-interpolation equivalence. The pulled-back direction $J\hat d_i$
is expanded in the group's own (numeric) direction basis, with weight-ratio factors
making the expansion invariant under the SVD scale/sign ambiguity of each node's
recovered $(w, D)$ factorization. The group must span the derivative jet, share its
points, and have pairwise-parallel weights; otherwise `NotImplementedError`.

Key facts the framework rests on:

* **FIAT normals are "UFC consistent":** computed from the tangents by the same formula
  on reference and physical cells (`UFCTriangle.compute_normal` rotates the scaled edge
  tangent; `UFCTetrahedron.compute_normal` is $-2\times$ the unit cross product of the
  scaled face tangents), with cell-independent magnitude. Hence $N = \kappa C/\|C\|$
  with $\kappa$ recoverable from reference data, and no orientation logic is needed
  (assumes $\det J > 0$). For the record, the fully simplified closed forms the solve
  reproduces are $a = \det J\sqrt{\det\hat G/\det G}$ and $b = G^{-1}T^TJ\hat n$ with
  Gram matrices of the (mapped) tangents.
* **Integral averages are push-forward invariant.** FIAT's Morley dofs are averages
  (`FacetQuadratureRule(..., avg=True)`), so physical nodes share the reference weights
  and no `physical_edge_lengths` appear. The framework *assumes* measure-intrinsic
  moments (the Brubeck & Kirby 2025 reference-node convention) — documented in
  `finat/functional.py`.
* **The conditioning $h$-scaling generalizes via `max_deriv_order`.** Column $j$ is
  scaled by $h_E^{-m}$ where $m$ is the derivative order of node $j$ and $h_E$ averages
  `cell_size()` over the vertices of its entity; reproduces every per-element
  hand-written scaling. `cell_size()` returns raw numpy values in the test mappings but
  GEM in Firedrake, so use operator arithmetic (`havg**(-m)`), not GEM constructors
  (GEM overloads `+ - * / ** @` with `Zero`/constant folding; keep numpy object arrays
  on the left when scaling by a GEM scalar).

Extensions beyond first order and Morley/Hermite:

* `PhysicallyMappedFunctional` directions live in derivative multi-index space (`multiindices`, axis
  order for $m=1$); `pullback` distributes them over a symmetric tensor (dividing by
  multiplicities), contracts every slot with $J$ (`numpy.tensordot` on object arrays),
  and collapses back. Point-jet groups are split per order; each order solves in its
  own multi-index direction basis (Argyris/Bell vertex jets: gradient + Hessian).
* Facet completions of Argyris edge moments couple to vertex jets and same-edge trace
  moments; the existing row recursion handles both with no new code (trace moments are
  order-0 and thus invariant; FIAT builds all these moments with
  `FacetQuadratureRule(avg=True)`, i.e. measure-intrinsic, as the framework assumes).
* `ScalarPhysicallyMappedElement.avg = False` (an instance attribute Argyris sets from
  its constructor kwarg) reproduces the legacy FInAT convention where physical facet
  moments are plain integrals: their columns are divided by the physical facet measure
  $\|C\| |\hat e| / \|\hat C\|$ (`FacetFrame.measure`). Single-point facet dofs
  (Argyris "point" variant) are unaffected.
* Bell is the extended-element pattern: FIAT.Bell is the 21-node quintic element with
  the constraint functionals as extra edge nodes; overriding `space_dimension()` to 18
  drops the constraint *columns* of $V$ at the end of the template method (their rows
  still contribute the $D$-matrix entries through the completion recursion), and the
  FInAT element overrides `entity_dofs`.
* Known convention change: the generic $h^{-m}$ conditioning scaling now also applies
  to integral-variant Argyris edge moments, which the hand-written code left unscaled
  (Morley scaled them; the legacy convention was inconsistent). Invisible when
  `cell_size == 1`; flag in PR review.

**Piola-mapped elements** (Aznaran, Kirby & Farrell 2022). `PhysicallyMappedFunctional` carries a
value rank: component weight profiles (nq x sd^rank) parsed from `pt_dict` component
tuples. Under contravariant Piola the roles of the scalar case are mirrored: the
*scaled* facet normal is the cofactor image $K\hat n_s$, $K = \mathrm{adj}(J)^T$
(exactly the physical `compute_scaled_normal`, cross product of mapped tangents), so
pure normal moments are invariant, while scaled tangents map by $J$. `_piola_facet_rows`
works with per-point *frame-coordinate profiles* (handles 3D MTW's point-varying
RT-mapped tangential directions): the pulled-back profile is contracted per value slot
with the mixing matrix $Y$; tangential profiles are matched within the facet group by a
numeric pseudo-inverse; the residual normal profile is eliminated by per-point normal
moments through the Vandermonde recursion (this is where e.g. tangential-to-normal
couplings emerge). Key subtlety (in any dimension > 2): FIAT builds tangential value
components on the **reciprocal basis** (`cross(n, t_k)`), which transforms in-plane
contravariantly: absorb $S^{-1} = (\det\hat G_t/\det G_t)\hat G_t^{-1} G_t$
(tangent Gram change) into $Y$'s tangential rows; in 2D $S = 1$, which hides the effect.
Interior value moments are Piola-invariant by construction (scaled-normal components for
JM; covariant Nedelec test fields for MTW order 2 cancel the contravariant trial
exactly). Double contravariant (tensor) elements use the same code: one contraction per
value slot. MTW (2D/3D) and Johnson-Mercier (2D/3D) are reimplemented this way, matching
the deleted hand code to machine precision; RaviartThomas-type elements come out as pure
identity (Piola-equivalent) automatically.

GuzmanNeilanFirstKindH1 (orders 0-2, 2D/3D) is also automatic: vertex/edge point
values are value point-groups mapping by $K$ (`_piola_point_rows`, the mirror of
`_scalar_point_rows`), and the trailing tangential facet constraints are dropped since
`PiolaBubbleElement.space_dimension()` already returns the reduced count (the Bell
pattern, inherited rather than passed as a parameter); `PiolaBubbleElement.__init__`
still provides the reduced `entity_dofs` bookkeeping. `GuzmanNeilanFirstKindH1` is
`class GuzmanNeilanFirstKindH1(PiolaPhysicallyMappedElement, PiolaBubbleElement)`: MRO
puts the automatic `basis_transformation` first, while `PiolaBubbleElement.__init__`
(reached via `super()`) still sets up `self._element` and the reduced dof bookkeeping. Its hand-derived vertex-facet coupling correction ("fix
discrepancy" in `finat/piola_mapped.py`) emerges automatically from the Vandermonde
residual elimination, since the per-point normal moments evaluate against vertex basis
functions of the extended element.

Next steps: remaining PiolaBubbleElement users (GN second kind, H1div: interior
derivative moments need the divergence detJ rule), ArnoldWinther (vertex tensor values
+ higher facet moments), the extended-element path for reduced HCT (macro polynomial
spaces); covariant elements.

The implementation mirrors the theory factor by factor:

1. **Sparsity from topology**: each row of $V$ is a reference node; its nonzero columns
   are the physical nodes on the same entity (via $V^c$) plus the nodes on adjacent
   entities pulled in by $D$ (e.g. edge rows couple to the edge's vertex dofs, and for
   hierarchical elements to the same edge's trace moments). Derive this from
   `entity_dofs()` and functional types (`FIAT/functional.py`), never hard-code indices.
2. **Completion from functional type**: a normal-derivative node's completion partner is
   the corresponding tangential-derivative node; the completion is a per-entity statement,
   determined by which components of the derivative jet the dual basis lacks.
3. **$D$ from unisolvence, not from closed-form rules**: the univariate exactness rules
   (FTC, quintic endpoint rule, integration by parts against trace moments) are all
   instances of one computation — express a completion functional applied to the
   polynomial space in the basis of the element's own nodes, i.e. solve with the
   generalized Vandermonde matrix. This is where GEM-based dual evaluation comes in.
4. Constrained spaces (reduced HCT, Bell) reuse the same machinery on the extended
   element; the FIAT element must expose the constraint functionals as extra dofs.

### FInAT implementation conventions

* `coordinate_mapping.jacobian_at(point)` returns $\partial x_{\text{phys}} / \partial
  x_{\text{ref}}$ — the **inverse** of the papers' $J$ (papers map physical → reference).
  So where the paper writes $J^{-T}$, the code uses the FInAT Jacobian transposed; e.g.
  Hermite's vertex blocks are `J[j, k]` entries of $M$ directly.
* `basis_transformation` builds the numpy object array `V` (row = reference node, column
  = physical node, entries are GEM scalars) and returns `gem.ListTensor(V.T)` = $M$.
  Start from `identity(ndof)` (in `finat/physically_mapped.py`) and overwrite the
  non-identity rows.
* Existing generalized helpers, all in `finat/argyris.py`:
  * `_jet_transform(J, order)`: chain-rule block for a symmetric derivative jet of any
    order (order 1 → Jacobian, order 2 → $\Theta$, …), handling symmetric-component
    flattening.
  * `_vertex_transform(V, vorder, cell, mapping)`: places jet blocks for every vertex.
  * `_normal_tangential_transform(cell, J, detJ, edge)`: returns $(B_{nn}, B_{nt}, Jt)$
    for an edge, expressing $B$ entries via $G^{TT}$ Gram data so only GEM-representable
    quantities appear (`detJ/beta`, `alpha/beta`); 3D variant for faces is
    `morley_transform` in `finat/morley.py`.
  * `_edge_transform(V, vorder, eorder, cell, mapping, avg)`: the Jacobi-moment edge rows
    for integral-variant Argyris/HCT, encoding the endpoint values
    $P_i^{(1,1)}(\pm 1)$ and the trace-moment coupling.
* Reduced/constrained elements (`ReducedHsiehCloughTocher` in `finat/hct.py`, `Bell`):
  the FIAT element is the *extended* element (12 basis functions for reduced HCT), the
  FInAT element exposes only $\nu$ dofs: `V = identity(numbf, ndof)` is rectangular,
  `entity_dofs()` is overridden to empty the constrained entities, and
  `space_dimension()` returns the reduced count.
* Conditioning convention: after assembling `V`, columns associated with derivative dofs
  are rescaled by powers of `coordinate_mapping.cell_size()` (a per-vertex $h$). This
  redefines the physical dofs as $h$-scaled derivatives — consistent across cells because
  the scaling depends only on shared vertices/edges. Any new transformation must follow
  the same convention or mass-matrix conditioning degrades
  (`test/finat/test_mass_conditioning.py`).
* Verification: `test/finat/test_zany_mapping.py::check_zany_mapping` computes the exact
  $M$ numerically by tabulating the FIAT element on a *physical* cell and least-squares
  solving against pulled-back (and Piola-mapped, if applicable) reference tabulations,
  then compares against `basis_transformation` evaluated through `gem.interpreter`. Its
  assertion message pretty-prints the relative-error matrix with row/column indices —
  read those indices to identify which dof couplings are wrong.

## Style and Conventions

When writing Python code for FIAT, maintain the ecosystem's structural and stylistic integrity:

* Class attributes must be declared in one visible location.
* Attributes must be initialized in the `__init__` constructor or declared as a `functools.cached_property` if they are expensive to compute.
* Ad hoc lazy initialization discovering attributes via `hasattr`, `setattr`, or `getattr` scattered across methods is strictly prohibited.
* Boolean attributes must be used to record initialization intent and state instead of probing for the presence of state built by an initialization function.
* New code must include type hints on all function and method signatures.
* Public-facing APIs must include properly formatted `numpydoc`-style docstrings.
