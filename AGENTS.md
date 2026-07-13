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

**Status (2026-07-13).** Prototype in `finat/zany.py`
(`zany_basis_transformation`); Morley works in 2D *and* 3D with one
dimension-independent code path, matching the hand-coded element to machine precision
(including the $h$-scaling) and passing `check_zany_mapping`. Tests:
`test/finat/test_zany_automation.py`; `check_zany_mapping` moved to the finat conftest
and is now provided to test modules as a pytest fixture (pytest runs with
`--import-mode=importlib`, so test modules cannot import from each other or from
conftest).

Key derivations that made the automation work (record of pattern-matching strategy):

* **Mapped-tangent completion makes $D$ purely numeric.** Choose the completion
  functionals to be derivative moments along the *push-forwards of the scaled reference
  tangents* $J\hat{t}_k$ (not unit physical tangents). Then the physical completion
  functional pulls back *exactly* to a reference functional, so its expansion
  coefficients in the element's nodes are reference-cell constants, computed by one
  generalized Vandermonde (dual evaluation) — this subsumes every univariate closed-form
  rule in the papers (FTC, quintic endpoint rule, integration by parts). The
  hand-written code confirms this reading: in `_edge_transform`, numeric factors like
  $-7/16$ multiply chain-rule factors `Bnt * Jt[i]`.
* **Frame decomposition by Gram algebra, no orientation logic.** The pullback of a
  reference facet-normal derivative expands as $J\hat{n} = a\,n_{\text{phys}} + \sum_k
  b_k\, J\hat{t}_k$. Because $n_{\text{phys}} \perp J\hat{t}_k$ and FIAT normals are
  "UFC consistent" (the *same* tangent-based formula on reference and physical cells —
  `UFCTriangle.compute_normal` is a rotation of the scaled edge tangent;
  `UFCTetrahedron.compute_normal` is $-2\times$ the unit cross product of scaled face
  tangents), all signs and normal-magnitude conventions cancel identically:
  $$a = \det J\, \sqrt{\det\hat{G}/\det G}, \qquad b = G^{-1} T^T J\hat{n},$$
  with $T = [J\hat{t}_k]$, $G = T^T T$ (GEM), $\hat{G}$ the reference tangent Gram
  (numeric). This is `_normal_tangential_transform` and `morley_transform`
  generalized and unified across dimensions. Assumes $\det J > 0$.
* **Integral averages are push-forward invariant.** FIAT's Morley dofs are averages
  (`FacetQuadratureRule(..., avg=True)` inside `IntegralMomentOfNormalDerivative`, and
  explicit `avg=True` codim-2 moments), so value-moment nodes need identity rows and the
  facet-measure Jacobians cancel from $a$ — no `physical_edge_lengths` needed anywhere.
* **Completion functionals are built from the node's own quadrature.**
  `IntegralMomentOfDerivative(ref_el, node.Q, node.f_at_qpts, t)` reuses the stored rule
  and weight of the normal node, guaranteeing the tangential partners carry identical
  scaling conventions.
* **The conditioning $h$-scaling generalizes via `max_deriv_order`.** Column $j$ is
  scaled by $h_E^{-m}$ where $m$ is the derivative order of node $j$ and $h_E$ averages
  `cell_size()` over the vertices of its entity. This reproduces the per-element
  hand-written scalings (Morley facet dofs, Hermite/Argyris vertex jets). Note
  `cell_size()` returns raw numpy values in the test mappings but GEM in Firedrake, so
  use operator arithmetic (`havg**(-m)`), not explicit GEM node constructors.

Next steps: vertex-jet groups (`PointDerivative`/`PointSecondDerivative` etc. →
`_jet_transform` blocks) to cover Hermite, then completion rows that couple to
non-invariant (jet) dofs via recursive substitution to cover HCT/Argyris, then the
extended-element path for reduced HCT/Bell.

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
