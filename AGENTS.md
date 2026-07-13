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

## Style and Conventions

When writing Python code for FIAT, maintain the ecosystem's structural and stylistic integrity:

* Class attributes must be declared in one visible location.
* Attributes must be initialized in the `__init__` constructor or declared as a `functools.cached_property` if they are expensive to compute.
* Ad hoc lazy initialization discovering attributes via `hasattr`, `setattr`, or `getattr` scattered across methods is strictly prohibited.
* Boolean attributes must be used to record initialization intent and state instead of probing for the presence of state built by an initialization function.
* New code must include type hints on all function and method signatures.
* Public-facing APIs must include properly formatted `numpydoc`-style docstrings.

## Pull Request Expectations

* All changes are expected to arrive through GitHub Pull Requests.
* Keep diffs reviewable and focused.
* Before concluding work verify that the relevant subset of the pytest test suite
  succeeds locally.
* CI (`.github/workflows/test.yml`) checks `flake8` and `pydocstyle .` (both
  configured in `setup.cfg`); run them locally and fix all findings before pushing.
* Watch for `pydocstyle` D413 (blank line after the last numpydoc section) and D417
  (every argument, including keyword arguments, needs a description) in new docstrings.

---

## Pattern Matching and Mathematical Reasoning

This section used to be a rough draft of aspirations. It is now grounded in the concrete
experience of automating the Kirby (2017) / Aznaran-Kirby-Farrell (2022) / Brubeck & Kirby
(2025) transformation theory (`finat/zany.py`, `finat/functional.py`) for Morley, Hermite,
Argyris, Bell, Mardal-Tai-Winther, Johnson-Mercier, and Guzman-Neilan. The lessons below are
about *how to design and debug this kind of code*, not just about this one project.

### Mathematical structures to recognize in FIAT/FInAT

* **A degree of freedom is fully described by five numbers, not by its FIAT class.** Every
  functional FIAT builds from `pt_dict`/`deriv_dict` reduces to (points, weights, derivative
  order $m$, a direction tensor of rank $m$, a value rank for vector/tensor-valued dofs).
  `IntegralMomentOfNormalDerivative`, `PointNormalDerivative`, `TensorBidirectionalIntegralMoment`,
  etc. are just different ways of *constructing* the same five numbers. Recognizing this
  collapses "N functional types to support" into "one shape to recover numerically"
  (`finat.PhysicallyMappedFunctional.from_fiat`, `finat/functional.py`).
* **Pullback is always "contract each tensor slot of the direction with a fixed matrix."**
  Order-0 (values) are invariant (zero slots to contract). Order-$m$ derivatives contract $m$
  slots with the Jacobian $J$ (the chain rule). Rank-$r$ Piola values contract $r$ slots with
  the cofactor matrix $K = \operatorname{adj}(J)^T$. This is *the same operation* with a
  different matrix, which is why the scalar and Piola code in `finat/zany.py` are mirror images
  of each other (`_scalar_point_rows` / `_piola_point_rows`, `_scalar_facet_rows` /
  `_piola_facet_rows`) rather than unrelated implementations.
* **Frame decomposition is the one computational primitive underlying every non-affine
  element.** Facet dofs (Morley/Argyris/Bell normal derivatives; MTW/JM/GN normal-tangential
  moments) and vertex-jet completions (Hermite/Argyris/Bell gradients and Hessians) are all
  solved the *same* way: split a direction/profile into an invariant part and a part that needs
  completing, express the pulled-back quantity in the frame built from the mapped generators of
  that split (`FacetFrame`, or the direction-basis inverse in `_scalar_point_rows`), and solve
  symbolically via `adjugate`/`determinant`. Once this pattern is visible, "add a new element"
  stops being "derive new math" and becomes "which invariant subspace, which frame."
* **Constrained/extended elements are the same construction as a restriction.** Bell and
  Guzman-Neilan are both "take the extended FIAT element with its constraint functionals as
  extra dofs, transform the whole thing, then keep only the first $\nu$ columns." This is
  Kirby (2017) §5's extended-element proposition, and in code it is nothing more than
  `space_dimension()` returning a smaller count than the FIAT element's — no special-casing
  needed in the transformation loop itself.

### Confusing mathematical ideas, clarified

* **The Jacobian direction is flipped between the papers and the code.** The papers define
  $F$ from physical to reference space, so their $J$ is FInAT's Jacobian *inverse*.
  `coordinate_mapping.jacobian_at(point)` returns $\partial x_{\text{phys}}/\partial
  x_{\text{ref}}$ (see "FInAT implementation conventions" below); a paper's $J^{-T}$ is FInAT's
  plain Jacobian, transposed. This is a constant source of "why does my formula look
  transposed" confusion — check this convention flip before suspecting a sign or index bug.
* **Components against a reciprocal/dual basis transform contragrediently — this is the single
  subtlety that broke 3D and is not in any of the papers.** FIAT builds the tangential
  component of 3D Piola-mapped dofs (MTW) using `cross(n, t_k)`, the *reciprocal* partner of
  the tangent frame, not the tangent frame itself. A general fact from differential geometry:
  if a frame transforms by a matrix $A$, components expressed against its *reciprocal* frame
  transform by $A^{-T}$ (up to a determinant factor), not by $A$. Kirby (2017)/Aznaran-Kirby-
  Farrell (2022) only work out the 2D case, where the tangent "plane" is 1-dimensional and this
  correction $S^{-1}$ collapses to $1$ — silently hiding the effect. **Whenever a FIAT dual set
  builds a direction via a cross product against the normal (a common way to get "the other
  in-plane directions" in 3D), check whether its transformation law is contragredient before
  assuming it matches the direct frame.**
* **Numerically recovered invariants (SVD, pinv) are only defined up to a group action, and
  formulas built from them must be invariant under that action.** `from_fiat`'s SVD recovers a
  direction/weight pair $(\hat n, w)$ up to a joint sign flip ($\hat n \to -\hat n$, $w \to
  -w$ leaves the functional unchanged). A formula like $r_k = a x_k + (1-c)\beta_k$ that
  implicitly assumes a particular sign will be wrong on exactly the subset of entities where
  SVD happened to pick the other sign — a bug that looks like it "mostly works" and is
  otherwise very hard to localize. The fix, $r_k = x_k - c\beta_k$, is invariant under the
  joint flip. **Always test the full topology of a non-degenerate, non-symmetric physical
  cell** (see `test/finat/conftest.py::MyMapping`'s deliberately irregular vertex coordinates) —
  a symmetric test cell can accidentally hide a sign bug that only shows up on a generic mesh.

### Design strategies that generalize

* **Design by duality, never by direct construction.** Do not try to write physical basis
  functions directly (that requires inverting a Vandermonde system). Write the physical *node*
  (a linear functional — point evaluation or moment, always computable), push it forward, and
  solve for its expansion in the reference nodes. This is the master strategy behind the whole
  framework and generalizes to any element whose nodes, not whose basis functions, are the
  simple objects.
* **Recover mathematical type from data, not from a class hierarchy.** `from_fiat` never asks
  "is this an `IntegralMomentOfNormalDerivative`?"; it reads `pt_dict`/`deriv_dict` and derives
  (order, direction, rank) numerically. This makes the framework forward-compatible with FIAT
  functional classes that do not exist yet, as long as they reduce to the same numeric shape —
  which essentially all "linear functional built from point/derivative data" dofs do. Prefer
  this kind of structural recovery over adding another `isinstance`/type-tag branch whenever a
  new functional needs support.
* **Turn a mathematical case-split into a template method, not a conditional.** When a theorem
  reads "for pullback type X do A, for pullback type Y do B" against an otherwise identical
  algorithmic skeleton, encode the skeleton once (`ZanyPhysicallyMappedElement
  .basis_transformation`) and the case-split as small, named hook methods on sibling mixins
  (`ScalarPhysicallyMappedElement`, `PiolaPhysicallyMappedElement`). Keep the underlying linear
  algebra as free functions taking plain arrays/GEM expressions with no `self`
  (`FacetFrame`, `_scalar_facet_rows`, `_piola_facet_rows`, ...): the class layer should only
  ever be a thin adapter from "element structure" to "which pure-math function to call," never
  a place where new mathematics is derived. This also keeps the generic infrastructure
  (`PhysicallyMappedElement` in `finat/physically_mapped.py`, used by hand-coded elements like
  Arnold-Winther and HCT) free of any awareness of the theory-specific case-split.
* **Validate one element at a time against an independently computed ground truth, not just
  against the closed-form answer you derived.** `test/finat/test_zany_mapping.py
  ::check_zany_mapping` tabulates the FIAT element on an actual physical cell and least-squares
  fits the physical values against the pulled-back reference tabulation — a computation
  completely independent of the symbolic derivation. Its assertion message pretty-prints the
  (row, column) location of the relative error; read those indices first, they usually
  localize the bug to one entity/dof-type combination before you write any new code.
* **Generalize from one worked example, verify to machine precision, then extend — never
  generalize speculatively ahead of a second concrete example.** The actual sequence that
  worked: derive the mechanism on Morley alone, check it reproduces the deleted hand-coded
  matrix bit-for-bit, *then* extend to Hermite, then Argyris/Bell, then the Piola family. Each
  extension needed a genuinely new piece (interior invariance for Piola interior moments,
  mixed-order jets for Bell/Argyris vertices, the reciprocal-basis correction for 3D) that a
  premature generalization from Morley alone would not have anticipated.

### Working with the human collaborator

* **The literature is necessary but not sufficient — treat a stuck derivation as a cue to ask,
  not to keep re-deriving.** Kirby (2017) and Aznaran-Kirby-Farrell (2022) work out 2D (or a
  simplification that hides a 3D-only effect); none of the papers mention the reciprocal-basis
  transformation law 3D Piola elements need. That fix came directly from the user's domain
  expertise ("the papers do not consider the reciprocal basis that FIAT implements... Consider
  this"), not from re-reading the papers harder. When a derivation matches every 2D case but
  fails exactly where a paper's simplifying assumption would hide something, that is the moment
  to ask what convention the *implementation* (not the paper) actually uses.
* **Expect, and design for, architectural correction mid-project.** This project's real
  trajectory was: prototype on Morley → generalize into a `Functional`/free-function design →
  "don't extend further, make the computation generic first, get rid of the very specific
  helpers" → extend to Argyris/Bell → extend to the Piola family → "refactor into
  `Scalar`/`PiolaPhysicallyMappedElement` mixins so the `if piola` disappears from the loop" →
  "move the method body into the Mixin" (keep the generic `PhysicallyMappedElement` free of
  zany-specific knowledge). None of these were bug fixes; each was the human reshaping the
  design once its true shape became visible. Treat working code as a draft the architecture
  will still move under, and re-run the full verification (test suite, `flake8`, `pydocstyle`)
  after every such reshaping, not only after functional changes.
* **Keep this file's *why*, not just the *what*.** Git history has the diffs; this file exists
  so the next session does not have to rediscover the reciprocal-basis subtlety, the
  sign-invariance requirement, or the identity-row idiom by re-reading a diff from scratch.
