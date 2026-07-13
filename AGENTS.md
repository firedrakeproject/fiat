# AGENTS.md for FIAT

This document outlines the guidelines and architectural context for AI agents assisting with the FIAT codebase, functioning as a core component within the broader Firedrake ecosystem. 

---

## AI Contribution Policy

When assisting with contributions to FIAT and the Firedrake project, AI agents and their human counterparts must adhere to the following strict policies:

* The use of AI tools must be explicitly declared alongside the specific tool used.
* Reviewer questions must be answered directly by the human, rather than acting as a relay to the AI.

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
