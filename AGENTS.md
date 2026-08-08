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

## Style and Conventions

When writing Python code for FIAT, maintain the ecosystem's structural and stylistic integrity:

* Class attributes must be declared in one visible location.
* Attributes must be initialized in the `__init__` constructor or declared as a `functools.cached_property` if they are expensive to compute.
* Ad hoc lazy initialization discovering attributes via `hasattr`, `setattr`, or `getattr` scattered across methods is strictly prohibited.
* Boolean attributes must be used to record initialization intent and state instead of probing for the presence of state built by an initialization function.
* New code must include type hints on all function and method signatures.
* Public-facing APIs must include properly formatted `numpydoc`-style docstrings.
* CI enforces `pydocstyle` (see the `[pydocstyle]` section of `setup.cfg` for the active ignore list)
  in addition to `flake8`; run `pydocstyle <changed files>` locally before finishing a change, since a
  clean `flake8` pass does not imply a clean `pydocstyle` pass.
* Every docstring or comment you write or touch must follow Simplified Technical English
  (ASD-STE100): short sentences, one idea per sentence, active voice, subject named up front instead
  of buried in a relative clause. Avoid the clause-stacking, inverted phrasing typical of unedited
  AI-generated prose.
* When fixing code that was wrong, do not leave comments or prose explaining what the removed,
  incorrect approach used to do or why it was wrong. Keep comments and documentation focused on the
  current, correct code. The test to apply: a reader who never saw the diff must not be able to tell
  that anything was removed.

## Anti-Patterns

These must be avoided when writing code, and flagged when reviewing it.

### Clause-Stacked Docstrings And Comments

WRONG — the subject hides inside a relative clause the reader must unwind before finding the verb:

```python
def scale_boundary_nodes(u, factor):
    """Give the nodes a boundary condition constrains their scaled values."""
```

RIGHT — subject named up front, one short sentence, active voice:

```python
def scale_boundary_nodes(u, factor):
    """Scale the values of the nodes that a boundary condition constrains."""
```

### Documenting Code That Is Not There

A reader has only the file in front of them. A comment can describe a removed approach. It can also
argue against a branch the code does not take. Either one sends the reader looking for something
that is not there.

WRONG — the first sentence describes deleted code, and the second argues with an absent branch:

```python
def barycentric_weights(points):
    # This no longer normalises the weights, which was wrong when the points
    # were not symmetric. A test for a repeated point here would divide by
    # zero.
    return 1.0 / numpy.prod(points[:, None] - points[None, :] + numpy.eye(len(points)), axis=1)
```

RIGHT — say what the present code does, and state the condition it relies on:

```python
def barycentric_weights(points):
    # The identity keeps the diagonal out of the product. The caller passes
    # distinct points.
    return 1.0 / numpy.prod(points[:, None] - points[None, :] + numpy.eye(len(points)), axis=1)
```

Some words give this away on sight: "used to", "previously", "no longer", "instead of", "we removed",
"this replaces". Watch equally for "would" when its subject is code that does not exist. An argument
against a branch that nobody can see is still a description of the past.
