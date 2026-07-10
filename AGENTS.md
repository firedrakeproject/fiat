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

## Style and Conventions

When writing Python code for FIAT, maintain the ecosystem's structural and stylistic integrity:

* Class attributes must be declared in one visible location.
* Attributes must be initialized in the `__init__` constructor or declared as a `functools.cached_property` if they are expensive to compute.
* Ad hoc lazy initialization discovering attributes via `hasattr`, `setattr`, or `getattr` scattered across methods is strictly prohibited.
* Boolean attributes must be used to record initialization intent and state instead of probing for the presence of state built by an initialization function.
* New code must include type hints on all function and method signatures.
* Public-facing APIs must include properly formatted `numpydoc`-style docstrings.
