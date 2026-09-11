### Confusing GEM patterns, clarified

* **Use operator overloading, not manual node construction.** `gem.Node` overloads
  `+ - * / ** @` (via `as_gem`/`componentwise`) to work transparently across GEM nodes,
  `gem.Literal`, and plain Python/numpy numbers, with automatic `Zero` folding. Write
  `havg**(-m)`, never `gem.Power(havg, gem.Literal(-m))`: the operator form also works when
  `havg` is a raw float (as in the test harness's `MyMapping.cell_size()`) whereas a manual
  `gem.Power` call assumes GEM operands and will not always coerce correctly.
* **Never call `numpy.linalg.inv`/`solve` on a GEM-valued matrix.** LAPACK cannot see inside
  GEM expressions. Symbolic linear solves use `adjugate(A) / determinant(A)`
  (`finat/physically_mapped.py`) — the symbolic Cramer's-rule equivalent — e.g. in
  `solve`, which inverts the diagonal blocks of the physical Vandermonde matrix. Numeric
  `numpy.linalg` calls are only valid when every entry is a plain number: reference-cell-only
  quantities such as the orthonormal facet tangents of `FlagFrame` (`numpy.linalg.qr`). Check
  which regime an array is in before reaching for a numpy linear-algebra routine.
* **Build sparse GEM-valued arrays with `numpy.full(shape, gem.Zero(), dtype=object)`**, never
  `numpy.zeros(shape)` — plain `0` is not interchangeable with `gem.Zero()` when the array will
  later be combined with GEM nodes via `+`.
* **"Start from identity, mutate only the rows that need work" is not an optimization, it is
  the mathematical content.** `V = identity(nbf)` (`finat/physically_mapped.py`) encodes
  "these dofs are push-forward invariant" directly; `PhysicalVandermonde.inverse` only replaces
  the rows of entities carrying a node that is not invariant (`PhysicalNode.is_invariant`),
  rather than writing `1`s explicitly. Treat the untouched identity rows as the base case of
  the block back-substitution.
* **Row/column convention, and where the one transpose happens.** Throughout assembly, row
  index = reference node, column index = physical node — i.e. the code builds $V$, never $M$
  directly. `ListTensor(V.T)` at the very end is the single place Kirby (2017) Theorem 3.1's
  $M = V^T$ gets applied. If something looks transposed, check this convention before
  suspecting a sign error. The dual evaluation applies $B = V^{-1}$ itself
  (`PhysicalVandermonde.matrix`, no transpose): never invert a GEM-valued $M^T$ to get it.
