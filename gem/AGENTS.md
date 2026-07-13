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
  `FacetFrame.decompose` and `_piola_facet_rows`. Numeric `numpy.linalg` calls are only valid
  when every entry is a plain number: reference-cell-only quantities
  (`FacetFrame.reference_coefficients`) or purely numeric direction-basis inversions
  (`_scalar_point_rows`, `_piola_point_rows`). Check which regime an array is in before
  reaching for a numpy linear-algebra routine.
* **Build sparse GEM-valued arrays with `numpy.full(shape, gem.Zero(), dtype=object)`**, never
  `numpy.zeros(shape)` — plain `0` is not interchangeable with `gem.Zero()` when the array will
  later be combined with GEM nodes via `+`.
* **"Start from identity, mutate only the rows that need work" is not an optimization, it is
  the mathematical content.** `V = identity(ndof)` (`finat/physically_mapped.py`) encodes
  "these dofs are push-forward invariant" directly; the invariant-dof detection
  (`_invariant_dofs`) simply chooses *not* to touch those rows, rather than writing `1`s
  explicitly. Treat the untouched identity rows as the base case of the assembly recursion.
* **Row/column convention, and where the one transpose happens.** Throughout assembly, row
  index = reference node, column index = physical node — i.e. the code builds $V$, never $M$
  directly. `ListTensor(V.T)` at the very end is the single place Kirby (2017) Theorem 3.1's
  $M = V^T$ gets applied. If something looks transposed, check this convention before
  suspecting a sign error.
