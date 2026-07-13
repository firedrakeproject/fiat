## FInAT implementation conventions

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
    `morley_transform` in `finat/walkington.py` (Morley itself is automated now; this
    3D helper survives only as Walkington's dependency).
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

## Mathematical structures to recognize in FIAT/FInAT

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
