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
`finat.PhysicallyMappedFunctional`. `finat/physically_mapped.py` stays fully generic:
`PhysicallyMappedElement` is unchanged from before this project, still an abstract
mixin with no knowledge of the zany theory, used as-is by hand-coded elements (AW,
HCT, PowellSabin, Walkington, ...). All the automation lives in `finat/zany.py`, as a
template method on `ZanyPhysicallyMappedElement(PhysicallyMappedElement)`: the
entity-by-entity assembly loop is implemented once, in its concrete
`basis_transformation`, calling four hooks (`_check_mapping`, `_invariant_dofs`,
`_facet_dof_rows`, `_point_dof_rows`) that carry ALL mapping-specific knowledge — the
loop itself contains no `if piola` anywhere. Two mixins implement those hooks,
`ScalarPhysicallyMappedElement` (affine pullback: Morley, Hermite, Argyris, Bell) and
`PiolaPhysicallyMappedElement` ((double) contravariant Piola: MTW, Johnson-Mercier,
Guzman-Neilan), plus the pure math functions they call (`FacetFrame`,
`_scalar_facet_rows`, `_scalar_point_rows`, `_piola_facet_rows`, `_piola_point_rows`) —
these take plain arrays/GEM expressions, no `self`, so the mathematics stays readable
independent of the class plumbing. Concrete elements (`finat.Morley`, etc.) are now
just a citation plus a FIAT constructor call: mixing in the right base class is enough,
`basis_transformation` is inherited (MRO example: `Morley -> ScalarPhysicallyMappedElement
-> ZanyPhysicallyMappedElement -> PhysicallyMappedElement -> ...`). `ndof` truncation is
no longer a parameter; the loop always slices by `self.space_dimension()`, which
constrained elements (Bell, GN) already override. Tests:
`test/finat/test_zany_automation.py`; `check_zany_mapping` lives in the finat conftest
and is provided to test modules as a pytest fixture (pytest runs with
`--import-mode=importlib`, so test modules cannot import from each other or from
conftest).

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
with the mixing matrix $Y$; tangential profiles are matched within the facet group by
solving the (small, square) Gram system $B B^T c = B\cdot(\text{target})$, where $B$
stacks the group's own reference tangential profiles and has full row rank by
unisolvence, so $B B^T$ is invertible and a rank deficiency (a genuine bug) surfaces as
a hard numerical error rather than passing silently; the residual normal profile is
eliminated by per-point normal moments through the Vandermonde recursion (this is where
e.g. tangential-to-normal couplings emerge).

**Sparsity gotcha: fold the quadrature-point sum numerically, never symbolically.** The
naive way to do that last elimination is to build the per-point residual (a GEM
expression, since it involves the symbolic mixing matrix $Y$) and dot it against the
tabulated-basis row `L[m, :]` (numeric) with a Python loop over points, accumulating
into `V[i] += V[m] * (residual[q] * L[m, q])`. This reproduces the right *numbers* but
not the right *sparsity*: whenever the true coupling to some dof `m` is zero, it is zero
only as an analytic identity in $J$ (a cancellation across quadrature points or across
frame directions), not as a literal `gem.Zero()` node — GEM has no polynomial-identity
simplifier, so `isinstance(x, gem.Zero)` (or eyeballing `x != 0`) cannot detect this, and
the resulting matrix keeps a dense cloud of numerically-tiny-but-symbolically-nonzero
entries (compare against the hand-coded matrix's exact sparsity, e.g. MTW's
`normal_tangential_transform`, to see the gap). The fix: contract the quadrature-point
axis while everything is still numeric — `Lmap[i] = L @ coords[i]` (both plain arrays)
— and only *afterwards* multiply by the (few, small) symbolic frame-mixing scalars
($Y[0, r]$, or $-c_j$ for each group member's own normal profile), one reference-frame
multi-index/group-member at a time rather than summed together first. Sparsity is then
decided by thresholding the numeric `Lmap` columns (`abs(...) < tol`), exactly the same
idiom already used for `Binv`/`B` above; a symbolic sum of several *different* GEM
scalars must never be built first and then inspected for zero-ness, since that sum's
GEM shape does not reveal whether it is analytically zero. Verified by comparing bit-for-
bit against the deleted hand-coded `MardalTaiWinther.basis_transformation` sparsity
pattern for order 1 (2D/3D) and order 2 (3D): zero extra and zero missing entries.

Key subtlety (in any dimension > 2): FIAT builds tangential value
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

**AlfeldSorokina and GuzmanNeilanH1div (divergence at vertices).** AlfeldSorokina's dual
basis interleaves, at every vertex, one `PointDivergence` node with `sd` `ComponentPointEvaluation`
nodes at the same point; GuzmanNeilanH1div wraps the same AlfeldSorokina dofs (plus GN
facet bubbles), so it needs the identical fix. `PointDivergence` does not fit the
existing `PhysicallyMappedFunctional` shapes: it has both `order = 1` (a derivative) and
`comp != ()` (a value component) on the *same* weight entries, in the pattern $\ell(v) =
\sum_i \partial_i v_i$ — a trace pairing an order-1 derivative multi-index with a rank-1
value component, not a rank-one `direction` the SVD factorization can recover. Recovered
in `from_fiat` by testing, per point, that the $(alpha, comp)$ weight matrix is a scalar
multiple of the identity; represented with a new `divergence: bool` flag (order and rank
are otherwise meaningless for this node, so reusing them would have made `evaluate`/
`pullback` silently wrong for it — a new field beats overloading old ones here). Handling
is a one-line closed form, not a generalization of the existing machinery: the
(contravariant) Piola pullback commutes with the divergence up to $\det J$, regardless of
which entity the node sits on, so `PiolaPhysicallyMappedElement._divergence_rows` strips
divergence nodes out of the group first and sets `V[i, i] = detJ * ell.weights[0]`
directly, before either `facet_dof_rows` or `point_dof_rows` runs.

Stripping the divergence node still left a `facet_dof_rows` bug: AlfeldSorokina's edge
dofs (`dim == sd - 1` in 2D) are `ComponentPointEvaluation`s too — plain Cartesian point
values that happen to sit on a codimension-1 entity, not genuine facet moments. Routing
them through the normal/tangential frame logic built for MTW/JM/GN produced a singular
Gram matrix (`B @ B.T`), because that logic assumes the tangential-only residual group
has exactly `sd - 1` members (the facet completion count), but `sd` unrelated Cartesian
components don't reduce to that shape. Fix: `_is_cartesian_point_group` recognizes a
group of rank-1, order-0 nodes sharing one single point (real facet moments are built
from multi-point quadrature, even at low order — verified by checking `len(ell.points)`
for BernardiRaugel/MTW/JM/GN facet dofs, all $>1$) and dispatches it to the same
`_piola_point_rows` Cartesian-component transform `point_dof_rows` uses, regardless of
the entity's topological dimension. This is the same lesson as the FacetFrame work:
dispatch on what the node's *data* looks like, never on which entity it happens to sit on.

Next steps: GN second kind (interior derivative moments need the divergence detJ rule,
same as above), ArnoldWinther (vertex tensor values + higher facet moments), the
extended-element path for reduced HCT (macro polynomial spaces); covariant elements.

## Paper process (Kirby, Marsden & Brubeck, "Automation of Finite Element
Transformations"): staged plan

Process agreed 2026-07-16 (full plan: `~/.claude/plans/lazy-yawning-quasar.md`): alternate
a *math stage* (derive/restate theory, hand-work an example, extract a glossary) with a
*code stage* (rename/refactor to match, verify via `test/finat/test_zany_mapping.py` +
`flake8` + `pydocstyle`), starting with the scalar case (Stages 0-3) and then testing the
hypothesis that Piola-mapped elements unify further by composing the Piola transform with
the scalar machinery (Stages 4-6), before assembling `fiat_zany_auto/paper.tex` (Stage 7).
Math work accumulates here before being promoted into the paper. Stage 0 (restate
pullback/push-forward duality, $M = V^T$, and $V = EV^cD$ precisely) is already satisfied
by the "Transformation Theory" section opening this file.

### Stage 1 (2026-07-16): worked example — Hermite vertex jets

The affine-interpolation-equivalent case (`ScalarPhysicallyMappedElement.point_dof_rows`,
`finat/zany.py:420-458`), worked by hand and checked to machine precision against
`finat.Hermite.basis_transformation` on a concrete triangle.

**Setup.** Reference cell $\hat K$ = UFC triangle, vertex $\hat v_0 = (0,0)$. FIAT's cubic
Hermite dual basis carries, at $\hat v_0$, one value node and two derivative nodes
$\hat\ell_1 = \partial/\partial\hat x_1$, $\hat\ell_2 = \partial/\partial\hat x_2$
(`FIAT.hermite.CubicHermite(ref).dual_basis()`, confirmed via `deriv_dict`). Physical
triangle vertices `((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))` (the `phys_el` fixture of
`test/finat/conftest.py`), giving the constant Jacobian
$$J = \begin{pmatrix} 1.17 & 0.15 \\ -0.19 & 1.74\end{pmatrix}$$
(`coordinate_mapping.jacobian_at`, i.e. $x = J\hat x + b$ — this is the *code's* $J$,
mapping reference to physical; see the reconciliation note below for how it relates to
the "paper's $J$" of the opening Transformation Theory section).

**Mechanism.** Away from a facet there is no geometric frame (`FacetFrame` does not
apply), so `point_dof_rows` treats the whole order-1 group at $\hat v_0$ as its own
completion, exactly the affine-interpolation-equivalent case of Kirby (2017): the group
must span the derivative jet, and $V$'s $2\times2$ block on $\{1,2\}$ is obtained by
expanding each node's *pulled-back* direction in the group's own direction basis.
Precisely (`finat/zany.py:437-458`):

1. `PhysicallyMappedFunctional.from_fiat` recovers each $\hat\ell_i$'s direction/weight
   pair $(d_i, w_i)$ from the reference dual basis by a rank-1 SVD of the derivative
   weight matrix — only determined up to a common sign/scale, so $w_i d_i$ (not $d_i$
   alone) is the invariant quantity, the *true* reference direction $e_i$. Here FIAT
   stores $\hat\ell_1$ directly as $(d_1,w_1)=((1,0),\,1)$, but recovers $\hat\ell_2$ as
   $(d_2,w_2) = ((0,-1),\,-1)$ (sign flipped by the SVD) — with $e_2 = w_2 d_2 = (0,1)$
   the correct Cartesian direction is recovered regardless.
2. `directions = [d_1; d_2]`, `Dinv = inv(directions.T)`: because $d_i$ (not $e_i$) enters
   here, $Dinv$ absorbs the same sign ambiguity that $w_i$ will later correct.
3. For each pair $(i,j)$: `s = _weight_ratio(w_i, w_j)` $= w_i/w_j$ recovers the *relative*
   sign/scale between the two recovered factorizations (`_weight_ratio` raises if the
   weights are not simply proportional — here trivially $\pm1/\pm1$), and
   `x = s * Dinv[col_j]`; then `V[i,j] = pullback(J, d_i) @ x`, where
   `pullback(J, d_i) = J @ d_i` for an order-1 direction (verified numerically: the chain
   rule for a first derivative is a plain linear map, no transpose, since `direction` is
   contravariant in the *entity* Jacobian at this stage — the transpose appears only once
   $d_i$ is re-expressed in the *dual* frame `Dinv` provides).

Composing steps 2-3 for the true (sign-corrected) directions $e_i$ collapses to the clean
statement $V[i,j] = e_j^T J\, e_i$ — i.e. on this basis $V\big|_{\{1,2\}} = E^TJE$ with
$E = [e_1\,|\,e_2]$ (here $E=I$, the Cartesian basis, so $V\big|_{\{1,2\}} = J^T$ exactly);
the $w_i/d_i$ split and `_weight_ratio` correction exist only so this holds *regardless*
of which sign/scale `from_fiat`'s SVD happens to recover, not because the mathematics
needs a sign at all.

**Verification.** Computed by hand from the formula above and checked bit-for-bit against
`finat.Hermite(ref).basis_transformation(mapping)` (evaluated via `gem.interpreter`):
$$
V\big|_{\{1,2\}} = J^T = \begin{pmatrix} 1.17 & -0.19 \\ 0.15 & 1.74 \end{pmatrix}
\quad\Longleftrightarrow\quad
M\big|_{\{1,2\}} = V^T\big|_{\{1,2\}} = J = \begin{pmatrix} 1.17 & 0.15 \\ -0.19 & 1.74 \end{pmatrix},
$$
matching `M[1:3,1:3]` from the running code exactly (both diagonal *and* off-diagonal
entries, so the sign/scale bookkeeping above is exercised nontrivially, not just checked
on a symmetric or diagonal case).

**Reconciliation with the opening Transformation Theory section.** Line 15 there states
the vertex-gradient block of $M$ as "$J^{-T}$ ... in the *paper's* $J$." The code's
`coordinate_mapping.jacobian_at` returns $J_{code}$ with $x = J_{code}\hat x + b$
(reference $\to$ physical), whereas Kirby (2017)'s $F$ maps physical $\to$ reference, so
its Jacobian is $J_{paper} = J_{code}^{-1}$; hence $J_{paper}^{-T} = J_{code}^T$, exactly
the $M$ block computed above. **Glossary note for Stage 3**: every code docstring should
state explicitly which of these two (inverse, transpose-of-each-other) conventions a
given $J$ is, since the two papers' $F$ and the code's `coordinate_mapping` point in
opposite directions.

### Stage 2 (2026-07-16): worked example — Morley edge completion

The general (non-equivalent) completion case (`ScalarPhysicallyMappedElement.facet_dof_rows`,
`finat/zany.py:368-418`, using `FacetFrame`), worked by hand for edge 0 (the hypotenuse,
joining vertices 1 and 2) on the same physical triangle as Stage 1, and checked bit-for-bit
against `finat.Morley`.

**Setup.** FIAT's Morley dual basis has, on edge 0, a single node $\hat\ell_3$: the
normal derivative at the midpoint $(0.5,0.5)$, stored as an *average*
(`deriv_dict` weights $(1,1)/\sqrt2$ on $(\partial_{\hat x_1}, \partial_{\hat x_2})$,
i.e. direction $\hat n = (1,1)/\sqrt2$, the reference normal itself — Morley has no
separate tangential-derivative node, so $\hat\ell_3$ must be *completed* using the
vertex value nodes ($\hat\ell_1,\hat\ell_2$ at $v_1=(1,0)$, $v_2=(0,1)$, already
processed since $\mathrm{order}=0$) before it can be expanded on the physical cell.
Reference edge-0 tangent (FIAT) $\hat t = (-1,1)$ (the chord $v_2-v_1$, unnormalized).

**Mechanism.** `FacetFrame(Mo, 0, J)` builds the reference frame
$[\hat n\,|\,\hat t\,]$ and its physical image $[C\,|\,J\hat t\,]$ with
$C = $ `generalized_cross`$(J\hat t)$ (here just a $90°$ rotation of $J\hat t$ in 2D).
`facet_dof_rows` then, for $\hat\ell_3$:

1. `reference_coefficients(\hat\ell_3.direction)` solves $\hat d = a\hat n + \beta\hat t$
   for $(a,\beta)$ in the *reference* frame — here $a=-1$ (sign from the SVD recovery
   of `from_fiat`, harmless: every downstream quantity built from $a$ is consistently
   rescaled), $\beta=0$ exactly, since $\hat\ell_3$'s direction *is* the reference normal
   with no tangential component by construction.
2. `decompose(\hat\ell_3.pullback(J).direction)` solves the *physical*, symbolic system
   $J\hat d = x_0 C + x_1 J\hat t$ via `adjugate`/`determinant` (Cramer's rule, since $J$
   is symbolic in general; numeric here because $J$ is the constant matrix of Stage 1)
   giving $(x_0,x_1) = (0.612629,\,-0.244111)$.
3. $c = x_0\cdot(\|C\|/\kappa)/a = 1.337343$ becomes $V[3,3]$ — the diagonal
   "own-node" coefficient, i.e. the $B_{nn}$ entry of the notebook's $2\times2$ block
   $B = \hat G J^{-T} G^T$ (line 34 of the Transformation Theory section above; that
   block's off-diagonal $B_{nt}$ entry is exactly $-c\beta/a$-type bookkeeping, here
   $0$ since $\beta=0$).
4. The tangential residual $r = x_1 - c\beta = -0.244111$ is *not itself* a new
   unknown: because $\hat t$ is a **mapped reference tangent**, the reference
   functional "derivative along $\hat t$ at the midpoint" coincides with a functional
   already expressible in the element's own reference basis — `ell.with_direction(\hat t)
   .evaluate(Mo)` (a numeric generalized-Vandermonde row) gives coefficients
   $(0,1,-1,0,0,0)$: the classical fact that for a quadratic, the derivative along the
   full chord at the midpoint equals $f(v_2)-f(v_1)$ exactly (Brubeck & Kirby 2025's
   univariate-exactness/FTC argument realizing $D$, here in its simplest 1-node form).
5. Elimination: $V[3,1] \mathrel{+}= r\cdot1 = -0.244111$, $V[3,2] \mathrel{+}= r\cdot(-1)
   = +0.244111$ — both already-`processed` vertex rows, so no `NotImplementedError`.

**Verification.** `finat.Morley(ref).basis_transformation(mapping)` (same physical
triangle as Stage 1) gives column 3 of $M=V^T$ as
$M[:,3] = (0,\,-0.244111,\,0.244111,\,1.337343,\,0,\,0)$, matching $V[3,\cdot]$ from
steps 3-5 to 6 decimal places.

**What this example newly exercises, vs. Stage 1.** Stage 1's Cartesian point-jet
group needed only $E=I$ (no frame) and a diagonal-plus-off-diagonal $V^c$ block coming
purely from the chain rule. Here, for the first time, $E$ (the frame's normal/tangential
extraction, keeping only the normal row) and $D$ (the numeric elimination of the
tangential residual through *already-assembled, lower-dimension* rows — vertex nodes,
processed before edges by the increasing-dimension loop of `basis_transformation`) are
both genuinely nontrivial and distinct, which is exactly the completion mechanism the
notebook's $V=EV^cD$ factorization (lines 21-36) describes in the abstract.

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

### Stage 3 (2026-07-16): scalar glossary + code refactor — done

Deliverables, applied to `finat/zany.py` only (zero semantic diff, no behavior change):

* A module-level glossary docstring mapping every recurring symbol (`V`, `E`, `V^c`, `D`,
  `J`, `K`, `normal`, `direction`/`weights`/`order`/`rank`/`divergence`, `a`, `beta`, `x`,
  `c`, `r`, `Dinv`, `B`/`Binv`/`L`/`Lmap`) to its code location and mathematical role, and
  stating explicitly which Jacobian-inverse convention `J` uses relative to Kirby (2017)'s
  $F$ (the Stage 1 finding).
* `_materialize_jacobian(J)`: a new free function replacing three copies of the same
  `numpy.array([[J[i, k] ...` snippet (`PiolaFacetFrame.__init__`, `_divergence_rows`,
  `_piola_point_rows`), so the numpy-materialized Jacobian has one name and one
  construction site instead of three silently-identical ones.
* `FacetFrame.normal`/`PiolaFacetFrame.normal`'s differing scaling convention
  (`compute_normal` vs `compute_scaled_normal`) is now stated in the module glossary,
  not just each class's own docstring, so it's visible without reading both classes.

Verified: `pytest test/finat/test_zany_mapping.py test/finat/test_zany_automation.py`
(124 passed), the full `test/finat/` suite (338 passed, 8 skipped), `flake8 finat/zany.py`,
`pydocstyle finat/zany.py` — all clean, no matrix entries changed.

This closes the first math-code loop (Stages 0-3, scalar case). Stages 4+ (Piola
composition hypothesis and beyond) remain only roughly sketched in the plan file and will
be planned in detail next.

### Paper migration (2026-07-16)

The plan now lives at `~/git/fiat_zany_auto/PLAN.md`, and the scalar theory
(Stages 0-3 above) has been drafted as Section 2 of `~/git/fiat_zany_auto/paper.tex`
("Automation of finite element transformations", Kirby, Marsden & Brubeck; acmart,
compiles cleanly with `make`). Key notation choices made in the draft, to keep
consistent going forward: $F: K \to \hat K$ physical-to-reference (Kirby 2017), but $J$
defined by $F^{-1}(\hat x) = J\hat x + b$ (the *code's* Jacobian, stated explicitly in
eq. 2.3); the generic node $\ell_{X,w,d}$; the push-forward closed form
$F_*(\ell_{X,w,d}) = \hat\ell_{\hat X, w, (J^{-1})^{\otimes m}d}$; the "design equation"
$J^{\otimes m}\hat d = \sum_j c_j d_j$ as the unifying statement of every row solve; and
a theory-to-code correspondence table (Table 1). The Piola/implementation/examples
sections are `\todo` stubs pending Stages 4+.

### Review response, RCK round 1 (2026-07-17)

Robert's remarks (`rck_remarks_for_claude.md`) addressed in `~/git/fiat_zany_auto/paper.tex`
(now 12 pp., compiles clean). Disposition:

- **det J > 0 dropped everywhere.** §2.1 now states both orientations occur under UFC;
  the normal formula (2.14) is presented as orientation-covariant (κ = ±1, ν flips with
  C exactly as the reference formula would on a reflected cell); (2.23) notes Gram
  determinants are positive and det J enters linearly. *Verified in code*: ran
  `check_zany_mapping` on a clockwise triangle (det J = −2.06) for Morley, Hermite,
  Bell, Argyris(5,6,7, avg=True), WuXuH3NC, WuXuRobustH3NC, and on a reflected tet
  (det J = −1.28) for Morley/Hermite — all pass. **Note: the test-suite fixtures only
  use positively oriented cells; suggest adding a flipped-cell parametrization.**
- **Notation**: physical/reference normals renamed N, n̂ → ν, ν̂ (avoids clash with
  nodes n̂_j and node-set 𝒩). Facet frame and scaled tangents (t̂_k = v̂_k − v̂_0)
  now defined explicitly (new eqs. framedef/scaledtangents).
- **(2.8) scope**: one direction + one order per functional stated as deliberate,
  no premature generalization; symmetry-WLOG + rank-deficiency remark (d = ν⊗ν rank 1;
  construction never uses rank; covers Wu–Xu-type nodes).
- **"dividing by J"** → multiplying each slot by J^{-1} (gradients by J^{-T});
  invariance/design-equation paragraphs rewritten ("conversely" contrast made explicit;
  numeric-vs-symbolic split cued early); no-type-dispatch point added ("a normal
  component is detected through a ≠ 0, not declared through a class hierarchy").
- **Binning**: §2.4 now states grouping = entity (element data) + order (numeric);
  shared points/weight-ratios/direction-basis are checked numerically, nothing else needed.
- **FTOC proof**: new Prop. 2.1 (exactness identities are recovered — unisolvence
  uniqueness, 1-line proof), with FTOC instance proved (average → all C¹; Morley
  midpoint → on P̂ via linearity of ∂_t f along the edge) and IBP/Stokes-type (Wu–Xu
  face moments) as further instances.
- **E, V^c, D separated then recursion (his 2-part request)**: §2.6 now defines the
  three factors exactly (V^c block per eq:Vcblock, E Boolean, D rows = w·V — self-
  referential but triangular by entity structure), §2.7 "Row recursion" evaluates the
  product without materializing (graph/topological-order remark included), with
  **Algorithm 1** (algpseudocode) mirroring `basis_transformation`/`facet_dof_rows`.
- **Worked examples**: explicit 6×6 Morley V with closed forms c_e = det J |t̂|/|Jt̂|,
  r_e = (Jν̂·Jt̂)/|Jt̂|² — *verified to 6.7e-16 against the code on both orientations*;
  hexic Argyris completion rows computed: ℓ̂⁽⁰⁾ = δ_{v+} − δ_{v−} (FTOC) and
  ℓ̂⁽¹⁾ = 3δ_{v+} + 3δ_{v−} − μ̂_ê (IBP, couples to same-edge trace moment; exact
  integers to machine precision).
- **Novel-element example**: sec:examples todo now points at WuXuH3NC/WuXuRobustH3NC,
  which the automation already handles (tested above).
- **Honesty fix**: scalar completions only ever couple to *identity* rows; the fully
  recursive regime (coupling into non-identity rows) is now correctly attributed to the
  Piola macroelements (§3, deferred).
- New bib entries: `ufc` (Alnæs–Logg–Mardal FEniCS-book chapter), `wu-xu` (Math. Comp. 2019).

Not done (open for co-authors): whether more calculations deserve proposition status
(only Prop 2.1 added so far); explicit Stokes-identity theorem for tet face moments
(covered as an instance of Prop 2.1, no standalone theorem); India's email still TODO.

### Stage 4 (2026-07-17): Piola theory derived as a composition with the scalar machinery

Goal (user hypothesis from PLAN.md): derive the Piola-mapped case *from* the scalar
case by composing basis transformations, as the route to unifying
`ScalarPhysicallyMappedElement` and `PiolaPhysicallyMappedElement`. Result: the
hypothesis is correct, the composition is exact, and it was verified to machine
precision against every Piola element in the FInAT test zoo (automated *and*
hand-coded), in 2D and 3D, on both orientations. Details below; conventions as in
the paper (`~/git/fiat_zany_auto/paper.tex` §2): $F : K \to \hat K$,
$F^{-1}(\hat x) = J\hat x + b$.

**1. Composition theorem.** On an affine cell every Piola pullback factors through
the componentwise scalar pullback $F^*_0(\hat f) = \hat f \circ F$:
$$F^*_{\mathrm{piola}} = \theta_A \circ F^*_0, \qquad (\theta_A f)(x) = A\,f(x),$$
with $A$ a *constant* matrix acting on the value components: $A = J/\det J$
(contravariant), $A = J^{-T}$ (covariant), slot-wise for double variants.
Dually, on functionals,
$$F_*^{\mathrm{piola}} = F_*^0 \circ \theta_A^*, \qquad \theta_A^*(n) = n \circ \theta_A,$$
and both factors act *slot-locally* on the generic node: extend the scalar node to
$\ell_{X,W}(f) = \sum_q \langle W_q, \nabla^m f(x_q)\rangle$ where $W_q$ now carries
$r$ value slots and $m$ derivative slots. Then the unified push-forward is
$$F_*(\ell_{X,W}) = \hat\ell_{\hat X, \hat W}, \qquad
\hat W = (\text{each contravariant value slot}) \cdot \Theta^T
       \;\otimes\; (\text{each covariant value slot}) \cdot J^{-1}
       \;\otimes\; (\text{each derivative slot}) \cdot J^{-1},$$
with $\Theta^T = J^T/\det J = K^{-1}$, $K = \det J \, J^{-T}$ the cofactor matrix.
Points and scalar weights are preserved (paper eq. 2.9 is the $r=0$ case). Note the
collapse: **covariant value slots transform exactly like derivative slots** — a
covariant element is "scalar theory with one more $J^{-1}$ slot".

**2. Mirror duality of invariance.** A physical slot direction $\delta$ is invariant
iff $\delta = \rho^{-1}\hat\delta$ for the slot map $\rho$:
- derivative / covariant slots: $\delta = J\hat\delta$ — mapped tangents (scalar case,
  paper §2.2);
- contravariant slots: $\delta = K\hat\delta$ — cofactor images.

The geometric content is one lemma used by both cases: the generalized cross
product intertwines $J$ and $K$, $\bigwedge_k J\hat t_k = K \bigwedge_k \hat t_k$
(i.e. $C = K\hat C$). In the scalar case this gave the physical unit normal
$\nu = \kappa C/|C|$; in the Piola case it gives the **cofactor lemma**: FIAT's
`compute_scaled_normal` is exactly $K$-covariant,
$$K \hat\nu^s = \nu^s_{\mathrm{phys}} \quad \text{(verified: error} \le 3\cdot10^{-16},
\text{all facets, 2D+3D, both orientations)},$$
so physical normal-flux moments are invariant *with no symbolic factor at all*
(the mirror of the scalar case's invariant mapped-tangential derivatives — but
exact, no $\kappa$ needed). Consequence: FIAT's physical scaled normal is NOT
always outward on negatively oriented cells; its sign rides the UFC vertex
ordering, which is what makes the invariance orientation-robust.

**3. Facet frame, mirrored, with closed-form net.** Reference frame
$[\hat\nu^s \,|\, \hat t_k]$ (scaled normal + scaled tangents). FIAT's physical
facet-dof conventions (read off `FIAT/mardal_tai_winther.py` and verified across
the zoo):
- normal profiles: against $\nu^s = K\hat\nu^s$ → invariant;
- tangential profiles, 2D: against plain mapped tangents $J\hat t$;
- tangential profiles, 3D: against $\nu^s \times (J\hat b)$, cross products of the
  scaled normal with mapped in-plane (RT-profile) vectors — *same formula on
  reference and physical cells*, the same principle as the scalar normal.

Pushing the 3D convention forward with the slot calculus, using the identity
$\Theta^T(Ka \times Jb) = (J^{-1}Ka)\times b$ and BAC-CAB, the net map on frame
coordinates is
$$N = \begin{pmatrix} 1 & -\det J\, \frac{\hat t_l \cdot M^{-1}\hat\nu^s}{|\hat\nu^s|^2} \\ 0 & s\,I_{d-1} \end{pmatrix},
\qquad s = \det J\, \frac{\hat\nu^s \cdot M^{-1} \hat\nu^s}{|\hat\nu^s|^2},
\qquad M = J^T J .$$
The tangential block is a **scalar multiple of the identity** — the
reciprocal-basis correction $S^{-1}$ and the mixing matrix $Y$ of
`PiolaFacetFrame` are derived consequences of this, not independent structure;
no in-plane matrix correction is needed. The top-right entries are the
completion residuals (the mirror of the scalar $r_k$): a pulled-back tangential
profile leaves behind a *normal*-direction functional, eliminated numerically
through the generalized Vandermonde row (`evaluate` on the nodal basis), exactly
as the scalar case eliminates *tangential* residuals. Spot-verified numbers
(fixture cells): MTW 3D face 0: $s = 1.730944$, completion row
$(0.006240, -0.020316)$, matching an unconstrained least-squares fit of FIAT's
actual physical duals to 9e-16; GN2 2D edge 0 tangential dof: pushed frame
coords $(0.398465, 1.154215)$ = closed form.

**4. Divergence nodes.** $\mathrm{div}$ = contraction of one contravariant value
slot with one derivative slot; the slot maps contract to
$(K^{-1})(J^{-1})^T \delta = \delta/\det J$, so
$F_*(\ell_{\mathrm{div}}) = (\det J)^{-1} \hat\ell_{\mathrm{div}}$ — invariant up
to the scalar $\det J$, independent of entity ("the divergence miracle", =
`_divergence_rows`). Tensor case (moments of $\mathrm{div}\,\sigma$ against
vector weights, the Arnold-Winther constraint nodes): one value slot survives
and maps by $\Theta^T$, same $1/\det J$; covered by the calculus, currently
blocked only by `from_fiat` not parsing `IntegralMomentOfTensorDivergence`.

**5. Point data vs moments.** Single-point order-0 nodes (vertex values, edge
midpoint values, interior point values — C0-type data) keep Cartesian components;
their push-forward is the plain $\Theta^T$ slot map and the group block-solve
gives $K$-blocks per slot (the exact mirror of the scalar vertex jets, whose
blocks are $J^T$ per slot). Interior *moments* are invariant by convention
(physical tests are Piola-mapped). **Bug-relevant discovery**: the current
`PiolaPhysicallyMappedElement.invariant_dofs` rule (`order == 0 and dim == sd`)
misclassifies interior Cartesian *point* data as invariant — it happens to be
correct for every currently automated element, but is wrong for Guzman-Neilan
second kind (interior barycenter values → need $K$ blocks, as its hand-coded
transformation does) and Hu-Zhang point variant (interior point values of
$\sigma$). Discriminator that works for the whole zoo: single-point order-0
nodes are point data; multi-point are moments. (HZ-point facet dofs additionally
show rank-2 single-point *frame* dofs, so the facet-level Cartesian test
`len(points)==1 and rank==1` is still needed there.)

**6. Unified formulation = duality.** Everything above collapses into one
statement: with slot-mapped weights, the generalized Vandermonde matrix
$$B_{ij} = F_*(n_i)(\hat\psi_j)$$
is computable for any pullback that factors through $\theta_A$, and $V = B^{-1}$,
inverted blockwise along the entity hierarchy — the row recursion of paper
§2.7 is precisely this block-triangular inversion. A ~60-line prototype
(numeric $J$; slot maps + frame net + divergence + point-data rules, nothing
else) reproduces `basis_transformation` to machine precision for:
MTW(1,2), JohnsonMercier, ArnoldWintherNC, AlfeldSorokina, BernardiRaugel(+Bubble),
GuzmanNeilan first kind (1,2), **GuzmanNeilan second kind**, GN Bubble, GN H1div,
ReducedArnoldQin, ChristiansenHu, **HuZhang 3/4 in both point and integral
variants** — 2D and 3D where defined, positive and negative orientation
(bold = elements the current automated class cannot handle; their hand-coded
transformations agree with the composition theory). All `check_zany_mapping`
ground-truth tests also pass on reflected cells (fixtures only test positive
orientation; worth adding).

**7. What this means for Stage 5 (code unification).** The
`{Scalar|Piola}PhysicallyMappedElement` split reduces to a small table keyed by
the FIAT mapping string:

| ingredient                  | affine (scalar)             | contravariant piola          | covariant piola |
|-----------------------------|-----------------------------|------------------------------|-----------------|
| value slot map              | $I$                         | $\Theta^T = K^{-1}$          | $J^{-1}$        |
| derivative slot map         | $J^{-1}$                    | $J^{-1}$                     | $J^{-1}$        |
| frame-stable directions     | tangents ($J\hat t$)        | scaled normal ($K\hat\nu^s$) | tangents        |
| completed directions        | unit normal $\nu$           | in-plane (2D: $J\hat t$; 3D: $\nu^s{\times}J\hat b$) | normal |
| point-data block            | $J^T$ per slot              | $K$ per slot                 | $J^T$ per slot  |
| special contractions        | —                           | div → $\det J$               | curl analog     |

plus one shared engine: recover node data numerically (`from_fiat` extended with
value slots — already done), decide invariance from frame profiles, expand the
non-stable part symbolically, eliminate residuals through `evaluate`, recurse
over entities. `FacetFrame`/`PiolaFacetFrame` merge into one class parametrized
by the stable subspace; the $S^{-1}$/`Y` machinery is replaced by the closed-form
$N$ above. Newly automatable for free: GN2, HuZhang (both variants), BR, CH, RAQ,
and AW/AWnc once `from_fiat` learns tensor-divergence moments. Covariant (Nedelec
-type zany) elements: theory ready, value slot $\equiv$ derivative slot.

Verification artifacts: the prototype now lives in the repo as
`test/finat/test_claude_zany_piola.py`, upgraded from the in-session scripts:
it computes $V = B^{-1}$ by *block back-substitution* (no dense inversion
anywhere) and asserts both the support law of $B$ (see the Stage 5 design
below) and agreement with `basis_transformation`, for the whole zoo, 2D and
3D, both orientations (56 parametrized cases).

### Stage 5 design (2026-07-17): sparsity, cost, and the unified elimination

Robert's (correct) objection to the first prototype: it formed $B$ densely and
called `inv`. That was a verification shortcut, not the algorithm. This section
records the sparsity theory that makes the duality formulation *no less sparse
and no more expensive* than the current hand-structured code, and the
implementation design in enough detail to be reviewed before touching
`finat/zany.py`.

**1. Support law (sparsity theorem for $B$).** Order the dofs by the entity
partial order: $(d, e) \preceq (d', e')$ iff $e \subseteq
\overline{e'}$ (closure). Claim: $B_{ij} = F_*(n_i)(\hat\psi_j) \ne 0$ only if
$\mathrm{entity}(j) \preceq \mathrm{entity}(i)$, so $B$ is block lower
triangular with per-entity diagonal blocks. Sharper, by dof kind:

* *Single-point rows* (vertex/edge/interior Cartesian point data): exactly
  block-diagonal within the point group. Proof: $F_*(n)$ at a point $\hat x$ is
  the $\Theta^T$-combination of the component-evaluation nodes at $\hat x$ — a
  pointwise identity of functionals on $C^0$, needing no polynomial exactness.
* *Divergence rows*: $B_i = e_i/\det J$ exactly. Proof: the divergence miracle
  $(K^{-1}\!\otimes\!J^{-1})\delta = \delta/\det J$ turns $F_*(\ell_{div})$
  into $\hat\ell_{div}/\det J$, and $\hat\ell_{div}$ *is* a reference node, so
  its generalized Vandermonde row is $e_i$.
* *Interior invariant moments*: $B_i = e_i$ by definition of invariance.
* *Facet rows*: support $\subseteq$ facet block $\cup$ dofs on the strict
  closure $\partial f$. Proof sketch: decompose the slot-mapped weight in the
  reference frame; the frame components give the facet-block entries (the net
  $N$); the residual $R$ is a moment supported on $f$, hence a functional of
  the trace of its argument on $f$. For $j \notin \overline{f}$, $\hat\psi_j$
  has vanishing dofs on $\overline{f}$, and *trace unisolvence* — the dofs on
  $\overline{f}$ determine the traces on $f$ that facet functionals sense,
  i.e. exactly the property that makes the element (H1- or H(div)-)
  assemblable across facets — forces $R(\hat\psi_j) = 0$. This is the Piola
  analogue of the scalar exactness identities (paper Prop 2.1): in Morley,
  "the trace on $e$ is determined by closure dofs" is implemented by FTOC
  along the edge. **Sparsity of $B$ = conformity of the dof layout.**

Verified numerically (scratch script, 2026-07-17): max $|B_{ij}|$ outside the
predicted support is $0$ to machine precision for all 28 element instances of
the zoo, on a negatively oriented cell; the test file asserts the same law on
both orientations on every run.

**2. Elimination = block back-substitution = the paper's row recursion.**
$BV = I$ with $B$ block triangular gives, per entity $e$ with closure dof set
$c$ (already processed, since dims are visited in increasing order):
$$V_e = B_{ee}^{-1}\,(I_e - B_{ec} V_c).$$
This is the scalar row recursion $V_i = c\,e_i + \sum_{j \in P} \gamma_{ij}
V_j$ in matrix clothing. Fill-in: $\mathrm{supp}(V_e) \subseteq e \cup c$,
because the closure relation is *transitive* — the predicted fill equals the
final fill, nothing grows during the elimination. Depth of the recursion
$\le sd$ (vertex → edge → face → interior). At no point is a dense matrix
formed or inverted; the prototype's `np.linalg.inv` is gone
(`composition_transformation` in `test/finat/test_claude_zany_piola.py` now
implements exactly this loop and asserts the support law en route).

**3. Diagonal blocks invert in closed form (the symbolic cost).** The only
$J$-dependent inverses ever needed are:

* point group: inverse slot map $= K$ per slot (already what
  `_piola_point_rows` emits; $K = \mathrm{adj}(J)$ is *polynomial* in $J$);
* divergence: $\det J$ (polynomial);
* facet frame: $N$ is block triangular
  $\begin{pmatrix}1 & r_l\\ 0 & sI\end{pmatrix}$ with **scalar** diagonal, so
  $N^{-1} = \begin{pmatrix}1 & -r_l/s\\ 0 & s^{-1}I\end{pmatrix}$, and the
  coefficients are Gram contractions of the $K$-mapped frame — no symbolic
  $M^{-1} = (J^TJ)^{-1}$ solve, since $J^{-T}\hat\nu = K\hat\nu/\det J$:
  $$s = \frac{|K\hat\nu^s|^2}{\det J\,|\hat\nu^s|^2}
      = \frac{|\nu^s|^2}{\det J\,|\hat\nu^s|^2}, \qquad
    r_l = -\frac{(K\hat t_l)\cdot(K\hat\nu^s)}{\det J\,|\hat\nu^s|^2}$$
  (verified to 1e-15, both orientations, 2D and 3D). Note the mirror of the
  scalar edge coefficient $c_e = \det J\,|\hat t_e|/|J\hat t_e|$: numerator
  and denominator swap roles under $J \leftrightarrow K$, tangent
  $\leftrightarrow$ normal — the covariant/contravariant duality again.

Everything else in $B_{ee}$ and $B_{ec}$ is *numeric and $J$-independent*:
restricted generalized Vandermonde data (reference tabulations contracted
with reference weights), computable and cacheable at element construction.
Each symbolic entry of $V_e$ is a sum of $\le$ (number of frame monomials)
terms (numeric constant) $\times$ (monomial in $\{1, s, r_l, \det J, K_{ab}\}$)
— for rank-$r$ elements the monomials come from $N^{\otimes r}$, still a fixed
finite set. This is the paper's $V = E\,V^c D$ factorization surfacing again:
numeric completion $E$, symbolic (block-)diagonal $V^c$, diagonal scaling $D$.
Per-dof symbolic cost: $O(sd^2)$ rational scalars; total GEM size $O(n\,sd^2)$
— no symbolic LU, and *cheaper* than the current `PiolaFacetFrame` path, which
builds a symbolic physical Gram $G$ and its determinant for `Sinv`.

**4. Implementation plan (what actually changes, reviewable pieces).**

1. `finat/functional.py`: give `PhysicallyMappedFunctional` typed value slots
   (mapping string per slot: affine / contravariant / covariant) and one
   `pushforward(J, K)` that applies $J^{-1}$ per derivative slot (the existing
   `pullback`) and $\Theta^T$ resp. $J^{-1}$ per value slot. Extend `from_fiat`
   to tensor-divergence moments (trace over one derivative+value slot pair,
   remaining value slot kept) — unlocks AW/AWnc.
2. `finat/zany.py`: one `basis_transformation` loop over entity blocks in
   increasing dimension (the skeleton already exists); per-block behavior
   chosen from *data* (divergence flag, single-point, entity dim, mapping
   string), not from the Scalar/Piola subclass. `FacetFrame` and
   `PiolaFacetFrame` merge into one frame class parametrized by the stable
   subspace; it precomputes the numeric restricted Vandermonde/profile data at
   construction and emits the $\{c\ \text{or}\ s, r_l, \det J\}$ closed forms
   at transformation time. Delete `Y`/`Sinv`; `_piola_point_rows` and
   `_divergence_rows` become instances of the generic slot-map inverse.
3. Fix the interior point-data rule (`invariant_dofs`): single-point order-0
   nodes are point data (K blocks), multi-point are invariant moments —
   unlocks GN2 and HuZhang-point.
4. Runtime guard = the support law: assert the numeric residual expansion
   vanishes outside the closure block (reference-time, once per element), so
   a non-conforming dof layout fails loudly instead of producing a dense or
   wrong $V$ (the generalization of Algorithm 1's assert line).
5. Tests: keep `test_claude_zany_piola.py` as the independent numeric ground
   truth (it shares no code path with `basis_transformation`); add
   reflected-cell fixtures to `test_zany_mapping.py`.

Risk framing unchanged from PLAN.md: abandon any piece that adds more
complexity than it removes; the theory above is the review artifact to be
signed off *before* production code moves.

---

## Stage 5 SHIPPED — duality engine as the default `basis_transformation` (2026-07-19)

The mixin-unification plan above (Stage 5c/"Implementation plan" items 1–3)
was **superseded by a course change**: instead of merging the frame classes,
the numeric prototype of `test/finat/test_claude_zany_piola.py` was turned
directly into product code.  `PhysicallyMappedElement.basis_transformation`
(`finat/physically_mapped.py`) now assembles $B_{ij} = n_i(\hat\psi_j \circ
F^{-1})$ symbolically in GEM and inverts it sparsely; `finat/zany.py` shrank
from ~900 lines (FacetFrame, PiolaFacetFrame, four hooks, two mixins) to two
thin configuration mixins (`_check_mapping` guards + the Piola `dof_scale`
override).  Paper §4 ("Assembly by duality and sparsity-preserving
inversion") documents the theory.

**Engine design (all in `finat/physically_mapped.py`):**

1. *Per-slot maps, adjoint on the tabulation.*  One effective matrix per
   slot: $G = J^{-1}\Phi$ (derivative slots), $G = (J/\det J)^T\Phi$ (value
   slots), applied to the reference direction/weights via the generalized
   `PhysicallyMappedFunctional.pullback(A)` (now accepts any numeric or GEM
   matrix); rows of $B$ are then numeric Vandermonde pairings.  Numerator/
   denominator form keeps numerators polynomial: $s_i = (\det J\,|K\hat n|)^m$
   for scalar facet rows, $\det J^{2r}$ for Piola facet rows (using the exact
   identity $J^T K\hat\nu^s = \det J\, \hat\nu^s$), $\det J$ for divergence.

2. *Sparsity and constants inferred from numerics.*  $B$ is assembled at 3
   deterministic sample Jacobians (`_sample_jacobians`, cosine-based, one
   orientation-reversing).  Entries below `pattern_tol = 1e-10` (row-relative)
   at every sample are structural `gem.Zero`; entries equal at every sample
   are constants (snapped to integers where they round), so invariant rows
   never touch symbolic algebra.  No closure bookkeeping, no invariance case
   analysis, no support-law *assumption* — the pattern is measured.

3. *SCC block triangularization + sparse back-substitution.*  Tarjan on the
   coupling digraph emits diagonal blocks in dependency order;
   $V_S = (B'_{SS})^{-1}(\mathrm{diag}(s_i) I_S - B'_{Sc}V_c)$ with one
   adjugate/determinant pair per block, cached across identical blocks
   (vertex jets of different vertices hit the cache).  GEM constant folding
   through `Zero` keeps everything sparse; gotcha: numpy object arrays must
   be the *left* operand of elementwise products, else GEM broadcasts a
   vector node (`V[j] * coef`, not `coef * V[j]`).

**Verification:** full finat suite 430 passed / 8 skipped (includes
`test_zany_mapping` direct physical fits, both orientations, and the
independent prototype cross-check in `test_claude_zany_piola.py`, which
still shares no code path with `basis_transformation`); flake8 + pydocstyle
clean; opaque `gem.Variable` Jacobian path verified bit-identical to the
literal path with max 3–9 nonzeros per row of $M$.

**Numeric groundwork:** pattern-only elimination validated over the entire
scalar + Piola zoos before the symbolic build (scratch `validate_pattern.py`):
max diagonal block 5×5 (BrambleZlamalC2 order-4 jets), pattern leak ≤ 1e-14
except BZC2 (~1e-10, its known FIAT-side conditioning).  Hand-coded
transformations (HCT, PS, C2 elements, WuXu, Walkington) still override the
engine and are untouched.
