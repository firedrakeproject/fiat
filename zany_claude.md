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
