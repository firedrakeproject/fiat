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

**Status (2026-07-14).** The symbolic dof lives in `finat/functional.py` as
`finat.PhysicallyMappedFunctional`. `finat/physically_mapped.py` stays fully generic:
`PhysicallyMappedElement` is unchanged from before this project, still an abstract
mixin with no knowledge of the zany theory, used as-is by hand-coded elements (AW,
HCT, PowellSabin, Walkington, ...). The automation lives in `finat/zany.py`, but the
two mixins no longer share one template-method loop: `PiolaPhysicallyMappedElement`
((double) contravariant Piola: MTW, Johnson-Mercier, Guzman-Neilan) keeps the original
entity-ordered `_check_mapping`/`_invariant_dofs`/`_facet_dof_rows`/`_point_dof_rows`
hook design (untouched); `ScalarPhysicallyMappedElement` (affine pullback: Morley,
Hermite, Argyris, Bell) was rewritten from scratch around a much simpler idea — see
below. `ZanyPhysicallyMappedElement` now only holds what both still share: `tol` and
`_rescale_derivative_dofs`. Concrete elements (`finat.Morley`, etc.) are still just a
citation plus a FIAT constructor call. Tests: `test/finat/test_zany_automation.py`;
`check_zany_mapping` lives in the finat conftest and is provided to test modules as a
pytest fixture (pytest runs with `--import-mode=importlib`, so test modules cannot
import from each other or from conftest).

**Framework design, rewritten for the scalar case.** The old design (frame
decomposition into normal/tangential coordinates, `FacetFrame`, `generalized_cross`,
per-entity completion recursion through already-assembled rows of $V$) is gone for
`ScalarPhysicallyMappedElement`. The replacement starts from duality directly: a
physical node $\ell_i$ restricted to the polynomial space equals $\sum_j
\ell_i(\hat\psi_j)\,\hat n_j$, i.e. row $i$ of $B$ with $B_{ij} = \ell_i(\hat\psi_j)$
is the *generalized Vandermonde* evaluation of the physical node against the
*reference* nodal basis $\hat\psi_j = F^*(\hat\psi_j)$ (the FIAT basis functions,
transplanted to physical space by the plain affine cell map, no Piola-style pullback
scaling). $B$ relates physical nodes to *reference basis functions*, not to reference
nodes, so it is $V = B^{-1}$ that is actually needed — computing $B$ directly is not
the free lunch it looks like; the matrix must still be inverted.

* `PhysicallyMappedFunctional.evaluate` (`finat/functional.py`) computes one row of
  $B$ directly, with no geometric frame: `self.direction` is the *physical* direction
  tensor (unchanged Cartesian tensor for vertex/interior jets; cofactor image of the
  reference normal, renormalized to unit length, for facet derivative dofs — see
  `ScalarPhysicallyMappedElement._physical_direction`), and the chain rule
  $\nabla_x^m(\hat\psi_j\circ F) = (J^{-1})^{\otimes m}:\hat\nabla^m\hat\psi_j$ is
  applied to the *tabulation* (not to `direction`, which stays untouched and is simply
  contracted with the raw weights, $W_q = w_q D$, at the end) — this avoids ever having
  to work out which side of a bilinear pairing needs $J^{-1}$ vs. $(J^{-1})^T$: `Tab`
  is built as a genuine (uncompressed, repeated-per-index-ordering) tensor of shape
  `(ndof, npoints) + (sd,)*order` and each of the `order` trailing slots is contracted
  with $J^{-1}$ in turn via `numpy.tensordot`, always targeting the axis right after
  the `(ndof, npoints)` prefix so that an unprocessed slot cycles into that position
  every iteration (get this axis position wrong — e.g. always contracting the *last*
  axis — and the same slot gets contracted twice while another is never touched; this
  is invisible for order 1 and silently wrong for order $\ge 2$, i.e. every vertex
  Hessian dof in Bell/Argyris, so always check against a genuine order-2 case, not
  just Morley/Hermite, before trusting a refactor of this contraction).
* $B$ is not diagonal (a physical derivative node's row is generally nonzero against
  many reference basis functions), but it *is* block lower triangular by increasing
  topological dimension: a physical node's row is nonzero only on its own entity and
  on strictly-lower-dimensional entities (this is the same structural fact the old
  per-entity completion recursion relied on, now discovered as a property of $B$
  rather than asserted by construction). `ScalarPhysicallyMappedElement.
  basis_transformation` therefore still visits entities in increasing dimension, but
  only to invert $B$ into $V$: each entity's own small diagonal block of $B$ is
  inverted with `finat.physically_mapped.inverse` (reused as-is; it already exploits
  further internal block/identity structure, e.g. order-0 trace moments sitting next
  to order-1 normal moments within one Argyris edge's block) and used to eliminate the
  already-known contribution of lower-dimensional entities from the target (`I`
  restricted to this entity's rows). Do not invert the whole $B$ at once with generic
  `adjugate`/`determinant` cofactor expansion — it is combinatorial in the matrix size
  and only tractable because each per-entity block is small.
* Order-0 (value) physical nodes are exactly their reference functional (no geometric
  counterpart to build), so `evaluate` returns *numeric* Python floats for that
  branch — remember to wrap them in `gem.Literal` before writing into the (GEM-typed)
  $B$/$V$ arrays, `ListTensor` will raise an opaque `AttributeError` on a bare float.

Key facts the framework rests on:

* **The physical facet normal is the cofactor image of the reference one, full stop.**
  No "UFC-consistent normal" recovery, `generalized_cross`, or cell-independent-$\kappa$
  argument is needed: $K\hat n$ (any reference vector proportional to the facet normal,
  $K = \operatorname{adj}(J)^T$) is already, exactly, the (non-unit) physical scaled
  normal — the identity `_piola_facet_rows` already relies on for the Piola case.
  Renormalizing by $\|K\hat n\|$ gives the physical *unit* normal FIAT's own facet dofs
  are built against (`ref_el.compute_normal`); the earlier session's `FacetFrame` reached
  the same fact through a much longer detour (mapped-tangent cross products, recovering
  a cell-independent scale constant $\kappa$ from reference data) that is no longer
  needed now that nothing needs to *decompose* a direction into a normal/tangential
  frame in the first place.
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
  on the left when scaling by a GEM scalar). Unaffected by the rewrite above.

Extensions beyond first order and Morley/Hermite (all still automatic under the new
design, no per-element special-casing beyond `_physical_direction`'s
facet-vs-vertex/interior split):

* `ScalarPhysicallyMappedElement.avg = False` (an instance attribute Argyris sets from
  its constructor kwarg) reproduces the legacy FInAT convention where physical facet
  moments are plain integrals rather than averages: `_physical_direction` scales the
  unit-normal direction back up by the *reference* facet volume instead of
  renormalizing by $\|K\hat n\|$, i.e. the moment keeps the shared reference weights
  but stops being measure-intrinsic. Single-point facet dofs (Argyris "point" variant)
  are unaffected (guarded by `len(ell.points) == 1`).
* Bell is still the extended-element pattern: FIAT.Bell is the 21-node quintic element
  with the constraint functionals as extra edge nodes; overriding `space_dimension()`
  to 18 drops the constraint *columns* of $V$ at the end (their rows are still needed
  to invert $B$'s lower-dimensional blocks, i.e. vertex rows still eliminate against
  them), and the FInAT element overrides `entity_dofs`.
* Known convention change carried over unchanged from the previous design: the generic
  $h^{-m}$ conditioning scaling also applies to integral-variant Argyris edge moments,
  which the hand-written code left unscaled (Morley scaled them; the legacy convention
  was inconsistent). Invisible when `cell_size == 1`; flag in PR review.

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
e.g. tangential-to-normal couplings emerge). Key subtlety (in any dimension > 2): FIAT builds tangential value
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

Next steps: remaining PiolaBubbleElement users (GN second kind, H1div: interior
derivative moments need the divergence detJ rule), ArnoldWinther (vertex tensor values
+ higher facet moments), the extended-element path for reduced HCT (macro polynomial
spaces); covariant elements.

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
