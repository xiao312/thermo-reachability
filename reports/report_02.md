# Second report — audit revision and the dimension question

**Status:** Revision 2 complete. All 15 audit defects (A01-A15) repaired and
regression-tested; claim C7 refuted and replaced; the Phase 3 coverage claim C20
downgraded to inconclusive; a new sensitivity experiment (Phase 3d) reframes the
open question. Phases 4-6 remain deferred.

**Baseline:** commit `1901996` (audited). **Revision branch:** `audit-revision`.
Report 01 and the original results are preserved with superseded-status notes;
nothing was silently rewritten. Server connection details that had leaked into
the public log were removed and the offending commit amended and force-pushed.

## 1. What the audit changed

An independent audit of `1901996` found the work to be a "useful computational
prototype with a substantially correct reactor model" but "not yet a validated
thermochemical-reachability study." Every concrete defect was verified against
the actual source before repair, and both mathematical refutations were
re-derived with this repository's own propagator. Summary:

| Audit finding | Effect | Fix and evidence |
|---|---|---|
| A01 server details in public log | publication issue | scrubbed; commit amended + force-pushed (`321e957`); runs now emit `host: redacted-host` |
| A02 phase3c crashed before saving | no results written | `regime` schema implemented; 8 cases × 5 levels now run end to end |
| A03 exposure Γ(t) double-counted | switched balances wrong | assignment form with immutable segment starts; extracted to Cantera-free `gamma_integral_of`; unit-tested on unequal/repeated/switched cases |
| A04 ReactorNet switch drifted with output grid | switching never validated | immutable cumulative switch times + `reinitialize()`; verified on 11- vs 4001-point grids; switched agreement with the custom RHS < 1e-3 K |
| A05 diagnostics not on the RHS path | C17 unsupported | raw sum-of-Y deviation, clipped negative, min raw component, eval count recorded per run |
| A06 invariants from the wrong matrix | reported 0 invariants | `ker(B^T)`, `B = diag(W)ν`: rank(ν)=6, **invariants = 4**, spanned by the rows of E |
| A07 inadmissible `const_zero_seg` | class silently widened | `AdmissibleControls` validates every history; 84/study, 0 violations; sampling covers the full [10, 1e5] |
| A08 trace-sensitive metric was engineering | dispatch error invisible | both metrics evaluated; scales/floors saved; deterministic dispatch test pins them apart |
| A09 distance inequality direction | coverage claim unsound | reference resolution and candidate coverage reported separately; no "Kelvin equivalent" |
| A10 steady acceptance not residual-based | unsteady points in library A | residual threshold with physical scales; unresolved points excluded and counted |
| A11 grid maximum sold as an upper bound | wrong side of the domain | feasible-T bracket via `h_min(T) = min_Y h_k(T)·Y` over the elemental polytope |
| A12 salted seeds / empty code identity | not reproducible | stable hashlib child seeds; git-or-source-tree hash in every manifest; pilot/main manifests separate |
| A13 analysis clouds not persisted | not auditable | extrema + A/B/C/HO clouds saved (NPZ + JSON) |
| A14 reference via linear interpolation | error contamination | dense output evaluated at the exact comparison times; domain-exit now raises |
| A15 tests skip toy suite without Cantera | failure paths untested | split `test_toy.py` (19) / `test_reactor.py` (14), all passing |

## 2. The main scientific correction: C7 is refuted

**The visited set does depend on the control bound.** The original claim reasoned
that because γ = 0 is admissible for every G, the common batch trajectory makes
the visited set G-insensitive. A common maximum does not imply identical sets.

*Witness (exact, verified with this repo's propagator).* γ = 0 for 4 time units,
then γ = 10 for δ = (1/11)·log((10/11 − e⁻⁴)/(10/11 − 1/2)) = 0.0707413427911034
reaches **(x, y) = (0.5, 0.04939624612802243)** at t = 4.0707 < 5.

*Exclusion.* With W = y − x(1−x) and x_G = G/(1+G),
dW/dt|_{W=0} = (1−x)[(1+γ)x − γ] ≥ 0 for all 0 ≤ γ ≤ G whenever x ≥ x_G, and x
cannot cross x_G upward. Hence every reachable state with x ≥ x_G has y ≥ x(1−x).
For G = 0.1, x_G = 0.0909, so at x = 0.5 any reachable state has y ≥ 0.25; the
witness (y = 0.0494) is excluded **at every time**. For G = 10, x_G = 0.909 > 0.5
and the witness is reachable. So R^G=0.1 ≠ R^G=10.

*Why the diagnostic lied.* `envelope_area_fraction_visited` is a **convex-hull**
area. The single zero-control batch curve (y = −x log x, zero two-dimensional
area) alone produces hull fractions **0.128906 / 0.932575 / 0.999999918** at
horizons 1 / 5 / 20 — reproducing the reported 0.129 / 0.933 / 1.000 while
covering no interior at all.

*Corrected diagnostic.* A one-sided fill distance over an interior grid
(`search.interior_coverage`). At horizon 5:

| G | covered fraction | fill distance | deprecated hull fraction |
|---|---|---|---|
| 0.1 | 0.085 | 0.340 | 0.933 |
| 10 | 0.665 | 0.073 | 0.926 |

The hull fraction is G-insensitive (0.93 both); the corrected diagnostic is
strongly G-sensitive. Evidence: `results/phase1b`, `tests/test_toy.py`.

*C6 completed.* The unrestricted interior is attained by the audit's
reaction-then-replace construction: age pure A for s ≥ −log x giving (a, sa) with
a = e^−s, then mix with fresh A at fresh fraction λ = (x−a)/(1−a); the resulting
intermediate concentration y(s) = (1−x)s/(e^s − 1) decreases continuously from
−x log x to 0. This holds under the unrestricted closure (γ ≥ 0, unlimited time,
arbitrarily rapid replacement) and does **not** transfer to fixed G or finite
horizon — which is precisely the unresolved useful question.

## 3. Corrected detailed-chemistry results

**Switched validation now exists.** The previously broken multi-segment exposure
and ReactorNet scheduling are fixed and tested: exact balances hold on a
three-segment switched history from a nontrivial initial state (b0 ≠ b_in,
h0 ≠ h_in) to < 1e-8 (elements) and < 1e-6 (enthalpy), and the two integration
pathways agree on a switched history to < 1e-3 K independent of the output grid.

**Chemical invariants.** rank(ν) = 6 over all 29 reactions; the invariant space
ker(B^T) with B = diag(W)ν has dimension **4** and is spanned by the rows of the
elemental matrix E. The state manifold at fixed feed inventory and enthalpy is
therefore 6-dimensional, with temperature derived.

**Temperature bounds, corrected.** The relaxed maximum is now *bracketed*
(900 K feed: [3010.4, 3011.2] K; 1200 K feed: [3270.4, 3271.2] K) via
h_min(T) ≤ h_in. The audit's independent continuous estimates (3010.95 K and
3270.72 K) fall **inside** these brackets — a clean cross-validation. The old
grid maxima (3010.35 / 3269.70 K) are retained as historical estimates; they are
attained values, not upper bounds. The mechanism's common thermodynamic range is
[300, 3500] K, not [200, 5000] K.

**Metric dispatch now real.** At the non-igniting 900 K condition, switching
states have engineering distance 0.006 from the constant-control library but
trace-sensitive distance **0.95** — the states are thermally nearly identical
but differ by orders of magnitude in radical composition. This is the audit's
point that temperature proximity can hide composition differences; it is
reported as an observation about the sampled states, not a coverage claim.

**Steady family.** 48/48 traced points pass the residual-based acceptance;
multistability is preserved (cold branch ~900 K vs hot branch ~1550 K at
γ = 1e5). No unstable-branch continuation, so no complete S-curve is claimed.

## 4. Phase 3d — the dimension question

The refuted framing ("does switching leave the constant-control family?") is
replaced by a sharper one: **does the endpoint map of a switched history have
image dimension exceeding the two-parameter constant-control family?**

At a feed-compatible state the manifold M = {(T,Y) : E·Y = b_in, h_k(T)·Y = h_in}
is 6-dimensional. The constant-control family is a 2-parameter image
(γ, t). For m-segment histories with levels θ, the endpoint Jacobian
J = ∂E_m/∂θ is computed by central finite differences, projected onto the tangent
space of M (so conserved directions cannot masquerade as history-generated ones),
and compared with the constant-control tangent span V.

**Result (all 18 cases, fresh and hot starts, m = 3/4/6, equal / varied-dwell /
pulse-hold patterns).** Every projected Jacobian has **exactly one significant
singular direction** (singular value 5e-3 to 6e-2, stable to four significant
figures across step sizes 1e-2, 1e-3, 1e-4); all remaining singular values sit at
machine precision (1e-16 to 1e-20). Moreover the single real direction lies
**inside the constant-control tangent span**: its sine to span(V) is 1e-12 or
smaller in all 18 cases. The local endpoint image is therefore a curve, and that
curve is tangent to the constant-control family. Zero of 18 cases show a stable
history-generated third direction (`results/phase3d`).

This is a *negative* answer at the tested base points: switching does not open a
locally transverse direction there. Caveats, stated explicitly: this is a *local*
rank statement at 18 specific admissible interior points under one mechanism, one
condition class (H2/O2, feed-compatible states, 0.1 s horizon) and a finite set of
base histories; it does not bound the global reachable set; and a floating-point
SVD is not a rank proof, which is why step-size refinement (agreement to 4 digits)
is the actual evidence, not the SVD itself.

An incidental methodological finding, confirmed by the corrected computation: the
span-vs-span principal angles between J and V are **meaningless** when J is
numerically rank-deficient, because QR-orthonormalization then fills the basis
with rounding-noise directions. The first implementation produced erratic values
(0, 0.78, 0.85, 2.6e-8) for cases whose true transversality is ~1e-12. The
corrected measure compares only singular directions above a noise threshold
against the family plane, and yields clean values. Recorded in `research_log.md`
as a trap for anyone repeating the analysis.

## 5. Splitting, interpreted correctly (Phase 3C, revised)

The pre-reaction stage displacement is **first order and expected**:
q_preR − q(t + dt/2) = −(dt/2)·f(q) + O(dt²). It is compatible with second-order
completed Strang steps and is not a convergence failure. The revised experiment
evaluates the reference by dense output at the exact comparison times and
classifies each case by regime rather than pass/fail:

| Case | Regime | err_T at dt=1.3e-3 | err_T at dt=7.8e-5 | final order |
|---|---|---|---|---|
| smooth, T_in=900 K, fresh | reference precision | 2.8e-7 | 7.1e-9 | n/a (noise) |
| γ=100 fresh/hot | asymptotic, order 2 | 81 | 2.5 / 1.1 | 1.94 |
| γ=1000 hot | asymptotic, order 2 | 611 | 10.2 | 1.91 |
| γ=1000 fresh | pre-asymptotic | 611 | 24 | 0.93 |
| γ=1e4 fresh/hot | pre-asymptotic | 1046 | 202 / 127 | 0.90 / 1.47 |
| switching 1e4→1e2, hot | asymptotic, order 2 | 710 | 13.2 | 1.61 |

The smooth case's error sits at the reference-tolerance floor, so its "observed
orders" (0.3-1.3) are noise — correctly labelled reference-precision, not a
failure. The stiffest cases (γ = 1e4) are genuinely pre-asymptotic at
dt ≥ 7.8e-5 s, with the order still climbing. The physical reachable set
R_physical and the chemistry-input domain D_chemistry(dt, scheme, class) remain
distinct objects; this study quantifies the latter's convergence, not the former.

## 6. Experiment summary

| Experiment | Run ID / path | Attempted | Success | Fail | Runtime | Strongest residual | Status |
|---|---|---|---|---|---|---|---|
| Phase 0 smoke | `results/phase0/smoke_results.json` | 3 blocks | 3 | 0 | < 1 s | 1.2e-9 | passed |
| Phase 1 toy (corrected diagnostics) | `results/phase1/phase1_results.json` | 9 cases | 9 | 0 | ~15 s | barrier 1e-16 | executed; hull diagnostic deprecated |
| Phase 1b toy correction | `results/phase1b/phase1b_results.json` | witness + 3 bounds + 2 clouds | all | 0 | 142 s | exact | executed, C7 refuted |
| Phase 2 reactor validate | `results/phase2/phase2_results.json` | 5 balances + switched + 6 xint + 8 net | all | 0 | 4.6 s | invariants dim 4 | executed, passed |
| Phase 3 envelopes (corrected) | `results/phase3/phase3_results.json` | 3 × (24 B + 43 C + 16 held-out) | 249 | 0 | 239 s | steady 48/48 accepted | executed |
| Phase 3C splitting | `results/phase3c/phase3c_results.json` | 8 cases × 5 levels | 8 | 0 | 2069 s | order → 1.94 | executed |
| Phase 3d sensitivity | `results/phase3d/phase3d_results.json` | 18 cases × 3 steps | 18 | 0 | 908 s | n_real=1, sin 1e-12 | executed |
| Unit tests | `tests/test_toy.py` + `tests/test_reactor.py` | 33 | 33 | 0 | ~15 s | — | all passing |

Replay (compute server, env `/data2/conda-envs/tr`):

```bash
$PY scripts/phase0_smoke.py            --results-dir results/phase0
$PY scripts/phase1_reachability.py     --results-dir results/phase1
$PY scripts/phase1b_toy_correction.py  --results-dir results/phase1b
$PY scripts/phase2_reactor_validate.py --results-dir results/phase2
$PY scripts/phase3_envelopes.py        --results-dir results/phase3
$PY scripts/phase3c_splitting.py       --results-dir results/phase3c
$PY scripts/phase3d_sensitivity.py     --results-dir results/phase3d
$PY -m pytest tests -q
```

## 7. Revised conclusions and what remains open

**Supported now:** the toy identities (C1-C5); the unrestricted enclosure with
its completion (C6); the G-dependence of the visited set with an exact witness
and an analytic exclusion (C7, revised); exact balances including switched
histories (C12); the invariant structure (C11, dim 4); the two-pathway
integration agreement including switching (C15); the conserved-manifold
dimension 6; multistability (C19, narrowed); the temperature brackets (C21,
corrected); and the local rank-1 endpoint image tangent to the constant-control
family at the 18 tested base points (C22).

**Retracted or narrowed:** C7 as originally stated (refuted); C20 as a coverage
claim (inconclusive — candidate coverage, not reference resolution, was the
binding limit); C19 as a complete S-curve (no unstable branches); C21 as an upper
bound (it is a bracket); C17 as previously evidenced (instrumentation was
inactive, now fixed).

**Open and worth pursuing:** the *global* reachable-set geometry at fixed G and
finite horizon (C7 shows it is G-dependent; its dimension is unresolved — the
Phase 3d rank-1 result is local only); the commutator mechanism
[f,g] = (1,−1), det(g,[f,g]) = x+y−1 = −Y_C, which predicts history-order effects
once product is present and gives concrete ordered-pulse protocols to test; and
an adaptive, sensitivity-driven outer enclosure (‖E(θ) − E(θ_c)‖ ≤ L_C‖θ−θ_c‖)
over cells of the control-parameter domain — the direct route to estimating the
envelope without enumerating histories.

**Honest scope, restated:** all numerical statements are traceable to saved
outputs; optimization optima are achieved objectives, not certified suprema; the
temperature bracket is a numerical bracket, not an interval-certified bound;
solver agreement is not mechanism validation; and a finite search over sampled
histories establishes nothing about the whole admissible class.
