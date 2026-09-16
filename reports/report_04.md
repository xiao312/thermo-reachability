# Report 04 — Memory calibration: matched exchange exposure, pulse order,
# and an exact tangent-linear transverse spectrum

Branch `memory-calibration`, from `sensitivity-v2` at `31ae274`.
Studies: `results/phase5`, `results/phase6`, `results/phase7`, `results/phase8`.

---

## 0. Missing handoff material (reported, not blocking)

The review's Task 0 points at `prior_handoff_verified/worker_revision_review/`
(`REVIEW.md`, `review_checks.py`, `linear_rank_three_benchmark()`, the prior
math note). That path does not exist in the repository, in the handoff zips, in
`Downloads`, or on the desktop. The three-state benchmark was therefore
reconstructed from the explicit formula in Task 1.1 and used as ground truth;
it is verified to reproduce the stated singular values
(0.5007801147013896, 0.08533145181532144, 0.006828632164138058) to 7 digits. Any
statement about the missing note's own derivations remains an unaudited
hypothesis.

## 1. Targeted defects fixed

| # | Defect | Fix | Test |
|---|---|---|---|
| 1.1 | `linear3_exact_jacobian` returned `-G_spec`; the endpoint map at all-ones control gave negative values instead of `(1-exp(-lam))/lam` | sign convention corrected; `linear3_ode_reference` added — an independently coded segment-restarted LSODA solver (a single-shot solve steps over the middle segment and returns 0, so per-segment restart was required) | endpoints match closed form and ODE to ~1e-12/1e-13; the three singular values match the stated ones to 7 digits |
| 1.2 | `scaled_tangent_space` formed `null(C W)`, which is the wrong subspace whenever `W` is not a multiple of the identity — always, here, since temperature and mass fraction carry different scales | rewritten as `null(C W^-1)` with row equilibration to unit infinity-norm per row; conditioning reported before and after | review test `C=[1,1]`, `W=diag(1e-2,100)`, `dq=[1,-1]`: the projection leaves `W dq = [0.01,-100]` unchanged instead of corrupting it to `[0.02,-2e-6]`; `‖C W^-1 N‖≈0`; conditioning 3.1e8 → 4.5 |
| 1.3 | reference tangents anchored at the wrong time/state; identity test applied to non-constant base histories where it is not valid | dimensionless time parameter `tau = t/T_ref` with `T_ref = T` recorded per case, so the time column is `T·F(q_base, γ*)` and carries the same units per unit parameter as the control columns; held endpoints anchored at total time `T+L` and flagged `reference_anchored_at_T_plus_L`; the identity test is `identity_applicable` only for equal base histories — a common scaling of a non-constant history is a different curve, not the constant-control derivative | `test_sensitivity.py`, and the phase 5/8 records carry the fields |
| 1.4 | threshold wording: "1e-6" read as 1e-6 mass fraction; machine rank reported as the rank; NaN stencil aggregated | `StateScaling.component_bound` states that the 1e-6 scaled threshold is **1e-4 K and 1e-8 mass fraction per unit log-control step**; `classify_ranks` separates `rank_machine` / `rank_derivative_noise_resolved` / `rank_application_effective` / `stencil_complete`, returns `None` (never 0) for the noise-resolved rank when no discrepancy is supplied, and gates `full_rank_claim_valid` on a complete stencil; NaN columns are excluded from the SVD rather than crashing it | 30→34 tests, including the four-notion separation on singular values (1.28e-1, 1.16e-5, 1.08e-12) → ranks (3, 2, 2) |
| 1.5 | cheap tests absent | the four-notion test, the component-bound test, the leakage-floor test, and two closed-form FSA tests were added; all 34 pass | `tests/test_sensitivity.py` |

## 2. Calibrating the weak second direction (Phase 5)

Six unique equal-control anchors (a prefix repeated under several hold choices is
not several independent rank findings). Per anchor the scaled singular values of
`W J` were computed over three FD steps, two integrators and two tolerance
settings, and the weak singular pair was validated by **replay**: perturbing the
levels along `v2` and differencing the endpoints must reproduce `sigma_2`.

| anchor | sv2 (total) | sv1 transverse | manifold leak | FD discrepancy | replay |
|---|---|---|---|---|---|
| fresh 100, 1e-3 | 1.16e-5 | 4.6e-8 | 2.3e-10 | 2.8e-6 | 1.156/1.155/1.174e-5 |
| fresh 1000, 1e-3 | 9.94e-5 | 1.31e-5 | 7.3e-8 | 4.7e-5 | 9.943/9.951/1.001e-4 |
| hot 100, 1e-3 | 1.16e-5 | 3.3e-8 | 2.3e-10 | 2.8e-6 | 1.155/1.157/1.174e-5 |
| hot 1000, 1e-3 | 9.94e-5 | 1.32e-5 | 7.3e-8 | 4.7e-5 | 9.945/9.953/1.001e-4 |
| fresh 100, 1e-2 | 4.8e-13 | 2.5e-8 | 2.3e-10 | 2.8e-6 | noise (unresolved) |
| fresh 1e4, 1e-3 | 1.2e-7 | 4.3e-6 | 1.4e-8 | 7.9e-6 | 1.197/1.196/1.196e-7 |

Two results.

**The replay validates the weak total direction.** `sigma_2` from the SVD is
reproduced by an independent directional finite difference to three decimals,
across three perturbation sizes and across Radau and LSODA independently. The
direction is real, not a decomposition artifact.

**But the second direction is not transverse to the constant-control family at
this resolution.** Its transverse component (1e-8 to 1e-5) lies between the
conserved-manifold leakage floor (1e-10 to 1e-7) and the finite-difference
refinement discrepancy (1e-6 to 1e-5). A direction below the estimator's own
differentiation noise is not a negative result — it is an unresolved one. The
FD discrepancy itself is a *refinement discrepancy*, not a certified noise bound:
Phase 5 measures 2.76e-6 between the two largest steps where Phase 4 reported
2.8e-8 from a different step pair, so the number depends on which pair is chosen
and is stored and labelled as such.

The transverse spectrum `(I - Q Qᵀ) W J` is therefore reported next to the
manifold-leakage spectrum `(I - N Nᵀ) W J` rather than being used to hide error.

## 3. Exchange exposure is not the mechanism (Phase 6)

The Phase 4 account said the weak direction is governed by the exchange time
`γ·T`. That is confounded: at fixed `γ`, changing `T` changes both the exposure
and the chemical relaxation. Phase 6 separates them with **matched pairs** —
cases sharing `γ·T` with different `γ` and `T`.

| init | γ·T | γ | T | T / t_ign | sv2 |
|---|---|---|---|---|---|
| fresh | 0.1 | 100 | 1e-3 | 23 | 1.16e-5 |
| fresh | 0.1 | 1000 | 1e-4 | 2.3 | 1.55e-3 |
| fresh | 0.1 | 1e4 | 1e-5 | 0.23 | 4.2e-8 |
| fresh | 1 | 100 | 1e-2 | 200 | **7.4e-12** |
| fresh | 1 | 1000 | 1e-3 | 23 | 9.9e-5 |
| fresh | 1 | 1e4 | 1e-4 | 2.3 | **8.8e-3** |
| fresh | 10 | 100 | 1e-1 | 399 | 4.4e-12 |
| fresh | 10 | 1000 | 1e-2 | 200 | 1.9e-12 |
| fresh | 10 | 1e4 | 1e-3 | 22 | 1.2e-7 |

At **fixed γ·T = 1 the second singular value spans 7.4e-12 to 8.8e-3 — a factor
of 1.2e9.** Exchange exposure does not control it. What tracks it is the ratio of
the horizon to the chemical relaxation time: for the fresh 1200 K feed the
declared ignition delay is `t_ign ≈ 4.3e-5 s` (first time the temperature rises
100 K above its initial value), and the direction is largest at
`T ≈ 2 t_ign` and has collapsed by `T ≈ 200 t_ign`. **This refutes the mechanism
stated in C23 and replaces it.**

Holds: every hold of at least one residence time collapses the prefix block to
~1e-12 scaled units with the endpoint already within 1e-12…1e-9 of the steady
reference, for both hold levels — memory is erased by roughly one exchange time
regardless of the level held. Cases where the hold level differs from the prefix
level are flagged `reference_anchored_at_T_plus_L: false`, so no unanchored angle
claim is made from them.

## 4. Pulse order and the Lie bracket (Phase 7)

The CSTR right-hand side is affine in the control, `F(q,γ) = γ v(q) + r(q)`
(verified to 0.0 against two levels), so the bracket of two constant levels has a
closed form independent of the absolute levels:

    [f_a, f_b] = (γ_a − γ_b) [r, v]

Two falsifiable predictions, tested by differencing the two-segment histories
`(a,b)` and `(b,a)` at equal segment durations.

**Both hold at short τ.** The same-level control returns **exactly zero**
(0.000e+00), so the order dependence is entirely due to the level difference.
At `τ ≤ 1e-6 s` the log-log slope of the endpoint difference is **2.03** against
the predicted 2, and the magnitude reproduces `τ² (γ_a−γ_b)[r,v]` to within 4 %
— 6.546e-14 observed versus 6.349e-14 predicted, with the `(γ_a−γ_b)`
proportionality exact across the three level pairs (900 : 400 : 500 gives
6.546 : 2.888 : 3.658e-14).

**Both fail above the ignition delay.** At `τ = 1e-4…1e-2 s` the difference
saturates (0.706 → 1.117, identical at 1e-3 and 1e-2), the log-log slope is 0.9,
and the magnitude is ~1e5 times the bracket prediction. The trajectories have
ignited and left the asymptotic regime. So the 0.7-scaled-unit order dependence
reported in Phase 4 is an **ignition effect, not a bracket effect** — and the
bracket itself is exactly as derived.

## 5. An exact estimator resolves one transverse direction (Phase 8)

Phase 5's limitation was the finite-difference noise floor of the estimator, not
of the system. Because the CSTR is affine in the control, the forward-sensitivity
equations are exact with a closed-form inhomogeneity and no control-step noise:

    dq/dt   = F(q, γ_k)
    dS_j/dt = F_q(q) S_j + γ_j F_γ(q)      (segment j only)

`endpoint_jacobian_fsa_system` integrates this augmented system. It is validated
against a matrix-exponential closed form for a linear affine-in-control system at
m = 1, 2, 3 segments to **1.2e-14** (`test_fsa_matches_closed_form_affine_linear`),
and a control-step FD is shown to be orders worse. `F_q` is still differenced in
*state* because Cantera exposes no Jacobian; that costs one FD pass per RHS
evaluation regardless of segment count and does not limit the control
sensitivity.

With the reference span also exact (`v_τ` closed form, `v_η` a one-segment
tangent-linear derivative), the transverse question becomes decidable:

| anchor | total spectrum | transverse | manifold leak | Radau vs LSODA | rank_app |
|---|---|---|---|---|---|
| **hot 1e4, 1e-5** | 9.5e-1, 1.2e-1, 6.0e-5 | **1.2e-4**, 2.0e-11, 9.2e-13 | 2.8e-12, 4.7e-13, 5.0e-14 | 1.5e-9 | **3** |
| hot 1e4, 1e-4 | 2.3, 3.3e-2, 3.1e-11 | 4.8e-10, 5.6e-11, 2.7e-11 | 1.8e-11, 1.2e-11, 1.8e-12 | 1.5e-7 | 2 |
| fresh 1e4, 1e-4 | 1.7, 8.8e-3, 1.1e-11 | 1.1e-10, 1.1e-11, 2.2e-12 | 8.4e-12, 2.9e-13, 2.4e-13 | 3.6e-8 | 2 |

In the hot, short-horizon case the transverse direction is **4×10⁴ times above
the conservation leakage floor and 8×10⁴ times above the independent-integrator
discrepancy, and 120 times above the declared application threshold**. That is a
genuine history-generated direction outside the two-parameter constant-control
family — `rank_application_effective = 3`. It exists only in this narrow window:
at every longer horizon the transverse component returns to the 1e-10…1e-11
floor, below the cross-integrator check.

### 5a. Where the off-family direction lives (window scan, `results/phase8b`)

| anchor | total spectrum | transverse | manifold leak | Radau vs LSODA | rank_app |
|---|---|---|---|---|---|
| hot 1e4, T=1e-6 | 1.5e-1, 1.1e-2, 1.1e-3 | **1.6e-3**, 7.8e-7, 1.2e-13 | 3.0e-13, 1.6e-13, 2.1e-15 | 3.6e-10 | **3** |
| hot 1e4, T=3e-6 | 3.7e-1, 6.2e-2, 6.8e-4 | **1.1e-3**, 1.3e-10, 1.9e-13 | 9.3e-13, 1.6e-13, 3.5e-15 | 3.5e-10 | **3** |
| hot 1e4, T=1e-5 | 9.5e-1, 1.2e-1, 6.0e-5 | **1.2e-4**, 2.0e-11, 9.2e-13 | 2.8e-12, 4.7e-13, 5.0e-14 | 1.5e-9 | **3** |
| hot 1e4, T=3e-5 | 1.9, 8.7e-2, 8.6e-10 | 1.8e-9, 1.1e-10, 1.6e-12 | 7.4e-12, 1.3e-12, 6.4e-14 | 1.1e-9 | 2 |
| hot 1e5, T=1e-6 | 1.4, 1.1e-1, 1.0e-2 | **1.5e-2**, 1.0e-5, 1.6e-12 | 3.4e-12, 4.2e-13, 9.2e-14 | 2.0e-9 | **3** |

The transverse component decays by six orders as the horizon grows from 1e-6 to
3e-5 s and crosses below the cross-integrator check between 1e-5 and 3e-5 s —
the same chemical-relaxation control found in Phase 6, now at the faster
chemistry of the hot state. At the shortest horizons a third *total* direction is
also above threshold (1.1e-3 at T=1e-6) while its transverse component is at
1.2e-13: it lies *inside* the family, so `rank_application_effective = 3` there is
a family direction, not a third off-family one. Only the second direction is
transverse.

The direction is independently confirmed by replay (`transverse_replay` in the
manifests): perturbing the levels along the second right singular vector and
differencing endpoints reproduces the transverse magnitude, stably across two
orders of perturbation size — 1.078e-4 / 1.249e-4 against an SVD value of 1.234e-4
at T=1e-5 — while at T=3e-5 the replay value grows with the perturbation size
instead of converging (2.9e-7 → 2.8e-5 against an SVD value of 1.8e-9), which is
the correct signature of an unresolved quantity. The two anchors at γ = 1e5 have
no replay because γ = 1e5 sits exactly on the admissible upper bound and admits
no perturbation; their `rank_application_effective = 3` rests on the exact
estimator plus the cross-integrator check (ratios 7×10⁶).

This direction was invisible to the FD estimator, whose noise floor (1e-6…1e-5)
stood 1…4 orders above it. Building the exact estimator is what turned an
unresolved quantity into a measurement.

## 6. Honest summary of what changed

- **Refuted (C23's mechanism):** exchange exposure `γ·T` does not control the
  weak direction; a 1.2e9-fold spread at fixed `γ·T` settles it.
- **Corrected (C27):** the observable pulse-order effect at chemically relevant
  times is ignition, not the Lie bracket. The bracket itself is exactly
  `(γ_a−γ_b)[r,v]`, confirmed to 3 % and slope 2.03.
- **Established (C26):** one resolved transverse, off-family direction, in a
  narrow short-horizon fast-chemistry window, at 1.2e-4 scaled units, validated
  against a manifold-leakage floor and two independent integrators.
- **Unchanged:** the weak total direction is real (replay-validated) and is
  erased by roughly one residence time of hold.

Acceptance does not require a positive third direction; it requires that the
experiment would detect one at its stated resolution. The positive controls and
the closed-form FSA validation establish that, and then found one.

## 7. Scope and provenance

All results are for the h2o2 mechanism (10 species, 29 reactions), ideal-gas
CSTR, the two declared initial states, `γ ∈ [10, 1e5]`, horizons 1e-6…1e-1 s, the
frozen scaling (1 scaled unit = 100 K or 1e-2 mass fraction) and the declared
1e-6 application threshold (= 1e-4 K, 1e-8 mass fraction per unit log-control
step). Singular values are of the *scaled* Jacobian `W J`; the transverse
spectrum is `(I - Q Qᵀ) W J` with `Q` an orthonormal basis of the scaled
constant-control tangent span at the same endpoint, anchored from the original
initial state; the manifold-leakage floor is `(I - N Nᵀ) W J` with `N` the basis
of `null(C W^-1)`. Solver settings, FD steps, and per-case wall time are recorded
in the JSON manifests under `results/phase{5,6,7,8}`. No server connection
details appear in the repository.
