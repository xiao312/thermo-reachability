# Report 05 — Flagship validation of the third direction, and a finite local
reachable patch

Branch `flagship-validation`, from `memory-calibration` at `962ba08`.
Studies: `results/phase9` (flagship validation), `results/phase10` (local patch).
Regression tests: `tests/test_sensitivity.py`, 46 passing.

This revision retains and validates the single most consequential result rather
than sweeping again: whether the hot-start, γ=1e4 1/s, horizon=1e-5 s,
three-equal-segment endpoint map has a reproducible third independent derivative
direction and a finite history-generated excursion from the constant-control
family.  **It does**, at a 635× error margin — and the finite excursion it
produces at admissible radii is real but small on the application scale.

---

## 1. Repairs, each with a regression test

| Repair | What was wrong | Test |
|---|---|---|
| Segment-restart FSA | one `solve_ivp` with a `searchsorted` control selector; the column receiving parameter forcing changes discontinuously at every boundary | the review's exact benchmark (`qdot=γ`, `S_j'=γ_j` on segment j, γ=(10,1e5,10), durations=(0.0499995,1e-6,0.0499995)) reproduces the endpoint 1.09999 and sensitivities (0.499995, 0.1, 0.499995) |
| Sensitivity tolerances | `np.full(n*m, atol.min())` applied a species-sized tolerance to the temperature sensitivity | `np.repeat(atol_state, m)` for the row-major (n,m) block |
| Replay direction | replaying along `v2(A)` tests `‖B v2‖`, not `σ1(B)` — ZERO on the counterexample `A=diag(3,2,1)`, `P=proj(e1,e2)` where `σ2(A)=2`, `σ1(B)=1` | replay along the leading transverse RIGHT singular vector of `B=(I-P)A`, with signed projection, cosine and full vector residual |
| Signed commutator | `[F_a,F_b]=(γ_b−γ_a)[r,v]` under `[f,g]=Dg f−Df g`; Phase 7 reported the opposite sign and compared only norms | linear benchmark `qdot=−q+γ`, a=1, b=3: exact difference `(b−a)(1−e^{−τ})²`, positive |
| Ignition time | a 400-point grid lookup quantizes `t_ign` to `T/399` and makes it horizon-dependent | dense-output root finding; the grid estimate is kept only to expose the quantization, censoring explicit |
| Reference basis | unfiltered QR had returned in the transverse analysis | the single rank-revealing `svd_basis` decides rank; an unresolved span yields transversality UNRESOLVED, never a fabricated projector |

A repair that would have been invisible without the review: the *norm-only*
commutator comparison.  Phase 7's coefficient had the wrong sign and the error was
invisible because both sides were passed through `‖·‖`.

## 2. Primary-anchor uncertainty table

Anchor: HP-equilibrium hot start of stoichiometric H2/air (h2o2 mechanism, hash
`0efc6c52862741a2`), Tin = 1200 K, p = 101325 Pa, γ = 1e4 1/s, horizon 1e-5 s,
m = 3 equal segments and equal base controls.  The state Jacobian `F_q` is itself
numerical, so it is varied (eps ladder 1e-5, 1e-6, 1e-7) independently of the ODE
tolerance, of the integrator (Radau, LSODA) and of the replay.

| Quantity | Value |
|---|---|
| total spectrum `σ(A)` | (0.9484, 0.1189, **5.963e-5**) |
| transverse spectrum `σ(B)` | (**1.234e-4**, 2.44e-9, 6.93e-13) |
| `‖(I−P)u₀‖`, `‖(I−P)u₁‖`, `‖(I−P)u₂‖` | 9.3e-6, 9.1e-4, **1.000** |
| δ_A (spread of A over methods × eps) | 3.07e-9 |
| δ_P·‖A‖ (spread of reference projectors) | 1.91e-7 |
| combined, `‖B̂−B‖ ≤ δ_A + δ_P‖A‖` | ≤ **1.94e-7** |
| margin of 1.234e-4 over the combined estimate | **635×** |
| margin of σ₃(A) over the combined estimate | 307× |
| 3-segment vs 1-segment endpoint agreement | 1.13e-9 |

**The third total direction is entirely transverse** (`‖(I−P)u₂‖ = 1.000`), and by
the *exact* inequality `‖(I−P)A‖₂ ≥ σ₃(A)` valid for ANY orthogonal projector P of
rank at most 2, a stable nonzero `σ₃(A)` is evidence for a third direction that
does not depend on how the reference plane was built at all.

The replay is now the correct quantity.  Perturbing along `v_perp`:

| ε | signed projection | relative deviation | cosine | vector residual |
|---|---|---|---|---|
| 0.1 | 1.325e-4 | +0.074 | +0.9993 | 8.4e-2 |
| 0.03 | 1.242e-4 | +0.007 | +1.0000 | 7.6e-3 |
| 0.01 | 1.235e-4 | +0.001 | +1.0000 | 8.4e-4 |
| 0.003 | 1.234e-4 | +0.000 | +1.0000 | 7.6e-5 |
| 0.001 | 1.2339e-4 | +0.000 | +1.0000 | 1.3e-5 |

The scalar, the direction and the full vector all converge with the expected
O(ε²) central-difference behaviour.  A separate replay along `v₃(A)` returns
6.006e-5 → 5.963e-5, confirming `σ₃` independently.

The secondary anchor (horizon 1e-6 s) is even cleaner: transverse 1.553e-3 against
a combined error estimate 1.81e-9, an **8.6×10⁵** margin.

### The negative case, correctly unresolved

At the fresh γ=1e4, horizon 1e-4 s anchor, the state-Jacobian eps spread is
**δ_A = 1.19e-3** — larger than both `σ₃` (7.6e-12) and the transverse value
(1.7e-10).  The replay does not converge: its signed projection is negative and
proportional to ε (−2.5e-4, −2.3e-5, −2.5e-6, −2.3e-7, −2.6e-8) with cosine −0.996.
This is the correct signature of a quantity below the estimator's resolution, and
it demonstrates that the replay diagnostic distinguishes signal from noise rather
than concealing it.

## 3. A finite local reachable patch

Radii r ∈ {0.003, 0.01, 0.03, 0.1} in log-control space, perturbing along the
three total right-singular directions and the transverse one, both signs.  Every
generated state is witnessed by an actual history from the original q0 and the
raw nonlinear endpoint is evaluated.  `A = W J` and `D = W V` are loaded from the
phase-9 NPZ so the patch is built on the exact validated matrices.

Local model check `E(η₀+dη) = E₀ + J dη + R`, transverse direction:

| r | ‖W d‖ | ‖W R‖ | dT (K) |
|---|---|---|---|
| 0.003 | 3.70e-4 | 2.58e-6 | +3.3e-3 |
| 0.01 | 1.22e-3 | 2.87e-5 | +9.5e-3 |
| 0.03 | 3.57e-3 | 2.58e-4 | +1.6e-2 |
| 0.1 | 1.10e-2 | 2.86e-3 | −9.8e-2 |

The residual grows as r² (a ×11 step for a ×3.3 radius), so the linear model
holds with a clean second-order correction.  The least-squares scale
`L = 2 Σ n²‖R‖ / Σ n⁴ = 1.60` in the physical control norm
`n = √(Σⱼ (durⱼ/T) dηⱼ²)` is recorded, and the resulting containing shell is
labelled **EMPIRICAL**, not certified: L is estimated from samples, not bounded.

**The honest limitation.**  The transverse direction's *finite* displacement is
dominated by its in-family component: the off-plane part is only `r·σ_perp`, i.e.
1.27e-5 scaled units at r = 0.1, which is **1.2e-3 K**.  One scaled unit is 100 K
or 1e-2 mass fraction, so at ε = 0.01 the first-order excursion is 0.01 times the
derivative magnitude.  The direction is mathematically real and well validated,
but the off-family excursion reachable at admissible radii is small on the
application scale — it would not by itself change a CFD calculation.  Significance
is judged on the finite state difference, not on the derivative per unit
parameter.

Distances to the continuous constant-control family `B(γ,t;q0)`, by global grid
over a declared domain plus local refinement and a fresh-integration replay of
the nearest candidate.  This is an **upper estimate** of the distance to the
continuous family: a grid plus a local minimizer can overestimate the minimum, so
it is not a certified global exclusion.

| direction, sign | tangent-plane distance | curved-family distance (upper est.) | ratio | nearest (γ, t) |
|---|---|---|---|---|
| v1 +1 | 1.63e-4 | 9.09e-7 | **0.006** | 1.07e4, 9.92e-6 |
| v1 −1 | 1.49e-4 | 8.48e-7 | **0.006** | 9.37e3, 1.01e-5 |
| v2 +1 | 1.05e-5 | 1.12e-5 | 1.07 | 1.08e4, 9.23e-6 |
| v2 −1 | 1.60e-5 | 1.03e-5 | 0.65 | 9.30e3, 1.09e-5 |
| v3 +1 | 5.58e-6 | 5.72e-6 | 1.03 | 1e4, 1.00e-5 |
| v3 −1 | 6.34e-6 | 6.20e-6 | 0.98 | 1e4, 1.00e-5 |
| v_transverse +1 | 1.27e-5 | **1.21e-5** | 0.95 | 9.34e3, 1.08e-5 |
| v_transverse −1 | 1.41e-5 | **1.26e-5** | 0.90 | 1.07e4, 9.32e-6 |

The in-family direction (v1) sits essentially ON the curved family — the family
closes the gap by a factor of ~180, i.e. it curves to meet it, and the nearest
point is the base point with γ and t nudged.  Every other direction is different:
the curved-family distance is no smaller than the tangent-plane distance (ratios
0.65–1.07), so the family does **not** curve to meet them.  The transverse
direction is off the continuous constant-control family by

    1.21e-5 / 1.26e-5 scaled units = **1.2e-3 K and ~1e-7 in mass fraction**

at control radius r = 0.1 (a 10 % change in each segment level), in both signs,
with the nearest family candidate confirmed by fresh integration.

## 4. What changed relative to Revision 4

* **Refuted as a mechanism claim, retained as a candidate:** the two
  γ=1000 / horizon=1e-3 "resolved but below threshold" directions (former C26)
  are narrowed to candidates.  They rested on projector-based inference without
  reference-uncertainty machinery, at reference-span condition numbers ~1.94e7
  and ~1.56e6 versus ~20 for the flagship.
* **Promoted:** the hot γ=1e4 case is no longer a candidate — it is validated as
  the flagship, with a 635× error margin, an exactly-inequality-backed σ₃, and a
  convergent signed replay (C29).
* **Repaired, not retracted:** the flagship's underlying numbers are unchanged
  (0.9484, 0.1189, 5.963e-5; 1.234e-4), so the earlier estimate was right for
  reasons that are now actually established rather than assumed.

## 5. One concise next experiment

With the third direction validated, the open question is no longer *whether* it
exists but *how much state it can move*: the off-family excursion at admissible
radii is ~1.2e-3 K, which is small.  The natural next experiment is therefore a
**radius escalation at the secondary anchor** (horizon 1e-6 s, where the margin
is 8.6×10⁵ and the transverse value 1.55e-3 is 13× the flagship's), pushing r
toward the admissible boundary while tracking `‖W R‖` to find where the linear
model breaks and the off-family excursion becomes application-relevant.  This is
a single one-parameter sweep at an already-calibrated anchor, not a broad study.

## 6. Scope

h2o2 mechanism (10 species, 29 reactions; hash `0efc6c52862741a2`), ideal-gas
CSTR, the two declared initial states, γ ∈ [10, 1e5], horizons 1e-6…1e-4 s, frozen
scaling (1 unit = 100 K or 1e-2 mass fraction).  `F_q` remains numerical; the
estimator is exact in the *control* derivative only.  Cross-method spreads are
empirical, not certified operator-error bounds.  Matrices, singular vectors,
projectors, endpoints and per-case settings are persisted in
`results/phase9/*.npz` and the JSON manifests; the requested/completed/failed
index and deliberate exclusions are recorded explicitly.
