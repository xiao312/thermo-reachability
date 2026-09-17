# Report 06 — Radius sweep, chemistry response, and a local coverage statement

Branch `radius-response`, from `flagship-validation@39afaa3`. Study: `results/phase11`.
Tests: 111 passing on the server (23 new in `tests/test_chemresponse.py`).

The rank-three result is an accepted numerical milestone; this revision asks the
question that follows it: *at permitted finite perturbations, can the canonical
constant-control family represent the generated states accurately enough for the
intended chemistry computation, and how well can a local state generator describe
the extra variation?*

**Answer, in scope.** Within the examined local class — the calibrated secondary
anchor (HP-equilibrium hot start, γ = 1e4 1/s, horizon 1e-6 s, three equal
segments) at time-norm control radii up to 1.0 — the family represents the
generated states to within **0.08 K and 1.0e-5 in mass fraction**, and the same
chemistry-only map applied to a generated state and to its nearest representative
agrees to within **0.11 K and 6.8e-6 in mass fraction** over dt = 1e-7…1e-4 s.
Both are small relative to every illustrative tolerance tried, and the chemistry
difference *decays* with dt. The off-family residual is **99.9 % concentrated in a
single direction**, so a minimal "curved reference + one normal direction" model
captures essentially all of it. This is a tolerance-qualified local statement, not
a global claim.

## 1. Metric, enclosure and uncertainty wording — addendum to Report 05

The following corrections are made to the Report 05 wording. Historical results
are preserved unchanged; only the claims attached to them are narrowed.

* Method/eps spreads (δ_A, δ_P·‖A‖) are **empirical uncertainty indicators**, not
  guaranteed operator-error bounds. The combination
  `‖B̂−B‖ ≤ δ_A + δ_P‖Â‖` is an inequality between the measured spread and the
  operator, and the spread itself certifies nothing beyond the settings tried.
* The weighted state norm with T_scale = 100 K and Y_scale = 0.01 implies
  `|ΔT| ≤ 100·d` and `|ΔY_k| ≤ 0.01·d` for a scaled magnitude `d` — an upper bound
  per component, **not equality**, and not a statement that both components are
  that large.
* The best *located* reference pair is **constructive evidence of closeness** in
  the selected norm. It is an upper estimate of the infimum over the reference
  domain; it is not a certified global exclusion, which would require a separately
  justified lower bound.
* **No CFD significance or insignificance claim has been established from state
  distance alone.** No target application tolerances have ever been supplied for
  this project; the tolerance table below spans explicitly illustrative choices.
* `L = 1.602` (Report 05 §3) is a **descriptive least-squares fit**, not a shell
  that contains the calibration samples. The review's warning is confirmed
  quantitatively below: samples do exceed it.

## 2. Repairs, each with a regression test

| Repair | What was wrong | Test |
|---|---|---|
| Ignition event dense-output extrapolation | the event was `terminal=True`, so dense output was defined only on `[0, t_event]` while the routine evaluated it out to the horizon — extrapolation, not solution | non-terminal event; dense evaluation restricted to `[0, t_end]`; **censored** (no crossing in the horizon) distinguished from **integration failure** (no dense evaluation attempted); tested on a known event at horizons 0.3 / 1.0 / 2.0 and at the boundary |
| Admissible radius interval | only two of the four cases were handled; for a **mixed-sign** direction a negative-*v* segment bounds *r* from *above* through the *lower* level bound and from *below* through the *upper* bound — the original returned a wrong interval | all four cases tested; bounds verified to hold exactly at both interval ends |
| Reference refinement | coordinate-wise golden section stalls when the coarse candidate is wrong in *both* γ and t; Nelder–Mead stalled at 1.2e-4 and a zoom grid at 1.2e-2 inside a bounded eval budget | replaced by **damped Gauss–Newton** on the scaled residual, whose Jacobian is the family tangent (`∂B/∂t` exact from the ODE RHS, `∂B/∂log γ` by central difference); recovers exact family points to **1.7e-14** |
| Zero-length integration | the search can evaluate `B(γ, 0)`, a degenerate interval that returns a malformed result | zero horizon is the identity `B(γ,0;q₀) = q₀`, returned directly |
| Jacobian column order | `local_family_tangent_at` returns columns `(dB/dt, dB/dlog γ)` but the parameter order is `(log γ, t)` — a silent transpose in the step | reordered with an explicit comment; the swap was the reason the first LM iterations rejected every step |
| Reference projector | unfiltered `np.linalg.qr` in three places filled a rank-deficient span with rounding noise | the single rank-revealing `svd_basis` builds every projector; rank deficiency raises rather than smooths; equivalence at full rank asserted against the committed NPZ |

A repair that would have stayed invisible: the Jacobian-column swap produced
*sensible-looking* output (the search converged to a worse point and stopped)
rather than an error.

## 3. The declared control norm

Everything is measured in the **time control norm**

    ‖δη‖²_time = Σⱼ (durⱼ/T) δηⱼ²,      H = diag(durⱼ/T)

which is portable across segment counts, with the whitened map `A H^(−1/2)` for
any cross-segment-count comparison. The phase-9 matrices were computed in the
Euclidean parameter norm; they are re-expressed here, and **both norms plus every
actual γⱼ are reported** so no past result changes meaning. For m = 3 equal
segments a Euclidean radius r is a time-norm radius r/√3.

    total spectrum        Euclidean  (0.1497, 0.01120, 0.001108)
                          time-norm  (0.2593, 0.01940, 0.001918)     × √3
    transverse spectrum   Euclidean  1.553e-3
                          time-norm  2.690e-3                          × √3

`D` (the 2-column family tangent) is deliberately **not** whitened: its columns
are family tangents in the scaled *state* space parametrized by the family's own
coordinates (t, γ), not by the m segment controls, so the state-space projector P
is invariant under a change of input norm.

## 4. Cached continuous reference

Each library γ (21 points, γ ∈ [2.2e3, 4.5e4]) is integrated **once** to
t_max = 100 × T with dense output retained, and every later evaluation at that γ
and any t is a dense-output call — the repeated 61×61 fresh-integration grid of
phase 10 is not reproduced. Cache identity (mechanism, thermodynamics, q₀, γ,
tolerances, method, domain) is recorded so a stale cache cannot be silently used.
The coarse scan is refined by damped Gauss–Newton with a one-time domain
expansion if the minimizer sits on a domain edge (**not triggered in this study**:
the library domain was adequate at every radius).

**Exact in-family positive control** (all three prefix levels equal and perturbed
together — the family contains that trajectory at its own γ and at t = T; `v1(A)`
is *not* a substitute and was not used):

| γ_control | recovered γ | recovered t | rel. err γ | distance |
|---|---|---|---|---|
| 10305 | 10304.5 | 1.000e-6 | 1.1e-12 | 7.7e-14 |
| 11052 | 11051.7 | 1.000e-6 | 1.8e-12 | 7.5e-14 |
| 13499 | 13498.6 | 1.000e-6 | 1.5e-12 | 1.1e-13 |

Recovered within verified numerical accuracy at every radius.

## 5. The radius sweep

Time-norm radii {0, 0.03, 0.1, 0.3, 0.6, 1.0}, both signs. The exact admissible
interval for the fixed leading transverse direction is (−2.956, +3.830), so no
radius in the schedule was trimmed and none was clamped. At each radius the
nonlinear history is integrated (Radau, rtol 1e-10) and the raw endpoint, enthalpy,
pressure, density, conservation residuals, physical-domain status and solver
statistics are recorded; the extremes are re-integrated with LSODA at tighter
tolerances (agreement 5.5e-10, reference distance reproduced to 4 digits).

| r | dist. to family | \|ΔT\| K | max \|ΔY\| | γ* | t* | R/lin |
|---|---|---|---|---|---|---|
| 0 | 7.6e-14 | 2.7e-12 | 4.4e-16 | 10000 | 1.00e-6 | — |
| 0.03 | 4.65e-5 | 2.4e-3 | 2.9e-7 | 9928 | 1.01e-6 | 0.105 |
| 0.1 | 1.55e-4 | 7.9e-3 | 9.6e-7 | 9780 | 1.02e-6 | 0.35 |
| 0.3 | 4.61e-4 | 2.4e-2 | 2.9e-6 | 9494 | 1.07e-6 | 1.02 |
| 0.6 | 9.18e-4 | 4.8e-2 | 5.7e-6 | 9408 | 1.12e-6 | 2.01 |
| 1.0 | 1.54e-3 | 8.0e-2 | 9.5e-6 | 9855 | 1.17e-6 | 3.30 |

Two separate facts, deliberately not conflated:

* **The family distance grows linearly in r** (4.65e-5 → 1.54e-3 over a 33× radius
  increase). The transverse direction is a genuine linear direction, and the
  *curved* family closes part of the gap (at r = 0.1 the distance is 58 % of
  σ_⊥·r = 2.69e-4).
* **The fixed-anchor linear model does not stay accurate**: `R/lin` — the
  whole-state Taylor residual over the linear prediction — exceeds **1.0 at
  r ≥ 0.3**. This is the review's warning made quantitative: at finite radius the
  weakest linear direction is dominated by quadratic curvature in the *other*
  directions, and smooth O(r²) growth of the residual says nothing about useful
  *relative* linear accuracy. The residual relative to the **moving** curved
  reference stays exactly at the family distance (the residual is purely normal at
  a minimum, which is the correct nearest-point condition), so the curved
  reference — not the fixed tangent plane — is the usable local model.

## 6. Chemistry response

The same chemistry-only map Φ_dt (adiabatic, constant pressure, exchange control
exactly zero) is applied to each generated state and to its nearest found
representative. Each state's own enthalpy and elemental inventory are preserved by
the map, and the baseline mismatch is **reported, not projected away** (enthalpy
preserved to 2e-8 J/kg in 1.34e6; elements to 5.6e-17). Initial source differences,
finite-time future-state differences and chemistry-increment differences are
reported separately.

No target application tolerances have been supplied, so the error metric
`E(dt,q,q_B) = max_l |Q_l(Φ_dt(q)) − Q_l(Φ_dt(q_B))| / ε_l` is tabulated over
explicitly illustrative tolerance choices; none is asserted to be a universally
acceptable combustion tolerance, and near-zero rates use an absolute-plus-relative
denominator.

Worst case over all radii and all dt (the r = 1.0 ray at dt = 1e-6 s):

| tolerance | E_max | E_T | E_Y |
|---|---|---|---|
| T 1 K / Y 1e-3 | 0.106 | 0.106 | 6.8e-3 |
| T 1 K / Y 1e-4 | 0.106 | 0.106 | 6.8e-2 |
| T 10 K / Y 1e-3 | 0.0106 | 0.0106 | 6.8e-4 |

The response difference is temperature-dominated and **decays with dt** (0.106 at
dt = 1e-6 → 0.023 at dt = 1e-4): the two states relax to the same equilibrium, so
the chemical map does not amplify the geometric difference. A numerically resolved
geometric difference and a consequential chemical-map difference are separate
outcomes; here both are small, and neither implies anything about surrogate
generalization — that is a separate later experiment.

## 7. A local model with honest coverage meaning

    L_fit        = 0.2576        least-squares curvature descriptor
    L_sample_max = 0.3429        worst observed sample, time norm
    ratio        = 1.33

**12 of 22 samples exceed the L_fit envelope, by up to 1.33×.** L_fit is therefore
confirmed to be a descriptive fit, not a containing shell; L_sample_max contains
the observed samples and is not a uniform derivative bound over the neighbourhood.
Neither is a substitute for a bound, and neither is comparable to the
Euclidean-norm L = 1.602 reported in phase 10.

**Structure of the normal residual.** Recomputing the normal basis at each located
point (the anchor normal is not assumed to remain correct), the SVD of the stacked
normal residuals over 23 samples is

    3.469e-3,  1.171e-4,  1.715e-6,  7.14e-7,  …
    99.89 %    100.0 %

— **99.9 % of the off-family energy lies in a single direction**. A minimal
"nearest curved reference + one normal direction" representation is therefore a
candidate local generator for this class, with the second direction at 3 % of the
first.

**Held-out check.** 12 *new* admissible three-segment histories on random
directions at r ∈ {0.1, 0.3, 0.6}, kept separate from the four original rays:

| radius | max dist. | max \|ΔT\| | max R/lin |
|---|---|---|---|
| 0.1 | 2.6e-4 | 1.3e-2 | 0.18 |
| 0.3 | 4.7e-4 | 2.5e-2 | 0.42 |
| 0.6 | 1.6e-3 | 8.2e-2 | 1.31 |

The held-out distances are comparable to the transverse ray's at the same radius,
and their `R/lin` is *smaller* — the fixed transverse direction is the hardest
direction for the linear model, as the rank analysis predicts. The approximation is
therefore not merely fitting the four original rays. This is an empirical
local-coverage check, not a global completeness claim.

## 8. Decision

Against the assignment's stopping rules, this lands in the first branch:
**geometry small and chemistry response small.** Within the examined class the
canonical constant-control family is an adequate representation to ~0.08 K and
1e-5 in mass fraction at the largest permitted radius, the chemical-map difference
is smaller still and decaying, and the residual off-family variation is
one-dimensional. The honest reading is a **tolerance-qualified canonical
approximation for these samples and this local class** — not a global statement,
and not a claim that this survives at other anchors, other initial states, or
longer horizons (phase 6 established that memory erases at holds beyond one
residence time, so the class boundary matters).

**One next experiment.** The single useful extension is a *tolerance-supplied*
study: have the application (a specified diffusion-flame/flamelet model) declare
its actual output tolerances and dt range, then repeat this sweep and report
against them instead of illustrative choices. Without supplied tolerances the
statement above is the strongest defensible one; with them it becomes a
specification. This is deliberately a re-run under declared tolerances, not a
broadening of the mechanism, the anchor set, or the segment count.

## 9. Scope and data

h2o2 mechanism (10 species, 29 reactions), ideal-gas CSTR, HP-equilibrium hot
start, Tin = 1200 K, p = 101325 Pa, γ ∈ [10, 1e5], horizon 1e-6 s, m = 3, frozen
scaling (1 unit = 100 K or 1e-2 mass fraction). `F_q` remains numerical. All
distances are upper estimates of the infimum over the declared reference domain;
no global nonmembership is claimed anywhere. Matrices, mechanism identity, states,
parameters, solver settings and code identity are in `results/phase11/` (JSON plus
`manifest.json`); the run index records 26 completed, 0 failed, 0 excluded, 0
unresolved.
