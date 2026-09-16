# Third report — sensitivity pipeline rebuilt, C22 withdrawn, terminal-memory experiment

**Status:** Revision 3. The Phase 3D sensitivity pipeline has been rebuilt after
a second independent review established that its conclusions were unsupported.
Claim C22 is **withdrawn**; claim C23 replaces it. Positive controls and the
corrected pipeline are complete and passing. The terminal-memory pilot is
reported here.

**Baseline:** `96d4bf2` (audited). **Branch:** `sensitivity-v2`. The superseded
Phase 3D results are retained unmodified with `results/phase3d/DEPRECATED.json`.

## 1. Why C22 is withdrawn

Seven defects, each verified in the source at `96d4bf2`:

| # | Defect | Consequence |
|---|---|---|
| 1 | The reference tangent span was differentiated from the *switched endpoint* rather than the original initial state | It did not differentiate B(γ,t;q_initial); tangency was never established |
| 2 | `h = rel·mean(|θ|)` exceeded the admissible interval for the smallest control; the column was left as **zero** | A fabricated zero sensitivity entered the rank analysis |
| 3 | V was orthonormalized by unfiltered QR | A rank-deficient span was filled with rounding noise; reported angles (0, 0.78, 0.85, 2.6e-8) were artifacts |
| 4 | The "third direction" index was `n−3` with `n = m` | For m = 3 this reads the *smallest* value, not the third largest |
| 5 | Increments mixed kelvin with mass fractions, differentiated w.r.t. raw γ | Singular values had no single physical meaning |
| 6 | Pulse-hold generator appended a remainder then sliced it away | Horizons 0.044 s / 0.064 s despite the 0.1 s label |
| 7 | Only a relative SVD threshold was used | Noise-floor directions (~1e-12 scaled) were counted as real |

The reported `n_real = 1` and `sine ≤ 1e-12` are therefore **not established**. A
singular direction must not be called "real" merely for passing a relative
threshold, and even an exact rank-one derivative at one point does not prove a
curve-shaped image without neighbourhood assumptions — the map
`(a,b) ↦ (a,b²)` has a rank-one Jacobian at the origin and a two-dimensional
image.

## 2. The corrected pipeline

`src/thermoreach/sensitivity.py` + `scripts/phase4_terminal_memory.py`.

* **Dimensionless controls** `η_j = log(γ_j/γ_ref)`. Per-coordinate steps stay
  strictly interior; a column whose stencil cannot be centred is **invalid**
  (NaN), never zero, and is excluded from every rank summary. A round-trip bug
  that returned `exp(η)` rather than `γ_ref·exp(η)` — a factor-1000 scaling
  error — was caught by a regression test, not by inspection.
* **Reference tangents from the original initial state**:
  `v_t = F(q_base, γ*)` and `v_logγ = dφ_γ(T;q₀)/d log γ`, with the exact
  identity `Σ_j dE_m/d(log γ_j) = dφ/d log γ` at constant-control base histories.
* **Rank filtering of both matrices** by SVD with explicit relative **and**
  absolute thresholds; a zero-rank matrix yields an empty basis, never an
  arbitrary orthonormal completion.
* **Conserved-manifold tangent space** as the null space of `C·W⁻¹` by a
  rank-revealing SVD, with conditioning recorded and conservation leakage
  reported *before* projection.
* **Declared frozen state scaling**: one scaled unit = 100 K or 1% mass fraction.
* **Three rank notions** reported separately: algebraic, relative-threshold, and
  absolute-threshold application rank, plus an empirical differentiation-noise
  scale from step refinement `‖J(h₁) − J(h₂)‖₂` (an estimate, not a bound).
* Exactly m durations summing to the recorded horizon; the third singular value
  at fixed index 2 for every m; `record_extrema=False` for endpoint-only work.

## 3. Positive controls (all passing)

`scripts/phase3e_positive_controls.py` → `results/phase3e`, plus
`tests/test_sensitivity.py` (23 tests).

* **Three-state linear benchmark** `dz_i/dt = −λ_i z_i + u(t)`, λ = (1,2,4),
  T = 1, three equal segments: exact endpoint control Jacobian has rank 3 and
  singular values **(0.5007801, 0.08533145, 0.006828632)**, reproduced to 7
  digits. (The review's `REVIEW.md` and `review_checks.py` were **not delivered**
  with the handoff; this benchmark was reconstructed from the stated values and
  the reconstruction confirmed by the exact match.)
* **Exact toy rank-2 history** plus terminal hold: algebraic prefix rank is
  preserved (2, through L = 10) while the application-scale effective rank
  decays (2 → 0); a ratio-preserving relative threshold alone keeps reporting
  rank 2 at any scale.
* **Same-initial-state identity** for m = 1, 3, 5 across γ and T, to machine
  precision (≤ 1.4e-17).
* **Angle tests**: `V = [e₁,0]` vs `J = [e₂]` gives sine **1**, not 0.

## 4. The terminal-memory experiment

`E(θ_prefix, γ_hold, L) = φ^L_{γ_hold}(q_prefix(θ_prefix))`, so
`dE/dθ_j = D_q φ^L · dq_prefix/dθ_j`: prefix sensitivity is *propagated* through
the hold and can become unresolved while γ_hold sensitivity approaches the
steady-branch tangent. Exponentially tiny is not exactly zero.

*Scope frozen:* ideal-gas h2o2, 101325 Pa, stoichiometric H2/air, T_in = 1200 K,
γ ∈ [10, 1e5]; fresh and HP-hot kept separate; 900 K carried as a separately
labelled weak-reaction comparison (it does not ignite: T stays at 900 K).
m = 3 prefix. Constant-control anchors γ = 1e2, 1e3, 1e4; horizons 1e-3, 1e-2,
0.1 s chosen from an actual time scan (fresh/1200 K ignites at ≈2.2 ms for
γ = 1e3). Hold durations L ∈ {0, 1e-2, 1e-1, 1.0} s. 144 cases.

**Validation first.** The same-initial-state identity
`Σ_j dE_m/d(log γ_j) = dφ/d log γ` holds over all 144 cases to a worst relative
error of **8.0e-5** — consistent with O(h²) finite-difference truncation of both
sides (on the toy, where the sensitivity is available in closed form, the same
identity holds to 1.4e-17). The independent cross-integration check (LSODA vs
Radau on the same reference derivative) agrees to 5.3e-7 relative at
γ = 1e2/T = 1e-3 and 8.5e-13 at γ = 1e4/T = 0.1. The differentiation-noise
scale `‖J(h₁) − J(h₂)‖₂` is 2.8e-8 to 4.8e-7 scaled units.

**Result — prefix rank depends on the number of exchange times γ·T.**
Singular values in scaled units (one unit = 100 K or 1% mass fraction);
`noise` is the empirical differentiation-noise scale:

| γ·T | γ | T (s) | sv₁ | sv₂ | sv₃ | noise | abs rank | above-noise rank |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 1e2 | 1e-3 | 1.28e-1 | **1.16e-5** | 6.5e-12 | 2.8e-8 | **2** | **2** |
| 1 | 1e3 | 1e-3 | 1.17 | **9.94e-5** | 5.3e-12 | 4.7e-7 | **2** | **2** |
| 10 | 1e4 | 1e-3 | 2.76 | 1.20e-7 | 1.3e-11 | 2.0e-7 | 1 | 1–2 (marginal) |
| 1 | 1e2 | 1e-2 | 1.29e-1 | 7.4e-12 | 2.9e-12 | 2.8e-8 | 1 | 1 |
| 10 | 1e3 | 1e-2 | 1.17 | 1.9e-12 | 4.6e-13 | 4.5e-7 | 1 | 1 |
| 100 | 1e4 | 1e-2 | 2.76 | 1.4e-11 | 7.9e-13 | 7.9e-8 | 1 | 1 |
| 10 | 1e2 | 1e-1 | 1.29e-1 | 4.4e-12 | 1.4e-12 | 2.8e-8 | 1 | 1 |
| 100 | 1e3 | 1e-1 | 1.17 | 4.7e-12 | 1.9e-12 | 4.5e-7 | 1 | 1 |
| 1000 | 1e4 | 1e-1 | 2.76 | 8.3e-12 | 2.0e-12 | 7.9e-8 | 1 | 1 |

(Identical for the HP-hot initial state and for both perturbation patterns;
the table uses the `early_high` pattern.)

**Finding 1 — real independent transient history effects exist.** At
T = 1e-3 s, for γ = 1e2 and 1e3 (both initial states, both patterns), the prefix
Jacobian has a second singular direction at 1.16e-5 / 9.94e-5 scaled units,
**400× / 200× above the differentiation-noise scale** and above the declared
1e-6 application threshold. This is a genuine history-generated direction, not a
numerical artifact. Its magnitude is modest in physical terms — roughly 0.01 K
(or 1e-6 in mass fraction) per unit step in log control — so it is resolved but
of small application scale. It is largest at about one exchange time
(γ·T ≈ 1) and decays on both sides.

**Finding 2 — decay of earlier-history sensitivity.** By T = 1e-2 s the second
direction has collapsed to ~1e-12 scaled units, two to five orders of magnitude
*below* the noise scale: unresolved, and indistinguishable from absence.

**Finding 3 — unresolved at the numerical accuracy.** At γ = 1e4, T = 1e-3,
sv₂ = 1.20e-7 sits at or below the noise scale (2.0e-7 for the `equal` pattern,
5.6e-8 for `early_high`): the second direction is at the resolution floor and
no claim about it is supportable in either direction.

**Finding 4 — terminal memory erasure.** For every hold L ≥ 1e-2 s — including
L = 1e-2 s, which is a single residence time at γ = 1e2 — the prefix block of
the full-history Jacobian collapses to ~1e-12 scaled units, absolute application
rank **0**, while the endpoint sits at the steady reference (distance to steady
~1e-13 to 1e-16, endpoint RHS norm ~1e-6 to 1e-10). The chain rule
`dE/dθ_j = D_q φ^L · dq_prefix/dθ_j` is confirmed: the hold propagates the prefix
sensitivity into nothing at application scale. This is erasure, not exact zero —
the state-transition map is locally invertible — and it matches the toy positive
control, where the algebraic rank survives while the application-scale rank
decays.

**Physical reading.** The reachable endpoint image is locally two-dimensional
only within roughly one exchange time of the initial state; beyond that, the
CSTR forgets the ordering of its control history and the endpoint image
collapses onto the one-dimensional constant-control/steady-branch curve. The
relevant dimensionless group is the number of exchange times γ·T, not the
horizon alone.

## 5. Interpretation: four separate phenomena

The review asked for these to be distinguished. With the three rank notions and
the noise scale, they are, and all four appear in this study:

1. **Real independent transient history effects** — Finding 1: sv₂ above both
   the noise scale and the application threshold, at γ·T ≈ 0.1–1.
2. **Decay of earlier-history sensitivity under a terminal hold** — Finding 4:
   the same direction collapsing below the application threshold for every
   L ≥ one residence time.
3. **Directions unresolved at the numerical accuracy** — Finding 3: sv₂ at the
   noise floor (γ = 1e4); reported as unresolvable, not as absent.
4. **Directions resolved but below application tolerance** — Finding 1's
   direction is above the threshold but only ~0.01 K per unit log control; its
   application relevance is modest and is stated as such.

## 6. Honest scope

All findings are confined to the tested map (the h2o2 CSTR endpoint map at the
stated conditions), the tested histories (m = 3 constant-control bases with
equal / early-high variation), the time window (1e-3 to 0.1 s plus holds), and
the declared tolerance. Nothing here bounds the global reachable set, and the
positive finding is a *local* statement about the endpoint map at short horizon.
Acceptance does not require finding a third direction; it requires that the
experiment would **detect** one when present at its stated resolution — which
the positive controls establish.

The tangent-linear (FSA) pathway was implemented and is available behind
`--fsa`, but is too slow for routine use; the independent cross-integration check
used here is LSODA vs Radau on the same derivative.

## 7. Experiment summary

| Experiment | Path | Outcome |
|---|---|---|
| Positive controls | `results/phase3e` | all 4 checks pass (0.1 s) |
| Sensitivity unit tests | `tests/test_sensitivity.py` | 23 pass |
| Terminal-memory pilot | `results/phase4` | 144 cases, 3405 s; rank 2 at γ·T≈0.1–1, erasure for every hold |
| CLI smoke tests | `tests/test_cli_smoke.py` | phase3e + phase4 included, with semantic assertions |

Replay (compute server, env `/data2/conda-envs/tr`):

```bash
$PY scripts/phase3e_positive_controls.py  --results-dir results/phase3e
$PY scripts/phase4_terminal_memory.py      --results-dir results/phase4
$PY -m pytest tests -q
```

## 8. Next experiment selected by the pilot

The pilot shows the interesting regime is γ·T ≈ 0.1–1, where the second
direction is largest, and that the controlling dimensionless group is the
number of exchange times. The natural next step is therefore NOT a larger
mechanism or an exhaustive history search, but a **focused scan of the
exchange-time group**: hold γ·T fixed at several values in [0.03, 3] and sweep
the *shape* of the prefix history (segment ratio and ordering) at m = 3 and 4,
with the hold sweep repeated inside the γ·T ≈ 1 window where memory survives.
The ordered-pulse a→b vs b→a commutator test (claim C21 area) should be run
there too, since the commutator `[f,g] = (1,−1)` predicts history-ordering
effects exactly when the second direction is present.
