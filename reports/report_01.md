# First report — thermochemical reachability (Phases 0–3)

**Status:** Phases 0–3 executed. Raw data, manifests, figures, replay commands
and a claim ledger are present. Phases 4–6 deferred by design.

All computation ran on a remote Linux compute server (112 cores, no cgroup
quota, no GPU used) in an isolated conda environment. The GitHub repository
holds code, results and logs; no CI is configured, and no code was executed
locally.

## 1. Environment and execution commands

Isolated env: `/data2/kexiao/conda-envs/tr` — Python 3.12.14, Cantera 3.2.0,
NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0, Matplotlib 3.11.1, pandas 3.0.5,
pytest 9.1.1. Audit: `results/phase0/environment.json` (incl. `h2o2.yaml`
sha256, species, 29 reactions, NASA range 200–5000 K).

```bash
export PY=/data2/kexiao/conda-envs/tr/bin/python
$PY scripts/phase0_smoke.py            --results-dir results/phase0
$PY scripts/phase1_reachability.py     --results-dir results/phase1
$PY scripts/phase2_reactor_validate.py --results-dir results/phase2 --horizon 0.1
$PY scripts/phase3_envelopes.py        --results-dir results/phase3            # main
$PY scripts/phase3_envelopes.py        --results-dir results/phase3 --pilot     # pilot
$PY scripts/phase3c_splitting.py       --results-dir results/phase3c
$PY -m pytest tests -q
```

Every run writes a `manifest.json` (timestamp, git revision, resolved config,
seed, mechanism provenance). The historical smoke-test `results.json` from the
handoff bundle was preserved and **not** relabelled as a new run.

## 2. Phase 0 — audit and smoke tests (executed)

- Environment audit recorded; Cantera available here (it was **not** in the
  original 4-core/4 GiB sandbox).
- Toy A→B→C: exact propagator vs Radau max error **7.4e-11**; max positive
  barrier violation **2.3e-11**; symbolic identity
  `dV/dt|_(V=0) = γ(log x + 1 − x)` verified, plus two *additional* audit checks
  not in the handoff (steady locus `y = x(1−x)`; batch trajectory `y = −x log x`).
- Robertson stiff system: Radau vs BDF max difference **1.2e-9**.
- `all_tests_passed = True`.

## 3. Phase 1 — toy bounded-control reachability (executed)

Setup: γ ∈ [0, G], G ∈ {0.1, 1, 10}; t_f ∈ {1, 5, 20}; 8 segments (nested
4/8/16/32); 24 constant controls, 32 random + 11 structured + 16 held-out
histories; direct-shooting optimization of 4 support directions.

| Quantity | t_f = 1 | t_f = 5 | t_f = 20 |
|---|---|---|---|
| Envelope area fraction visited (G=1) | 0.129 | 0.933 | 1.000 |
| Terminal y_max, G = 0.1 / 1 / 10 | 0.368 | 0.101 / 0.250 / 0.352 | 0.083 / 0.250 / 0.342 |

**Key results.**

1. **The visited set is blind to the control bound.** `visited_y_max = 0.3679
   = 1/e` for *every* G, because γ = 0 is admissible for all G and the
   zero-control batch trajectory traces the analytical upper curve
   `y = −x log x` exactly. Only the *terminal* set R(t_f) distinguishes G.
   This is a property of the pure-A feed toy, not a general statement.
2. The unrestricted enclosure `{0 ≤ x ≤ 1, 0 ≤ y ≤ −x log x}` is a containing
   bound (proved via the barrier), and its upper face is *attained* by γ = 0;
   interior attainability needs a reaction-then-replace construction whose
   infinite-time / unbounded-feed limits were not separately formalized.
3. Barrier V = y + x log x ≤ 0 held to ~1e-16 over all histories; conservation
   triangle likewise. Inclusion under class enlargement held in all tests.
4. **Resolution caution:** terminal extent is noisy in the segment count
   (terminal area 0.0485/0.0557/0.0518/0.0429 for 4/8/16/32), so terminal
   optima are achieved lower estimates, not certified suprema.

## 4. Phase 2 — detailed-chemistry CSTR (executed, validated)

h2o2.yaml, adiabatic, fixed p, constant mass, variable volume, matched in/out
flows. Governing equations as specified; both algebraically equivalent energy
forms verified to relative < 1e-9.

| Check | Result |
|---|---|
| Energy-form equivalence | rel. < 1e-9 |
| E·r = 0 (elemental source) | rel. < 1e-9 |
| Closed-reactor enthalpy drift | rel. ~1e-9 |
| Exact balances b(t), h(t) — **nontrivial** case (T_in=1000 ≠ T0=1100, different comp.) | elem. rel. < 1e-8, enth. rel. < 1e-6 ✓ |
| Radau vs BDF (γ ∈ {10,1e3,1e5}, fresh+hot) | < 1e-6 K, < 3e-11 species |
| **Custom RHS vs Cantera ReactorNet** (γ ∈ {10,1e2,1e3,1e4}, fresh+hot) | **≤ 4e-7 K, ≤ 3e-11 species** |

The fresh/hot-start balance cases have b0 = b_in and h0 = h_in *by
construction*; each is flagged `trivial_consistency_case` in the results, and
the nontrivial case above is the one that actually tests the balance law.

The ReactorNet cross-check validates two independent integration pathways
against each other; it is **not** independent validation of the chemical
mechanism. The Cantera pressure-controller example was not assumed to be an
exact fixed-pressure implementation; matched `MassFlowController`s on an
`IdealGasConstPressureReactor` were used.

## 5. Phase 3 — empirical envelopes (executed)

Frozen condition: h2o2, 1 atm, T_in = 900 K, φ = 1, γ ∈ [10, 1e5] s⁻¹
(residence 1e-5–1e-1 s), t_f = 0.1 s. Three labelled studies (fresh feed;
HP-equilibrium hot start — a *separate permitted initial condition*, not
evidence of reachability from fresh feed; and a pilot-adjusted T_in = 1200 K
condition). Per study: 24 constant controls (set B), 32 random + 11 structured
switching histories (set C), 16 whole-history held-out, and a steady family
(set A) traced from cold and hot inits.

| Study | A: steady T | B: const-ctrl T | C: switching T | C→B max dist. | failures |
|---|---|---|---|---|---|
| fresh (900 K) | 900–2653 K | 900–900 K | 900–900 K | 0.116 | 0 |
| hot (HP eq.) | 900–2653 K | 1550–2655 K | 1550–2655 K | 0.173 | 0 |
| fresh (1200 K, adjusted) | 1727–2762 K | 1200–2762 K | 1200–2763 K | 0.285 | 0 |

**Key results.**

1. **Default fresh feed does not ignite in 0.1 s** (T rises < 0.3 K,
   Y_OH ~ 1e-10). Reported, not hidden; the 1200 K condition was added and the
   original preserved.
2. **The steady family is an S-curve**: at γ = 1e5 the cold branch terminates
   at the unreacted feed (~900 K) and the hot branch at a burning state
   (~1550 K). Steady residuals < 6e-9. A converged root is a mathematical
   solution, not evidence of accessibility within the horizon.
3. **Main result — negative.** Within γ ∈ [10, 1e5] and t_f = 0.1 s, switching
   histories do **not** reach states far outside the constant-control transient
   library from the same initial condition and horizon (C→B max distance
   0.12–0.29 dimensionless ≈ 120–290 K equivalent; held-out consistent).
4. **Audit incident worth reporting.** An *apparent* large excursion (hot start
   → 900 K under switching) was initially observed. It was traced to a
   control-generation bug that drew γ up to 1e6 — outside the declared
   admissible class. After enforcing γ ≤ 1e5, the excursion vanished: no
   switching history from a hot start goes below 1550 K. **An unverified
   control bound can manufacture a false "switching reaches new states"
   result.** This is the most transferable methodological finding of the
   tranche.
5. **Resolution trap caught.** Constant γ = 1e5 from a hot start cools to
   1550 K within ~1e-4 s; a coarse output grid misses the dip and would
   misattribute the state to switching. Extrema are now captured on a grid
   resolving 1/γ (`reactor.integrate(record_extrema=True)`).
6. **LP conservation bounds** (relaxation, kinetics ignored): Y_N2 ≤ 0.7451;
   energy-conserving T bound 3010 K (900 K feed) / 3270 K (1200 K feed).
   Equilibrium temperature was not assumed to be a universal attainable
   maximum.

## 6. Phase 3C — Strang splitting (see `results/phase3c`)

X_{dt/2} → R_dt → X_{dt/2}, with X the exact exchange map (preserves the exact
mixing invariant) and R the closed-reactor reaction step. Compared against the
unsplit high-accuracy CSTR solution under timestep refinement, with steps
aligned to control discontinuities. On a smooth (non-igniting) control case the
error is ~1e-5 K and clean; on igniting mixtures the error is large at coarse
dt and converges at an observed order approaching the formal value of 2
(≈1.9 at the finest level), with a long pre-asymptotic regime extending to
dt ~ 1e-4 s. Chemistry-substep inputs (pre-R states) are recorded separately
from exact unsplit physical states.

## 7. Experiment summary table

| Experiment | Run ID / path | Attempted | Success | Fail | Runtime | Strongest residual | Status |
|---|---|---|---|---|---|---|---|
| Phase 0 smoke tests | `results/phase0/smoke_results.json` | 3 blocks | 3 | 0 | < 1 s | 1.2e-9 | executed, passed |
| Phase 1 toy reachability | `results/phase1/phase1_results.json` | 9 cases + 4 nested | 13 | 0 | 15 s | barrier 1e-16 | executed |
| Phase 2 reactor validation | `results/phase2/phase2_results.json` | 5 balances, 6 xint, 8 net-checks | 19 | 0 | 3.6 s | elem. rel. 1e-9 | executed, passed |
| Phase 3 envelopes (main) | `results/phase3/phase3_results.json` | 3×(24 B + 43 C + 16 held-out) | 249 | 0 | 227 s | steady res. 6e-9 | executed |
| Phase 3 envelopes (pilot) | `results/phase3/phase3_results_pilot.json` | 3×(8+24+8) | 120 | 0 | 113 s | — | executed |
| Phase 3C Strang splitting | `results/phase3c/phase3c_results.json` | 8 cases × 5 levels | 8 | 0 | see manifest | order → 2 | executed |
| Unit tests | `tests/test_thermoreach.py` | 16 | see run | — | < 2 min | — | executed |

## 8. Replay of the most informative case

The audit incident (item 5 above) is the most informative, and it is replayable:

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0,'src')
import numpy as np
from thermoreach.reactor import CSTR, ReactorConfig, hot_hp_state
from thermoreach.controls import random_history
c = CSTR(ReactorConfig(mechanism='h2o2.yaml'))
rng = np.random.default_rng(20260916)
for i in range(60):                      # admissible class enforced: gamma <= 1e5
    h = random_history(rng, 1e5, 8, 0.1, zero_prob=0.0)
    assert h.values.max() <= 1e5
    r = c.integrate(h, hot_hp_state(c), samples_per_segment=41)
    print(i, round(float(r['states'][0].min()), 1))   # never below 1550 K
EOF
```

## 9. Limitations — what was shown vs. what remains untested

**Shown:** exact toy propagator and invariant; G-sensitivity of the terminal
set; a validated detailed-chemistry CSTR (two independent integration pathways,
exact balances incl. a nontrivial case); the steady S-curve; and the *absence*
of a switching excursion within the declared admissible class.

**Not shown / caveats:** no certified (interval) outer bounds — ordinary
optimization optima are achieved objectives only; the toy envelope's interior
attainability is not formally proved; reference-set distances are bounded
below by reference-library resolution (mitigated, not eliminated, by the
extrema capture); the Phase 3 conclusion is for one mechanism, one operating
condition, one horizon and one control resolution class; solver agreement is
not mechanism validation; a failed simulation is not evidence of unreachability.

**Not reached:** Phases 4–6 (counterflow continuation, reduced flamelet
transients, reaction-invariant convex enclosures), the GRI-3.0 scaling check,
and any GPU/ML acceleration — all deferred until after this first report, as
the brief directs.

## 10. Closing assessment

- **What is supported?** The toy mathematics (C1–C6), the reactor implementation
  and its validation (C10–C15), the S-curve structure (C19), and the *negative*
  switching result (C20) — all numerically supported within the stated scope.
- **What was refuted / ambiguous?** The motivating idea that switching
  histories reach states far outside the constant-control transient library is
  *not* supported at this condition; the apparent support was a control-bound
  bug. Whether this generalizes is open.
- **What would most efficiently reduce remaining uncertainty?** A
  resolution-and-bound sweep on the *adjusted* (1200 K) condition: repeat the
  C-vs-B comparison while tightening γ bounds, segment count, and reference
  resolution, and add interval-arithemetic bounds on the toy barrier to convert
  C5 from a symbolic-plus-sampling argument into a certified enclosure.
- **Single best next experiment.** A **two-segment quench-then-hold protocol
  sweep** at the 1200 K condition: parameterize γ_high (flush), hold time, and
  γ_low, and measure the minimum achievable T against the constant-control
  library. Expected discriminating outcome: if switching can cross from the
  burning branch to the unreacted branch, the minimum T will fall below the
  constant-control floor as hold time increases past the ignition delay; if it
  cannot, the minimum T will plateau at the constant-control floor — directly
  testing whether branch switching is possible at fixed control bound, which
  is the crux of the history-dependence question.
