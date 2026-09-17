# Research log — thermochemical reachability

Chronological record of the first tranche (Phases 0–3). All commands were
executed on a remote Linux compute server (connection details deliberately
withheld from this public repository) in `/data2/ThermoReachability`, unless
noted. Reproduction commands are at the end.

## 2026-09-16 — environment

- SSH access via an existing `~/.ssh/config` key-auth entry; no VPN required.
- Server: Ubuntu 20.04, 112 cores, 251 GiB RAM, 2.2 TiB free on `/data2`,
  `/home` 100% full. System Python 3.8 only (no pip).
- Network: PyPI and the Tsinghua mirror reachable; GitHub, astral.sh and
  raw.githubusercontent.com blocked. OpenSSL/zlib dev headers absent, so a
  source build of CPython was not viable.
- Discovered an existing Miniconda (`/data2/miniconda3`, Python 3.12.2)
  already configured to use the Tsinghua conda-forge mirror via `~/.condarc`.
- Created an isolated env under `/data2` (package cache kept off the full
  `/home`): `/data2/conda-envs/tr` — Python 3.12.14, Cantera 3.2.0,
  NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0, Matplotlib 3.11.1, pandas 3.0.5,
  pytest 9.1.1. Both `h2o2.yaml` and `gri30.yaml` are available.
- **Incident (resolved):** an early file-sync command was run without a target
  directory and streamed the local home directory to the server. It was
  aborted partway; only dot-directories landed (no `.ssh`, no credentials),
  nothing reached the public repo, and the server directory was wiped and
  verified empty. Subsequent syncs use an explicit tarball built with `-C`.

## Phase 0 — executed

`python scripts/phase0_smoke.py --results-dir results/phase0`
- Environment audit written to `results/phase0/environment.json` (mechanism
  sha256, species list, 29 reactions, thermo range 200–5000 K).
- Smoke tests re-executed fresh: toy exact-propagator error vs Radau 7.4e-11;
  barrier max positive violation 2.3e-11; Radau/BDF Robertson max diff 1.2e-9.
  `all_tests_passed = True`.
- Historical `results.json` from the handoff preserved separately.

## Phase 1 — executed

`python scripts/phase1_reachability.py --results-dir results/phase1`
- G in {0.1, 1, 10}, t_f in {1, 5, 20}, 8 segments; nested 4/8/16/32 comparison
  at G=1, t_f=5; 24 constant controls, 32 random + 11 structured + 16 held-out
  histories per case; direct-shooting L-BFGS-B optimization of 4 support
  directions (terminal and max-over-time objectives).
- **First run was degenerate:** visited-set diagnostics (y_max, projected area)
  were identical across G because gamma = 0 is admissible for every G and the
  zero-control batch trajectory traces the envelope's upper boundary. Added
  separate *terminal*-set diagnostics, which are G-sensitive.
- **Bug found and fixed (control bound):** `random_history` drew
  gamma_max * 10^U(-3, 1), i.e. up to 10 × gamma_max, violating the declared
  admissible class. Exponents changed to U(-3, 0). Phase 1 and Phase 3 were
  re-run after the fix.
- Results after fix: terminal y_max at t_f = 5 is 0.1009 / 0.2502 / 0.3520 for
  G = 0.1 / 1 / 10; envelope area fraction visited 0.129 / 0.933 / 1.000 for
  t_f = 1 / 5 / 20; barrier violations ~1e-16; inclusion checks all pass.
- Terminal area is noisy in the segment count (0.0485 / 0.0557 / 0.0518 /
  0.0429 for 4/8/16/32) → search-resolution caution (claim C9).

## Phase 2 — executed

`python scripts/phase2_reactor_validate.py --results-dir results/phase2 --horizon 0.1`
- **Threshold calibration incident:** the first run "failed" enthalpy balance
  checks with absolute residuals ~3e-2. This was a threshold error, not a model
  error — the balances involve h ~ 1e7 J/kg, so rtol=1e-8 implies ~1e-1 absolute
  roundoff. Criteria were made relative and fixed in the script before re-running
  (not retrofitted silently); per-case `trivial_consistency_case` flags mark the
  fresh/hot cases where b0 = b_in and h0 = h_in by construction.
- **API fix:** Cantera 3.2 `MassFlowController` uses the `mass_flow_rate`
  property; `ReactorNet.advance(t)` used for the cross-check.
- Final: energy forms agree to relative < 1e-9; E.r relative < 1e-9; closed-reactor
  h drift relative ~1e-9; exact balances (incl. nontrivial T_in=1000/T0=1100 case)
  pass; Radau vs BDF < 1e-6 K; custom RHS vs Cantera ReactorNet ≤ 4e-7 K and
  ≤ 3e-11 in species over gamma in {10, 1e2, 1e3, 1e4}, fresh and hot.
  `all_passed = True`, run time 3.6 s.

## Phase 3 — executed

`python scripts/phase3_envelopes.py --results-dir results/phase3` (pilot then main)
- Pilot first (8 constant / 16 switching), then main: 24 constant controls,
  32 random + 11 structured switching, 16 held-out, per initial-condition study;
  three studies: fresh (T_in = 900 K), hot (HP-equilibrium), and a separately
  named pilot-adjusted condition at T_in = 1200 K.
- **Default fresh study shows no ignition in 0.1 s** (T rises < 0.3 K,
  Y_OH ~ 1e-10) — reported, not hidden; the 1200 K condition was added.
- **Resolution trap caught:** constant control at gamma = 1e5 from a hot start
  cools to 1550 K within ~1e-4 s. A coarse output grid would miss the dip and
  misattribute the state to switching. Fixed by recording per-segment extrema on
  a grid resolving 1/gamma (`reactor.integrate(record_extrema=True)`); the
  plateau at 1550 K is genuine (verified with a 2001-point trajectory).
- **Main audit finding:** an apparent large excursion (hot start → 900 K under
  switching) was traced to the control-bound bug (gamma up to 1e6). With
  gamma <= 1e5 enforced, no switching history from a hot start goes below
  1550 K — the excursion vanishes. Recorded as a negative result (C20) and a
  methodological caution.
- Steady family traced from cold and hot inits: multiple steady states at the
  same gamma (S-curve); residuals < 6e-9.
- LP conservation bounds: Y_N2 <= 0.7451; energy-conserving T bound 3010.35 K
  (900 K feed) and 3269.7 K (1200 K feed).
- Final main-study distances (dimensionless): C_to_B_max = 0.116 (fresh),
  0.173 (hot), 0.285 (adjusted); held-out behaves consistently. Zero failures.
  6 figures in `results/phase3/figures/`.

## Reproduction

On the server, from the repository root:

```bash
export PY=/data2/conda-envs/tr/bin/python
$PY scripts/phase0_smoke.py   --results-dir results/phase0
$PY scripts/phase1_reachability.py --results-dir results/phase1
$PY scripts/phase2_reactor_validate.py --results-dir results/phase2 --horizon 0.1
$PY scripts/phase3_envelopes.py --results-dir results/phase3            # main
$PY scripts/phase3_envelopes.py --results-dir results/phase3 --pilot     # pilot
$PY scripts/phase3c_splitting.py --results-dir results/phase3c
$PY -m pytest tests -q
```

Each run writes a `manifest.json` (timestamp, git revision, resolved config,
seed, mechanism provenance) alongside its results.

## Incident log — phase3c pass/fail criterion (2026-09-16)

The committed `results/phase3c/phase3c_results.json` was produced by
`scripts/phase3c_splitting.py` as of 14:02 server time, which classified each
case with a boolean `passed` requiring `err_T < 50 K` **and** finest-level
observed order >= 1.5. That criterion is too coarse and produced misleading
`passed=false` labels:

- `smooth_Tin900K_fresh` is **converged to reference precision** (err_T ~7e-9 K,
  i.e. at the 1e-12 reference-tolerance floor), so its observed "orders"
  (0.3-1.3) are noise, not a convergence failure.
- `const_g1000_fresh` (finest err 21.8 K) and the two `const_g10000_*` cases
  (finest err 202 / 127 K) are genuinely **pre-asymptotic**: at the finest tested
  dt = 7.8e-5 s the order is still climbing toward the formal value of 2
  (0.92 / 0.92 / 1.47), and has not yet levelled off.

The error and order *numbers* in that file are correct and reproducible; only
the boolean labelling is superseded. The script has since been edited to report
a `regime` field (`reference_precision` / `asymptotic_order2` /
`pre_asymptotic`) instead of the boolean, but the committed results were
generated by the earlier version, so re-running the current script will
reproduce the same `err_T` and `observed_order_T` values with different
`passed`/`regime` labels.

## Revision 2 — audit-driven correctness release (2026-09-16)

An independent audit of commit `1901996` found 15 defects (A01-A15) and refuted
claim C7. Every concrete defect was verified against the actual source before
repair; the two mathematical refutations were re-derived independently with this
repository's own propagator. Entries below record cause and fix.

### Publication incident (A01) — resolved
`research_log.md` committed the compute server's address, port and username,
contradicting the stated assurance. These were removed from `research_log.md`,
`reports/report_01.md` and the `host` fields of six result JSONs; the offending
commit was amended and force-pushed (tip `321e957`, branch `audit-revision`).
The prior commit is unreachable from `main`. No credentials or private keys were
present in any committed file. Subsequent runs write `host: redacted-host`.

### Bug A03 — switched exposure double-counted
`CSTR.gamma_integral` accumulated per segment with `+=`, so a later segment's
entry already contained the preceding contributions: gamma [1,2], durations
[1,1] gave Gamma(2) = 4 instead of 3. Fixed to the assignment form
Gamma(t) = sum_j gamma_j clip(t - t_j, 0, dt_j) with immutable segment starts.
Extracted to a Cantera-free `controls.gamma_integral_of` so it is unit-testable
without Cantera. The balance *laws* were never wrong; only the multi-segment
evaluator was. Masked until now because feed-compatible initial states have
b0 = b_in and h0 = h_in, so the wrong exponential factor multiplied zero.

### Bug A04 — ReactorNet switch timing drifted with the output grid
`seg_end` was derived from the previous *output* time, so the switch moved
forward as outputs advanced and, with frequent sampling, was never reached.
Fixed with precomputed immutable cumulative switch times, advancing exactly to
each discontinuity, updating both mass-flow controllers together, and calling
`ReactorNet.reinitialize()` at the switch. Verified: the solution no longer
depends on the output grid (11 vs 4001 points) and the switched history now
agrees with the custom RHS to < 1e-3 K. The old constant-control comparisons
were never affected.

### Bug A02 — phase3c crashed before writing results
`compare()` never created the `regime` field that `main()` read (a two-part edit
failed atomically and only the reader half applied). Fixed and now covered by
regression tests; the labelling distinguishes integration failure,
reference-precision, asymptotic-order-2, and pre-asymptotic regimes.

### Bug A06 — chemical invariants computed from the wrong matrix
The code reported the left nullspace of the elemental matrix E (trivial, dim 0)
and interpreted it as chemical invariants. The correct space is ker(B^T) with
B = diag(W)·nu, dimension 10 - 6 = 4, spanned by the ROWS of E. Recomputed,
cross-checked three ways, and unit-tested. The old field is retained but
renamed `_DEPRECATED`.

### Bugs A05/A07/A08/A10/A11/A12/A13/A14
- A05: state-setter diagnostics now run on the actual RHS path (raw sum-of-Y
  deviation, largest clipped negative, minimum raw component, eval count),
  reset per run.
- A07: `const_zero_seg` (gamma = 0) removed from the Phase 3 class where the
  declared bound is [10, 1e5]; an `AdmissibleControls` object now validates
  every structured, random, held-out and constant history before execution and
  raises on violation. Random sampling now covers the FULL declared range.
- A08: `C_to_B_max_trace_sensitive` was computed with `kind="engineering"`, i.e.
  it repeated the first metric. Both representations are now actually evaluated,
  with scales and floors recorded; a deterministic dispatch test pins them apart.
- A10: steady acceptance is residual-based (integration success no longer
  suffices); unresolved points are excluded from reference library A and kept
  separately.
- A11: the temperature scan now returns a feasible/infeasible BRACKET via
  h_min(T) = min_Y h_k(T).Y over the elemental polytope, not a grid maximum;
  the common thermodynamic range is corrected to [300, 3500] K with no 6000 K
  extrapolation; the exchange map raises a visible recorded error instead of
  silently returning a bracket endpoint.
- A12: salted `abs(hash(label)) % 1000` seed offsets replaced with stable
  hashlib child seeds; manifests record the git revision or a source-tree sha256
  (never empty); pilot and main runs get distinct manifest files.
- A13: extrema states and the A/B/C/HO analysis clouds are now persisted (NPZ),
  so distances are recomputable without rerunning trajectories.
- A14: the phase3c reference is evaluated by dense output at the exact
  comparison times rather than by linear interpolation of a stored grid.

### Test restructure (A15)
Toy/bookkeeping tests moved to `tests/test_toy.py` (Cantera-free, 19 tests);
Cantera-dependent tests moved to `tests/test_reactor.py` (14 tests). The
module-level `importorskip` no longer skips toy tests when Cantera is absent.
New regression tests cover the actual failure paths: switched exposure,
unequal durations, repeated switch times, ReactorNet scheduling on two grids,
switched ReactorNet-vs-custom-RHS agreement, admissibility bounds for every
structured family, metric dispatch, steady acceptance, invariant dimensions,
the temperature bracket, and the domain-status reporting.

### Scientific correction — claim C7 refuted
The visited set DOES depend on the control bound. Witness: gamma = 0 for 4 units
then gamma = 10 for 0.0707413427911034 reaches (0.5, 0.04939624612802243)
before t = 5; for G = 0.1 the conditional lower barrier
dW/dt|_(W=0) = (1-x)[(1+gamma)x - gamma] >= 0 on x >= x_G forces y >= 0.25 at
x = 0.5, excluding the witness at every time. The original
`envelope_area_fraction_visited` diagnostic was degenerate: the single
zero-control batch curve alone reproduces 0.129 / 0.933 / 1.000 while having zero
area. Replaced by a one-sided fill-distance diagnostic, which is G-sensitive
(covered fraction 0.085 vs 0.665 for G = 0.1 vs 10). See `results/phase1b` and
claim C7 (REVISED).

### New experiment — Phase 3d
Endpoint-sensitivity study (`scripts/phase3d_sensitivity.py`): the endpoint
Jacobian of an m-segment switched history, projected onto the tangent space of
the conserved manifold, compared by principal angles with the two-parameter
constant-control tangent span. This replaces the refuted coverage framing with a
direct dimension question; see claim C22 and `reports/report_02.md`.

### Tooling
Local `rsync` is now used for server sync. An early `rsync --delete` invocation
removed copies of local home-directory content that shared the server project
directory (tool caches/configs such as `.codex`, `.pi`, `.lark-cli`,
`AppData`, and two project folders). All originals are intact locally; no
research data was affected; `--delete` is no longer used.

## Revision 3 — sensitivity pipeline rebuilt; C22 withdrawn (2026-09-17)

A second independent review inspected commit `96d4bf2` and established that the
Phase 3D sensitivity pipeline could not support its own conclusions. Seven
defects were verified in the actual source:

1. `case()` passed the switched endpoint `z0` to `constant_control_tangent()`
   as that routine's initial state, so the "reference tangent span" was
   integrated from the wrong state and did not differentiate
   B(gamma, t; q_initial). Tangency was therefore never established.
2. The absolute perturbation `h = rel * mean(|theta|)` exceeded the admissible
   interval for the smallest control (30 - 114.5 < 10); the column was skipped
   and left as a ZERO, then included in the rank analysis as a fabricated zero
   sensitivity.
3. `direction_alignment()` filtered J but orthonormalized V by unfiltered QR,
   which fills a rank-deficient span with rounding-noise directions. The reported
   angles (0, 0.78, 0.85, 2.6e-8) were artifacts.
4. The "third direction" index was `n-3` with `n = m`, i.e. the SMALLEST value
   for m = 3, not the third largest.
5. State increments mixed kelvin with mass fractions and were differentiated
   w.r.t. raw gamma, so singular values had no single physical meaning.
6. The pulse-hold generator appended a remaining dwell and then sliced it away,
   producing horizons 0.044 s and 0.064 s for m = 3 and 4 despite the 0.1 s label.
7. Only a relative SVD threshold was used, so noise-floor directions
   (~1e-12 scaled) were counted as real, reporting "rank 3" after a hold.

Consequences: claim C22 is WITHDRAWN (marked SUPERSEDED) and replaced by C23.
The Phase 3D results are retained unmodified with a `DEPRECATED.json` manifest.

The corrected pipeline is `src/thermoreach/sensitivity.py` plus
`scripts/phase4_terminal_memory.py`, with positive controls in
`scripts/phase3e_positive_controls.py` and regression tests in
`tests/test_sensitivity.py` (23 tests, all passing).

Corrections applied, each with a regression test:
* Dimensionless controls `eta = log(gamma/gamma_ref)`; per-coordinate steps that
  stay strictly interior; a column that cannot be centred is INVALID (NaN), never
  zero, and is excluded from every rank summary. A round-trip bug that returned
  `exp(eta)` instead of `gamma_ref * exp(eta)` - a factor-1000 scaling error -
  was caught by the round-trip test, not by inspection.
* Reference tangents from the ORIGINAL initial state:
  v_t = F(q_base, gamma_star) and v_log_gamma = d phi/d log gamma, with the exact
  identity `sum_j dE_m/d(log gamma_j) = d phi/d log gamma` at constant-control
  base histories (verified to machine precision on the toy).
* Rank filtering of BOTH matrices by SVD with explicit relative AND absolute
  thresholds; zero-rank matrices yield an empty basis, never an arbitrary
  orthonormal completion.
* The conserved-manifold tangent space is the null space of `C W^-1` by a
  rank-revealing SVD with `full_matrices=True` (the reduced Vt of a wide matrix
  has only rank(A) rows and silently yields an EMPTY null space - caught by a
  test), with conditioning recorded and conservation leakage reported BEFORE
  projection.
* A declared frozen state scaling (one scaled unit = 100 K or 1% mass fraction);
  three separate rank notions: algebraic, relative-threshold, and
  absolute-threshold application rank.
* Exactly m durations summing to the recorded horizon; the third singular value
  at fixed index 2 for every m.
* An empirical differentiation-noise scale from step refinement
  `||J(h1) - J(h2)||_2`, labelled as an estimate, not a rigorous bound.
* `record_extrema=False` for endpoint-only work; the dense extrema grid was
  multiplying the cost of every perturbation.

Positive controls (section C of the review), all passing:
* The three-state linear benchmark `dz_i/dt = -lam_i z_i + u(t)`,
  `lam = (1, 2, 4)`, `T = 1`, three equal segments. Its exact endpoint control
  Jacobian has rank 3 and singular values (0.5007801, 0.08533145, 0.006828632),
  reproduced to 7 digits. This benchmark was reconstructed from the review's
  stated values alone (REVIEW.md and review_checks.py were NOT delivered with
  the handoff) and the reconstruction was confirmed by the exact match.
* The exact toy rank-2 history, plus a terminal hold showing that algebraic
  prefix rank is preserved by the invertible finite-time flow (2, through
  L = 10) while the application-scale effective rank decays (2 -> 0), and that a
  ratio-preserving relative threshold alone would keep reporting rank 2 at any
  scale.
* The same-initial-state identity for m = 1, 3, 5 and several gamma and T.
* Orthogonality and rank-deficient-reference angle tests: V = [e1, 0] versus
  J = [e2] gives sine 1, not 0.

An exact closed-form tangent-linear sensitivity was derived for the toy
(segment-map chain rule) and validated against finite differences at O(h^2);
this is the ground truth that the detailed-chemistry finite differences are
checked against.

The main experiment (section D) separates transient geometry from
terminal-memory erasure via
`E(theta_prefix, gamma_hold, L) = phi^L_{gamma_hold}(q_prefix(theta_prefix))`,
so `dE/dtheta_j = D_q phi^L . dq_prefix/dtheta_j`. See `reports/report_03.md`.

Tooling note: the tangent-linear (FSA) solve for the detailed-chemistry
reference derivative was implemented (`constant_control_fsa`, exact F_gamma in
closed form because the CSTR RHS is affine in gamma) but is too slow for
routine use; it is available behind `--fsa`. The default cross-integration check
compares Radau against LSODA on the same reference derivative instead.


## Revision 4 - memory calibration (branch `memory-calibration`)

Followed the review's Tasks 0-5. Task 0's `prior_handoff_verified/...` tree does
not exist anywhere (repo, handoff zips, Downloads, desktop); the three-state
benchmark was reconstructed from Task 1.1's explicit formula and reproduces the
stated singular values to 7 digits.

Targeted defects, each with a regression test:
* `linear3_exact_jacobian` returned `-G_spec` (sign error) - corrected, and an
  independently coded segment-restarted LSODA reference was added because a
  single-shot solve steps over the middle control segment and silently returns 0.
* `scaled_tangent_space` formed `null(C W)` instead of `null(C W^-1)`. With
  unequal temperature/mass-fraction scales this returns the wrong subspace
  always. Rewritten with row equilibration; conditioning 3.1e8 -> 4.5.
* Reference anchoring: dimensionless time parameter tau = t/T_ref so the time
  column carries the same units per unit parameter as the control columns; held
  endpoints anchored at T+L and flagged; the sum-of-columns identity marked
  applicable only for equal base histories.
* Threshold wording corrected: the 1e-6 scaled threshold is 1e-4 K and 1e-8 mass
  fraction per unit log-control step. `classify_ranks` separates machine /
  noise-resolved / application rank and gates full-rank claims on a complete
  stencil, and never lets a NaN column reach the SVD.

Experiments (results/phase5..8):
* Phase 5 - six unique anchors. The weak TOTAL direction is validated by replay
  across three perturbation sizes and two integrators. Its TRANSVERSE component
  sits between the manifold-leakage floor and the FD refinement discrepancy, so
  the FD estimator cannot decide it. The FD discrepancy itself is a refinement
  discrepancy, not a bound: 2.76e-6 here versus 2.8e-6... 2.8e-8 in Phase 4 from
  a different step pair.
* Phase 6 - matched gamma*T pairs REFUTE the Phase 4 mechanism: at fixed gamma*T
  = 1 the second singular value spans 7.4e-12 to 8.8e-3 (1.2e9-fold). What tracks
  it is T / t_ign: largest near 2x the ignition delay, collapsed by 200x.
* Phase 7 - [f_a, f_b] = (gamma_a - gamma_b)[r, v] because the RHS is affine in
  gamma. Same-level control gives exactly zero; at tau <= 1e-6 the log-log slope
  is 2.03 and the magnitude matches the closed form to 3%. Above the ignition
  delay the slope is ~0.9 and the effect saturates: the observable order
  dependence at chemically relevant times is ignition, not the bracket.
* Phase 8 - the exact tangent-linear estimator (no control-step noise floor,
  validated to 1.2e-14 against a matrix-exponential closed form at m = 1, 2, 3)
  resolves what the FD run could not: a transverse off-family direction in a
  narrow window, 1.2e-4 scaled units at (hot, gamma=1e4, T=1e-5), 4e4x above the
  leakage floor and 8e4x above the cross-integrator discrepancy. A window scan
  shows it decaying by six orders between T=1e-6 and T=3e-5 s, and a replay along
  the singular vector reproduces it while correctly failing to converge at the
  unresolved anchor.

The FSA is no longer too slow to use: the multi-segment `endpoint_jacobian_fsa`
makes it tractable by integrating the (state, sensitivity) pair in one system,
which costs one state-Jacobian FD pass per RHS evaluation regardless of segment
count. 76 tests pass on the server (34 in test_sensitivity alone).


## Revision 5 - flagship validation and a finite local patch (branch
`flagship-validation`)

The assignment was to retain and validate the single most consequential result
rather than sweep again. It survives.

Repairs, each with a regression test (46 in test_sensitivity, 88 in the whole
suite on the server):
* The tangent-linear estimator now integrates each segment as its own solve_ivp
  call with the segment index fixed inside the RHS closure, instead of one call
  with a searchsorted control selector (the parameter-forcing column changes
  discontinuously at every boundary). The review's exact benchmark reproduces
  the endpoint 1.09999 and sensitivities (0.499995, 0.1, 0.499995). Sensitivity
  absolute tolerances now map the per-state vector as np.repeat(atol_state, m)
  for the row-major (n, m) block.
* The replay perturbs along the leading transverse RIGHT singular vector of
  B = (I-P)A, comparing the signed projection, the cosine and the FULL vector -
  not along v2(A), which maximizes neither ||Bv|| nor sigma1(B) and gives zero on
  the review counterexample.
* Signed commutator: [F_a,F_b] = (gamma_b-gamma_a)[r,v]. Phase 7 had the sign
  backwards and the norm-only comparison concealed it.
* Ignition time by dense-output root finding, not a 400-point grid lookup that
  quantizes to T/399 and depends on the horizon.
* One rank-revealing reference-basis routine everywhere; an unresolved reference
  span yields transversality UNRESOLVED, never a fabricated projector.

Flagship (hot, gamma=1e4, horizon 1e-5): total (0.9484, 0.1189, 5.963e-5),
transverse 1.234e-4, and ||(I-P)u2|| = 1.000, so the third total direction is
entirely transverse. The state Jacobian is itself numerical, so it was varied
(eps ladder 1e-5/1e-6/1e-7) independently of the integrator and tolerance:
delta_A = 3.07e-9 and delta_P*||A|| = 1.91e-7 measured separately, combined
<= 1.94e-7 - a 635x margin over the claim, and 307x over sigma3(A), which the
exact inequality ||(I-P)A||_2 >= sigma3(A) makes reference-plane independent.
The replay converges (reldev 0.074 -> 0.000, cosine +1.0000, vector residual
1.3e-5 with O(eps^2) behaviour). The secondary anchor at horizon 1e-6 has an
8.6e5 margin. The fresh negative case is correctly unresolved: delta_A = 1.19e-3
dwarfs the signal and the replay diverges (cosine -0.996). A = WJ and D = WV are
persisted as NPZ so the patch is built on the validated matrices.

Local patch (phase 10): 32 finite excursions at radii 0.003-0.1. ||W R|| grows as
r^2 so the linear model holds; L = 1.60 (EMPIRICAL, not certified). Distance to
the CONTINUOUS constant-control family, by global grid + local refinement + replay
(an upper estimate, not a global exclusion): the in-family direction sits
essentially ON the curved family (gap closed ~180x) while the transverse
direction stays off it at 1.21e-5/1.26e-5 scaled units = 1.2e-3 K in both signs.
That is a genuine but application-small excursion - the honest conclusion, since
one scaled unit is 100 K.
