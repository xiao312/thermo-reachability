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
