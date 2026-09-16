# ThermoReachability

Code, script results, and logs for a study of **thermochemical reachability**:
given initial conditions, feed/boundary conditions, operating constraints and a
time horizon, which thermochemical states can a reacting system reach — and can
we construct defensible inner estimates and outer bounds without enumerating
every history?

Research specification: `thermochemical_reachability_handoff/LOCAL_AGENT_PROMPT.md`.

**First tranche (Phases 0–3) is complete.** See `reports/report_01.md` (original),
`reports/report_02.md` (first audit revision), `reports/report_03.md` (sensitivity
pipeline rebuild) and **`reports/report_04.md` (memory calibration — read this for
the current sensitivity claims)**. `claims.md` is the claim ledger, including every
claim retracted or narrowed after audit: **C22 is withdrawn** (superseded by C23),
and **C23's mechanism is refuted in Revision 4** (C25: exchange exposure does not
control the weak direction; C26 reports a resolved off-family direction found with
the exact tangent-linear estimator; C27 confirms the closed-form Lie bracket).
Phases 4–6 (counterflow flamelet continuation, reduced flamelet transients,
reaction-invariant enclosures) remain deferred by design.

## What is here

| Path | Contents |
|------|----------|
| `src/thermoreach/` | Package: toy reactor + exact propagator, control histories, detailed-chemistry CSTR, Cantera cross-check, envelopes/metrics, Strang splitting |
| `scripts/` | Executable experiments, Phases 0–4 (the reproducible source of truth) |
| `tests/` | pytest suite: `test_toy.py`, `test_sensitivity.py` (34), `test_reactor.py`, `test_cli_smoke.py` — 76 passing on the server |
| `results/<phase>/` | Raw results, manifests, figures; `results/phase3d/DEPRECATED.json` marks the superseded study |
| `reports/report_01.md` | First report |
| `reports/report_02.md` | First audit revision report |
| `reports/report_03.md` | Sensitivity rebuild + terminal-memory experiment |
| `reports/report_04.md` | **Memory calibration: matched exchange exposure, pulse-order bracket, exact tangent-linear transverse spectrum (read first for current claims)** |
| `claims.md` | Claim ledger with assumptions, status and evidence pointers |
| `sources.md` | Sources, versions, access dates |
| `research_log.md` | Chronological log, including bugs found and fixed |

## Layout of the science

1. **Phase 0** — environment audit; smoke tests re-executed locally (exact
   propagator vs Radau/BDF, invariant barrier, stiff-ODE cross-method).
2. **Phase 1** — isothermal A→B→C toy: analytical audit plus a bounded-control
   reachability study (γ ∈ [0, G], horizons, switching/optimized histories).
3. **Phase 2** — adiabatic fixed-pressure detailed-chemistry CSTR (h2o2.yaml),
   validated by exact conservation balances and an independent Cantera
   ReactorNet cross-check.
4. **Phase 3** — empirical envelopes: steady family vs constant-control
   transients vs switching histories, with distance metrics, LP conservation
   bounds and a counterexample search.
5. **Phase 3C** — Strang splitting of the validated CSTR (solver-interface study).

## Headline results

- The toy's **visited** set is insensitive to the control bound, because the
  zero-control trajectory traces the analytical envelope; only the **terminal**
  set distinguishes control bounds.
- The detailed-chemistry CSTR is validated to ≤ 4e-7 K and ≤ 3e-11 in species
  against an independent Cantera integration pathway.
- Within the declared admissible class, **switching histories do not reach
  states far outside the constant-control transient library** at the studied
  condition. An apparent excursion was traced to a control-bound violation —
  see `research_log.md` and claim C20.
- At the default condition a fresh feed does **not** ignite within 0.1 s; a
  separately labelled pilot-adjusted condition (T_in = 1200 K) is included.

## Reproduction

Code is developed here and executed on the compute server. To reproduce on a
machine with Cantera 3.2 and a Python ≥ 3.10 environment:

```bash
python -m pip install -e .            # optional; scripts add src/ to the path
python scripts/phase0_smoke.py   --results-dir results/phase0
python scripts/phase1_reachability.py --results-dir results/phase1
python scripts/phase2_reactor_validate.py --results-dir results/phase2 --horizon 0.1
python scripts/phase3_envelopes.py --results-dir results/phase3            # main
python scripts/phase3_envelopes.py --results-dir results/phase3 --pilot     # pilot
python scripts/phase3c_splitting.py --results-dir results/phase3c
python -m pytest tests -q
```

Each run writes a `manifest.json` with timestamp, git revision, resolved
configuration, seed and mechanism provenance (path + sha256).

## Honest scope

All numerical statements are traceable to saved outputs. Distances are lower
estimates; optimization optima are achieved objectives, not certified suprema;
no interval-validated outer bounds are provided. Solver agreement checks
integration consistency, not chemical-mechanism validity. A failed simulation is
not evidence that a state is physically unreachable.
