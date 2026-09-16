# Sources

Versions, access dates, and what each source supports. Documentation was consulted
for the *installed* version rather than assuming the moving `stable` pages match
local APIs. No research-paper citations were invented; where a paper-level claim
was considered, it was left un-cited rather than guessed.

## Software environment (actually used)

| Component | Version | Role |
|-----------|---------|------|
| Python | 3.12.14 (Miniconda) | interpreter |
| Cantera | 3.2.0 | thermodynamics, kinetics, ReactorNet cross-check |
| NumPy | 2.5.3 | arrays |
| SciPy | 1.18.1 | `solve_ivp` (Radau/BDF), `linprog`, `brentq`, `ConvexHull` |
| SymPy | 1.14.0 | symbolic identity checks |
| Matplotlib | 3.11.1 | figures |
| pandas | 3.0.5 | tabular reporting |
| pytest | 9.1.1 | test suite |

Mechanism: `h2o2.yaml` (10 species: H2, H, O, O, O2, OH, H2O, HO2, H2O2, AR, N2;
29 reactions; NASA polynomials valid 200–5000 K), sha256
`0efc6c52862741a2…` recorded in `results/phase0/environment.json`.
Path: `<env>/lib/python3.12/site-packages/cantera/data/h2o2.yaml`.

Environment audit: `results/phase0/environment.json`. The machine exposes 112
logical CPUs, no cgroup CPU/memory quota, and no GPU was used.

## Cantera documentation (consulted 2026-09-16, cantera.org)

- S1 — Constant-pressure reactor equations and temperature formulation.
  `https://cantera.org/stable/reference/reactors/constant-pressure-reactor.html`
  `https://cantera.org/stable/reference/reactors/ideal-gas-constant-pressure-reactor.html`
  *Supports:* the Phase 2A governing equations (`reactor.py` docstring).

- S2 — Combustor residence-time example and pressure-control implementation.
  `https://cantera.org/stable/examples/python/reactors/combustor.html`
  *Supports:* the decision NOT to treat the pressure-controller example as an
  exact algebraically-fixed-pressure implementation; matched in/out
  `MassFlowController`s on an `IdealGasConstPressureReactor` were used instead
  (`reactornet_check.py`, claim C16).

- S3 — Properties/kinetics with an external SciPy ODE integrator (custom ODE example).
  `https://cantera.org/stable/examples/python/reactors/custom.html`
  *Supports:* implementation reference for the transparent custom RHS.

- S4 — Transport models, including the separate unity-Lewis-number model.
  `https://cantera.org/stable/reference/transport/index.html`
  *Relevance:* Phase 4+ (not executed in this tranche). Recorded so the
  unity-Lewis vs mixture-averaged distinction is not silently conflated later.

- S5 — Counterflow diffusion-flame continuation with two-point flame control.
  `https://cantera.org/stable/examples/python/onedim/diffusion_flame_continuation.html`
  *Relevance:* Phase 4+ continuation reference (not executed).

- S6 — 1-D nonlinear solver and its time-stepping procedure.
  `https://cantera.org/stable/reference/onedim/nonlinear-solver.html`
  *Relevance:* the caution that steady-solver internal time steps are not
  physical strain-response data (Phase 5, not executed).

- S7 — Thermodynamic state definitions and reactor tutorial.
  `https://cantera.org/stable/python/thermo.html`
  `https://cantera.org/stable/userguide/reactor-tutorial.html`
  *Supports:* `Solution` state accessors, `set_equivalence_ratio`,
  `equilibrate('HP')`, `partial_molar_enthalpies`, `net_production_rates`.

Cantera 3.2.0 API notes discovered during implementation (installed-version
behaviour): `MassFlowController` uses the `mass_flow_rate` property, not
`set_mass_flow_rate`; `ReactorNet.advance(t)` integrates to a specified time;
`get_data_directories()` can return `'.'` as its first entry, so mechanism files
must be searched across all returned directories.

## Provided handoff bundle

`thermochemical_reachability_handoff/` — the research brief
(`LOCAL_AGENT_PROMPT.md`), the start message, and
`thermochemical_compute_check.zip` (historical smoke tests executed in a
4-core / 4 GiB chat sandbox with **no Cantera**; Python 3.13.5).

- The historical `results.json` was preserved unmodified and was **not**
  relabelled as a new local run.
- The smoke tests were re-executed freshly on the server as
  `results/phase0/smoke_results.json`, with the equations reproduced from the
  brief rather than copied blindly, plus additional audit checks (steady locus,
  batch trajectory).

## Literature (topics identified, not cited as specific claims)

Attainable-region theory; invariant regions of reaction–diffusion systems;
controlled reachability; flamelet continuation; chemical-manifold reduction.
A broad literature survey was deliberately not allowed to displace the first
computational tranche; no specific paper titles are asserted here.
