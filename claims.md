# Claim ledger — thermochemical reachability

Each claim lists assumptions, status, and pointers to evidence. Statuses:
`proved`, `numerically supported`, `conjectured`, `refuted`, `not tested`.

Numerical support states the tested scope. Solver agreement is *not* a rigorous
error enclosure. A failed simulation is not evidence that a state is physically
unreachable.

---

## Phase 1 — toy A→B→C (isothermal, k1=k2=1, pure-A feed)

**C1. Conservation triangle invariance.** For gamma >= 0 and
(x0,y0) = (1,0), the triangle {x>=0, y>=0, x+y<=1} is invariant.
Status: *proved* (the RHS points inward on each face; verified symbolically and
numerically). Evidence: `tests/test_thermoreach.py::test_conservation_triangle_invariance`;
`results/phase1/phase1_results.json` `min_fraction_search` ~ 1e-16.

**C2. Steady-state locus.** Constant-gamma steady states are
x* = gamma/(1+gamma), y* = gamma/(1+gamma)^2 and lie on y = x(1-x).
Status: *proved* (algebraic; SymPy-verified in `scripts/phase0_smoke.py`).
Evidence: `tests::test_steady_locus`.

**C3. Batch trajectory.** With gamma = 0, the trajectory through (1,0) is
y = -x log x. Status: *proved*. Note this *coincides* with the analytical
upper curve (C6), which is essential to interpreting C7.

**C4. Exact propagator.** The constant-control map in `toy.exact_segment` solves
the linear ODE exactly. Status: *proved*, and *numerically supported* against
independent Radau and BDF runs over gamma in {0, 1e-3, 1, 10, 100}, horizons up
to 3, and initial states near all boundaries: max error < 1e-8
(Phase 0 smoke runs: ~7e-11). Evidence: `results/phase0/smoke_results.json`,
`results/phase1/*`, `tests::test_exact_propagator_matches_ode`.

**C5. Barrier invariance.** V(x,y) = y + x log x <= 0 along all admissible
piecewise-constant histories, with the boundary condition
dV/dt|_(V=0) = gamma (log x + 1 - x) <= 0 for gamma >= 0, 0 < x <= 1
(by log x <= x - 1), using the continuous extension of x log x at x = 0.
Status: *proved* for the boundary derivative; the *global* invariance
additionally requires the other two faces (C1) and the vertex (0,0), where V=0
is attained by extension. Numerically supported: max positive violation ~1e-16
over all Phase 1 histories. Evidence: `tests::test_barrier_*`,
`results/phase1/phase1_results.json` `max_barrier_search`.
Scope: piecewise-constant gamma in [0, 10], horizons <= 20, <= 32 segments.
Not tested: measurable (non-piecewise-constant) controls — the analytic
argument is pointwise in gamma(t), so it extends, but this was not formalized.

**C6. Unrestricted enclosure.** With gamma >= 0, unlimited duration and
arbitrarily rapid idealized feed replacement, the closure of the reachable set is
{0 <= x <= 1, 0 <= y <= -x log x}.
Status: *numerically supported as a containing bound* (C5 gives the upper face;
C1 the others). **Attainability is only partially established**: the upper face
is *exactly* attained by the gamma = 0 batch trajectory (C3) in infinite time;
interior points require a reaction-then-replace construction whose precise
limits (infinite time or unbounded feed rate) were not separately formalized.
Evidence: `results/phase1` `envelope_area_fraction_visited` → 1.000 at t_f = 20.

**C7 (the main Phase 1 negative result).** The *visited* set R_[0,t_f] is
essentially insensitive to the control bound G, because gamma = 0 is admissible
for every G and the zero-control trajectory traces the envelope's upper
boundary. Therefore G-restriction is invisible in visited-set diagnostics; only
the *terminal* set R(t_f) and the horizon distinguish cases.
Status: *numerically supported* (G in {0.1, 1, 10}; visited y_max = 0.3679 = 1/e
for all G, while terminal y_max = 0.1009 / 0.2502 / 0.3520 at t_f = 5).
Evidence: `results/phase1/phase1_results.json`.
Interpretive caution: this is specific to the *pure-A feed* toy. It does not
license transferring the envelope to bounded controls or finite time in general.

**C8. Monotone inclusion under class enlargement.** Enlarging G and/or the
horizon cannot shrink the reachable set; the numerical search reproduced
non-decreasing visited extent in all tested comparisons.
Status: *numerically supported*. Note: a failure of a numerical search to show
inclusion would indicate inadequate search, not refute the set-theoretic
inclusion. Evidence: `results/phase1` `inclusion_checks`.

**C9. Search-resolution sensitivity.** Terminal-set extent depends on the number
of control segments and on continued search; 4→16 segments changed the terminal
projected area (0.0485→0.0557 at G=1, t_f=5), and results are noisy rather than
monotone. Status: *numerically supported*. Any terminal-set "boundary" is
therefore a lower estimate of the true supremum, not a certified bound.

---

## Phase 2 — detailed-chemistry CSTR (h2o2.yaml, 10 species, 29 reactions)

**C10. Both energy forms agree.** cp dT/dt = -Σ h_k r_k + γ(h_in - Σ h_k(T) Y_in,k)
and cp dT/dt = γ(h_in - h) - Σ h_k dY_k/dt are algebraically identical;
verified to relative roundoff (< 1e-9) at representative admissible states and
γ in {0, 10, 1e3, 1e4}. Status: *proved + numerically supported*.
Evidence: `results/phase2/phase2_results.json::energy_forms`, `tests::test_energy_forms_agree`.

**C11. Elemental matrix.** E[e,k] = n_atoms(e,k) W_e / W_k has unit column sums
and E.r = 0 to roundoff (relative < 1e-9). Status: *proved + numerically supported*.
Evidence: `results/phase2::elemental`, `tests::test_elemental_matrix_normalization`,
`tests::test_elemental_source_conservation`.

**C12. Exact open-reactor balances.** b(t) = b_in + e^{-Γ}(b0-b_in) and
h(t) = h_in + e^{-Γ}(h0-h_in) hold regardless of chemistry, for matched
in/out flows at rate gamma. Status: *proved*; numerically verified including a
*nontrivial* case with T_in = 1000 K ≠ T0 = 1100 K and a different initial
composition (b0 ≠ b_in, h0 ≠ h_in): relative residuals < 1e-8 (elements) and
< 1e-6 (enthalpy). Evidence: `results/phase2::balance_checks`,
`tests::test_exact_mixing_balances_nontrivial`.
Scope note: the fresh-start and HP-hot-start cases have b0 = b_in and h0 = h_in
*by construction*, so they are consistency checks only; this is recorded per-case
in the results under `trivial_consistency_case`.

**C13. Closed-reactor energy conservation.** With gamma = 0, total enthalpy is
conserved to relative ~1e-9 over 1 ms from a 1200 K random composition.
Status: *numerically supported*. Evidence: `results/phase2::closed_energy`.

**C14. Cross-integration agreement.** Radau and BDF agree to < 1e-6 K and
< 3e-11 in species on the tested cases (γ in {10, 1e3, 1e5}, fresh and hot
starts, t_f = 0.1 s), and both agree with a tighter-tolerance reference.
Status: *numerically supported*. This checks integration consistency, not
mechanism validity. Evidence: `results/phase2::cross_integration`.

**C15. Independent ReactorNet cross-check.** The custom SciPy-integrated RHS
agrees with a separately configured Cantera `ReactorNet`
(IdealGasConstPressureReactor, matched inlet/outlet MassFlowControllers,
constant gamma) to ≤ ~4e-7 K and ≤ ~3e-11 in species over γ in {10, 1e2, 1e3,
1e4}, fresh and hot starts. Status: *numerically supported*.
Evidence: `results/phase2::reactornet_crosscheck`.
This validates the two independent integration pathways against each other; it
is not independent validation of the chemical mechanism.

**C16. Reactor realization.** Constant pressure + constant mass + variable volume
with matched in/out flows is the realization used throughout; fixed p, fixed m
and fixed V cannot all be imposed independently during arbitrary transients.
Status: *proved* (balance derivation) and *consistently implemented*
(C12, C15). The Cantera pressure-controller example was NOT assumed to be an
exact algebraically-fixed-pressure implementation; the matched-flow
IdealGasConstPressureReactor construction was used instead.

**C17. State-setter policy.** Y is clipped at zero only inside the Cantera state
setter (which requires a physical composition); raw accepted solver states are
preserved, and the largest Y renormalization applied by the setter is recorded
per run (`max_state_renorm`, ~0 in all runs). No abort-on-Newton-excursion logic
is hidden in the RHS. Status: *numerically supported*.

---

## Phase 3 — empirical detailed-reactor envelopes

**C18. Default fresh-feed study shows weak reaction.** At the brief's defaults
(h2o2, 1 atm, T_in = 900 K, phi = 1, gamma in [10, 1e5] s^-1, t_f = 0.1 s) a
fresh start does not ignite: temperature rises < 0.3 K and Y_OH stays ~1e-10.
Status: *numerically supported*; reported rather than hidden. A separately
named pilot-adjusted condition (T_in = 1200 K) was added where ignition is
observable; the original result is preserved.
Evidence: `results/phase3` `studies.fresh`.

**C19. Steady family is an S-curve with multiple steady states.** Time-marching
from cold (fresh) and hot (HP-equilibrium) initializations gives different
steady states at the same gamma where both converge, e.g. at gamma = 1e5 the
cold branch terminates at the unreacted feed (~900 K) and the hot branch at a
burning state (~1550 K). Steady residuals < 5e-9 on all traced points.
Status: *numerically supported*. A converged root is a mathematical solution,
not evidence of accessibility within the horizon. Extending steady integration
beyond the horizon does not extend the finite-time study.
Evidence: `results/phase3` `steady_family`, `steady_residual_max`.

**C20 (the main Phase 3 negative result).** Within the admissible class
gamma in [10, 1e5] s^-1 and t_f = 0.1 s, switching histories do NOT reach states
far outside the constant-control transient library from the same initial
condition and horizon: all switching trajectories from a hot start remain at
T >= 1550 K, i.e. on the burning branch that constant controls also occupy.
Status: *numerically supported (negative result)*.
Evidence: `results/phase3` `distances.C_to_B_max`; direct diagnostic in
`research_log.md`.
**Important audit note:** an *apparent* large excursion to 900 K was initially
observed, then traced to a control-generation bug that permitted gamma up to
1e6, i.e. outside the declared admissible class. After enforcing gamma <= 1e5,
the excursion disappeared. This is recorded as a methodological finding: an
unverified control bound can manufacture a false "switching reaches new states"
result.

**C21. LP conservation bounds are loose.** Maximizing Y_k by linear programming
under nonnegativity, normalization and elemental constraints gives
Y_N2 <= 0.7451 (the only binding nontrivial bound) and an energy-conserving
temperature bound of 3010.35 K for the 900 K feed (3269.7 K for the 1200 K feed).
Status: *numerically supported as a relaxation*. These ignore kinetics and
(partially) energy feasibility and are labelled as such; equilibrium temperature
was NOT assumed to be a universal attainable maximum.
Evidence: `results/phase3` `lp_bounds`.

---

## Not tested / deferred

- Certified (interval/validated) outer bounds; all bounds here are analytic
  (C5, C12) or LP relaxations (C21). Ordinary optimization optima are labelled
  as achieved objectives, not guaranteed maxima.
- Attainability of every interior point of the toy enclosure (C6).
- Phases 4–6 (counterflow flamelet continuation, reduced flamelet transients,
  reaction-invariant convex enclosures): deferred by design until the first
  report, per the brief.
- Counterflow/flamelet transport (unity-Lewis vs mixture-averaged): not reached.
