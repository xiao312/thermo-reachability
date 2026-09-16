# Claim ledger — thermochemical reachability

Each claim lists assumptions, status, and pointers to evidence. Statuses:
`proved`, `numerically supported`, `conjectured`, `refuted`, `not tested`,
`inconclusive`.

Numerical support states the tested scope. Solver agreement is *not* a rigorous
error enclosure. A failed simulation is not evidence that a state is physically
unreachable.

> **Revision 2 (2026-09-16, post-audit).** This ledger was revised after an
> independent audit of commit `1901996`. The audit refuted **C7**, found
> **C17** unsupported by its then-inactive instrumentation, and found **C20**
> inconclusive as a coverage claim. **C19** was narrowed (no unstable-branch
> continuation was performed) and **C21** was corrected (a grid maximum is not
> an upper bound; the thermodynamic range is [300, 3500] K, not [200, 5000] K).
> Superseded claims retain their number with a REVISED marker so report_01's
> provenance stays readable. The audit's own dispositions are recorded inline.

---

## Phase 1 — toy A→B→C (isothermal, k1=k2=1, pure-A feed)

**C1. Conservation triangle invariance.** For gamma >= 0 and
(x0,y0) = (1,0), the triangle {x>=0, y>=0, x+y<=1} is invariant.
Status: *proved* (the RHS points inward on each face; verified symbolically and
numerically). Evidence: `tests/test_toy.py::test_exact_propagator_steady_and_barrier`;
`results/phase1/phase1_results.json` `min_fraction_search` ~ 1e-16.

**C2. Steady-state locus.** Constant-gamma steady states are
x* = gamma/(1+gamma), y* = gamma/(1+gamma)^2 and lie on y = x(1-x).
Status: *proved* (algebraic; SymPy-verified in `scripts/phase0_smoke.py`).
Evidence: `tests/test_toy.py`.

**C3. Batch trajectory.** With gamma = 0, the trajectory through (1,0) is
y = -x log x. Status: *proved*. Note this *coincides* with the analytical
upper curve (C6) — this is exactly what made the C7 diagnostic degenerate.

**C4. Exact propagator.** The constant-control map in `toy.exact_segment` solves
the linear ODE exactly. Status: *proved*, and *numerically supported* against
independent Radau and BDF runs over gamma in {0, 1e-3, 1, 10, 100}, horizons up
to 3, and initial states near all boundaries: max error < 1e-8
(Phase 0 smoke runs: ~7e-11). Evidence: `results/phase0/smoke_results.json`,
`tests/test_toy.py`.
The audit independently integrated the C7 witness with Radau and BDF and agreed
with this propagator to ~1.3e-15 and ~1.4e-11 respectively.

**C5. Barrier invariance (upper face).** V(x,y) = y + x log x <= 0 along all
admissible piecewise-constant histories, with
dV/dt|_(V=0) = gamma (log x + 1 - x) <= 0 for gamma >= 0, 0 < x <= 1
(by log x <= x - 1), using the continuous extension of x log x at x = 0.
Status: *proved* for the boundary derivative; the *global* invariance
additionally requires the other two faces (C1) and the vertex (0,0), where V=0
is attained by extension. Numerically supported: max positive violation ~1e-16.
Scope: piecewise-constant gamma in [0, 10], horizons <= 20, <= 32 segments.
Not tested: measurable (non-piecewise-constant) controls — the analytic argument
is pointwise in gamma(t), so it extends, but this was not formalized.

**C6. Unrestricted enclosure.** With gamma >= 0, unlimited duration and
arbitrarily rapid idealized feed replacement, the closure of the reachable set is
{0 <= x <= 1, 0 <= y <= -x log x}.
Status: REVISED to *proved under the stated closure assumptions*.
The upper face is contained by C5 and *exactly attained* by the gamma = 0 batch
trajectory (C3). The interior is attained by the reaction-then-replace
construction supplied by the audit: age pure A for s >= -log x giving
(a, sa) with a = e^-s, then mix with fresh A using fresh fraction
lambda = (x-a)/(1-a); the resulting intermediate concentration is
y(s) = (1-x) s / (e^s - 1), which decreases continuously from -x log x (at
s = -log x) to 0 (as s -> inf), covering the interior. Arbitrarily rapid
idealized feed replacement is approached as a limit of increasing finite
exchange rates over shrinking durations.
This does NOT transfer to fixed G or a finite horizon — see C7.

**C7. REVISED — the visited set depends on the control bound.**
The original claim ("the visited set is essentially insensitive to G because
gamma = 0 is admissible for every G") is **REFUTED** as a statement about sets.
A common maximum of y does not imply identical sets.

*Explicit counterexample.* gamma = 0 for 4 time units, then gamma = 10 for
delta = (1/11) log((10/11 - e^-4)/(10/11 - 1/2)) = 0.0707413427911034 reaches
(x, y) = (0.5, 0.04939624612802243) at t = 4.0707 < 5. Verified with this
repository's own exact propagator in `scripts/phase1b_toy_correction.py`.

*Exclusion for smaller bounds.* With W = y - x(1-x) and x_G = G/(1+G),
dW/dt|_(W=0) = (1-x)[(1+gamma)x - gamma] >= 0 for all 0 <= gamma <= G whenever
x >= x_G, and a state that crosses below x_G cannot cross upward again. Hence
every reachable state with x >= x_G satisfies y >= x(1-x). For G = 0.1,
x_G = 0.0909, so at x = 0.5 any reachable state has y >= 0.25 — the witness
(y = 0.0494) is excluded **at every time**, not merely before t = 5. For
G = 10, x_G = 0.909 > 0.5 and the barrier does not apply, so the witness IS
reachable. Status: *proved* (the exclusion is analytic; the witness is exact).

*Why the original diagnostic misled.* The reported
`envelope_area_fraction_visited` values 0.129 / 0.933 / 1.000 at horizons
1 / 5 / 20 are the convex-hull area of the SINGLE zero-control batch curve,
whose own two-dimensional area is zero. A one-dimensional curve cannot cover
interior area, but its convex hull fills most of the envelope. The corrected
diagnostic is a one-sided fill distance over an interior grid
(`search.interior_coverage`), which is G-sensitive: at horizon 5 the covered
fraction is 0.085 (G = 0.1) versus 0.665 (G = 10), with fill distance 0.34
versus 0.073, while the deprecated hull fraction is 0.93 for both.
Status of the *replacement* claim (the diagnostics were degenerate): *proved
and numerically supported*. Evidence: `results/phase1b`, `tests/test_toy.py`.

**C8. Monotone inclusion under class enlargement.** Enlarging G and/or the
horizon cannot shrink the reachable set; the numerical search reproduced
non-decreasing visited extent in all tested comparisons.
Status: *numerically supported*; the set-theoretic statement needs no proof from
the search. A search failure would indicate inadequate search, not refute it.

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
Evidence: `results/phase2/phase2_results.json::energy_forms`,
`tests/test_reactor.py::test_energy_forms_agree`.

**C11. Elemental matrix and chemical invariants.** E[e,k] = n_atoms(e,k) W_e/W_k
has unit column sums and E.r = 0 to roundoff (relative < 1e-9). REVISED: the
chemical invariants of the source are ker(B^T) with B = diag(W)·nu, dimension
n_species - rank(B) = 10 - 6 = **4**, and are spanned by the ROWS of E
(elemental mass-fraction vectors). The earlier code reported the left nullspace
of E itself — which is trivial (dim 0) for this mechanism — and attached a
chemical interpretation to it; that was wrong and is deprecated.
Status: *proved + numerically supported* (rank(nu) = 6, invariant dim = 4,
rows of E annihilate B and span the invariant space; all unit-tested).
Evidence: `results/phase2::elemental.chemical_invariants`,
`tests/test_reactor.py::test_chemical_invariants_dim_and_elemental_span`.

**C12. Exact open-reactor balances, including switched histories.**
b(t) = b_in + e^{-Γ}(b0-b_in) and h(t) = h_in + e^{-Γ}(h0-h_in) hold regardless
of chemistry, for matched in/out flows at rate gamma, with
Γ(t) = Σ_j gamma_j clip(t - t_j, 0, Δt_j).
Status: *proved*; numerically verified for a *nontrivial* initial state
(b0 ≠ b_in, h0 ≠ h_in) under BOTH a constant and a three-segment switched
history (relative residuals < 1e-8 elements, < 1e-6 enthalpy).
REVISED: the multi-segment case was previously masked by a double-counting bug
in Γ(t) (fixed; see `research_log.md`) that feed-compatible initial states
render invisible because the multiplying initial difference vanishes.
Evidence: `results/phase2::balance_checks`,
`tests/test_reactor.py::test_exact_mixing_balances_nontrivial`,
`test_exact_balances_switched_nontrivial`.

**C13. Closed-reactor energy conservation.** With gamma = 0, total enthalpy is
conserved to relative ~1e-9 over 1 ms from a 1200 K random composition.
Status: *numerically supported*. Evidence: `results/phase2::closed_energy`.

**C14. Cross-integration agreement.** Radau and BDF agree to < 1e-6 K and
< 3e-11 in species on the tested cases (γ in {10, 1e3, 1e5}, fresh and hot
starts, t_f = 0.1 s). Status: *numerically supported*. This checks integration
consistency, not mechanism validity. Evidence: `results/phase2::cross_integration`.

**C15. Independent ReactorNet cross-check, including switched controls.**
The custom SciPy-integrated RHS agrees with a separately configured Cantera
`ReactorNet` to ≤ ~4e-7 K and ≤ ~3e-11 in species over γ in {10, 1e2, 1e3, 1e4},
fresh and hot starts. REVISED: a switched-history agreement check was added
(two-segment γ = [100, 2000], endpoint agreement < 1e-3 K over 201 output times)
after the scheduling bug (A04) was fixed. The switch now occurs at immutable
cumulative times and the result no longer depends on the output sampling
(verified on grids of 11 and 4001 points).
Status: *numerically supported*. Evidence: `results/phase2::reactornet_crosscheck`,
`tests/test_reactor.py::test_reactornet_switch_occurs_on_coarse_and_fine_grids`,
`test_reactornet_matches_custom_rhs_switched`.

**C16. Reactor realization.** Constant pressure + constant mass + variable volume
with matched in/out flows is the realization used throughout; fixed p, fixed m
and fixed V cannot all be imposed independently during arbitrary transients.
Status: *proved* (balance derivation) and *consistently implemented*
(C12, C15). The Cantera pressure-controller example was NOT assumed to be an
exact algebraically-fixed-pressure implementation; the matched-flow
IdealGasConstPressureReactor construction was used instead.

**C17. REVISED — raw-state instrumentation.** The earlier claim that the Y
clipping/renormalization policy is auditable rested on a counter
(`max_state_renorm`) that the RHS never incremented, because `rhs`/`rhs_alt` set
`gas.TPY` directly and bypassed `state()`. **That evidence was inactive.**
The instrumentation now runs on the actual state-setting path and records, per
run and reset: raw sum-of-Y deviation, largest clipped negative component,
minimum raw component, and the thermo-evaluation count
(`raw_state_diagnostics` in every integrate result).
Status: *numerically supported* (the policy is now actually measured).
Evidence: `results/phase2::balance_checks.*.raw_state_diagnostics`,
`tests/test_reactor.py::test_rhs_diagnostics_record_raw_state_quality`.

---

## Phase 3 — empirical detailed-reactor envelopes

**C18. Default fresh-feed study shows weak reaction.** At the brief's defaults
(h2o2, 1 atm, T_in = 900 K, phi = 1, gamma in [10, 1e5] s^-1, t_f = 0.1 s) a
fresh start does not ignite: temperature rises < 0.3 K and Y_OH stays ~1e-10.
Status: *numerically supported for the sampled histories* (no universal
quantification over all admissible histories is claimed); reported rather than
hidden. A separately named pilot-adjusted condition (T_in = 1200 K) was added
where ignition is observable; the original result is preserved.
Evidence: `results/phase3` `studies.fresh`.

**C19. REVISED — multistability evidence, not a complete S-curve.**
Time-marching from cold (fresh) and hot (HP-equilibrium) initializations gives
different long-time states at the same gamma where both settle, e.g. at
gamma = 1e5 the cold branch terminates at the unreacted feed (~900 K) and the
hot branch at a burning state (~1550 K). This is evidence of **distinct
long-time branches / multistability**. It is NOT a complete S-curve
reconstruction: no unstable branch was continued and no fold was located.
Steady acceptance is now residual-based (integration success alone no longer
suffices), and points failing the residual threshold are excluded from the
reference library A and retained separately.
Status: *numerically supported (multistability only)*.
Evidence: `results/phase3` `steady_family`, `steady_selection`,
`tests/test_reactor.py::test_steady_acceptance_requires_small_residual`.

**C20. REVISED — inconclusive as a coverage claim.** The original statement
("switching histories do not reach states far outside the constant-control
transient library") is **inconclusive** and is downgraded to a narrowly scoped
observation of the sampled histories. Three reasons:
 (i) *Candidate coverage, not reference resolution, is the binding limit.* The
     sampled set (43 histories on 8 equal 0.0125 s segments, randomized over
     only [100, 1e5] of the declared [10, 1e5] range) cannot support a universal
     quantifier over the admissible class. The sampling law now covers the full
     range, but one corrected sweep does not establish coverage.
 (ii) *Distance direction.* For a sampled reference B_s ⊂ B,
      d(q, B_s) >= d(q, B), so a SMALL measured distance does upper-bound the
      true distance (the "close to the family" direction is sound for sampled
      candidates), but a LARGE distance does not certify excursion — sparsity
      alone inflates it. The joint dimensionless distance is no longer expressed
      as a "Kelvin equivalent".
 (iii) The advertised detailed-reactor optimizer was never implemented; the
      workflow went from trajectory studies straight to LP bounds and figures.
 What remains supported: for the *sampled* histories, nearest-reference
 distances in both metrics are recorded with their scales and floors, and the
 corrected study reports reference resolution and candidate coverage separately.
 Status: *inconclusive*; superseded by the Phase 3d sensitivity approach.
 Evidence: `results/phase3` `metric_scales`, `distances`, `admissibility`.

**C21. REVISED — grid-feasible temperature estimates, not upper bounds.**
The routine now reports a FEASIBLE-TEMPERATURE BRACKET: a state with h = h_in at
temperature T requires h_min(T) <= h_in, where
h_min(T) = min_Y h_k(T)·Y over the elemental polytope; under the
positive-heat-capacity assumption the feasible set is an interval whose upper
edge is bracketed by the sign change of h_min(T) - h_in. The earlier "highest
feasible grid point" (3010.35 K at 900 K feed; 3269.7 K at 1200 K feed) is an
attained value in the relaxed problem, NOT a containing upper bound — the audit
constructed feasible states slightly above it (continuous relaxed boundary
~3010.95 K and ~3270.72 K). The bracket contains the true continuous maximum;
it is an ordinary numerical bracket, not an interval-certified bound.
Also corrected: the mechanism's common thermodynamic range is **[300, 3500] K**
(the earlier [200, 5000] K was the union of per-species endpoints), the scan no
longer extrapolates to 6000 K, and the exchange map now raises a visible,
recorded error rather than silently returning a bracket endpoint.
Status: *numerically supported as a bracketed relaxation*.
Evidence: `results/phase3` `lp_bounds.*.temperature`,
`tests/test_reactor.py::test_lp_temperature_bound_returns_bracket`,
`test_temperature_from_enthalpy_reports_domain_status`.

---

## Phase 3d — history-generated dimensions (new, audit Part C)

**C22. REVISED — local rank-1 endpoint image, tangent to the constant-control
family.** For a feed-compatible initial state the state manifold
M = {(T,Y) : E.Y = b_in, h_k(T).Y = h_in} has dimension 6. A constant-control
transient family is the image of two parameters (gamma, t), so its smooth local
dimension is at most two. Whether m-segment switched histories generate a
higher-dimensional endpoint image was tested by the rank of the endpoint
Jacobian dE_m/dtheta, projected onto the tangent space of M and compared with
the constant-control tangent span.

*Result (18 cases: fresh and hot starts, m = 3/4/6, equal / varied-dwell /
pulse-hold patterns, horizon 0.1 s, gamma in [10, 1e5]):* every projected
Jacobian has exactly ONE significant singular direction, with all remaining
singular values at machine precision (1e-16 to 1e-20) at every step size. The
single real direction lies inside the constant-control tangent span: its sine to
span(V) is <= 1e-12 in all 18 cases. Zero of 18 cases show a stable third
direction under step-size refinement (steps 1e-2, 1e-3, 1e-4 agree to four
significant figures). So at these base points switching opens no locally
transverse direction — the local endpoint image is a curve tangent to the
constant-control family.

This is a LOCAL rank statement at 18 specific admissible interior points under
one mechanism, one condition class and one horizon; it does not bound the global
reachable set. No formal rank claim is made from a floating-point SVD alone —
the step-size agreement is the actual evidence. Evidence: `results/phase3d`,
`scripts/phase3d_sensitivity.py`. A methodological corollary is recorded in
`research_log.md`: span-vs-span principal angles are meaningless for a
rank-deficient matrix (QR fills the basis with rounding noise) — the first
implementation produced erratic angles (0, 0.78, 0.85, 2.6e-8) where the true
transversality is ~1e-12.

---

## Not tested / deferred

- Certified (interval/validated) outer bounds; all bounds here are analytic
  (C5, C6, C12), bracketed relaxations (C21), or achieved objectives. Ordinary
  optimization optima are labelled as achieved objectives, not guaranteed maxima.
- The full toy reachable set at fixed G and finite horizon (C7 shows it is
  G-dependent; its exact geometry is open — the useful unresolved question).
- Complete S-curve / unstable-branch continuation (C19).
- Phases 4–6 (counterflow flamelet continuation, reduced flamelet transients,
  reaction-invariant convex enclosures): deferred; the audit directs a
  correctness revision and the sensitivity experiment first.
