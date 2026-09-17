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

**C22. SUPERSEDED — the Phase 3D result is withdrawn.** The saved Phase 3D
endpoint finite-difference matrices have one dominant singular value under the
implemented scaling and thresholds. **These data do not establish the local
image dimension or tangency to the original fixed-initial-state constant-control
family**, for these reasons:

* the reference tangent span was differentiated from the *switched endpoint*
  rather than the original initial state, so it did not differentiate the family
  B(gamma, t; q_initial) that was being claimed about;
* a common absolute perturbation (mean of theta) left inadmissible columns as
  *zeros* inside the Jacobian, which then entered the rank analysis as
  fabricated zero sensitivities;
* the reference matrix was orthonormalized by unfiltered QR, which fills a
  rank-deficient span with rounding-noise directions;
* the singular-value index used for the "third direction" was n-3, which reads
  the *smallest* value when m = 3 rather than the third largest (index 2);
* state increments mixed kelvin with mass fractions and were differentiated
  with respect to raw gamma, so the singular values had no single
  application-independent meaning;
* several base histories had horizons 0.044 s and 0.064 s despite the global
  0.1 s label.

A singular direction must not be called "real" merely because it passes a
relative SVD threshold. Algebraic rank, numerical rank and application-scale
effective rank are distinct notions, and even an exact rank-one derivative at a
single point does not prove a curve-shaped local image without further
neighborhood assumptions (the map (a,b) -> (a,b^2) has a rank-one Jacobian at the
origin but a two-dimensional image). Long terminal holds make loss of
earlier-history sensitivity a specific hypothesis for the observed effective
rank. Evidence retained as a historical record: `results/phase3d` (marked
superseded), `scripts/phase3d_sensitivity.py`.

**C23. NEW — corrected sensitivity pipeline and the terminal-memory experiment
(Phase 4).** Replaces C22. The corrected pipeline is `thermoreach.sensitivity`
plus `scripts/phase4_terminal_memory.py`, with positive controls in
`scripts/phase3e_positive_controls.py` (all passing, `results/phase3e`).

Corrections: dimensionless controls eta = log(gamma/gamma_ref) with
per-coordinate interior steps (a column that cannot be centred is INVALID, never
zero); reference tangents from the ORIGINAL initial state
(v_t = F(q_base, gamma_star) and v_log_gamma = d phi/d log gamma); rank filtering
of BOTH matrices by SVD with explicit relative AND absolute thresholds, never
filling a basis with arbitrary orthogonal vectors; the conserved-manifold
tangent space as the null space of C W^-1 by rank-revealing SVD with the
conditioning recorded and conservation leakage reported BEFORE projection; a
declared frozen state scaling; exactly m durations summing to the recorded
horizon; the third singular value at fixed index 2; and three separate rank
notions (algebraic / relative-threshold / absolute application threshold).

Positive controls (all pass): the three-state linear benchmark
(dz_i/dt = -lambda_i z_i + u, lambda = (1,2,4), T = 1, three equal segments)
whose exact endpoint Jacobian has rank 3 and singular values
(0.5007801, 0.08533145, 0.006828632) - reproduced to 7 digits; the exact toy
rank-2 history with terminal hold, showing algebraic prefix rank preserved while
the application-scale effective rank decays; the same-initial-state identity
sum_j dE_m/d(log gamma_j) = d phi/d log gamma to machine precision; and the
orthogonality/rank-deficient-reference angle tests (V = [e1,0] vs J = [e2]
gives sine 1, not 0).

Scientific framing: for the terminal-hold experiment,
E(theta_prefix, gamma_hold, L) = phi^L_{gamma_hold}(q_prefix(theta_prefix)), so
dE/dtheta_j = D_q phi^L . dq_prefix/dtheta_j — prefix sensitivity is PROPAGATED
through the hold and can become unresolved while gamma_hold sensitivity
approaches the steady-branch tangent. Exponentially tiny is not exact zero.

Status: results in `results/phase4`; see `reports/report_03.md`.

*Findings (144 cases: 2 inits x 3 anchors x 3 horizons x 2 patterns x 4 hold
lengths, 3405 s).* Validation: the same-initial-state identity holds over all
cases to a worst relative error of 8.0e-5 (FD truncation, not machine precision;
the toy version with the closed-form sensitivity holds to 1.4e-17), and LSODA
vs Radau agree on the reference derivative to 5.3e-7 / 8.5e-13 relative.

The prefix endpoint rank is governed by the number of EXCHANGE TIMES gamma*T:
at T = 1e-3 s the second singular direction is 1.16e-5 (gamma = 1e2) and 9.94e-5
(gamma = 1e3) scaled units - 400x / 200x above the measured differentiation-noise
scale and above the declared 1e-6 application threshold - so a REAL
history-generated direction exists there, identical for both initial states and
both patterns. It is largest near gamma*T ~ 1 and decays on both sides. At
gamma = 1e4 the same direction sits AT the noise floor (1.20e-7 vs 2.0e-7) and is
unresolved. By T = 1e-2 s it has collapsed to ~1e-12 for every anchor. Every
terminal hold of at least one residence time reduces the prefix block to ~1e-12
scaled units (absolute application rank 0) with the endpoint already at the
steady reference, confirming the chain-rule propagation
`dE/dtheta_j = D_q phi^L . dq_prefix/dtheta_j`.

**C23. SUPERSEDED IN PART (Revision 4) - the mechanism is refuted, the
existence claim survives only in a narrower form.** The Phase 4 statement above,
that the weak history direction is "largest near gamma*T ~ 1", is REFUTED by the
matched-pair experiment of Phase 6 (see C25): at FIXED gamma*T = 1 the second
scaled singular value spans a 1.2e9-fold range (7.4e-12 at gamma=100, T=1e-2
versus 8.8e-3 at gamma=1e4, T=1e-4). Exchange exposure does NOT control the
direction. What the data support instead is that it is largest when the horizon
is comparable to the chemical relaxation time (a few times the ignition delay)
and collapses once the trajectory has relaxed. The existence of a resolved
second TOTAL direction at short horizons, and its erasure by any hold beyond
about one residence time, are unaffected.


---

---

## Revision 4 claims (memory-calibration branch)

**C24. NEW - four rank notions must be reported separately, and the machine rank
is not the answer.** `classify_ranks` reports (i) `rank_machine`,
numpy's `matrix_rank`, which counts a direction at 1e-12 because numpy's
tolerance is ~max_sv * shape * eps ~ 1e-15; (ii) `rank_derivative_noise_resolved`,
directions above the measured refinement discrepancy ||J(h) - J(h')||, which is a
REFINEMENT DISCREPANCY and not a certified noise bound - Phase 5 measured
2.76e-6 between the two largest FD steps where Phase 4 had reported 2.8e-8 from a
different step pair, so the number depends on which pair is used and is stored
and labelled as such; (iii) `rank_application_effective`, directions above the
DECLARED 1e-6 scaled threshold, which with the frozen scaling
(1 unit = 100 K or 1e-2 mass fraction) means **1e-4 K and 1e-8 mass fraction per
unit log-control step, not 1e-6 mass fraction**; (iv) `stencil_complete`, which
gates every full-rank claim: an invalid stencil EXCLUDES THE WHOLE MATRIX, it is
never aggregated with nansum into a rank count, and a NaN column must not reach
the SVD (numpy does not converge on it).

The withdrawn Phase 3D claim reported the machine rank as though it were (ii)
and (iii). At the Phase 5 anchor (gamma=100, T=1e-3) the three notions give
rank_machine = 3, rank_noise_resolved = 2, rank_application = 2 on singular
values (1.28e-1, 1.16e-5, 1.08e-12).

**C25. NEW - exchange exposure does not control the weak history direction
(matched pairs).** Nine (gamma, T) pairs at three fixed exchange exposures
gamma*T in {0.1, 1, 10} for two initial states (Phase 6). At FIXED gamma*T = 1
the second scaled singular value spans 7.4e-12 to 8.8e-3, a factor of 1.2e9.
The same held exposure can give a resolved direction or an erased one. What
tracks the direction instead is the ratio of the horizon to the chemical
relaxation time: for the fresh 1200 K feed the declared ignition delay is
t_ign ~ 4.3e-5 s, and the direction is largest at T ~ 2 t_ign (8.8e-3) and has
collapsed by T ~ 200 t_ign (7.4e-12). This refutes the mechanism stated in C23
and replaces it.

**C26. NEW - a resolved transverse (off-family) direction exists in a narrow
window (Phase 8).** Phase 5 could not decide whether the second TOTAL direction
lies outside the two-parameter constant-control family: its transverse
component (1e-8 to 1e-5 scaled) sat between the conserved-manifold leakage
floor (1e-10 to 1e-7) and the finite-difference refinement discrepancy
(1e-6 to 1e-5), so the FD estimator was not resolving it - a direction below
its own noise floor is not a negative result, it is an unresolved one. Because
the CSTR right-hand side is affine in the control, the forward-sensitivity
equations are exact with a closed-form inhomogeneity
dS_j/dt = F_q S_j + gamma_j F_gamma(q) and NO control-step noise floor
(`endpoint_jacobian_fsa_system`, validated against a matrix-exponential closed
form to 1.2e-14 in the test suite, at m = 1, 2, 3 segments). With that estimator
the hot initial state at gamma = 1e4, T = 1e-5 s gives

    total spectrum       (9.5e-1, 1.2e-1, 6.0e-5)
    transverse spectrum  (1.2e-4, 2.0e-11, 9.2e-13)
    manifold leakage     (2.8e-12, 4.7e-13, 5.0e-14)
    Radau vs LSODA       1.5e-9

so the transverse direction is 4e4 times above the conservation leakage floor,
8e4 times above the independent-integrator discrepancy, and 120 times above the
declared application threshold: `rank_application_effective = 3`. This is a
genuine history-generated direction outside the constant-control family, but
only in this narrow window (short horizon, fast chemistry); at the longer
horizons the transverse component returns to the 1e-11..1e-12 floor.

The same estimator separates three outcomes that the FD run could not
distinguish. At (gamma=1000, T=1e-3), for BOTH initial states, a transverse
direction is RESOLVED - 4.5e-7 and 1.8e-8 scaled units, 225x and 2.9x above the
independent-integrator discrepancy - and is independently reproduced by replay
along the singular vector to ~30%; yet both lie BELOW the declared 1e-6
application threshold, so `rank_application_effective = 2`. These are real
off-family directions judged insignificant by the declared tolerance, not failed
detections. Reporting them as zero would repeat the error of the withdrawn
Phase 3D claim in the opposite direction. The remaining anchors are unresolved
(transverse below the cross-integrator check, and the replay fails to converge to
the SVD value). Scope: the h2o2 mechanism, the two declared initial states, the
tested (gamma, T) grid and the declared scaling and threshold. The fully
igniting-and-relaxing anchor (fresh, gamma=100, T=1e-2) did not complete within
the compute budget and is excluded rather than extrapolated; Phase 6 had already
established it as the memory-erased control.

**C27. NEW - the Lie bracket is exactly as the affine structure predicts
(Phase 7).** The CSTR right-hand side is affine in the control,
F(q, gamma) = gamma v(q) + r(q) (verified to 0.0 against two levels), so
[f_a, f_b] = (gamma_a - gamma_b) [r, v] independently of the absolute levels.
Tested by direct endpoint differencing of the two-segment histories (a,b) and
(b,a): (i) the SAME-LEVEL control returns exactly zero, so the order dependence
is entirely due to the level difference; (ii) at tau <= 1e-6 s the log-log slope
of the endpoint difference versus tau is 2.03 against the predicted 2, and the
magnitude reproduces tau^2 (gamma_a - gamma_b)[r, v] to within 4%
(6.55e-14 observed versus 6.35e-14 predicted), with the (gamma_a - gamma_b)
proportionality holding across the three level pairs; (iii) at tau above the
ignition delay the slope is 0.9-1.5 rather than 2 and the difference is ~1e5
times the bracket prediction, because the trajectories have ignited and
saturated out of the asymptotic regime. The bracket is real and exactly
modelled, but at chemically relevant segment times its observable magnitude is
set by ignition, not by the small-tau bracket. Phase 4's claim of order
dependence at 0.7 scaled units is an ignition effect, not a bracket effect.

---

## Revision 5 claims (flagship-validation branch)

**C28. NEW - repaired sensitivity machinery, with a regression test for every
repair.**
*Segment restarts.* The tangent-linear estimator now integrates each segment as
its own solve_ivp call with the segment index fixed inside the RHS closure,
rather than one call with a searchsorted control selector - the column receiving
parameter forcing changes discontinuously at each boundary. On the review's
exact benchmark (qdot = gamma; S_j' = gamma_j on segment j; gamma = (10, 1e5, 10);
durations = (0.0499995, 1e-6, 0.0499995)) it reproduces the endpoint 1.09999 and
the sensitivities (0.499995, 0.1, 0.499995). *Tolerances.* The row-major (n, m)
sensitivity block maps a per-state atol vector as np.repeat(atol_state, m), not
np.full(n*m, atol.min()), which had applied a species-sized tolerance to the
temperature sensitivity.
*Correct replay.* The second right singular vector of A maximizes neither
||B v|| nor sigma1(B) for B = (I - P) A; replaying along v2(A) tests ||B v2||,
which is ZERO on the review counterexample A = diag(3,2,1), P = proj(e1,e2) where
sigma2(A) = 2 and sigma1(B) = 1. The replay now perturbs along the leading
transverse RIGHT singular vector of B and compares the SIGNED projection, the
cosine, and the FULL vector residual against B v_perp. A separate replay along
v3(A) validates the third total direction.
*Signed commutator.* With [f,g] = Dg f - Df g and F_gamma = r + gamma v,
[F_a, F_b] = (b - a)[r, v] - the coefficient is (gamma_b - gamma_a), and Phase 7
had reported the opposite sign while comparing only NORMS, which concealed it.
Verified on the linear benchmark qdot = -q + gamma, a = 1, b = 3, whose exact
two-segment difference is (b - a)(1 - exp(-tau))^2, POSITIVE and tending to b - a.
*Ignition time.* Now located by dense-output root finding, not a 400-point grid
lookup that quantizes the estimate to T/399 and makes it horizon-dependent; the
grid estimate is retained only to expose the quantization and censoring is
explicit.
*Rank-revealing reference basis everywhere.* Unfiltered QR is gone from the
transverse analysis; the one svd_basis routine decides the rank, and an
unresolved reference span yields transversality UNRESOLVED, never a fabricated
projector.

**C29. NEW - the flagship third direction survives validation (Phase 9).**
Primary anchor: HP-equilibrium hot start of stoichiometric H2/air, Tin = 1200 K,
p = 101325 Pa, gamma = 1e4 1/s, horizon 1e-5 s, m = 3 equal segments. The state
Jacobian is itself numerical, so it is varied (eps ladder 1e-5, 1e-6, 1e-7)
independently of the ODE tolerance, of the integrator (Radau, LSODA) and of the
replay, and A = W J and D = W V are PERSISTED as NPZ with singular vectors and
endpoints.

    total spectrum       (0.9484, 0.1189, 5.963e-5)   [reproduces the historical values]
    transverse spectrum  (1.234e-4, 2.44e-9, 6.93e-13)
    sigma3(A)            5.963e-5     ||(I-P)u2|| = 1.000: the third TOTAL
                        direction is entirely transverse
    delta_A              3.07e-9      (spread of A across methods x eps ladder)
    delta_P * ||A||      1.91e-7      (spread of reference projectors)
    combined bound       <= 1.94e-7   via ||Bhat-B|| <= ||Ahat-A|| + ||Phat-P|| ||Ahat||

The claimed transverse value 1.234e-4 is therefore 635 times above the combined
operator-error estimate, and sigma3(A) = 5.96e-5 - which by the EXACT inequality
||(I-P)A||_2 >= sigma3(A) for ANY rank <= 2 projector is evidence independent of
the reference-plane construction - is 307 times above it. The replay along
v_perp converges with signed relative deviation 0.074 -> 0.000 and cosine
+1.0000 as epsilon shrinks, and the separate v3(A) replay gives 5.963e-5.
MATHEMATICS: the exact inequality and the affine structure of F in gamma.
NUMERICAL EVIDENCE: the spectra, replays and cross-method spreads above. The
secondary anchor (horizon 1e-6 s) is even cleaner: transverse 1.55e-3 against a
combined error estimate 1.8e-9, an 8.6e5 margin.

**C30. NEW - the negative case is now correctly UNRESOLVED, and the diagnostic
demonstrably distinguishes signal from noise.** At the fresh gamma = 1e4,
horizon 1e-4 s anchor, the state-Jacobian eps spread is delta_A = 1.19e-3, which
exceeds both sigma3 (7.6e-12) and the transverse value (1.7e-10). The replay does
not converge: its signed projection is NEGATIVE and proportional to epsilon
(-2.5e-4, -2.3e-5, -2.5e-6, -2.3e-7, -2.6e-8) with cosine -0.996 against
sigma_perp = 1.7e-10. This is the correct signature of a quantity below the
estimator's resolution. The previous revision's claim that the two
gamma = 1000 / horizon 1e-3 cases have "resolved but below application
threshold" transverse directions is therefore NARROWED TO CANDIDATE status: it
rested on projector-based inference without reference-uncertainty machinery, and
those anchors have reference-span condition numbers ~1.94e7 and ~1.56e6 versus
~20 for the flagship. They are not withdrawn - they are unconfirmed pending that
machinery.

**C31. NEW - a finite local reachable patch exists and is genuinely off the
constant-control family, but it is small on the application scale (Phase 10).**
Built on the validated matrices loaded from the phase-9 NPZ (not recomputed),
with finite histories at radii r in {0.003, 0.01, 0.03, 0.1} along the three total
and the transverse right-singular directions, both signs, every state witnessed
by an actual history from the original q0 and the raw nonlinear endpoint
evaluated.

Local model E(eta0 + deta) = E0 + J deta + R: ||W R|| grows as r^2
(2.58e-6, 2.87e-5, 2.58e-4, 2.86e-3) for the transverse direction, so the linear
model holds with a clean second-order correction.  The least-squares
second-derivative scale in the physical control norm
sqrt(sum_j (dur_j/T) deta_j^2) is L = 1.60 and the resulting containing shell is
labelled EMPIRICAL, not certified - L is estimated from samples, not bounded.

Distance to the CONTINUOUS constant-control family B(gamma, t; q0) by global grid
over a declared domain plus local refinement and a fresh-integration replay of the
nearest candidate.  This is an UPPER ESTIMATE of the distance to the continuous
parent family, since a grid plus a local minimizer can overestimate the minimum;
it is not a certified global exclusion.

    direction        tangent plane   curved family   ratio
    v1  +1 / -1      1.63e-4/1.49e-4  9.09e-7/8.48e-7  0.006
    v2  +1 / -1      1.05e-5/1.60e-5  1.12e-5/1.03e-5  1.07/0.65
    v3  +1 / -1      5.58e-6/6.34e-6  5.72e-6/6.20e-6  1.03/0.98
    v_perp +1 / -1   1.27e-5/1.41e-5  1.21e-5/1.26e-5  0.95/0.90

The in-family direction v1 sits essentially ON the curved family - the family
closes the gap by a factor ~180.  Every other direction stays off it: the
curved-family distance is no smaller than the tangent-plane distance.  The
transverse direction is off the continuous constant-control family by
1.21e-5 / 1.26e-5 scaled units = 1.2e-3 K and ~1e-7 in mass fraction at control
radius r = 0.1 (a 10% change in each segment level), in both signs.

APPLICATION SIGNIFICANCE: this is a genuine but SMALL excursion.  One scaled unit
is 100 K or 1e-2 mass fraction, so the reachable off-family displacement at the
largest admissible-in-the-linear-model radius is ~1e-3 K.  The direction is
mathematically real (C29) but would not by itself change a CFD calculation.
Significance is judged on the finite state difference, not on the derivative per
unit log-control norm.

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
