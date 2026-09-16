# Local-agent research brief: thermochemical reachability and chemistry-integration domains

## 1. Your role and the research objective

Act as a computational research collaborator with access to my local development environment. You must derive, implement, execute, test, and critically interpret results—not merely propose a plan or create a software skeleton.

My broader research concerns stiff chemistry ODE integration in turbulent reacting flows, including physics-aware sampling and machine-learning acceleration. The immediate question is more fundamental:

> Given specified initial conditions, feed/boundary conditions, operating constraints, and a time horizon, what thermochemical states can the system reach? Can we construct useful, mathematically defensible inner estimates and outer bounds without enumerating every possible history?

Start with a well-stirred reactor; subsequently study counterflow diffusion flames and an explicitly defined unsteady flamelet model. Ultimately, connect the results to the states actually supplied to a chemistry integrator by operator splitting.

Do not assume the answer is a low-dimensional manifold. Determine which object is appropriate: a trajectory, steady-solution family, reachable region, invariant enclosure, manifold approximation, or a neighborhood of a manifold. Reachability is existential; it is not a probability distribution unless a probability law for histories has separately been specified.

Treat the mathematical claims below as hypotheses or derivations to audit. Challenge them. A reproducible counterexample or a corrected statement is a successful research outcome. Do not force the experiments to support the motivating idea, and do not claim novelty before checking relevant primary literature.

## 2. Scope, autonomy, and the first stopping point

Execute Phases 0–3 below as the first research tranche. Produce an actual report and reproducible results before expanding the project. Phases 4–6 define the next research direction; do not let their implementation delay the first report. A small, validated CSTR splitting experiment is a useful optional addition to Phase 3.

Work in a new `thermochemical-reachability` directory unless an explicitly designated existing project is already present. Inspect repository instructions and the working-tree status first. Preserve unrelated changes. Use an isolated environment; do not alter global Python, system packages, existing solver installations, or unrelated repositories. Do not use administrative privileges, provision paid compute, publish data, push commits, or launch unattended jobs without explicit authorization.

For unspecified scientific choices, make conservative, documented defaults and proceed. Ask only about genuinely blocking permissions or missing information that cannot be resolved safely. Do not stop after environment inspection, package installation, a literature summary, or an architecture proposal.

Start serially. Default to at most four workers when parallelism is useful and permitted by actual CPU/memory limits. Set thread counts to avoid oversubscription. Give each worker its own mutable thermochemistry objects. Run pilots before expanding sweeps. Make budgets configurable and checkpoint completed cases. When a resource limit prevents further work, return completed evidence and exact restart commands rather than an unbounded promise.

Do not build a web interface, agent orchestration platform, distributed scheduler, neural network, or comprehensive CFD framework in this tranche. Plain Python modules, tests, configuration files, raw arrays, and a report are sufficient.

## 3. Scientific definitions and evidence standards

Use a single-phase, ideal-gas, one-temperature mixture at prescribed pressure for the first study. Represent its thermochemical state conceptually as

    q = (h, p, Y),    Y_k >= 0,    sum_k Y_k = 1.

Here `h` is mixture-specific enthalpy including formation enthalpies. It is not sensible enthalpy alone. Recover temperature and density from the same thermodynamic model. Temperature may be the numerical ODE coordinate, but save both T and h. Do not count T, h, p, rho, and Y as independent coordinates.

For a controlled reactor `dq/dt = F(q,u(t))`, distinguish

    R(t_f; Q0,U)       = closure of terminal states at exactly t_f;
    R_[0,t_f](Q0,U)    = closure of states visited at any 0 <= t <= t_f.

Every experiment must specify the initial-state set Q0, admissible controls U, control bounds, switching or slew restrictions, and horizon. A fixed-control trajectory family is not the same problem as arbitrary time-varying controls. A finite piecewise-constant parameterization is only a subset of a larger admissible history class unless equality has been established.

Maintain separate labels for:

- Conservation/thermodynamic domain D: compatible with specified algebraic constraints.
- Witnessed trajectory samples S: numerically obtained, with provenance and numerical uncertainty.
- Empirical enclosure K_emp: fitted around data; not automatically an outer bound for unsampled trajectories.
- Certified enclosure K_cert: proven to contain the stated reachable set under explicit assumptions and error bounds.
- Steady family M_ss: solutions of the steady equations, not automatically accessible from the allowed initial conditions or within the finite horizon.
- Chemistry-substep domain: numerical input states generated by a named splitting method and timestep range.

Do not call the convex hull of finitely sampled trajectories a certified outer bound. Do not assume that interpolating between witnessed states produces an admissible reactor trajectory. Conversely, arbitrary reaction-and-mixing constructions may relax the actual reactor or flow constraints and overestimate physical reachability.

Maintain `claims.md`. Each substantive claim must have its assumptions, status (`proved`, `numerically supported`, `conjectured`, `refuted`, or `not tested`), and pointers to the derivation and evidence. Numerical support must state the tested scope. Solver agreement is not a rigorous error enclosure. A failed simulation is not evidence that the state is physically unreachable.

## Phase 0 — Audit the environment and reproduce the existing checks

Inspect the actual OS, Python interpreters, available CPU quota/affinity, memory, disk space, numerical libraries, and Cantera installation. Record the versions and actual limits in `environment.json`. Do not assume my machine matches a previous chat sandbox, and do not assume a GPU is available or necessary.

Use an existing suitable environment when available; otherwise create an isolated one with compatible versions of NumPy, SciPy, SymPy, Matplotlib, pytest, and Cantera. Consult documentation matching the installed version. Record the environment actually used. If online installation is unavailable, try available local environments or approved local package caches; do not fabricate a successful import or claim a setup-only result is a combustion experiment.

If `thermochemical_compute_check.zip` is provided in the workspace, inspect and unpack it safely. Preserve its original files and historical JSON. Rerun `verify.py` to a new local results file. Its pinned dependency versions are a historical reproducibility option, not a requirement to break the main research environment. Record any compatibility adaptation.

That package contains small-model smoke tests: the A→B→C reactor and a three-variable stiff ODE. It does not contain a completed reachable-set study or any detailed-chemistry/counterflow validation. Do not relabel historical results as new local runs. If the package is absent, reconstruct equivalent tests from the equations below and explicitly label the reconstruction.

Deliver the environment audit, executed test commands, pass/fail details, and measured runtime. Proceed to substantive experiments after the checks.

## Phase 1 — Audit the toy mathematics and compute bounded-control reachability

### 1A. Analytical reference

Use the isothermal equal-molecular-weight sequence A→B→C with equal, nondimensional reaction rates k=1, pure-A feed, x=Y_A, y=Y_B, and Y_C=1-x-y:

    dx/dt = -x + gamma(t)*(1-x)
    dy/dt =  x - (1+gamma(t))*y
    (x(0),y(0)) = (1,0).

Derive and test the conservation triangle, the steady CSTR locus `y=x(1-x)`, the batch trajectory `y=-x*log(x)`, and the candidate invariant inequality

    V(x,y) = y + x*log(x) <= 0.

Check the boundary identity

    dV/dt |_(V=0) = gamma*(log(x)+1-x) <= 0.

Handle x=0 through the continuous extension of x log x, rather than evaluating log(0). Check the other boundaries too; one inward-pointing calculation does not prove invariance of an incompletely specified set.

Audit the stronger assertion that, with gamma>=0, unrestricted duration, and arbitrarily rapid idealized feed replacement, the closure of the reachable set is

    0 <= x <= 1,    0 <= y <= -x*log(x).

Separate the containing-bound proof from the constructive attainability argument. Examine reacting pure A to an intermediate state and then replacing an appropriate fraction with fresh A. Clearly state which limits need infinite time or unbounded feed rate. Do not carry this exact envelope over to bounded controls or finite time without justification.

For constant gamma on a segment, derive and use the exact propagator. A reference expression to verify is

    a = 1 + gamma
    x_star = gamma/a
    y_star = gamma/a**2
    x(t) = x_star + (x0-x_star)*exp(-a*t)
    y(t) = y_star + ((y0-y_star)+(x0-x_star)*t)*exp(-a*t).

Compare it with independently implemented Radau and BDF integrations. Include gamma=0, nearly zero gamma, large gamma, short segments, long segments, and initial states near boundaries. The exact map should make the reachability search inexpensive.

### 1B. A genuinely new bounded-control experiment

Use gamma in [0,G], with initial exploratory G values 0.1, 1, and 10, and nondimensional horizons 1, 5, and 20. These are research defaults, not physical combustion times. Begin with 8 equal-duration control segments; compare nested 4-, 8-, and 16-segment parameterizations in selected cases.

Combine constant-control baselines, deterministic bang-bang/pulse histories, reproducibly randomized histories, and direct-shooting optimization of selected extrema. Distinguish terminal objectives from maximum-over-time objectives. Save every candidate's controls, objective, feasibility checks, and trajectory.

For linear support objectives, use at least a modest collection of directions in (x,y). An achieved objective is a lower estimate of the true supremum, not a certified upper bound. Do not label a local optimum “the reachable boundary.” If implementing interval/validated bounds is tractable, keep them separate from ordinary optimization.

Check sampled within-segment extrema, not only segment endpoints. Use the exact segment solution or event/root calculations where practical. Avoid missing narrow peaks through coarse output sampling.

Construct separate training/search and held-out history sets. Show how observed extent changes with G, horizon, control resolution, and additional search. Test the expected inclusion when the admissible class or cumulative horizon is enlarged. A failure of numerical approximations to show that inclusion can indicate inadequate search; it does not refute the exact set-theoretic inclusion.

Do not use area in a two-dimensional projection as a general high-dimensional reachable volume. For this toy problem only, a carefully defined projected-area or support-gap diagnostic is reasonable.

Required result: at least one quantitative comparison between the steady curve, bounded finite-time witnessed states, and the unrestricted analytical enclosure. Report a negative or inconclusive result honestly if the search does not establish the anticipated distinction.

## Phase 2 — Implement and independently validate detailed-chemistry CSTR dynamics

### 2A. Governing equations

Use an adiabatic, fixed-pressure, well-stirred model with one feed, no surface chemistry, and gamma=mass_inflow/reactor_mass:

    dY/dt = r(T,p,Y) + gamma(t)*(Y_in-Y)
    dh/dt = gamma(t)*(h_in-h).

The mass-specific chemical source is

    r_k = W_k * omega_dot_k / rho,

with compatible units. If integrating (T,Y), derive

    cp*dT/dt = gamma*(h_in-h) - sum_k h_k(T)*dY_k/dt.

The equivalent expanded form is

    cp*dT/dt = -sum_k h_k(T)*r_k
                + gamma*(h_in-sum_k h_k(T)*Y_in,k).

Here h_k is species-specific enthalpy in J/kg, not molar enthalpy. Verify both forms numerically at representative admissible states. This model is consistent with the constant-pressure reactor balances in source S1.

Be explicit about the reactor realization. Fixed pressure, fixed mass, and fixed volume cannot all be independently imposed during arbitrary reacting transients. A constant-mass, variable-volume, constant-pressure reactor with matched inlet/outlet flows is one valid realization. If using a fixed-volume reactor with pressure regulation instead, quantify pressure deviations and derive which equations actually apply. Do not assume the published pressure-controller example is an exact implementation of an algebraically fixed-pressure model; see S2.

Implement a transparent RHS using Cantera thermodynamics/kinetics and a stiff SciPy integrator. Independently cross-check against a correctly configured Cantera ReactorNet for identical equations and representative histories. The official custom-ODE example S3 is a useful implementation reference, not a substitute for checking the open-reactor energy equation.

Align integration intervals with control discontinuities and reset/reinitialize integrator history correctly. Do not let a discontinuity be accidentally smoothed by output interpolation. Preserve accumulated physical time when restarting segments.

Audit state setters. Normalizing or clipping Y on every RHS evaluation can change the modeled ODE and hide errors. Use a documented strategy suitable for solver trial states, preserve raw accepted states, and quantify any correction. Do not abort on every tiny internal Newton trial excursion or silently accept a materially negative physical output. Keep this numerical-policy distinction explicit.

### 2B. Mechanism and conservative initial defaults

Start with an installed small hydrogen mechanism such as `h2o2.yaml`; verify the actual available phase, species, reactions, thermo range, and source. It is a tractable detailed-kinetics test, not validation of my ammonia mechanism.

Use these initial defaults, subject to a documented pilot:

    pressure = 101325 Pa
    inlet_temperature = 900 K
    equivalence_ratio = 1.0
    fuel = H2
    oxidizer mole ratio = O2:1, N2:3.76
    residence_time bounds = [1e-5, 1e-1] s
    gamma bounds = [10, 1e5] s^-1
    primary finite horizon = 0.1 s.

Use the library's composition utilities and record the resolved mass fractions. Do not hard-code remembered mechanism sizes. Store mechanism file hashes, all referenced definitions when practical, phase name, species ordering, and resolved reactions/thermodynamics provenance.

Define two separate initial-condition studies: fresh inlet mixture, and the HP-equilibrium composition obtained from the same feed enthalpy, pressure, and elemental inventory. The hot initial state is an explicitly permitted initial condition in its own study, not evidence that it is reachable from fresh feed within the primary horizon. Do not mix these data sets without labeling the union.

Run a pilot before freezing the matrix. If these defaults produce only weak reaction or immediate relaxation, report it; add a separately named pilot-adjusted condition near observable ignition/extinction behavior. Preserve the original result. Do not move the operating point invisibly to manufacture interesting plots.

After the hydrogen model passes, an available methane mechanism such as `gri30.yaml` is an optional scaling check. Do not delay the first report to obtain a large ammonia or aviation-fuel mechanism. Do not assume an unavailable user mechanism can be substituted without changing the scientific scope.

### 2C. Conservation and solver checks

Construct the elemental mass matrix from the loaded mechanism:

    E_e,k = atom_count(e,k)*atomic_weight(e)/molecular_weight(k).

Check normalization, elemental-source conservation E*r≈0, and closed-reactor energy conservation. Inspect the stoichiometric left nullspace in consistent mass coordinates for additional linear invariants, including inert species. Avoid counting redundant constraints as independent dimensions.

For the open reactor, verify the exact balances for fixed inlet data:

    Gamma(t) = integral_0^t gamma(s) ds
    b(t) = b_in + exp(-Gamma(t))*(b0-b_in)
    h(t) = h_in + exp(-Gamma(t))*(h0-h_in),    b=E*Y.

When b0=b_in and h0=h_in, those coordinates should remain fixed. Add a dedicated admissible test with different inlet/initial enthalpy or elemental inventory to ensure the implementation is not passing only because the expected derivatives vanish.

Begin with rtol around 1e-8 and component-dependent absolute tolerances, approximately 1e-6 K for T and 1e-14 for species, adjusted to documented scales and trace-species objectives. Rerun selected trajectories with tighter tolerances and another integration method. Preserve raw conservation residuals, minimum Y, solver statistics, and failures. Define acceptance thresholds in configuration before the main sweep; do not retrospectively loosen them without reporting the change.

Compare trajectories on shared physical times, ignition metrics where applicable, species peaks, and enthalpy/element residuals. Use absolute and suitably scaled species errors; naive relative error near zero is misleading. Agreement between solvers sharing Cantera kinetics checks integration consistency, not independent validation of the chemical mechanism.

Required result: a tested detailed-chemistry reactor implementation, a cross-integrator comparison, and exact-balance checks that are strong enough to support subsequent geometric claims.

## Phase 3 — Construct and challenge empirical detailed-reactor envelopes

### 3A. Compare the right reference sets

At one frozen operating condition, build three separately labeled data sets:

    A: steady solutions across constant residence time;
    B: finite-time trajectories under constant residence time;
    C: finite-time trajectories under admissible switching histories.

Keep fresh and hot initial-condition studies separate. Start with a pilot of roughly 8 constant controls and 16 switching histories. Once validated, an initial main study might use 24 constant-control values, 32 random switching histories, 16 structured histories, and 16 held-out histories per initial-condition study. These are configurable starting budgets, not a claim of sufficient coverage. Record actual counts and costs.

Trace the steady family from hot and cold initializations where practical. Verify a scaled steady residual and allow sufficient convergence time. Steady calculations are a separate asymptotic reference; extending their integration beyond the reachability horizon does not extend the finite-time study. Record roots, initialization dependence, failure, and unresolved branches. Do not infer a complete S-curve or linear stability merely because time marching converged. Report periodic/nonconvergent behavior when observed rather than forcing it into a steady branch.

Structured histories should include slow-to-fast and fast-to-slow exchange, isolated pulses, alternating levels, and schedules with different dwell times. Random histories should have a declared law, but interpret their density as sampling density—not physical probability. Ensure every switch obeys the named admissible control class. Use separate seeds and whole-history holdouts; do not randomly split neighboring points of the same trajectory into training and validation.

The first question is whether transients extend beyond the steady library. The stronger question is whether switching reaches states not well represented even by constant-control transients from the same allowed initial condition and horizon. Do not present ordinary startup off the steady branch as evidence that switching is necessary.

### 3B. Geometry and active counterexample search

Analyze joint states, not just independent min/max species boxes. Use meaningful projections such as (T,Y_OH), (T,Y_HO2), and a declared progress-variable/radical pair when those species exist. Preserve the full composition for every plotted point.

Define a dimensionless metric before comparing distances. Include both an engineering-scaled representation and a trace-sensitive representation; document species scales, temperature scale, any log floor or transform, and sensitivity to these choices. A PCA projection is a visualization/approximation, not proof of intrinsic dimension. Do not assume learned projections preserve physical convexity or reachability.

For selected candidate states, compute distance to the steady reference and the constant-control transient reference. A distance to a finite point cloud can reflect under-resolved reference sampling. Refine the reference near nearest matches, interpolate only with appropriate constraints, and when feasible minimize distance over continuous reference parameters such as residence time and elapsed time.

State the strongest conclusion actually supported: “outside this sampled library” is weaker than “outside the entire continuous steady family.” Initial endpoints should be represented in comparisons so trivial omissions are not mistaken for new geometry.

Use a modest bounded optimization budget to seek histories maximizing one selected intermediate/radical peak or distance from a reference library. Start from structured and randomized seeds. Save failed evaluations. Report achieved objectives, not guaranteed global maxima. Validate each leading candidate by replaying its exact controls with tighter tolerances and an independent integration method.

Add at least one simple conservation-based outer comparison. For example, maximize selected Y_k by linear programming under nonnegativity, normalization, elemental constraints, and applicable additional linear invariants. Such bounds can be loose because they ignore kinetics and possibly energy feasibility; label the relaxation explicitly. Do not assume equilibrium temperature is a universal maximum attainable temperature.

Energy-constrained temperature/composition bounds are optional. If implemented, respect the mechanism's thermodynamic validity domain, and justify any temperature search or monotonicity assumption. Do not mistake an arbitrary user temperature cutoff for a physically proved bound.

Require numerical-resolution checks before accepting an apparent excursion: trajectory tolerance, temporal extrema resolution, reference-library resolution, and metric sensitivity. A near-zero measured difference should be compared against those uncertainties rather than rounded into a discovery.

Required result: a quantitative table and a few interpretable figures showing what each reference covers, what additional histories reveal, and how robust the difference is. If no robust excursion is found, document the tested region, search effort, and remaining alternatives.

### 3C. Optional small solver-aware extension

A useful first connection to chemistry integration is to split the validated CSTR ODE itself. Let R_dt be adiabatic fixed-pressure reaction, and X_dt be exact fixed-feed exchange at constant gamma:

    Y_new = Y_in + exp(-gamma*dt)*(Y_old-Y_in)
    h_new = h_in + exp(-gamma*dt)*(h_old-h_in).

Implement the explicitly named Strang sequence

    X_(dt/2) -> R_dt -> X_(dt/2).

Record the state immediately before R_dt, its output, and the completed-step state. Compare against the unsplit high-accuracy CSTR solution at common physical times and under timestep refinement. Align all steps with control discontinuities; assess the expected order on smooth segments rather than across unresolved jumps.

Keep numerical chemistry-substep inputs separate from exact unsplit physical states. Do not assume an arbitrary chemistry image is an actual CFD chemistry input. This small experiment is a valid solver-interface study for this CSTR splitting only, not a claim about OpenFOAM or a turbulent flame.

## Phase 4 — Next tranche: steady counterflow diffusion-flame families

After the first report, study an explicitly defined opposed-flow configuration with fixed inlet compositions, temperatures, pressure, geometry, and a controlled inlet mass-flux scaling. Begin with the validated small mechanism where appropriate.

Use unity-Lewis-number transport for the first idealized study, and verify that the installed solver actually selected the intended model. Mixture-averaged transport must not silently stand in for unity Lewis number. Cantera documents a separate unity-Lewis-number transport model; see S4. Save the actual transport setting, Soret setting, radiation setting, domain width, inlet fluxes, and mesh.

Under adiabatic common-diffusivity assumptions and compatible initial/boundary data, examine

    b(Z) = Z*b_F + (1-Z)*b_O
    h(Z) = Z*h_F + (1-Z)*h_O.

Check these relations numerically rather than merely plotting mixture fraction. Define which elemental/Bilger mixture fraction is used and verify consistency. When later introducing differential diffusion, do not impose these single-Z relations by normalization or reconstruction; their deviations become part of the question.

Use continuation and branch-aware initialization. Distinguish actual branch folds, extinction criteria, and solver failure. Investigate stable/unstable burning branches only with an appropriate method and explicitly reported stability evidence. Cantera's two-point flame-control example provides a continuation reference; see S5. A converged unstable steady root is a mathematical solution, not proof of accessibility from the specified initial flame.

Do not treat “strain rate” as one unambiguous scalar. Report imposed mass-flux scaling, the exact strain-rate diagnostic used, and, where appropriate,

    chi = 2*D*|grad Z|^2,

including how D and gradients are evaluated. Save chi_st and its relation to the chosen operating parameter when meaningful. Do not equate imposed strain and scalar dissipation without deriving or measuring their relation.

Extract full local states q(x;a,branch) and construct the steady family. Record mesh convergence near steep gradients and extinction. Quantify component extrema but retain joint-state information. Apply the same reference-resolution caution used in Phase 3.

## Phase 5 — Next tranche: physical transients in an explicitly reduced flamelet model

A sequence of separately converged steady flames is not a transient simulation. Cantera's one-dimensional steady solver uses time stepping as part of its nonlinear solution procedure; do not reuse intermediate solver states as validated physical strain-response data without separately establishing the corresponding unsteady model and its consistency. See S6.

For the first transient experiment, implement and label a reduced unity-Lewis-number flamelet model, for example

    partial_t Y_k = (chi(Z,t)/2)*partial_ZZ Y_k + r_k(h,p,Y)

on Z in [0,1], with explicitly specified boundary compositions, initial profile, fixed pressure, and an enthalpy field consistent with the same idealization. If h is fixed to the affine inlet-mixing profile, show why it is a solution of the model and enforce compatible initial data.

Specify a nonnegative chi profile and its normalization, e.g. chi(Z,t)=chi_st(t)*g(Z) with g(Z_st)=1. State how it relates—or does not relate—to physical counterflow strain. Prescribing chi independently is a modeling assumption, not a solved momentum equation.

Do not accidentally replace `(chi/2)*Y_ZZ` by `partial_Z[(chi/2)*Y_Z]`; these differ when chi varies with Z. Derive the chosen model and discretization. Check conservation, positivity behavior, boundary implementation, and grid/time convergence. Use sparse Jacobian structure where useful.

Build steady references from this same reduced model before assessing its transient excursions. A difference from a full counterflow library may reflect a model mismatch rather than dynamical history. Keep those comparisons separate.

Study ramps, pulses, periodic forcing, and bounded switching in chi_st with a declared horizon and control limits. Include rate-of-change restrictions when claiming a physically limited forcing class. Compare forcing timescales with observed response times rather than assuming quasi-steady tracking.

The desired result is a validated comparison between a steady family and its history-dependent transient state set, including replayable forcing histories for leading excursions. It is not yet a bound on all turbulent flames.

## Phase 6 — Longer-term link to chemistry sampling and mathematical enclosures

Extend the solver-aware CSTR construction to a validated spatial split model or an actual CFD interface only after specifying the transport update and thermodynamic constraint. Fixed-pressure enthalpy-conserving chemistry and fixed-volume internal-energy-conserving chemistry define different maps. Do not silently transfer a domain from one to the other.

Keep distinct: coupled physical states, chemistry-substep inputs, chemistry outputs, and internal solver trial states. For a named timestep range, ask which states require accurate chemistry integration and whether a steady-flame-only library misses them. Evaluate practical consequences such as source-term variation, trajectory error, and integration cost—not just geometric distance.

No neural-network training is required for the initial evidence. A later sampling study can compare equal-budget sampling on steady states, admissible transient histories, and counterexample-guided expansions, with held-out trajectories and solver-level validation.

For the mathematical outer-bound direction, investigate reaction-invariant convex sets containing the initial/boundary states. Under a common scalar diffusion operator, compatible boundary conditions, and suitable regularity, an inward-pointing chemistry condition on a convex boundary may provide a flow-wide invariant enclosure. Derive the assumptions precisely. Do not claim that this argument survives differential diffusion, heat loss, variable-pressure work, or phase change unchanged.

Reaction-plus-arbitrary-mixing closure is another relaxed model. Distinguish it from bounded-residence-time, finite-rate, geometry-constrained reachability. Track time and ancestry if combining operations. Mixing is affine in h and Y at the assumed common pressure, not generally in T. A conservation-valid mixture may still be inaccessible to the target geometry.

If fitting a barrier V(q)<=0 for controlled dynamics, the relevant boundary test involves all admissible controls:

    max_u grad(V(q)) dot F(q,u) <= 0 on V(q)=0,

together with appropriate regularity and inclusion of the allowed initial set. Testing this on finitely many samples is not certification. Distinguish analytic proof, validated global computation, and empirical tests. A counterexample-guided search is useful even when it cannot certify the final candidate.

If using timescale separation to propose a manifold neighborhood, treat any contraction inequality and tube thickness as hypotheses requiring verification. Do not equate a low-rank data projection, a slow chemical manifold, and a reachable set.

## 4. Reproducible implementation and data contract

Keep the implementation lean. A suggested layout is:

    thermochemical-reachability/
      README.md
      pyproject.toml or documented environment files
      configs/
      src/thermoreach/
      tests/
      scripts/
      results/<run_id>/
      reports/
      claims.md
      sources.md
      research_log.md

Create only modules needed for executed phases. Avoid empty placeholder implementations. Core capabilities should cover state conversion/invariants, toy exact maps, reactor RHS/integration, control histories, sampling/search, metrics, and reporting. Use type hints, clear units, input checks, deterministic seeds, and explicit error handling.

Make scripts the reproducible source of truth; notebooks are optional views, not hidden execution dependencies. Provide a smoke mode and a first-study mode. The documented commands must be the commands actually executed, not illustrative commands that reference missing entry points.

For each run, retain a manifest with timestamp, code revision or source hash, resolved configuration, seed, package versions, mechanism identity, solver settings, budget, outcome, and output locations. Avoid collecting credentials or unrelated environment variables. Hashes should identify the actual mechanisms/configuration/code used; do not build an elaborate tracking service.

For each trajectory, save at least time, raw T/Y, derived h/p/rho, species ordering, inlet/initial condition ID, exact control segments, solver status/statistics, and conservation/error diagnostics. For spatial cases add physical coordinate or Z, mesh, local transport quantities, and branch/forcing IDs. Use a simple inspectable format such as NPZ plus JSON/CSV, or HDF5 when already available.

Save failures with exception messages and the last available state. Record partial horizons explicitly. Do not omit hard cases from denominators or let output directories containing only successful trajectories disguise failure rates.

Separate fresh/hot starts, mechanisms, transport models, physical model variants, and time horizons in metadata. Never pool them silently into a larger-looking envelope. Deduplicate repeated segment endpoints for sample counts, while retaining trajectory reconstruction information.

Generate figures from saved data, with units, model assumptions, control bounds, and legend labels that distinguish sampled evidence from analytical/certified boundaries. Include uncertainty/resolution comparisons. Use a few figures that answer the research questions rather than a dashboard of unrelated projections.

## 5. Required first report and acceptance criteria

The first report must contain the actual environment and execution commands; the toy derivation and bounded-control results; the detailed-reactor equations and solver validation; the steady/constant-control/switching comparison; and a limitations section distinguishing what has been shown from what remains untested.

Include a compact table with each experiment, configuration/run ID, attempted/successful/failed cases, runtime, strongest numerical residuals, and evidential status. Include replay instructions for the most informative trajectory or apparent counterexample. Every numerical statement must be traceable to saved outputs.

The first tranche is complete when:

1. The small analytical benchmarks have been rerun locally and their assumptions audited.
2. At least one bounded-time/control toy reachability study has been executed beyond the original smoke tests.
3. Detailed-reactor dynamics have passed cross-integration and exact-balance checks, or a genuine environment/model blocker is precisely demonstrated.
4. A pilot detailed-reactor reference-versus-history comparison has produced inspectable data, including failures and uncertainty checks; do not demand a positive discovery as a condition of completion.
5. The report, claim ledger, raw data, and restart/reproduction commands are present and consistent with the actual execution.

If a blocker prevents items 3–4, finish the independent toy/mathematical work and clearly mark the detailed work incomplete. Do not fabricate Cantera results, replace detailed chemistry with the toy model without disclosure, or stop at a generic request for more compute.

End by answering: What is supported? What was refuted or remains ambiguous? What computation would most efficiently reduce the remaining uncertainty? What is the single best next research experiment, with its expected discriminating outcome rather than a promised positive result?

## 6. Primary-source starting points

Consult documentation for the installed version rather than assuming the moving `stable` pages match local APIs. Record versions, access dates, and which statements or code choices each source supports. Verify research-paper references directly; do not invent titles or citations. Related topics for a focused literature check are attainable-region theory, invariant regions of reaction–diffusion systems, controlled reachability, flamelet continuation, and chemical-manifold reduction. Do not let a broad literature survey displace the first computational tranche.

S1 — Cantera constant-pressure reactor equations and temperature formulation:

    https://cantera.org/stable/reference/reactors/constant-pressure-reactor.html
    https://cantera.org/stable/reference/reactors/ideal-gas-constant-pressure-reactor.html

S2 — Cantera combustor residence-time example and pressure-control implementation:

    https://cantera.org/stable/examples/python/reactors/combustor.html

S3 — Cantera properties/kinetics with an external SciPy ODE integrator:

    https://cantera.org/stable/examples/python/reactors/custom.html

S4 — Cantera transport models, including the separate unity-Lewis-number model:

    https://cantera.org/stable/reference/transport/index.html

S5 — Counterflow diffusion-flame continuation with two-point flame control:

    https://cantera.org/stable/examples/python/onedim/diffusion_flame_continuation.html

S6 — Cantera one-dimensional nonlinear solver and its time-stepping procedure:

    https://cantera.org/stable/reference/onedim/nonlinear-solver.html

S7 — Thermodynamic state definitions and reactor integration documentation:

    https://cantera.org/stable/python/thermo.html
    https://cantera.org/stable/userguide/reactor-tutorial.html

## Start now

Inspect the workspace and environment, then execute Phases 0–3 within a bounded, checkpointed initial run. Give concise progress updates at meaningful milestones. Do not return only a plan. Return the first reproducible scientific result set, or the completed independent work and an evidence-backed blocker, before attempting the optional larger program.
