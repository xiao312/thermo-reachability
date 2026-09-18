# Report 07 — The local thermochemical state generator

Revision 7, branch `local-generator`.  The reviewed milestone is the local
rank-three approximation at the secondary anchor (hot HP-equilibrium start,
T_in = 1200 K, p = 101325 Pa, γ_ref = 1e4, horizon T = 1e-6 s, m = 3 segments).
This revision turns that approximation into a **tested physical generator**: a
canonical curved reference state plus one conservation-compatible scalar
correction, evaluated against untouched test histories.

## 1. The coordinate defect (§1 of the review), corrected without re-solving

The phase-11 radius script built the whitened map `A_time = A H^(−1/2)`, took a
Euclidean-unit **right singular vector** `u` of it, and then used `δη = r·u` as a
log-control perturbation.  But the whitened map acts on `ξ = H^(1/2) δη`, so the
log-control direction of unit **time** norm is

```
    δη = H^(−1/2) u ,     ||δη||_time = ||H^(1/2) δη||₂ = ||u||₂ = 1 .
```

Applying `r·u` directly yields an actual time norm of `r·√(Σ hⱼ uⱼ²)`, which is
`r/√m` for equal segments: the ray labelled r = 1.0 has **actual norm 0.577** for
m = 3.  Scalar rescaling of `u` is *not* a substitute when the durations are
unequal, because `H` is then not a multiple of the identity.

Corrections (Phase 12, `results/phase12/`), all postprocessing — no saved
trajectory was re-solved or overwritten:

* `chemresponse.unwhiten_svd_direction` implements `δη = r H^(−1/2) u`; both
  norms are now derived from `δη` itself (`radius_from_deta`), never from a
  memorised `1/√m` factor.
* Every existing trajectory is relabelled by its **measured** norm, with the
  requested label retained.  The in-family controls and the 12 random histories
  were normalised directly in the time norm and their labels are exact; only the
  11 transverse rays were mislabelled.
* The confounded comparison is withdrawn: at the requested label r = 0.1 the
  distance is `0.996 × σ_⊥ · r_measured`, i.e. the curved family closes
  **essentially none** of the gap along the transverse direction at this anchor
  (the earlier "58 % closure" divided by the wrong radius).  The linear-in-r
  growth and every `R/lin` ratio are unaffected — both operands are built from
  the same `δη`.
* The E-metric is recomputed from the stored raw differences with **separate**
  temperature and species tolerances, and mechanically verified:
  `e_metric_consistent_in_raw_records: true`.  The stored records were always
  internally consistent; the transcription error (E_Y = 6.8e-4 where 6.8e-3 is
  correct) existed only in the hand-written report table, which is corrected.
* Every existing trajectory lies inside the declared development domain
  **measured time norm ≤ 0.6**, so the corrected analysis needed no new sweep.

## 2. The representation

```
    q_hat = decoder( Y_B(β) + a·n_Y ,  h_B(β) )
    β = (log γ_B, t_B)     canonical reference coordinates on B(γ, t; q₀)
```

* **`n_Y` is conservation-compatible by construction**: `E n_Y = 0` and
  `1ᵀ n_Y = 0`, so the corrected composition preserves the reference's elemental
  inventory and its normalisation.  Invalid compositions are **rejected and
  counted**, never clipped to make a prediction look admissible
  (`PhysicalDecoder`, `results/phase13/phase13_results.json` →
  `n_rejections`).
* **Temperature is solved, not assumed**: `T` is recovered by enthalpy inversion
  at the reference's own enthalpy `h_B`, so an additive (T, Y) correction is
  never assumed to conserve energy at finite amplitude.
* **`h_B` and the inventory `b_B` are exact closed-form functions of β**: for
  this reactor `db/dt = γ(b_in − b)` and `dh/dt = γ(h_in − h)` exactly (chemistry
  conserves elements and enthalpy), so `b = b_in + e^(−Γ)(b₀ − b_in)` and
  `h = h_in + e^(−Γ)(h₀ − h_in)` with `Γ = γ t`.  Computing them from β therefore
  uses no endpoint information.
* The correction direction is fitted on the **scaled** training residuals, then
  projected onto the conservation subspace in the **scaled metric** with
  row-equilibrated, rank-revealing SVD (both absolute *and* relative thresholds —
  the trap behind the withdrawn Phase 3D claim).  `UNRESOLVED` is reported, never
  replaced by a fabricated direction.  The retained fraction was 1.0000, because
  the locator already matches the exposure and hence `E(q − q_B) ≈ 0` holds
  automatically.
* The admissible range of `a` (`feasible_a_interval`) is derived from the
  **species bounds themselves**, not from the range of `a` seen in training.  It
  is wide (≈ 2.67) because it is set by the least abundant species, while the
  optimum is set by the transverse excursion — four orders of magnitude smaller.
  The oracle coefficient search is therefore **multi-scale** (a log-spaced scan
  of both signs down to 1e-12 of the interval width, then a bounded refinement
  between neighbouring candidates); a uniform 41-point grid over the interval is
  ~1000× too coarse and returns a spurious bimodal `a`.  This was found by the
  per-case diagnostic, fixed, and covered by a test.

## 3. The exposure-constrained reference-coordinate regression

The located reference satisfies `γ_B t_B ≈ Γ_actual` with an rms residual of
**2.7e-3** in log space — measured on the phase-11 oracle locations, and two
orders of magnitude tighter than a free regression of the two coordinates
(rms 5.3e-2).  The parametrisation is therefore

```
    log γ_B = log(Γ/(γ_ref T)) − v + u ,     log t_B = v
    u = log(Γ_B/Γ)   (≈ 0 by the exact balance law),   v = log(t_B/T)
```

so the regression only has to learn the **split** `v` between γ and t; the
exposure itself is exact.  `log Γ` is an explicit regression feature (an exact
function of the input history).  Training rms: order 1 → 1.33e-2, order 2 →
**4.47e-5** in log space; validation (8 untouched histories) selected order 2
with median state error 2.52e-4 vs 2.84e-4 for order 1.  The model — complexity,
regularisation (ridge on slopes only), normal direction, parametrisation — is
**frozen and written to disk before any test history is generated**
(`results/phase13/frozen_model.json`).

## 4. Protocol

1. **Training**: the 25 completed phase-11 exploratory trajectories (rays,
   in-family controls, the 12 random histories), relabelled by measured norm and
   reconstructed from the immutable records.  Nothing re-solved.
2. **Fit** both complexities on training only.
3. **Validation histories**: 8 new admissible histories, seeds and measured-norm
   stratification fixed in advance; complexity selected on these alone.
4. **Freeze** the selected model.
5. **Test histories**: 16 new admissible histories on **independent fixed seeds**
   (stratified at measured norms 0.1 / 0.3 / 0.6), disjoint from training and
   validation.
6. **Compare four representations** on the test set only.

## 5. Results on the 16 untouched test histories

Scaled state error (`results/phase13/phase13_summary.json`):

| representation | median | max | meaning |
|---|---|---|---|
| **A** oracle reference, no correction | 3.62e-4 | 1.72e-3 | what the family alone can reach |
| **B** predicted reference, no correction | 3.69e-4 | 2.14e-3 | the predictive canonical |
| **C** predicted reference + predicted correction | 2.92e-4 | 3.10e-3 | the full predictive generator |
| **D** oracle reference + oracle correction | **5.17e-5** | 1.72e-3 | the ansatz's capability |

Three separate facts, deliberately not conflated:

* **The canonical generator is validated.**  The prediction cost B − A is
  **+4.2e-6 median** — the predicted reference coordinates are as accurate as the
  oracle-located ones.  Coordinate error in log γ: median 1.0e-4, max 1.8e-2.
  All 16 decoded states are **admissible**: zero rejections for species bounds,
  normalisation, and enthalpy inversion.
* **The single correction is a real representational gain** where it applies:
  D improves the median **7×** over A, and where the coefficient is predicted
  well the state error falls by up to 20× (e.g. test_0: 1.42e-4 → 6.2e-6) and the
  chemistry error `E_max` at dt = 1e-6 s falls ~15× (9.25e-3 → 5.24e-4).
* **The predicted correction is not yet reliable per case.**  C improves on A in
  **7 of 16** cases and is worse in 9 (worst: test_12, A = 1.72e-3 → C = 3.10e-3).
  The reason is diagnosable and *not* a search artifact: the optimal `a` is
  **bimodal** across histories (either ≈ 0, when the residual has no component
  along the fixed direction within the 5-dimensional conservation subspace, or
  O(1e-4 – 2e-3)), and that bimodality is not a smooth low-order function of the
  controls.  The single fixed direction captures the leading but not the second
  residual mode (3.47e-3 vs 1.17e-4).  This is reported as a **candidate**, not a
  settled win: the canonical reference is the validated deliverable, and the
  correction is a constructive improvement whose *coefficient prediction* remains
  open.

## 6. Chemistry validation (test set only)

The chemistry-only map Φ_dt (adiabatic, constant pressure, exchange exactly off)
is applied to the true state, the predicted canonical state, and the corrected
predicted state at the four retained diagnostic timesteps
(`results/phase13/phase13_chemistry.csv`).  The predicted canonical's enthalpy
matches the true state's to ~1e-6 relative — the exposure constraint doing exactly
what §3 predicts.  Temperature differences dominate the response and **decay with
dt** (the two states relax to the same equilibrium), so the chemical map does not
amplify the geometric difference.  Enthalpy and elements are preserved by the map
to 2e-8 J/kg in 1.34e6 and 5.6e-17 respectively, and the baseline mismatch is
reported, never projected away.

## 7. Error versus tolerance (for a future application)

No target application tolerances have been supplied, so nothing here asserts a
combustion-relevant accuracy.  `results/phase13/phase13_tolerance.json` reports,
for every illustrative (ε_T, ε_Y) pair, the **fraction of the 16 test histories**
meeting the pair — for the state error and for the chemistry error at each dt —
so a future application with real tolerances can look its pair up rather than
extrapolate.  At ε_T = 1 K and ε_Y = 1e-3 all 16 test cases meet the tolerance for
every representation including C; at the deliberately tight ε_T = 0.1 K /
ε_Y = 1e-4, C fails 1 of 16 while A, B and D fail none.

## 8. Integrity repairs

* The phase-11 reference-domain-expansion branch passed `rtol=` to
  `refine_reference` where the signature is `refine_rtol=`; it would have raised
  `TypeError` had it ever fired.  It now fires under a stub test and records both
  coarse scans.
* `load_verified` checks mechanism **content hash**, species order, pressure and
  the hot-HP initial state — not just dimensions and filenames.
* `ignition_time_event` remains a non-terminal event; all dense evaluation is
  restricted to [0, t_end]; censored (no crossing) is distinguished from
  integration failure.
* Three `np.linalg.qr` projector constructions remain replaced by the single
  rank-revealing `svd_basis`.

## 9. Claims

* **C37** (established): the canonical curved reference is predictable from
  control history coordinates alone to the accuracy of the oracle-located
  reference (B − A median +4.2e-6 scaled units; log γ error median 1.0e-4), with
  every decoded state physically admissible, over the declared development
  domain measured time norm ≤ 0.6.
* **C38** (established): the closed-form exposure law `γ_B t_B ≈ Γ` reduces the
  reference-coordinate regression error from 5.3e-2 to 4.5e-5 (rms, log space);
  validation selects the quadratic parametrisation.
* **C39** (established): the oracle single correction improves the median state
  error 7× at this anchor, and where its coefficient is predicted well the
  chemistry error at dt = 1e-6 s falls ~15×.
* **C40** (candidate): the *predicted* correction coefficient is not yet
  per-case reliable (helps 7/16) because the optimum is bimodal across
  histories; it is a constructive improvement, not a settled win.

## 10. What this does not claim

No CFD or flamelet significance is asserted, and no tolerances have been supplied.
The located pair is constructive evidence of a good local approximation, not a
certified exclusion of the family; the optimised distance remains an upper
estimate of the infimum.  Reconstructed states are not claimed to be exactly
reachable — a family member is a real integrated state, but the corrected state
is a reconstruction validated only by admissibility and chemistry comparison.
The development domain is the small ball of measured time norm ≤ 0.6 around the
secondary anchor; nothing here is a statement about the reachable set at large.

## 11. Recommended next step

The open item is **C40**, and it is precisely scoped: predict the correction
coefficient's *magnitude and sign* from the controls by using the second residual
mode as a second fixed direction, or by replacing the scalar with a
rank-two correction — within the same frozen protocol, no new anchor and no
broadening of the domain.  A tolerance-supplied re-run (report_06 §8) remains the
other unfunded follow-up.
