# Report 08 — the structural face, the signed scalar correction, and reloadable inference

Revision 8, branch `revision8-structural-face`.  This tranche fixes a
demonstrated feasible-set defect, re-evaluates the scalar correction, corrects the
exposure-law interpretation, and makes the frozen predictor reloadable.  The
outcome is decisive for the open candidate C40: **the correction's coefficient was
never the bottleneck — the feasible set was one-sided, and half the optima lay on
the side that had been truncated to zero.**

Artifacts: `results/phase14/` (structural face, diagnostic), `results/phase15/`
(protocol + reloadable model), `results/phase14/face_verify_local.json`
(server-free verification), `results/phase16/` (exposure ablation, diagnostic).

## 1. The structural-zero defect (section A)

### 1.1 The defect

The frozen Revision-7 correction direction carried

```
n_AR = +9.759474952003097e-19      n_N2 = +6.923631936456039e-19
```

not physical entries but SVD numerical noise.  A conservation-compatible
direction already forces both to zero — `E[Ar,:]` and `E[N,:]` are unit rows on
AR and N2, so `E n = 0` requires `n_AR = n_N2 = 0` — but only to ~1e-18.  Against
the exactly zero `Y_AR`, that noise becomes an *active* bound through
`(0 - 0) / 9.76e-19 = 0`, so `a_lo = 0` and the feasible interval was
**one-sided**.  Every negative correction was silently truncated.

The frozen direction is bit-identical to a fresh recomputation of the Revision-7
code path (`recomputed_revision7_code_path` in `phase14_structural_face.json`),
so this is the fitted model, not a reconstruction artifact.

### 1.2 The classification

Three independent rigorous routes to a structurally constant species, none of
which clips a physical trace radical:

* **(a)** the feed carries none of any element the species is built from, so no
  admissible trajectory has the material to form it;
* **(b)** the species takes part in no reaction and its initial and feed mass
  fractions coincide, so the chemistry and exchange terms both vanish exactly;
* **(c)** the species is the **unique carrier of some element e**: then `E Y = b`
  forces `Y_k = b_e / E[e,k]` for *every* state with that inventory — zero when
  `b_e = 0`, constant otherwise.  Route (c) uses no stoichiometry at all.

On the real mechanism (h2o2.yaml, 10 species, 29 reactions, feed
H2 / O2:1,N2:3.76 at phi = 1):

| class | species | reason |
|---|---|---|
| absent | AR | unique carrier of Ar; `b_Ar = 0` (routes a and c) |
| fixed | N2 | unique carrier of N; `b_N = 0.745124` (route c) |
| active | H2, H, O, O2, OH, H2O, HO2, H2O2 | the only species the correction may move |

Element inventories `b_in = {O: 0.226354, H: 0.0285224, Ar: 0.0, N: 0.745124}`;
`inconsistent: []` (no zero-species/nonzero-feed conflict).

Both structural facts are confirmed empirically on the committed records:
`Y_AR = 0` for all 25 located references and `Y_N2` constant to 3.4e-15.

### 1.3 The repair and its consequences

Exact zeros are embedded for the absent and fixed species and the direction is
projected onto the conservation subspace of the *active* species.  The
corrected direction differs from the old one by at most 1.73e-18 — the change is
entirely in the **feasible set**, which is the point:

```
old interval:  [a_lo 0.000e+00, a_hi 2.6678]   binding species on the low side: AR (species_8)
new interval:  [a_lo -0.657850, a_hi 2.6678]   binding species on the low side: H  (species_1)
```

One-sided (`a_lo == 0`) intervals before the repair: **49 of 49** diagnostic
cases.  The repair is verified twice and independently: server-free on the
committed phase-11 records
(`results/phase14/face_verify_local.json`, reproduces without Cantera) and on the
real mechanism through Cantera (`results/phase14/phase14_structural_face.json`).

## 2. Re-evaluation of the scalar oracle (section B)

With the two-sided interval the oracle optimum is no longer pinned at zero:

| diagnostic set | n | negative | zero | positive | at an active bound |
|---|---|---|---|---|---|
| phase-11 training (25) + phase-13 histories regenerated from their seeds (24) | 49 | 25 | 0 | 24 | — |
| the 12 new final-test histories | 12 | 6 | 0 | 6 | 0 |

**The negative side — entirely unavailable in Revision 7 — carries 51 % of the
diagnostic optima and half the test optima, and no optimum now sits at a bound.**
The oracle `a` in the test set spans [-7.87e-4, +1.42e-3] with median -4.35e-5,
and the exact local linear estimate `a_lin = j^T W^T W (q - q_B) / (j^T W^T W j)`
with `dT/da = -sum_k h_k n_k / cp` is reported alongside it.

This resolves candidate C40.  In Revision 7 the oracle coefficient looked
bimodal (zero, or O(1e-4 - 2e-3)) and the predicted coefficient improved the state
error in only 7 of 16 cases; the second mode was blamed.  The zeros were the
truncation.  The decisive measurement is the new variant comparison (section 5):
the same correction, with a *predicted* coefficient, now improves the error in
11 of 12 cases at a 7.6x median gain, and swapping the predicted coefficient for
the oracle coefficient moves the median by only 1.44x.  A better coefficient is
not where the remaining error lives.

## 3. The exposure law is an inductive bias, not an invariant match (section C)

The exact balance laws are `b = b_in + e^-Gamma (b_0 - b_in)` and
`h = h_in + e^-Gamma (h_0 - h_in)` with `Gamma = sum_j gamma_j dur_j`.  The
initialized hot-HP state has `h_0 = h_in` and `b_0 = b_in` identically, so **both
invariants are constant for every control and cannot identify Gamma**: the law is
degenerate at this anchor.  The observed `gamma_B t_B ~ Gamma` is therefore an
empirical regularity, not a consequence of invariant matching, and the code now
regresses both `u = log(Gamma_B/Gamma)` and `v = log(t_B/T)` rather than enforcing
`u = 0`.

The first-moment ablation (`M1 = int (t_f - s) gamma(s) ds`, exact) on the 25
committed diagnostic cases, at unchanged order-1 complexity and on the same
targets and split:

| inductive bias | features | in-sample log-gamma rms | leave-one-out |
|---|---|---|---|
| free (no exposure) | 4 | 7.091e-2 | 9.363e-2 |
| exposure | 5 | 1.898e-2 | 3.617e-2 |
| exposure + log M1 | 6 | 2.182e-3 | 5.712e-3 |
| pure M1 asymptotic (no fitting) | 0 | 1.333e-2 | — |

Exposure is a real 2.6x inductive gain at equal complexity, and the first moment
adds a further 6.3x in leave-one-out error; the *zero-parameter* asymptotic
baseline is already as accurate as the fitted exposure model, so the time-profile
of the exposure carries most of the gamma/t split information.  This is
diagnostic only — 0 of 25 estimates fall out of domain, and adopting M1 as a
frozen feature is a Revision-9 decision to be validated on untouched data.  This
revision's frozen model does not do it.

## 4. The reloadable frozen model (section D)

`results/phase15/frozen_model_v2.json` serializes both coefficient arrays, the
feature specification, the correction direction, the species report, the decoder
policy, the reference coordinates, the training domain, the normal-subspace
report, the model and library identities, the problem definition, the complexity
selection, the training and validation case ids, and a code hash.  The run
reports **"the frozen model is reloadable with no identity warnings"**, and
`load_local_generator` reproduces predictions without refitting, with any
mismatch reported rather than silently absorbed.  The complexity choice (order 2)
was made on the existing validation set regenerated from its recorded seeds —
which are development data from this point on — and the declaration is recorded in
the results:

> the model (complexity, regularization, the correction direction and both
> coefficient arrays) was frozen to `frozen_model_v2.json` BEFORE any of the new
> final-test histories above was generated; the test seeds are new and disjoint
> from phase 13's; the oracle values are conditional on the fixed located
> reference and are not a joint optimum over (beta, a).

## 5. The bounded protocol (section E)

12 new final-test histories on a new seed, stratified at measured time norms
0.1 / 0.3 / 0.6: **12 requested, 12 completed, 0 failed, 0 excluded** (failures
are counted, never filtered out).  Decoder rejections: 0 (species-negative 0,
normalization 0, enthalpy-inversion 0).  Coefficients landing outside the feasible
interval: 0.  Every decoded state is admissible; nothing was clipped to look
admissible.

Scaled state error by variant (A oracle reference; B predicted reference;
C predicted reference + predicted correction; D oracle reference + oracle
correction; E predicted reference + oracle correction):

| case | norm | A | B | C | D | E | oracle a |
|---|---|---|---|---|---|---|---|
| test_0 | 0.10 | 2.653e-04 | 2.653e-04 | 6.178e-06 | 5.059e-06 | 5.689e-06 | -2.304e-04 |
| test_1 | 0.10 | 7.361e-05 | 7.714e-05 | 2.496e-05 | 2.324e-06 | 2.462e-05 | +6.391e-05 |
| test_2 | 0.10 | 5.817e-06 | 1.753e-05 | 1.698e-05 | 1.493e-07 | 1.663e-05 | +5.052e-06 |
| test_3 | 0.10 | 1.060e-04 | 1.065e-04 | 1.044e-05 | 2.461e-06 | 8.795e-06 | -9.210e-05 |
| test_4 | 0.30 | 4.643e-04 | 4.696e-04 | 6.468e-05 | 1.040e-05 | 6.332e-05 | -4.032e-04 |
| test_5 | 0.30 | 3.010e-05 | 3.092e-05 | 9.894e-06 | 4.067e-07 | 7.169e-06 | +2.614e-05 |
| test_6 | 0.30 | 3.452e-04 | 3.463e-04 | 5.636e-05 | 1.197e-06 | 2.720e-05 | -2.999e-04 |
| test_7 | 0.30 | 4.580e-04 | 4.593e-04 | 4.953e-05 | 1.491e-05 | 4.640e-05 | +3.977e-04 |
| test_8 | 0.60 | 8.463e-05 | 4.828e-04 | 5.125e-04 | 6.142e-07 | 4.754e-04 | +7.352e-05 |
| test_9 | 0.60 | 9.058e-04 | 1.193e-03 | 8.175e-04 | 2.462e-05 | 7.943e-04 | -7.865e-04 |
| test_10 | 0.60 | 1.633e-03 | 2.576e-03 | 2.013e-03 | 3.281e-05 | 2.012e-03 | +1.419e-03 |
| test_11 | 0.60 | 5.576e-04 | 5.862e-04 | 1.782e-04 | 6.691e-06 | 1.767e-04 | -4.843e-04 |

| variant | min | q25 | median | q75 | max | mean |
|---|---|---|---|---|---|---|
| A | 5.817e-06 | 8.188e-05 | 3.053e-04 | 4.876e-04 | 1.633e-03 | 4.108e-04 |
| B | 1.753e-05 | 9.917e-05 | 4.028e-04 | 5.086e-04 | 2.576e-03 | 5.509e-04 |
| C | 6.178e-06 | 1.534e-05 | 5.294e-05 | 2.617e-04 | 2.013e-03 | 3.133e-04 |
| D | 1.493e-07 | 1.051e-06 | 3.760e-06 | 1.153e-05 | 3.281e-05 | 8.471e-06 |
| E | 5.689e-06 | 1.467e-05 | 3.680e-05 | 2.514e-04 | 2.012e-03 | 3.048e-04 |

Aggregate readings:

* **C improves on B in 11 of 12 cases** (worst ratio 1.06x, best 0.023x);
  median 4.028e-04 -> 5.294e-05, a **7.6x gain**.  Revision 7 managed 7 of 16.
* **E vs C: 3.680e-05 vs 5.294e-05 — only 1.44x.**  Replacing the predicted
  coefficient with the oracle coefficient buys almost nothing, so the
  coefficient is *not* the bottleneck.  (This is the measurement that settles
  C40.)
* **B vs A: 1.32x.**  Predicting the reference coordinates costs little.
* **C vs D: 14.1x.**  The remaining gap to the oracle lives in the located
  reference and the single fixed direction, not in the coefficient.

By measured time norm:

| norm | A | B | C | D | B/A | B->C | C/D |
|---|---|---|---|---|---|---|---|
| 0.10 | 8.983e-05 | 9.183e-05 | 1.371e-05 | 2.392e-06 | 1.02x | 6.7x | 5.7x |
| 0.30 | 4.016e-04 | 4.028e-04 | 5.294e-05 | 5.800e-06 | 1.00x | 7.6x | 9.1x |
| 0.60 | 7.317e-04 | 8.897e-04 | 6.650e-04 | 1.566e-05 | 1.22x | 1.3x | 42.5x |

At norm 0.6 the correction stops being the dominant error source: the reference
prediction itself degrades (B/A 1.22x) and the fixed direction cannot represent a
much larger residual (C/D 42.5x).  This is the honest boundary of the local class,
already announced as the development domain (actual time norm <= 0.6); it is
reported, not repaired by widening the domain.

The chemistry comparison on the same pairs is reported per case in
`phase15_results.json` (same-pair gain relative to the dt -> 0 limit, with the
amplifying flag and both temperatures), computed as postprocessing over the
illustrative dt schedule; no application tolerances have ever been supplied.

## 6. The cost (section F)

Both integrators are timed at the SAME rtol (1e-10), so the comparison is honest:

| stage | s per call |
|---|---|
| regression (the closed-form balances + exposure features) | 8.854e-05 |
| reference ODE | 5.848e-02 |
| enthalpy inversion + correction | 1.760e-05 |
| **full prediction** | **5.935e-02** |
| original switched-history integration | 1.032e-01 |

The predictor is **1.74x faster** than the switched-history integration at the
same rtol, while achieving a scaled state error of 6.178e-06 on the timed case,
with the reference library built once in 6.05 s.  The dominant cost is the
reference ODE; the regression and decoder are ~600x cheaper and are not the
bottleneck.

## 7. The decision

**The signed scalar correction is adequate at this local class** once the feasible
set is two-sided.  The evidence: it improves the prediction in 11 of 12 untouched
cases at a 7.6x median gain, it never drives a state out of admissibility (0
rejections, 0 out-of-interval coefficients), and the oracle coefficient is worth
only 1.44x more than the predicted one — so there is no case for the rank-two
correction the review excluded.  The old Revision-7 result stands as a
restricted-oracle historical record: its bimodality and its 7-of-16 success rate
are now traced to a truncated feasible set, not to a second residual mode.

What this does *not* claim: the correction is not certified globally optimal (the
oracle is conditional on the fixed located reference and is not a joint optimum
over (beta, a)); the model is local to the development domain (actual time norm
<= 0.6) and the norm-0.6 degradation is reported rather than repaired; no
CFD/flamelet tolerance has been supplied, so the chemistry table spans
illustrative dt values only.

## 8. The next research direction

Bounded, and deliberately not launched here: the remaining 14.1x gap to the oracle
sits in the located reference and the single fixed direction.  Two candidate
moves, either of which fits inside the frozen protocol without a new anchor or a
broader domain:

1. make the correction direction a low-order function of the controls rather than
   a single frozen vector, validated on the existing development data before any
   new history; or
2. adopt the first-moment feature (section 3) as a frozen regressor, which cut the
   leave-one-out coordinate error 6.3x and would narrow the B/A gap that dominates
   the norm-0.6 error.

Both are Revision-9 decisions.  This revision closes with an executed comparison
and no outstanding sweep.
