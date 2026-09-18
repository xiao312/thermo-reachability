# Report 08 — the structural face, the signed scalar correction, and reloadable inference

Revision 8, branch `revision8-structural-face`.  This tranche fixes a
demonstrated feasible-set defect, re-evaluates the scalar model, corrects the
exposure-law interpretation, and makes the frozen predictor reloadable.  A
positive correction result was not required, and the outcome is reported as found.

## 1. The structural-zero defect (section A)

### 1.1 The defect

The frozen Revision-7 correction direction carried `n_AR = 9.76e-19` and
`n_N2 = 6.92e-19`: not physical entries, but SVD numerical noise.  A
conservation-compatible direction already forces both to zero - `E[Ar,:]` and
`E[N,:]` are unit rows on AR and N2, so `E n = 0` requires `n_AR = n_N2 = 0` -
but only to ~1e-16.  Against the exactly zero `Y_AR`, that noise becomes an
ACTIVE bound through `(0 - 0)/9.76e-19 = 0`, so `a_lo = 0` and the feasible
interval was **one-sided**.  Every negative correction was silently truncated.

[VERIFIED ON THE MECHANISM: exact species list, inventories, direction entries]

### 1.2 The classification

Three independent rigorous routes to a structurally constant species, none of
which clips a physical trace radical:

* (a) the feed carries none of any element the species is built from, so no
  admissible trajectory has the material to form it;
* (b) the species takes part in no reaction and the feed carries none of it, so
  the chemistry and exchange terms both vanish exactly;
* (c) **the species is the unique carrier of some element e**: then `E Y = b`
  forces `Y_k = b_e / E[e,k]` for every state with that inventory - zero when
  `b_e = 0`, constant otherwise.  This route uses no stoichiometry.

[RESULTS: which species are absent / fixed / active, and the element carriers]

### 1.3 The repair and its consequences

Exact zeros are embedded for the absent and fixed species, and the direction is
projected onto the conservation subspace of the ACTIVE species.  The feasible
interval becomes two-sided again.  `feasible_a_interval` reports the binding
species by ORIGINAL index and name, skips the exact-zero entries, and REPORTS
(rather than accepts) any zero-species/nonzero-direction inconsistency.

## 2. Re-evaluation of the scalar oracle (section B)

The signed objective near zero, the exact local linear estimate
`a_lin = j^T W^T W (q - q_B) / (j^T W^T W j)` from `dT/da = -sum_k h_k n_k / cp`,
and the active-bound status separate a genuinely curved objective from one the
truncated interval was holding at a bound.

[RESULTS: how many previously-zero optima become negative; the A/B/C/D/E table;
the old-D numbers retained as a restricted-oracle historical result]

## 3. The exposure law is an inductive bias, not an invariant match (section C)

The exact balance laws are `b = b_in + e^-Gamma (b_0 - b_in)` and
`h = h_in + e^-Gamma (h_0 - h_in)`.  The initialized hot HP state has `h_0 = h_in`
and `b_0 = b_in` identically, so **both invariants are constant for every control
and cannot identify Gamma**: the law is degenerate at this anchor.  The observed
`gamma_B t_B ~ Gamma` is therefore an empirical regularity, not a consequence of
invariant matching, and floating-point invariant mismatches were not promoted to
physical information.  Both `u = log(Gamma_B/Gamma)` and `v = log(t_B/T)` are
regressed; `u` is not an enforced zero.

The M1 first-moment ablation (`M1 = int (t_f - s) gamma(s) ds`, exact) and the
short-time asymptotic baseline it suggests (`t_B ~ 2M1/Gamma`,
`gamma_B ~ Gamma^2/(2M1)`, an `O(t_f^3)` model, never an exact constraint), on the
25 committed diagnostic cases at unchanged order-1 complexity and on the same
targets and split:

| inductive bias | features | log-gamma rms (in-sample) | (leave-one-out) |
|---|---|---|---|
| free (no exposure) | 4 | 7.09e-2 | 9.36e-2 |
| exposure | 5 | 1.90e-2 | 3.62e-2 |
| exposure + log M1 | 6 | 2.18e-3 | 5.71e-3 |
| pure M1 asymptotic (no fitting) | 0 | 1.33e-2 | - |

The exposure feature is a real 2.6x inductive gain at equal complexity, and the
first moment adds a further 6.3x in leave-one-out error; the zero-parameter
asymptotic baseline is already as accurate as the fitted exposure model, so the
time-PROFILE of the exposure carries most of the gamma/t split information.  This
is diagnostic only - adopting M1 as a frozen feature is a Revision-9 decision to
be validated on untouched data, and this revision's frozen model does not do it.
No out-of-domain estimates occurred (0/25).

## 4. The reloadable frozen model (section D)

[THE RELOAD TEST AND THE HASHES]

## 5. The bounded protocol (section E)

[THE 12 NEW TEST HISTORIES, A/B/C/D/E, FAILURES AND PROJECTIONS COUNTED]

## 6. The cost (section F)

[THE MATCHED-ACCURACY TIMINGS]

## 7. The decision

[WHETHER THE SIGNED SCALAR CORRECTION IS ADEQUATE AT THIS LOCAL CLASS]

## 8. The next research direction

[BOUNDED, BEYOND THIS ANCHOR]
