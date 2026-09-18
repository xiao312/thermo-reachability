#!/usr/bin/env python3
"""Phase 13: the local thermochemical state generator.

The accepted local rank-three approximation is turned into a TESTED physical
generator: a canonical curved reference state plus ONE conservation-compatible
scalar correction, with the temperature recovered by enthalpy inversion.

Protocol, in order, and this order is the point:

  1. The training set is the EXISTING phase-11 exploratory trajectories (rays,
     in-family controls, the 12 random histories), relabelled by their measured
     control norm.  Nothing is re-solved for training.
  2. The generator is FIT and then FROZEN (reference coordinates regression,
     normal direction, correction regression, complexity, regularization).  The
     frozen model identity is written to disk before any new history exists.
  3. NEW histories are generated: 8 validation (complexity selection only) and
     16 test, with fixed seeds and a stratification in the MEASURED time norm
     fixed in advance.
  4. Four representations are compared on the untouched test set:
       A  oracle reference coordinates, no correction   (what the family alone can do)
       B  predicted reference coordinates, no correction (the predictive canonical)
       C  predicted coordinates + predicted correction   (the full predictive generator)
       D  oracle coordinates + oracle correction         (the ansatz's capability)
  5. Chemistry is validated at the retained diagnostic timesteps on the test
     histories only: the true state, the predicted canonical state, and the
     corrected predicted state, as states and as increments.

Old trajectories are immutable; the corrected analysis lives under a new run
identifier (results/phase13).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls, stable_child_seed  # noqa: E402
from thermoreach.chemresponse import (  # noqa: E402
    ChemistryConfig, ReferenceLibrary, chemistry_response, hdiag,
    located_reference, time_norm,
)
from thermoreach.controls import History  # noqa: E402
from thermoreach.generator import (  # noqa: E402
    LinearRegressor, LocalGenerator, PhysicalDecoder,
    exact_balances_from_controls, exposure_integral, feasible_a_interval,
    fit_local_generator, oracle_correction_coefficient,
)
from thermoreach.io_utils import manifest, utc_now  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, hot_hp_state  # noqa: E402
from thermoreach.sensitivity import StateScaling  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "phase13"

# the secondary anchor, unchanged from phase 11
GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = 1e4
HORIZON = 1e-6
M = 3
T_IN = 1200.0
P = 101325.0
LIB_GAMMA_POINTS = 21
LIB_GAMMA_HALF_WIDTH_LOG = 1.5
LIB_T_MAX_FACTOR = 100.0

# the frozen development domain: measured time norm <= 0.6 contains every
# existing trajectory (rays up to 0.577, random histories up to 0.6 exactly)
DOMAIN_MAX_MEASURED = 0.6
VAL_RADII = (0.1, 0.3, 0.6)          # validation, 3 per stratum = 9 -> trimmed to 8
TEST_RADII = (0.1, 0.3, 0.6)         # test, 5/5/6 = 16
N_VAL = 8
N_TEST = 16


def make_cstr() -> CSTR:
    cfg = ReactorConfig(mechanism="h2o2.yaml", pressure=P,
                        inlet_temperature=T_IN, fuel="H2",
                        oxidizer="O2:1,N2:3.76", equivalence_ratio=1.0)
    return CSTR(cfg)


def integrate_history(cstr, theta, durations, q0):
    from thermoreach.reactor import CSTR as _C          # noqa: F401
    return cstr.integrate(History(theta, durations), q0,
                          method="Radau", samples_per_segment=2,
                          rtol=1e-10, atol_T=1e-12, atol_Y=1e-16)


def build_library(cstr, q0):
    g_lo = GAMMA_REF * np.exp(-LIB_GAMMA_HALF_WIDTH_LOG)
    g_hi = GAMMA_REF * np.exp(LIB_GAMMA_HALF_WIDTH_LOG)
    grid = np.exp(np.linspace(np.log(g_lo), np.log(g_hi), LIB_GAMMA_POINTS))
    return ReferenceLibrary(cstr, q0, grid,
                            t_max=LIB_T_MAX_FACTOR * HORIZON,
                            t_anchor=HORIZON, method="Radau", rtol=1e-10,
                            atol_T=1e-12, atol_Y=1e-16, t_grid_points=21)


# ---------------------------------------------------------------------------
# 1. training set from the immutable phase-11 records
# ---------------------------------------------------------------------------


def training_case_from_phase11(rec, durations, T, gamma_ref):
    """Reconstruct a training record (eta, q_target, oracle reference) from a
    committed phase-11 case.  The eta is recovered from the stored controls."""
    ref = rec.get("reference") or {}
    q_t = rec.get("q_target")
    q_B = ref.get("q_B")
    if q_t is None or q_B is None or ref.get("gamma_best") is None:
        return None
    theta = np.asarray(rec["theta_actual"], dtype=float)
    eta = np.log(theta / gamma_ref)
    return {"label": rec["label"], "kind": rec["kind"], "eta": eta.tolist(),
            "q_target": q_t, "q_B": q_B,
            "measured_time_norm": time_norm(eta, durations, T),
            "located_reference": {
                "log_gamma_B": float(math.log(ref["gamma_best"] / gamma_ref)),
                "log_t_B_over_T": float(math.log(
                    max(ref["t_best"], 1e-30) / T))},
            "dist_scaled": ref.get("dist_scaled_upper_estimate")}


def load_training(phase11_json, durations, T, gamma_ref):
    raw = json.loads(Path(phase11_json).read_text(encoding="utf-8"))
    out = []
    for rec in raw["cases"]:
        if rec.get("status") != "completed":
            continue
        c = training_case_from_phase11(rec, durations, T, gamma_ref)
        if c is not None and 0.0 < c["measured_time_norm"] <= DOMAIN_MAX_MEASURED + 1e-9:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# 3. new admissible histories with a FROZEN stratification
# ---------------------------------------------------------------------------


def new_admissible_history(cstr, adm, theta0, durations, T, radius, rng,
                           gamma_ref, q0, index, role):
    """A new admissible history on a random unit-time-norm direction, generated
    with a seed fixed before any result was inspected."""
    for _trial in range(400):
        v = rng.standard_normal(len(theta0))
        v = v / np.sqrt(np.sum(hdiag(durations, T) * v ** 2))
        theta = np.asarray(theta0, dtype=float) * np.exp(radius * v)
        ok, reason = adm.validate(History(theta, durations))
        if ok:
            break
    else:
        return {"label": f"{role}_{index}", "role": role, "status": "excluded",
                "exclusion_reason": f"no admissible direction: {reason}"}
    hist = integrate_history(cstr, theta, durations, q0)
    if not hist["success"]:
        return {"label": f"{role}_{index}", "role": role, "status": "failed",
                "failure": hist["message"]}
    eta = np.log(theta / gamma_ref)
    return {"label": f"{role}_{index}", "role": role, "status": "completed",
            "eta": eta.tolist(),
            "theta": theta.tolist(), "durations": durations.tolist(),
            "direction_time_norm_coords": v.tolist(),
            "measured_time_norm": time_norm(eta, durations, T),
            "stratified_radius": float(radius),
            "q_target": np.asarray(hist["terminal_state"], dtype=float).tolist(),
            "admissible": True}


def _distribute(n, n_strata):
    """Split n over the strata as evenly as possible, larger strata first."""
    if n <= 0:
        return []
    base = n // n_strata
    extra = n - base * n_strata
    return [base + (1 if i < extra else 0) for i in range(n_strata)]


def new_history_schedule(n_val, n_test, durations, T, gamma_ref, cstr, adm,
                         theta0, q0):
    """Fixed seeds and a stratification in the MEASURED time norm, decided here
    before the model is evaluated on any of them.  A zero count generates no
    histories for that role.  With the nominal counts the split is validation
    3/3/2 = 8 and test 5/5/6 = 16."""
    val_counts = _distribute(n_val, len(VAL_RADII))
    test_counts = _distribute(n_test, len(TEST_RADII))
    out = []
    seed_val = stable_child_seed(20240, "phase13-validation")
    seed_test = stable_child_seed(20241, "phase13-test")
    rng_v = np.random.default_rng(seed_val)
    rng_t = np.random.default_rng(seed_test)
    i = 0
    for r, k in zip(VAL_RADII, val_counts):
        for _ in range(k):
            out.append((new_admissible_history(cstr, adm, theta0, durations, T,
                                               r, rng_v, gamma_ref, q0, i,
                                               "validation"), "validation"))
            i += 1
    i = 0
    for r, k in zip(TEST_RADII, test_counts):
        for _ in range(k):
            out.append((new_admissible_history(cstr, adm, theta0, durations, T,
                                               r, rng_t, gamma_ref, q0, i,
                                               "test"), "test"))
            i += 1
    return out


# ---------------------------------------------------------------------------
# 4/5. the four representations and the chemistry validation
# ---------------------------------------------------------------------------


def state_error(q_true, q_hat, scaling):
    q_true = np.asarray(q_true, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    W = np.asarray(scaling, dtype=float)
    d = W * (q_true - q_hat)
    return {"scaled_norm": float(np.linalg.norm(d)),
            "abs_dT_K": float(abs(q_true[0] - q_hat[0])),
            "max_abs_dY": float(np.max(np.abs(q_true[1:] - q_hat[1:]))),
            "rel_dT": float(abs(q_true[0] - q_hat[0]) / max(abs(q_true[0]),
                                                            1e-12))}


def oracle_reference_for(cstr, lib, q_target, scaling):
    """Variant A's reference search: locate the canonical family member nearest
    the true endpoint (oracle coordinates)."""
    return located_reference(lib, q_target, scaling, cstr)


def evaluate_case(cstr, lib, scaling, gamma_ref, T, durations, gen, case,
                  chem_cfg):
    """The A/B/C/D comparison for one test history, plus chemistry validation."""
    eta = np.asarray(case["eta"], dtype=float)
    q_true = np.asarray(case["q_target"], dtype=float)
    res = {"label": case["label"],
           "measured_time_norm": case["measured_time_norm"],
           "stratified_radius": case["stratified_radius"]}

    # ---- A: oracle reference coordinates, no correction --------------------
    refA = oracle_reference_for(cstr, lib, q_true, scaling)
    q_B_A = np.asarray(refA["q_B"], dtype=float)
    res["A_oracle_reference"] = {
        "gamma_B": refA.get("gamma_best"), "t_B": refA.get("t_best"),
        "error": state_error(q_true, q_B_A, scaling),
        "dist_note": refA.get("dist_note", "")}

    # ---- B: predicted reference coordinates, no correction -----------------
    # B is a family member: a real integrated state, admissible by construction,
    # so no decoder is involved and there is nothing to reject.
    pred = gen.predict(eta, durations)
    q_B_B = np.asarray(pred["canonical_state_before_correction"]["q_B"],
                       dtype=float)
    beta_pred = pred["predicted_reference"]
    res["B_predicted_reference"] = {
        "gamma_B": beta_pred["gamma_B"], "t_B": beta_pred["t_B_s"],
        "coordinate_error_vs_oracle": {
            "log_gamma": float(beta_pred["log_gamma_B"]
                               - math.log(float(refA["gamma_best"]) / gamma_ref)),
            "log_t_over_T": float(beta_pred["log_t_B_over_T"]
                                  - math.log(max(float(refA["t_best"]), 1e-30)
                                             / T))},
        "error": state_error(q_true, q_B_B, scaling)}

    # ---- D: oracle reference + oracle correction --------------------------
    n_Y = np.asarray(gen.n_Y, dtype=float)
    balA = exact_balances_from_controls(
        cstr, lib.q0, [float(refA["gamma_best"])], [float(refA["t_best"])])
    a_or = oracle_correction_coefficient(q_true, q_B_A[1:], n_Y,
                                         balA["h_J_kg"], gen.decoder,
                                         scaling=scaling)
    res["D_oracle_reference_plus_oracle_correction"] = {
        "a": a_or.get("a"), "resolved": a_or.get("resolved"),
        "feasible_interval": a_or.get("interval"),
        "error": (state_error(q_true, np.asarray(gen.decoder.decode(
            q_B_A[1:] + float(a_or["a"]) * n_Y, balA["h_J_kg"])["q_hat"],
            dtype=float), scaling) if a_or.get("resolved") else None)}

    # ---- C: predicted coordinates + predicted correction ------------------
    q_hat_C = np.asarray(pred["decoded_state"].get("q_hat"), dtype=float) \
        if pred["decoded_state"].get("status") == "admissible" else None
    res["C_predicted_reference_plus_predicted_correction"] = {
        "a": pred["predicted_correction"]["a"],
        "a_raw": pred["predicted_correction"]["a_raw"],
        "a_clipped_to_feasible":
            pred["predicted_correction"]["a_clipped_to_feasible"],
        "decoded_status": pred["decoded_state"].get("status"),
        "rejection_reason": pred["decoded_state"].get("reason")
        if pred["decoded_state"].get("status") != "admissible" else None,
        "feasible_interval": pred["predicted_correction"]["feasible_interval"],
        "error": state_error(q_true, q_hat_C, scaling) if q_hat_C is not None
        else None,
        "uncertainty": pred["uncertainty_indicator"]}

    # ---- chemistry validation on the three states -------------------------
    chem = {"predicted_canonical": chemistry_response(cstr, q_true, q_B_B,
                                                       chem_cfg)}
    if q_hat_C is not None:
        chem["predicted_corrected"] = chemistry_response(cstr, q_true, q_hat_C,
                                                         chem_cfg)
    chem["oracle_canonical"] = chemistry_response(cstr, q_true, q_B_A, chem_cfg)
    res["chemistry"] = chem
    res["provenance"] = pred["provenance"]
    return res


# ---------------------------------------------------------------------------
# summary statistics with honest denominators
# ---------------------------------------------------------------------------


def quantiles(xs):
    xs = np.asarray([x for x in xs if x is not None and math.isfinite(x)],
                    dtype=float)
    if xs.size == 0:
        return {"n": 0}
    q = np.quantile(xs, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"n": int(xs.size), "min": float(q[0]), "q25": float(q[1]),
            "median": float(q[2]), "q75": float(q[3]), "max": float(q[4]),
            "mean": float(xs.mean())}


def summarize(results, key, field="scaled_norm"):
    return quantiles([r.get(key, {}).get("error", {}).get(field)
                      for r in results])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase11", default=str(HERE.parent / "results" / "phase11"
                                             / "phase11_results.json"))
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cstr = make_cstr()
    q0 = hot_hp_state(cstr)
    durations = np.full(M, HORIZON / M)
    T = HORIZON
    scaling = StateScaling(T_interval=100.0, Y_interval=0.01,
                           n_species=cstr.gas.n_species).W
    adm = AdmissibleControls(gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI,
                             horizon=HORIZON)
    print("building the reference library (21 constant-exchange integrations)...")
    lib = build_library(cstr, q0)
    chem_cfg = ChemistryConfig()
    decoder = PhysicalDecoder(cstr)

    # ---------- 1. training set from the immutable phase-11 records ----------
    train = load_training(args.phase11, durations, T, GAMMA_REF)
    print(f"training cases reconstructed from phase 11: {len(train)}")
    if not train:
        raise SystemExit("no training cases reconstructed from phase 11")
    train_norms = [c["measured_time_norm"] for c in train]
    print(f"  training measured-norm range "
          f"[{min(train_norms):.4f}, {max(train_norms):.4f}]")

    # ---------- 2. fit both complexities on TRAINING only ------------------
    models, reports = {}, {}
    for order in args.orders:
        m, r = fit_local_generator(cstr, lib, train, durations, GAMMA_REF,
                                   HORIZON, scaling, decoder, order=order,
                                   ridge=args.ridge)
        models[order], reports[order] = m, r
        print(f"  order {order}: beta rms {r['beta_train_rms']:.3e}, "
              f"a rms {r['a_train_rms']:.3e}, cond(XtX) {r['cond_XtX']:.2e}, "
              f"normal kept {m.normal_report['retained_fraction']:.4f}")

    # ---------- 3. validation histories, seeds fixed in advance ------------
    # The seeds and the measured-norm stratification are fixed HERE, before any
    # validation result is inspected and before the model is frozen.
    val_sched = new_history_schedule(args.n_val, 0, durations, T, GAMMA_REF,
                                     cstr, adm, np.full(M, GAMMA_REF), q0)
    val_cases = [h for h, role in val_sched if role == "validation"
                 and h.get("status") == "completed"]
    print(f"validation histories generated: {len(val_cases)} completed")

    # ---------- 4. complexity selection on VALIDATION only -----------------
    val_eval = {order: [] for order in args.orders}
    for case in val_cases:
        for order in args.orders:
            val_eval[order].append(evaluate_case(
                cstr, lib, scaling, GAMMA_REF, T, durations, models[order],
                case, chem_cfg))
    order_scores = {}
    for order in args.orders:
        errs = [r["C_predicted_reference_plus_predicted_correction"].get("error")
                for r in val_eval[order]]
        order_scores[order] = quantiles([e.get("scaled_norm") if e else None
                                         for e in errs])
    scored = [o for o in args.orders if order_scores[o].get("n", 0) > 0]
    best_order = (min(scored, key=lambda o: order_scores[o]["median"])
                  if scored else args.orders[0])
    print(f"validation selection: order {best_order} "
          f"(median C error {order_scores[best_order]['median']:.3e})")

    # ---------- 5. FREEZE the selected model before any test history -------
    model, fit_report = models[best_order], reports[best_order]
    frozen = {"frozen_at_utc": utc_now(),
              "n_train": len(train),
              "order": best_order, "ridge": args.ridge,
              "selection": {"orders_tried": list(args.orders),
                            "selection_set": "validation only",
                            "n_validation": len(val_cases),
                            "order_scores": order_scores,
                            "selected": best_order},
              "fit_report": fit_report,
              "model_identity": model.model_identity,
              "training_domain": model.training_domain,
              "normal_report": model.normal_report,
              "n_Y": model.n_Y,
              "declaration": ("the model above is frozen AFTER validation-based "
                              "complexity selection and BEFORE any test history "
                              "is generated; no refit happens afterwards")}
    (OUT / "frozen_model.json").write_text(json.dumps(frozen, indent=1) + "\n",
                                           encoding="utf-8")

    # ---------- 6. test histories, independent fixed seeds -----------------
    test_sched = new_history_schedule(0, args.n_test, durations, T, GAMMA_REF,
                                      cstr, adm, np.full(M, GAMMA_REF), q0)
    test_cases = [h for h, role in test_sched if role == "test"
                  and h.get("status") == "completed"]
    print(f"test histories generated: {len(test_cases)} completed")

    # ---------- 7. the A/B/C/D comparison on the untouched TEST set --------
    results = []
    for case in test_cases:
        results.append(evaluate_case(cstr, lib, scaling, GAMMA_REF, T, durations,
                                     model, case, chem_cfg))
        (OUT / "phase13_results.json").write_text(
            json.dumps({"run": {"identifier": "phase13",
                                "timestamp_utc": utc_now(),
                                "n_completed_test": len(results),
                                "n_val_completed": len(val_cases)},
                        "test_results": results,
                        "validation_results": val_eval,
                        "frozen_model": frozen},
                       indent=1) + "\n", encoding="utf-8")
        r = results[-1]
        print(f"  test {r['label']}: norm={r['measured_time_norm']:.3f} "
              f"A={r['A_oracle_reference']['error']['scaled_norm']:.3e} "
              f"B={r['B_predicted_reference']['error']['scaled_norm']:.3e} "
              f"C={(r['C_predicted_reference_plus_predicted_correction'].get('error') or {}).get('scaled_norm')}")

    # ---------- 8. summary --------------------------------------------------
    summary = {
        "run": {"identifier": "phase13-generator", "timestamp_utc": utc_now(),
                "wall_seconds": time.perf_counter() - t0},
        "frozen_model": {"order": best_order, "ridge": args.ridge,
                         "n_train": len(train),
                         "selection": frozen["selection"],
                         "normal_retained_fraction":
                             model.normal_report["retained_fraction"],
                         "n_Y": model.n_Y},
        "n_train": len(train),
        "n_val": len(val_cases),
        "n_test": len(results),
        "errors_scaled_norm": {
            "A_oracle_reference": summarize(results, "A_oracle_reference"),
            "B_predicted_reference": summarize(results, "B_predicted_reference"),
            "C_predicted_plus_correction":
                summarize(results,
                          "C_predicted_reference_plus_predicted_correction"),
            "D_oracle_reference_plus_oracle_correction":
                summarize(results,
                          "D_oracle_reference_plus_oracle_correction")},
        "errors_abs_dT_K": {
            "A_oracle_reference": summarize(results, "A_oracle_reference",
                                            "abs_dT_K"),
            "B_predicted_reference": summarize(results, "B_predicted_reference",
                                               "abs_dT_K"),
            "C_predicted_plus_correction":
                summarize(results, "C_predicted_reference_plus_predicted_correction",
                          "abs_dT_K"),
            "D_oracle_reference_plus_oracle_correction":
                summarize(results,
                          "D_oracle_reference_plus_oracle_correction",
                          "abs_dT_K")},
        "errors_max_abs_dY": {
            "A_oracle_reference": summarize(results, "A_oracle_reference",
                                            "max_abs_dY"),
            "C_predicted_plus_correction":
                summarize(results, "C_predicted_reference_plus_predicted_correction",
                          "max_abs_dY")},
        "coordinate_error_vs_oracle": quantiles([
            r["B_predicted_reference"]["coordinate_error_vs_oracle"]["log_gamma"]
            for r in results]),
        "n_rejections": {
            k: int(sum(1 for r in results
                       if r.get("C_predicted_reference_plus_predicted_correction",
                                {}).get("decoded_status") == k))
            for k in ("admissible", "rejected_species_bounds",
                      "rejected_normalization", "rejected_enthalpy_inversion")},
        "domain": {"max_measured_time_norm_declared": DOMAIN_MAX_MEASURED,
                   "test_norm_range":
                       [min(r["measured_time_norm"] for r in results),
                        max(r["measured_time_norm"] for r in results)],
                   "strata": list(TEST_RADII)},
        "note": ("D is a representational measurement made with the TRUE "
                 "endpoint; C is a prediction from the controls alone.  The "
                 "gap C - D is the prediction cost, the gap A - D is what the "
                 "single correction buys, and A - B is the reference-coordinate "
                 "prediction cost."),
    }
    (OUT / "phase13_summary.json").write_text(json.dumps(summary, indent=1)
                                             + "\n", encoding="utf-8")
    manifest("phase13-generator", OUT, {
        "mechanism": cstr.cfg.mechanism,
        "mechanism_sha256_16": cstr.mech.get("sha256", "")[:16],
        "species": cstr.gas.species_names,
        "anchor": {"state": "hot HP equilibrium of the feed",
                   "T_in_K": T_IN, "p_Pa": P, "gamma_ref": GAMMA_REF,
                   "horizon_s": HORIZON, "m_segments": M,
                   "initial_state_sha256_16":
                       __import__("hashlib").sha256(
                           np.asarray(q0).tobytes()).hexdigest()[:16]},
        "training_source": args.phase11,
        "n_train": len(train), "n_val": len(val_cases),
        "n_test": len(results),
        "frozen_order": best_order,
        "domain_max_measured_time_norm": DOMAIN_MAX_MEASURED,
    }, cwd=HERE.parent, extra={"host": "redacted-host"})
    print(json.dumps(summary["errors_scaled_norm"], indent=1))


if __name__ == "__main__":
    main()
