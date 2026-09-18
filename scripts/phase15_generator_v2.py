"""Phase 15: the signed scalar correction, re-tested under the repaired feasible
set, and the reloadable frozen model (Revision 8, sections E and F).

Protocol (held to the brief):

  1. Diagnose on the existing records; the phase-13 validation/test histories are
     regenerated from their RECORDED fixed seeds and are development data from
     this point on.
  2. Fit the corrected one-direction model on the declared training set (the
     immutable phase-11 records); complexity is selected on the EXISTING
     validation set for a FIXED low-complexity choice, never against the new test
     performance.
  3. Freeze the full RELOADABLE model (coefficients, features, direction,
     structural classification, policy, hashes) to disk.
  4. Generate at most 12 NEW final-test histories on NEW recorded fixed seeds,
     four near each measured time norm 0.1 / 0.3 / 0.6, AFTER freezing.  The
     bounds, the initial state, the mechanism and the horizon are unchanged.
  5. Compare A/B/C/D/E - E = predicted reference plus ORACLE coefficient, which
     separates reference-coordinate error from scalar-coefficient prediction
     error - with failures and coefficient projections explicitly counted and
     never filtered out of the score denominators.
  6. The oracle values are conditional on the fixed located reference and are NOT
     a joint optimum over (beta, a); a local optimizer is not a certified global
     one.
  7. Chemistry reuses the existing diagnostic intervals and reports future-state
     AND increment errors; the same-pair gain is reported relative to dt = 0, and
     any AMPLIFICATION is preserved rather than reported as decay.
  8. Cost benchmark: regression, reference ODE, enthalpy inversion, library
     construction and the original switched-history integration, timed
     separately at MATCHED accuracy.  No speedup is claimed without equal-accuracy
     timings.

Outputs results/phase15/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from thermoreach.admissibility import AdmissibleControls, stable_child_seed  # noqa: E402
from thermoreach.chemresponse import (  # noqa: E402
    ChemistryConfig, ReferenceLibrary, chemistry_response, located_reference,
    time_norm)
from thermoreach.controls import History  # noqa: E402
from thermoreach.generator import (  # noqa: E402
    PhysicalDecoder, active_species_report, apply_structural_zeros,
    exact_balances_from_controls, feasible_a_interval, fit_local_generator,
    load_local_generator, model_record, oracle_correction_coefficient)
from thermoreach.io_utils import manifest, utc_now  # noqa: E402
from thermoreach.summary_utils import quantiles as _quantiles,     same_pair_gain as _same_pair_gain  # noqa: E402
from thermoreach.reactor import hot_hp_state  # noqa: E402
from thermoreach.sensitivity import StateScaling  # noqa: E402

from phase13_generator import (  # noqa: E402
    GAMMA_HI, GAMMA_LO, GAMMA_REF, HORIZON, M, build_library, integrate_history,
    load_training, make_cstr, new_history_schedule, state_error)

OUT = HERE.parent / "results" / "phase15"
TEST_RADII = (0.1, 0.3, 0.6)
N_TEST = 12                    # four per stratum; the brief's ceiling
SEED_TEST = 20250              # NEW seeds, disjoint from phase 13's 20240/20241


def _new_test_schedule(n_test, durations, T, cstr, adm, theta0, q0):
    """New final-test histories on new recorded seeds, stratified in the measured
    time norm.  The seeds are fixed before any result is inspected."""
    counts = [0, 0, 0]
    for i in range(n_test):
        counts[i % len(TEST_RADII)] += 1
    rng = np.random.default_rng(stable_child_seed(SEED_TEST, "phase15-test"))
    out = []
    for r, k in zip(TEST_RADII, counts):
        for _ in range(k):
            for _trial in range(400):
                v = rng.standard_normal(len(theta0))
                v = v / np.sqrt(np.sum(
                    np.asarray(durations, dtype=float) / T * v ** 2))
                theta = np.asarray(theta0, dtype=float) * np.exp(r * v)
                ok, reason = adm.validate(History(theta, durations))
                if ok:
                    break
            else:
                out.append({"label": f"test_{len(out)}", "role": "test",
                            "status": "excluded",
                            "exclusion_reason": f"no admissible direction: {reason}",
                            "stratified_radius": float(r)})
                continue
            hist = integrate_history(cstr, theta, durations, q0)
            eta = np.log(theta / GAMMA_REF)
            rec = {"label": f"test_{len(out)}", "role": "test",
                   "status": hist["success"] and "completed" or "failed",
                   "eta": eta.tolist(), "theta": theta.tolist(),
                   "durations_s": list(durations),
                   "direction_time_norm_coords": v.tolist(),
                   "measured_time_norm": time_norm(eta, durations, T),
                   "stratified_radius": float(r),
                   "q_target": (np.asarray(hist["terminal_state"],
                                           dtype=float).tolist()
                                if hist["success"] else None),
                   "admissible": bool(hist["success"])}
            if not hist["success"]:
                rec["failure"] = hist.get("message")
            out.append(rec)
    return out


def evaluate_case(cstr, lib, scaling, gamma_ref, T, durations, gen, case,
                  chem_cfg):
    """The A/B/C/D/E comparison for one test history, plus chemistry."""
    eta = np.asarray(case["eta"], dtype=float)
    q_true = np.asarray(case["q_target"], dtype=float)
    res = {"label": case["label"],
           "measured_time_norm": case["measured_time_norm"],
           "stratified_radius": case["stratified_radius"]}

    refA = located_reference(lib, q_true, scaling, cstr)
    resolved = refA.get("q_B") is not None    # success is a located family member
    res["A_oracle_reference"] = {
        "gamma_B": refA.get("gamma_best"), "t_B": refA.get("t_best"),
        "resolved": resolved,
        "error": (state_error(q_true, np.asarray(refA["q_B"], dtype=float),
                              scaling) if refA.get("resolved") else None),
        "dist_note": refA.get("dist_note", "")}

    pred = gen.predict(eta, durations)
    beta_pred = pred["predicted_reference"]
    q_B_B = np.asarray(pred["canonical_state_before_correction"]["q_B"],
                       dtype=float)
    res["B_predicted_reference"] = {
        "gamma_B": beta_pred["gamma_B"], "t_B": beta_pred["t_B_s"],
        "coordinate_error_vs_oracle": {
            "log_gamma": float(beta_pred["log_gamma_B"]
                               - math.log(max(float(refA["gamma_best"]), 1e-300)
                                          / gamma_ref)),
            "log_t_over_T": float(beta_pred["log_t_B_over_T"]
                                  - math.log(max(float(refA["t_best"]), 1e-300)
                                             / T))},
        "error": (state_error(q_true, q_B_B, scaling)
                  if pred["canonical_state_before_correction"].get("q_B")
                  is not None else None)}

    n_Y = apply_structural_zeros(np.asarray(gen.n_Y, dtype=float),
                                 gen.species_report)
    species_names = gen.species_report["species_names"]
    fixed_mask = gen.species_report["structurally_zero_mask"]

    # ---- D: oracle reference + oracle correction --------------------------
    if resolved:
        balA = exact_balances_from_controls(
            cstr, lib.q0, [float(refA["gamma_best"])],
            [float(refA["t_best"])])
        a_or = oracle_correction_coefficient(q_true, refA["q_B"][1:], n_Y,
                                             balA["h_J_kg"], gen.decoder,
                                             scaling=scaling)
        decD = gen.decoder.decode(
            np.asarray(refA["q_B"], dtype=float)[1:]
            + float(a_or.get("a") or 0.0) * n_Y, balA["h_J_kg"])
        res["D_oracle_reference_plus_oracle_correction"] = {
            "a": a_or.get("a"), "resolved": a_or.get("resolved"),
            "a_lin": (a_or.get("linear_estimate") or {}).get("a_lin"),
            "active_bound": a_or.get("active_bound"),
            "feasible_interval": a_or.get("interval"),
            "decoded_status": decD.get("status"),
            "error": (state_error(q_true, np.asarray(decD["q_hat"], dtype=float),
                                  scaling)
                      if decD.get("status") == "admissible" else None)}

    # ---- C: predicted coordinates + predicted correction ------------------
    q_hat_C = np.asarray(pred["decoded_state"].get("q_hat"), dtype=float) \
        if pred["decoded_state"].get("status") == "admissible" else None
    res["C_predicted_reference_plus_predicted_correction"] = {
        "a": pred["predicted_correction"]["a"],
        "a_raw": pred["predicted_correction"]["a_raw"],
        "a_raw_outside_feasible_interval":
            bool(pred["predicted_correction"]["a_clipped_to_feasible"]),
        "decoded_status": pred["decoded_state"].get("status"),
        "rejection_reason": pred["decoded_state"].get("reason")
        if pred["decoded_state"].get("status") != "admissible" else None,
        "feasible_interval": pred["predicted_correction"]["feasible_interval"],
        "error": state_error(q_true, q_hat_C, scaling) if q_hat_C is not None
        else None,
        "uncertainty": pred["uncertainty_indicator"]}

    # ---- E: predicted reference + ORACLE correction -----------------------
    # separates reference-coordinate error from scalar-coefficient prediction
    # error; conditional on the PREDICTED reference, not a joint optimum
    if pred["canonical_state_before_correction"].get("q_B") is not None:
        balB = exact_balances_from_controls(
            cstr, lib.q0, [float(beta_pred["gamma_B"])],
            [float(beta_pred["t_B_s"])])
        a_orE = oracle_correction_coefficient(q_true, q_B_B[1:], n_Y,
                                              balB["h_J_kg"], gen.decoder,
                                              scaling=scaling)
        decE = gen.decoder.decode(
            q_B_B[1:] + float(a_orE.get("a") or 0.0) * n_Y, balB["h_J_kg"])
        res["E_predicted_reference_plus_oracle_correction"] = {
            "a": a_orE.get("a"), "resolved": a_orE.get("resolved"),
            "active_bound": a_orE.get("active_bound"),
            "decoded_status": decE.get("status"),
            "error": (state_error(q_true, np.asarray(decE["q_hat"], dtype=float),
                                  scaling)
                      if decE.get("status") == "admissible" else None)}

    # ---- chemistry on the states (future state AND increments) ------------
    chem = {}
    if pred["canonical_state_before_correction"].get("q_B") is not None:
        chem["predicted_canonical"] = chemistry_response(cstr, q_true, q_B_B,
                                                         chem_cfg)
    if q_hat_C is not None:
        chem["predicted_corrected"] = chemistry_response(cstr, q_true, q_hat_C,
                                                         chem_cfg)
    if resolved:
        chem["oracle_canonical"] = chemistry_response(
            cstr, q_true, np.asarray(refA["q_B"], dtype=float), chem_cfg)
    res["chemistry"] = chem
    res["provenance"] = pred["provenance"]
    return res


# the gain and quantile helpers live in thermoreach.summary_utils so their
# arithmetic is unit-testable without Cantera


def _cost_benchmark(cstr, lib, gen, theta, durations, q0, scaling):
    """Separate timings at MATCHED accuracy.  No speedup is claimed without
    equal-accuracy timings; the achieved accuracy is reported alongside."""
    eta = np.log(np.asarray(theta, dtype=float) / GAMMA_REF)
    bench = {}

    t0 = time.perf_counter()
    for _ in range(20):
        _ = gen.beta_regressor.predict(eta)
        _ = gen.a_regressor.predict(eta)
    bench["regression_s_per_call"] = (time.perf_counter() - t0) / 20.0

    t0 = time.perf_counter()
    for _ in range(5):
        _ = gen.reference_state(0.0, HORIZON)
    bench["reference_ode_s_per_call"] = (time.perf_counter() - t0) / 5.0

    q_B = np.asarray(gen.library.evaluate_fresh(GAMMA_REF, HORIZON)["q_B"],
                     dtype=float)
    h_B = exact_balances_from_controls(cstr, lib.q0, [GAMMA_REF],
                                       [HORIZON])["h_J_kg"]
    n_Y = apply_structural_zeros(np.asarray(gen.n_Y, dtype=float),
                                 gen.species_report)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = gen.decoder.decode(q_B[1:] + 1e-4 * n_Y, h_B)
    bench["enthalpy_inversion_and_correction_s_per_call"] = \
        (time.perf_counter() - t0) / 20.0

    hist = History(np.asarray(theta, dtype=float), durations)
    t0 = time.perf_counter()
    for _ in range(3):
        r = cstr.integrate(hist, q0, method="Radau", samples_per_segment=2,
                           rtol=1e-10, atol_T=1e-12, atol_Y=1e-16)
    bench["original_switched_history_integration_s_per_call"] = \
        (time.perf_counter() - t0) / 3.0

    t0 = time.perf_counter()
    full = gen.predict(eta, durations)
    bench["full_predict_s_per_call"] = time.perf_counter() - t0

    bench["accuracy_of_the_timed_paths"] = {
        "reference_ode_rtol": 1e-10,
        "history_integration_rtol": 1e-10,
        "decoded_state_status": full["decoded_state"].get("status"),
        "state_error_scaled_norm_of_full_prediction":
            float(np.linalg.norm(scaling * (
                np.asarray(full["decoded_state"].get("q_hat",
                                                     np.zeros(q_B.size)),
                            dtype=float)
                - np.asarray(r["terminal_state"], dtype=float))))
            if r.get("success") else None,
        "note": ("the two integrators are timed at the SAME rtol; the reported "
                 "accuracy is the achieved scaled state error of the full "
                 "prediction against the switched-history integration")}
    return bench


def quantiles(xs):
    return _quantiles(xs)


VARIANTS = ["A_oracle_reference", "B_predicted_reference",
            "C_predicted_reference_plus_predicted_correction",
            "D_oracle_reference_plus_oracle_correction",
            "E_predicted_reference_plus_oracle_correction"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase11", default=str(HERE.parent / "results" / "phase11"
                                             / "phase11_results.json"))
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ridge", type=float, default=1e-8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cstr = make_cstr()
    q0 = hot_hp_state(cstr)
    durations = np.full(M, HORIZON / M)
    T = HORIZON
    W = StateScaling(T_interval=100.0, Y_interval=0.01,
                     n_species=cstr.gas.n_species).W
    adm = AdmissibleControls(gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI,
                             horizon=HORIZON)
    t0 = time.perf_counter()
    lib = build_library(cstr, q0)
    lib_time = time.perf_counter() - t0
    print(f"reference library built in {lib_time:.2f}s")
    decoder = PhysicalDecoder(cstr)
    chem_cfg = ChemistryConfig()

    report = active_species_report(cstr, q0)
    print(f"structural face: absent {[cstr.gas.species_names[k] for k in report['absent_indices']]} "
          f"fixed {[cstr.gas.species_names[k] for k in report['fixed_indices']]} "
          f"active {[cstr.gas.species_names[k] for k in report['active_indices']]}")

    # ---------- 1/2. training from the immutable records, fit, validate -----
    train = load_training(args.phase11, durations, T, GAMMA_REF)
    print(f"training cases reconstructed from phase 11: {len(train)}")
    models, reports = {}, {}
    for order in args.orders:
        m, r = fit_local_generator(cstr, lib, train, durations, GAMMA_REF,
                                   HORIZON, W, decoder, order=order,
                                   ridge=args.ridge)
        models[order], reports[order] = m, r
        print(f"  order {order}: beta rms {r['beta_train_rms']:.3e}, "
              f"a rms {r['a_train_rms']:.3e}, cond(XtX) {r['cond_XtX']:.2e}")

    # the EXISTING validation set, regenerated from its recorded seeds; it is
    # development data, used for the fixed low-complexity choice only
    val_sched = new_history_schedule(8, 0, durations, T, GAMMA_REF, cstr, adm,
                                     np.full(M, float(GAMMA_REF)), q0)
    val_cases = [h for h, role in val_sched if role == "validation"
                 and h.get("status") == "completed"]
    print(f"validation histories regenerated from their seeds: {len(val_cases)}")
    order_scores = {o: [] for o in args.orders}
    # the complexity choice is made on the CANONICAL prediction (the B variant):
    # it is the model's core output, and mixing corrected with uncorrected errors
    # here would compare different quantities across orders
    for case in val_cases:
        eta = np.asarray(case["eta"], dtype=float)
        q_true = np.asarray(case["q_target"], dtype=float)
        for order in args.orders:
            pred = models[order].predict(eta, durations)
            q_hat = pred["canonical_state_before_correction"].get("q_B")
            order_scores[order].append(None if q_hat is None else
                                      state_error(q_true,
                                                  np.asarray(q_hat, dtype=float),
                                                  W)["scaled_norm"])
    for order in args.orders:
        print(f"  validation order {order}: median "
              f"{quantiles(order_scores[order]).get('median', float('nan')):.3e}")
    selected = min(args.orders,
                   key=lambda o: quantiles(order_scores[o]).get("median",
                                                                float("inf")))
    print(f"complexity selected on the existing validation set: order {selected}")
    gen = models[selected]

    # ---------- 3. freeze the RELOADABLE model ------------------------------
    record = model_record(
        gen,
        training_case_ids=[c["label"] for c in train],
        validation_case_ids=[c["label"] for c in val_cases],
        config={"orders_tried": list(args.orders), "ridge": args.ridge,
                "n_test_requested": int(args.n_test)},
        selection={"selected": int(selected),
                   "selection_set": ("the existing validation set, regenerated "
                                     "from its recorded seeds (development "
                                     "data); fixed low-complexity choice"),
                   "orders_tried": list(args.orders),
                   "order_scores": {str(o): quantiles(order_scores[o])
                                    for o in args.orders},
                   "n_validation": len(val_cases)})
    (OUT / "frozen_model_v2.json").write_text(json.dumps(record, indent=1),
                                              encoding="utf-8")
    reloaded = load_local_generator(record, cstr, lib, decoder)
    assert not reloaded.reload_warnings, reloaded.reload_warnings
    print("the frozen model is reloadable with no identity warnings")

    # ---------- 4. NEW final-test histories, AFTER freezing ----------------
    test_sched = _new_test_schedule(args.n_test, durations, T, cstr, adm,
                                    np.full(M, float(GAMMA_REF)), q0)
    n_requested = len(test_sched)
    n_excluded = sum(1 for c in test_sched if c.get("status") == "excluded")
    n_failed = sum(1 for c in test_sched if c.get("status") == "failed")
    test_cases = [c for c in test_sched if c.get("status") == "completed"]
    print(f"new final-test histories: {n_requested} requested, "
          f"{len(test_cases)} completed, {n_failed} failed, {n_excluded} "
          f"excluded (failures are counted, never filtered out)")

    # the raw trajectories are stored in full for reproducibility
    (OUT / "test_trajectories.json").write_text(
        json.dumps({"seeds": {"test": stable_child_seed(SEED_TEST,
                                                        "phase15-test")},
                    "strata": list(TEST_RADII), "cases": test_sched},
                   indent=1), encoding="utf-8")

    # ---------- 5. A/B/C/D/E ----------------------------------------------
    results = []
    for case in test_cases:
        r = evaluate_case(cstr, lib, W, GAMMA_REF, T, durations, gen, case,
                          chem_cfg)
        results.append(r)
        errs = {k.split("_")[0]: (r.get(k) or {}).get("error", {}) or {}
                for k in VARIANTS}
        print(f"  {case['label']}: norm={case['measured_time_norm']:.2f} "
              + " ".join(f"{k}={((v or {}).get('scaled_norm') if v else None)}"
                         for k, v in errs.items()))

    # ---------- 6/7. chemistry same-pair gains ------------------------------
    for r in results:
        r["chemistry_gains"] = {
            k: _same_pair_gain(r["chemistry"].get(k))
            for k in r["chemistry"]}

    # ---------- 8. the cost benchmark --------------------------------------
    theta0 = np.asarray(test_cases[0]["theta"], dtype=float) \
        if test_cases else np.full(M, float(GAMMA_REF))
    bench = _cost_benchmark(cstr, lib, gen, theta0, durations, q0, W)
    bench["reference_library_construction_s"] = float(lib_time)
    print(f"cost: regression {bench['regression_s_per_call']:.3e}s, "
          f"reference ODE {bench['reference_ode_s_per_call']:.3e}s, "
          f"decoder {bench['enthalpy_inversion_and_correction_s_per_call']:.3e}s, "
          f"history integration "
          f"{bench['original_switched_history_integration_s_per_call']:.3e}s")

    # ---------- honest denominators ----------------------------------------
    summary = {
        "n_test_requested": int(n_requested),
        "n_test_completed": int(len(test_cases)),
        "n_test_failed": int(n_failed),
        "n_test_excluded": int(n_excluded),
        "n_a_raw_outside_feasible_interval": int(sum(
            1 for r in results
            if (r.get("C_predicted_reference_plus_predicted_correction") or {})
            .get("a_raw_outside_feasible_interval"))),
        "n_decoder_rejections": {
            k: int(v) for k, v in decoder.n_rejected.items()},
        "errors_scaled_norm": {
            k: quantiles([(r.get(k) or {}).get("error", {}) and
                          (r.get(k) or {}).get("error", {}).get("scaled_norm")
                          for r in results]) for k in VARIANTS},
        "coordinate_errors": quantiles([
            (r.get("B_predicted_reference") or {})
            .get("coordinate_error_vs_oracle", {}).get("log_gamma")
            for r in results]),
        "oracle_a_stats": quantiles([
            (r.get("D_oracle_reference_plus_oracle_correction") or {}).get("a")
            for r in results]),
        "n_oracle_a_negative": int(sum(
            1 for r in results
            if ((r.get("D_oracle_reference_plus_oracle_correction") or {})
                .get("a") or 0.0) < 0.0)),
        "n_oracle_a_at_active_bound": int(sum(
            1 for r in results
            if (r.get("D_oracle_reference_plus_oracle_correction") or {})
            .get("active_bound") is not None)),
        "selected_order": int(selected),
        "selection": record["selection"],
        "structural_classification": {
            "absent": [cstr.gas.species_names[k]
                       for k in report["absent_indices"]],
            "fixed": [cstr.gas.species_names[k]
                      for k in report["fixed_indices"]],
            "active": [cstr.gas.species_names[k]
                       for k in report["active_indices"]]},
        "cost_benchmark": bench,
        "declaration": (
            "the model (complexity, regularization, the correction direction "
            "and both coefficient arrays) was frozen to frozen_model_v2.json "
            "BEFORE any of the new final-test histories above was generated; "
            "the test seeds are new and disjoint from phase 13's; the oracle "
            "values are conditional on the fixed located reference and are not a "
            "joint optimum over (beta, a)"),
    }
    for k in VARIANTS:
        summary["errors_scaled_norm"][k]["n_with_error"] = int(sum(
            1 for r in results if (r.get(k) or {}).get("error") is not None))

    (OUT / "phase15_results.json").write_text(
        json.dumps({"run": {"identifier": "phase15",
                            "timestamp_utc": utc_now(),
                            "frozen_model": "frozen_model_v2.json",
                            "phase13_frozen_model_for_comparison":
                                "results/phase13/frozen_model.json"},
                    "test_results": results, "summary": summary},
                   indent=1), encoding="utf-8")

    with (OUT / "phase15_per_case.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["label", "norm"] + [f"{v}_err" for v in VARIANTS]
                     + ["a_raw", "a_applied", "a_raw_outside", "decoded_status",
                        "D_a", "D_active_bound", "E_a"])
        for r in results:
            C = r.get("C_predicted_reference_plus_predicted_correction") or {}
            D = r.get("D_oracle_reference_plus_oracle_correction") or {}
            E = r.get("E_predicted_reference_plus_oracle_correction") or {}
            wtr.writerow([r["label"], f"{r['measured_time_norm']:.4f}"]
                         + [((r.get(v) or {}).get("error") or {})
                            .get("scaled_norm") for v in VARIANTS]
                         + [C.get("a_raw"), C.get("a"),
                            C.get("a_raw_outside_feasible_interval"),
                            C.get("decoded_status"), D.get("a"),
                            D.get("active_bound"), E.get("a")])

    manifest("phase15", OUT,
             {"script": "scripts/phase15_generator_v2.py",
              "frozen_model": "frozen_model_v2.json",
              "test_trajectories": "test_trajectories.json",
              "protocol": ("fit on phase-11 training; complexity on the existing "
                           "validation set; freeze; then 12 new test histories "
                           "on new seeds")},
             cwd=HERE.parent)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
