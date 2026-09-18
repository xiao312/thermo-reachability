"""Phase 14: the structural face of the species polytope, and a re-evaluation of
the scalar oracle (Revision 8, sections A and B).

DIAGNOSTIC ONLY.  No model is fit here and no new untouched history is generated.
The inputs are the immutable committed records (phase 11, and the phase-13
validation/test histories regenerated from their recorded fixed seeds, which are
henceforth DEVELOPMENT data - the correction and the oracle are re-evaluated on
them as diagnostic data, not as untouched evidence).

It reports

  A.1 the source species, the feed elemental inventories, the numerical
      direction entries, and the binding constraint for selected positive,
      negative and zero-optimum cases;
  A.2/A.3 the structural classification (absent / fixed / active) and the
      corrected feasible interval with ORIGINAL species indices and names;
  A.7 the separation of the raw from the applied correction coefficient;
  B   the signed scalar objective near zero, the local linear estimate
          a_lin = j^T W^T W (q - q_B) / (j^T W^T W j),   j = (dT/da, n_Y)
      from dT/da = -sum_k h_k(T) n_k / cp(T, Y), the active-bound status, and the
      recomputed oracle with the corrected interval - with an explicit count of
      how many previously-ZERO optima become NEGATIVE.

Outputs results/phase14/phase14_structural_face.json plus a CSV table.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from thermoreach.admissibility import AdmissibleControls  # noqa: E402
from thermoreach.chemresponse import (  # noqa: E402
    ChemistryConfig, ReferenceLibrary, chemistry_response, hdiag,
    located_reference, time_norm)
from thermoreach.controls import History  # noqa: E402
from thermoreach.generator import (  # noqa: E402
    PhysicalDecoder, active_species_report, apply_structural_zeros,
    exact_balances_from_controls, feasible_a_interval,
    oracle_correction_coefficient, project_onto_active_conservation_subspace,
    project_onto_conservation_subspace)
from thermoreach.io_utils import manifest  # noqa: E402
from thermoreach.reactor import hot_hp_state  # noqa: E402
from thermoreach.sensitivity import StateScaling  # noqa: E402

from phase13_generator import (  # noqa: E402
    DOMAIN_MAX_MEASURED, GAMMA_HI, GAMMA_LO, GAMMA_REF, HORIZON, M,
    build_library, load_training, make_cstr, new_history_schedule)

OUT = HERE.parent / "results" / "phase14"


def _fit_direction(train_cases, W, cstr, species_report, *, active: bool):
    """Recompute the fitted direction from the STORED residuals.  `active=False`
    reproduces the Revision-7 code path (projection onto the full conservation
    subspace, which zeroes AR/N2 only to SVD noise); `active=True` uses the
    structural mask with exact zeros."""
    R = np.stack([W * (np.asarray(c["q_target"], dtype=float)
                       - np.asarray(c["q_B"], dtype=float))
                  for c in train_cases], axis=1)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    w_Y = U[1:, 0]
    y_interval = 1.0 / W[1]
    if active:
        rep = project_onto_active_conservation_subspace(
            w_Y, cstr.E, species_report["active_mask"], y_interval=y_interval)
    else:
        rep = project_onto_conservation_subspace(w_Y, cstr.E,
                                                y_interval=y_interval)
    if not rep["resolved"]:
        raise RuntimeError(f"direction unresolved: {rep['reason']}")
    n = np.asarray(rep["n_Y"], dtype=float)
    if active:
        n = apply_structural_zeros(n, species_report)
    return n, S, rep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cstr = make_cstr()
    q0 = hot_hp_state(cstr)                       # the anchor start
    durations = np.full(M, HORIZON / M)
    T = HORIZON
    W = StateScaling(T_interval=100.0, Y_interval=0.01,
                     n_species=cstr.gas.n_species).W
    lib = build_library(cstr, q0)
    decoder = PhysicalDecoder(cstr)
    adm = AdmissibleControls(gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI,
                             horizon=HORIZON)

    # ---------------- A.2/A.3 the structural classification ---------------
    report = active_species_report(cstr, q0)
    names = report["species_names"]
    print("structural classification:")
    print(f"  absent : {[names[k] for k in report['absent_indices']]}")
    print(f"  fixed  : {[names[k] for k in report['fixed_indices']]}")
    print(f"  active : {[names[k] for k in report['active_indices']]}")
    print(f"  element carriers: {report['element_carriers']}")
    if report["inconsistent"]:
        print(f"  INCONSISTENT: {report['inconsistent']}")

    # the committed frozen direction, for the noise-entry report
    n_frozen = np.asarray(json.loads(
        (HERE.parent / "results" / "phase13" / "frozen_model.json"
         ).read_text(encoding="utf-8"))["n_Y"], dtype=float)

    # ---------------- the two directions -----------------------------------
    train = load_training(str(HERE.parent / "results" / "phase11"
                              / "phase11_results.json"),
                          durations, T, GAMMA_REF)
    print(f"training cases reconstructed: {len(train)}")
    n_old, S_old, rep_old = _fit_direction(train, W, cstr, report, active=False)
    n_new, S_new, rep_new = _fit_direction(train, W, cstr, report, active=True)
    for k in report["absent_indices"] + report["fixed_indices"]:
        print(f"  {names[k]:>6s}: frozen n = {n_frozen[k]:+.3e}, "
              f"recomputed n = {n_old[k]:+.3e}, corrected n = {n_new[k]:+.3e}")
    print(f"  max |n_new - n_old| = {np.max(np.abs(n_new - n_old)):.3e}")

    # ---------------- A.1/A.7/B the per-case diagnostic --------------------
    def diagnose(case, n_dir, label, role):
        q = np.asarray(case["q_target"], dtype=float)
        Y_B = np.asarray(case["q_B"], dtype=float)[1:]
        lr = case["located_reference"]
        bal = exact_balances_from_controls(
            cstr, lib.q0,
            [GAMMA_REF * math.exp(float(lr["log_gamma_B"]))],
            [T * math.exp(float(lr["log_t_B_over_T"]))])
        h_B = bal["h_J_kg"]
        n_z = apply_structural_zeros(n_dir, report)
        iv_old = feasible_a_interval(Y_B, n_dir)
        iv_new = feasible_a_interval(Y_B, n_z, names,
                                     fixed_mask=report["structurally_zero_mask"])
        orc = oracle_correction_coefficient(q, Y_B, n_z, h_B, decoder, scaling=W)
        # a small signed scan of the objective near zero, both signs; a REJECTED
        # decode is infinite distance, never a finite-looking number
        width = (iv_new["interval_width"] or 1.0)

        def _dist_at(a):
            dec = decoder.decode(Y_B + a * n_z, h_B)
            if dec.get("status") != "admissible":
                return None                      # rejected, not infinite-looking
            return float(np.linalg.norm(
                W * (q - np.asarray(dec["q_hat"], dtype=float))))

        scan = [{"a": float(a), "dist": _dist_at(a),
                 "rejected": _dist_at(a) is None}
                for a in np.linspace(-0.05 * width, 0.05 * width, 21)]
        return {"label": label, "role": role,
                "measured_time_norm": float(case["measured_time_norm"]),
                "interval_old": {k: iv_old.get(k) for k in
                                 ("a_lo", "a_hi", "bounded", "interval_width",
                                  "binding_name_low", "binding_name_high",
                                  "inconsistent")},
                "interval_new": {k: iv_new.get(k) for k in
                                 ("a_lo", "a_hi", "bounded", "interval_width",
                                  "binding_name_low", "binding_name_high")},
                "oracle_a": orc.get("a"),
                "oracle_active_bound": orc.get("active_bound"),
                "a_lin": (orc.get("linear_estimate") or {}).get("a_lin"),
                "dT_over_da": (orc.get("linear_estimate") or {}).get("dT_over_da"),
                "cp": (orc.get("linear_estimate") or {}).get("cp_J_kg_K"),
                "dist_at_oracle": orc.get("dist_scaled_raw"),
                "signed_objective_near_zero": scan,
                "n_candidates": orc.get("n_candidates")}

    rows = [diagnose(c, n_old, c["label"], "train") for c in train]

    # the phase-13 validation/test histories are regenerated from their recorded
    # fixed seeds; from now on they are development/diagnostic data
    sched = new_history_schedule(8, 16, durations, T, GAMMA_REF, cstr, adm,
                                 np.full(M, float(GAMMA_REF)), q0)
    regen = []
    for case, role in sched:
        if case.get("status") != "completed":
            continue
        q = np.asarray(case["q_target"], dtype=float)
        loc = located_reference(lib, q, W, cstr)
        if loc.get("q_B") is None:          # success is a located family member
            continue
        c = {"label": case.get("label", role),
             "eta": case["eta"], "q_target": case["q_target"],
             "q_B": loc["q_B"],
             "measured_time_norm": case["measured_time_norm"],
             "located_reference": {
                 "log_gamma_B": float(math.log(
                     max(loc["gamma_best"], 1e-300) / GAMMA_REF)),
                 "log_t_B_over_T": float(math.log(
                     max(loc["t_best"], 1e-300) / T))}}
        regen.append((c, role))
    print(f"phase-13 histories regenerated from their seeds: {len(regen)}")
    rows += [diagnose(c, n_old, c["label"], f"phase13-{role}")
             for c, role in regen]

    # ---------------- the correction-to-claim ledger -----------------------
    neg = [r for r in rows if (r["oracle_a"] or 0.0) < 0.0]
    pos = [r for r in rows if (r["oracle_a"] or 0.0) > 0.0]
    zeros = [r for r in rows if abs(r["oracle_a"] or 0.0) <= 1e-14]
    train_zero = [r for r in rows if r["role"] == "train"
                  and abs(r["oracle_a"] or 0.0) <= 1e-14]
    print(f"oracle coefficient over {len(rows)} diagnostic cases:")
    print(f"  negative: {len(neg)}   zero: {len(zeros)}   positive: {len(pos)}")
    print(f"  previously-zero TRAINING optima: {len(train_zero)}; of those, "
          f"{sum(1 for r in train_zero if (r['oracle_a'] or 0.0) < 0)} now "
          f"negative")
    # how many intervals were one-sided before the repair
    onesided = [r for r in rows
                if (r["interval_old"].get("a_lo") is not None
                    and float(r["interval_old"]["a_lo"]) == 0.0)]
    print(f"  one-sided (a_lo == 0) intervals before the repair: "
          f"{len(onesided)} of {len(rows)}")

    def pick(pred):
        return next((r for r in rows if pred(r)), None)
    selected = {
        "positive_optimum": pick(lambda r: (r["oracle_a"] or 0.0) > 0),
        "negative_optimum": pick(lambda r: (r["oracle_a"] or 0.0) < 0),
        "zero_optimum": pick(lambda r: abs(r["oracle_a"] or 0.0) <= 1e-14)}

    csv_path = OUT / "phase14_per_case.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["label", "role", "measured_norm",
                      "iv_old_lo", "iv_old_hi", "iv_old_width",
                      "iv_new_lo", "iv_new_hi", "iv_new_width",
                      "binding_lo", "binding_hi",
                      "a_oracle", "a_lin", "active_bound", "dT_over_da"])
        for r in rows:
            io, inew = r["interval_old"], r["interval_new"]
            wtr.writerow([r["label"], r["role"],
                          f"{r['measured_time_norm']:.4f}",
                          io["a_lo"], io["a_hi"], io["interval_width"],
                          inew["a_lo"], inew["a_hi"], inew["interval_width"],
                          inew["binding_name_low"], inew["binding_name_high"],
                          r["oracle_a"], r["a_lin"], r["oracle_active_bound"],
                          r["dT_over_da"]])

    payload = {
        "structural_classification": {
            "species_names": names,
            "absent": [names[k] for k in report["absent_indices"]],
            "fixed": [names[k] for k in report["fixed_indices"]],
            "active": [names[k] for k in report["active_indices"]],
            "b_in": report["b_in"],
            "element_carriers": report["element_carriers"],
            "inconsistent": report["inconsistent"],
            "note": report["note"],
        },
        "directions": {
            "frozen_revision7": n_frozen.tolist(),
            "recomputed_revision7_code_path": n_old.tolist(),
            "corrected": n_new.tolist(),
            "max_abs_difference_new_vs_old":
                float(np.max(np.abs(n_new - n_old))),
            "noisy_entries": {names[k]: {"frozen": float(n_frozen[k]),
                                         "recomputed": float(n_old[k]),
                                         "corrected": float(n_new[k])}
                              for k in report["absent_indices"]
                              + report["fixed_indices"]},
            "residual_singular_values": S_old.tolist(),
            "projection_old": {"retained_fraction":
                               rep_old["projection_retained_fraction"],
                               "subspace": rep_old["subspace"]},
            "projection_new": {"retained_fraction":
                               rep_new["projection_retained_fraction"],
                               "subspace": rep_new["subspace"]},
        },
        "ledger": {
            "n_diagnostic_cases": len(rows),
            "n_negative": len(neg), "n_zero": len(zeros), "n_positive": len(pos),
            "previously_zero_training_optima": len(train_zero),
            "now_negative_among_those":
                sum(1 for r in train_zero if (r["oracle_a"] or 0.0) < 0),
            "one_sided_intervals_before_repair": len(onesided),
        },
        "selected_cases": selected,
        "per_case": rows,
        "protocol_note": (
            "diagnostic only; no model fit and no new untouched history.  The "
            "phase-13 validation/test histories were regenerated from their "
            "recorded fixed seeds and are development data from this point on.  "
            "The oracle values are conditional on the fixed located reference and "
            "are NOT a joint optimum over (beta, a); a local optimizer is not a "
            "certified global one."),
    }
    def _sanitize(o):
        """Non-finite numbers are replaced by null with a marker, so the strict
        JSON dump never silently encodes an infinity as a finite-looking value."""
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    (OUT / "phase14_structural_face.json").write_text(
        json.dumps(_sanitize(payload), indent=1, allow_nan=False),
        encoding="utf-8")
    manifest("phase14", OUT,
             {"script": "scripts/phase14_structural_face.py",
              "inputs": ["results/phase11/phase11_results.json",
                         "results/phase13/frozen_model.json"],
              "diagnostic_only": True},
             cwd=HERE.parent)
    print(f"wrote {OUT / 'phase14_structural_face.json'} and {csv_path}")


if __name__ == "__main__":
    main()
