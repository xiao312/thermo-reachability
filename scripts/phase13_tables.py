#!/usr/bin/env python3
"""Phase 13 tables: per-case and per-tolerance error tables from the phase-13
results, generated as POSTPROCESSING (no re-solving).

Emits
  results/phase13/phase13_per_case.csv     one row per test case x representation
  results/phase13/phase13_chemistry.csv    one row per test case x state x dt
  results/phase13/phase13_tolerance.json   the error-versus-tolerance surface:
    for every illustrative (eps_T, eps_Y) pair, the fraction of test cases whose
    scaled state error (and whose chemistry E_max at each dt) is below the pair.
A future reader with actual application tolerances can look their pair up.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "phase13"
EPS_T_GRID = (0.1, 1.0, 10.0)
EPS_Y_GRID = (1e-4, 1e-3, 1e-2)

VARIANTS = [
    ("A", "A_oracle_reference"),
    ("B", "B_predicted_reference"),
    ("C", "C_predicted_reference_plus_predicted_correction"),
    ("D", "D_oracle_reference_plus_oracle_correction"),
]


def _e(rec, key):
    e = rec.get(key, {}).get("error")
    return e or {}


def main() -> None:
    res = json.loads((OUT / "phase13_results.json").read_text(encoding="utf-8"))
    tests = res["test_results"]

    with (OUT / "phase13_per_case.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "measured_time_norm", "variant", "scaled_error",
                    "abs_dT_K", "max_abs_dY", "rel_dT",
                    "log_gamma_error_vs_oracle", "a", "decoded_status"])
        for r in tests:
            for name, key in VARIANTS:
                e = _e(r, key)
                w.writerow([r["label"], r["measured_time_norm"], name,
                            e.get("scaled_norm"), e.get("abs_dT_K"),
                            e.get("max_abs_dY"), e.get("rel_dT"),
                            (r.get("B_predicted_reference", {})
                             .get("coordinate_error_vs_oracle", {})
                             .get("log_gamma") if name == "B" else ""),
                            (r.get("C_predicted_reference_plus_predicted_"
                                   "correction", {}).get("a") if name == "C"
                             else ""),
                            (r.get("C_predicted_reference_plus_predicted_"
                                   "correction", {}).get("decoded_status")
                             if name == "C" else "simulated_or_oracle")])

    with (OUT / "phase13_chemistry.csv").open("w", newline="",
                                              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "measured_time_norm", "reference_used",
                    "dt_s", "future_dT_K", "future_max_abs_dY",
                    "increment_dT_K", "increment_max_abs_dY",
                    "E_max_T1_Y1e-3", "E_max_T1_Y1e-4", "E_max_T10_Y1e-3",
                    "enthalpy_mismatch_J_kg", "elemental_mismatch_max"])
        for r in tests:
            ch = r.get("chemistry", {})
            for label in ("predicted_canonical", "predicted_corrected",
                          "oracle_canonical"):
                blk = ch.get(label)
                if not blk:
                    continue
                base = blk.get("baseline", {})
                for d in blk.get("dt_results", []):
                    em = {row["tolerance_label"]: row["E_max"]
                          for row in d.get("E_metric", [])}
                    w.writerow([r["label"], r["measured_time_norm"], label,
                                d["dt"], d.get("future_state_dT_K"),
                                max(abs(x) for x in d.get("future_state_dY", [])
                                    ) if d.get("future_state_dY") else "",
                                d.get("increment_dT_K"),
                                max(abs(x) for x in d.get("increment_dY", []))
                                if d.get("increment_dY") else "",
                                em.get("T 1 K / Y 1e-3"),
                                em.get("T 1 K / Y 1e-4"),
                                em.get("T 10 K / Y 1e-3"),
                                max([abs(x) for x in
                                     d.get("enthalpy_mismatch_J_kg", [])]
                                    or [0.0]),
                                base.get("elemental_inventory_mismatch_max")])

    # ---- the error-versus-tolerance surface ------------------------------
    surf = {"note": ("fraction of the 16 untouched test histories whose error "
                     "is below each illustrative tolerance pair; a future "
                     "application with real tolerances looks its pair up "
                     "instead of extrapolating"),
            "state_error_scaled": {},
            "chemistry_E_max": {}}
    errs = {name: np.array([_e(r, key).get("scaled_norm", np.nan)
                            for r in tests]) for name, key in VARIANTS}
    # the scaled error is measured in units of (dT/100 K, dY/0.01); a tolerance
    # pair is met when BOTH components are met
    for name, key in VARIANTS:
        dT = np.array([_e(r, key).get("abs_dT_K", np.nan) for r in tests])
        dY = np.array([_e(r, key).get("max_abs_dY", np.nan) for r in tests])
        for et in EPS_T_GRID:
            for ey in EPS_Y_GRID:
                ok = np.mean((dT <= et) & (dY <= ey))
                surf["state_error_scaled"].setdefault(
                    f"T_{et:g}_K_Y_{ey:g}",
                    {})[name] = float(ok)
    for label in ("predicted_canonical", "predicted_corrected",
                  "oracle_canonical"):
        for dt_s in (1e-7, 1e-6, 1e-5, 1e-4):
            row = {}
            for r in tests:
                blk = r.get("chemistry", {}).get(label)
                if not blk:
                    continue
                d = next((x for x in blk.get("dt_results", [])
                          if abs(x["dt"] - dt_s) < 0.5 * dt_s), None)
                if d is None:
                    continue
                dT = abs(d["future_state_dT_K"])
                dY = max(abs(x) for x in d["future_state_dY"])
                for et in EPS_T_GRID:
                    for ey in EPS_Y_GRID:
                        k = f"T_{et:g}_K_Y_{ey:g}"
                        row.setdefault(k, []).append(dT <= et and dY <= ey)
            if row:
                surf["chemistry_E_max"][f"{label}_dt_{dt_s:g}"] = {
                    k: float(np.mean(v)) for k, v in row.items()}
    (OUT / "phase13_tolerance.json").write_text(json.dumps(surf, indent=1) + "\n",
                                                encoding="utf-8")

    # ---- the honest-degradation summary ----------------------------------
    a_err = np.array([_e(r, "A_oracle_reference").get("scaled_norm", np.nan)
                      for r in tests])
    b_err = np.array([_e(r, "B_predicted_reference").get("scaled_norm", np.nan)
                      for r in tests])
    c_err = np.array([_e(r, "C_predicted_reference_plus_predicted_correction")
                      .get("scaled_norm", np.nan) for r in tests])
    d_err = np.array([_e(r, "D_oracle_reference_plus_oracle_correction")
                      .get("scaled_norm", np.nan) for r in tests])
    worst = int(np.argmax(c_err))
    print(f"cases: {len(tests)}")
    print(f"median error  A={np.nanmedian(a_err):.3e}  B={np.nanmedian(b_err):.3e}"
          f"  C={np.nanmedian(c_err):.3e}  D={np.nanmedian(d_err):.3e}")
    print(f"C improves on A in {int(np.sum(c_err < a_err))}/{len(tests)} cases;")
    print(f"C worse than A in {int(np.sum(c_err > a_err))} cases "
          f"(worst case {tests[worst]['label']}: "
          f"A={a_err[worst]:.3e} -> C={c_err[worst]:.3e})")
    print(f"prediction cost B-A: median {np.nanmedian(b_err - a_err):+.3e}")
    print(f"prediction cost C-D: median {np.nanmedian(c_err - d_err):+.3e}")


if __name__ == "__main__":
    main()
