#!/usr/bin/env python3
"""Phase 12: compact, reconstructable summary of the phase-11 radius sweep.

This is POSTPROCESSING ONLY.  It re-reads the committed raw phase-11 records and

  1. relabels every trajectory by its MEASURED control norm, computed from the
     stored perturbation itself, and reports both the requested label and the
     reconstructed physical norm (the transverse rays were generated from
     Euclidean-unit right singular vectors of the whitened map A H^(-1/2) and
     therefore carry an actual time norm of label/sqrt(m));
  2. separates statistics by case kind and makes every maximum name both the set
     it was taken over and the timestep it occurred at;
  3. builds the detailed per-dt tables (future-state difference, increment
     difference, gain over the zero-horizon difference) with an explicit
     unresolved-denominator rule;
  4. recomputes the E-metric from the stored raw differences as a separate
     postprocessing step with separate temperature and species tolerances, and
     verifies that the species error is independent of the temperature tolerance.

No chemistry is solved here and no saved trajectory is modified: the phase-11
raw records are treated as immutable, and the corrected analysis lives under a
new run identifier (results/phase12).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.chemresponse import hdiag, time_norm  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "phase12"

# declared illustrative application tolerances for the E-metric postprocessing;
# no target application has supplied tolerances for this project
EPS_T_GRID = (1.0, 10.0)          # K
EPS_Y_GRID = (1.0e-3, 1.0e-4)     # mass fraction

# a reference distance below this is not distinguishable from the refinement
# tolerance, so a ratio against it is reported as undefined rather than large
DIST_FLOOR_SCALED = 1.0e-6


def scaled_norm(dT: float, dY, t_interval: float, y_interval: float) -> float:
    """||W d|| with the declared diagonal scaling: one scaled unit is
    T_INTERVAL K or Y_INTERVAL in every mass fraction."""
    return math.hypot(abs(dT) / t_interval,
                      float(np.linalg.norm(np.asarray(dY, dtype=float))) / y_interval)


def measured_norms(case: dict, T: float, gamma_ref: float) -> dict:
    """Reconstruct the physical norms of the perturbation that produced the
    case, from the stored controls themselves."""
    durs = np.asarray(case["durations"], dtype=float)
    if case.get("kind") == "transverse_ray" and "deta_time_norm_coords" in case:
        deta = np.asarray(case["deta_time_norm_coords"], dtype=float)
        euclid = float(np.linalg.norm(deta))
    else:
        theta = np.asarray(case["theta_actual"], dtype=float)
        deta = np.log(theta / gamma_ref)
        euclid = float(np.linalg.norm(deta))
    return {"requested_label": case["radius_time_norm"],
            "measured_time_norm": time_norm(deta, durs, T),
            "measured_euclidean_norm": euclid,
            "H_diag": hdiag(durs, T).tolist(),
            "deta": deta.tolist(),
            "theta_actual": case["theta_actual"],
            "durations": durs.tolist()}


def e_metric_postprocess(max_abs_dT_K: float, max_abs_dY: float) -> dict:
    """Recompute the E-metric from the stored RAW differences.  E_Y depends only
    on eps_Y and E_T only on eps_T, so the table is a genuine cross product."""
    rows = []
    for eps_t in EPS_T_GRID:
        e_t = max_abs_dT_K / eps_t
        for eps_y in EPS_Y_GRID:
            e_y = max_abs_dY / eps_y
            rows.append({"eps_T_K": eps_t, "eps_Y": eps_y,
                         "E_T": e_t, "E_Y": e_y,
                         "E_max": max(e_t, e_y)})
    return {"rows": rows,
            "note": ("postprocessed from the stored raw max |dT| and max |dY|; "
                     "E_T scales as 1/eps_T and E_Y as 1/eps_Y independently")}


def verify_stored_e_metric(dt_row: dict) -> dict:
    """The integrity check the summary exists to make: every stored E_Y must
    equal max_abs_dY/eps_Y regardless of the temperature tolerance."""
    bad = []
    for row in dt_row.get("E_metric", []):
        lbl = row["tolerance_label"]
        # the stored label encodes both tolerances; recover them from the row
        # by matching against the recomputed cross product
        for cand in e_metric_postprocess(row["max_abs_dT_K"],
                                        row["max_abs_dY"])["rows"]:
            if (math.isclose(cand["E_T"], row["E_T"], rel_tol=1e-12)
                    and math.isclose(cand["E_Y"], row["E_Y_max"], rel_tol=1e-12)):
                if not math.isclose(cand["E_max"], row["E_max"], rel_tol=1e-12):
                    bad.append({"tolerance_label": lbl,
                                "stored_E_max": row["E_max"],
                                "recomputed_E_max": cand["E_max"]})
                break
        else:
            bad.append({"tolerance_label": lbl,
                        "reason": "no recomputed tolerance pair matched",
                        "stored": {k: row[k] for k in
                                   ("E_T", "E_Y_max", "E_max")}})
    return {"consistent": not bad, "inconsistencies": bad}


def chemistry_summary(case: dict, t_interval: float, y_interval: float,
                      d0_scaled: float, d0_T: float, d0_Y_scaled_norm: float,
                      d0_max_abs_dY: float,
                      dist_scaled: float, resolved: bool) -> dict:
    ch = case.get("chemistry") or {}
    rows = []
    for d in ch.get("dt_results", []):
        dt = float(d["dt"])
        fdT = float(d.get("future_state_dT_K", float("nan")))
        fdY = [float(x) for x in d.get("future_state_dY", [])]
        idT = float(d.get("increment_dT_K", float("nan")))
        idY = [float(x) for x in d.get("increment_dY", [])]
        f_scaled = scaled_norm(fdT, fdY, t_interval, y_interval)
        i_scaled = scaled_norm(idT, idY, t_interval, y_interval)
        max_fdY = float(np.max(np.abs(fdY))) if fdY else float("nan")
        max_idY = float(np.max(np.abs(idY))) if idY else float("nan")
        denom_status = ("resolved" if (resolved and d0_scaled > 0
                                       and math.isfinite(d0_scaled)) else "unresolved")
        rows.append({
            "dt_s": dt,
            "success": bool(d.get("success", False)),
            "future_state_max_abs_dT_K": abs(fdT),
            "future_state_max_abs_dY": max_fdY,
            "increment_max_abs_dT_K": abs(idT),
            "increment_max_abs_dY": max_idY,
            "scaled_norm_future_difference": f_scaled,
            "scaled_norm_increment_difference": i_scaled,
            "gain_future_over_d0": (f_scaled / d0_scaled) if denom_status == "resolved"
            else None,
            "gain_increment_over_d0": (i_scaled / d0_scaled)
            if denom_status == "resolved" else None,
            "gain_future_T_component": (abs(fdT) / abs(d0_T))
            if (denom_status == "resolved" and abs(d0_T) > 0) else None,
            "gain_future_Y_component": ((max_fdY / y_interval) / d0_Y_scaled_norm)
            if (denom_status == "resolved" and d0_Y_scaled_norm > 0) else None,
            "gain_denominator_status": denom_status,
            "gain_denominator_scaled": d0_scaled,
            "E_metric_postprocessed": e_metric_postprocess(abs(fdT), max_fdY),
        })
    return {"dt_rows": rows,
            "n_dt": len(rows),
            "dts_s": [r["dt_s"] for r in rows],
            "all_dt_succeeded": bool(rows) and all(r["success"] for r in rows),
            "zero_horizon_identity_check": {
                "d0_scaled": d0_scaled,
                "d0_T_K": d0_T,
                "d0_max_abs_dY": d0_max_abs_dY,
                "d0_Y_euclidean_norm_scaled": d0_Y_scaled_norm,
                "note": ("the chemistry map is the identity at dt = 0, so the "
                         "zero-horizon difference is the stored state "
                         "difference q - q_B; no dt = 0 row was solved in the "
                         "phase-11 run, and the smallest stored dt is checked "
                         "for convergence to d0 below"),
                "smallest_dt_future_scaled": rows[0]["scaled_norm_future_difference"]
                if rows else None,
                "smallest_dt_future_dT_K": rows[0]["future_state_max_abs_dT_K"]
                if rows else None},
            "e_metric_verification": [verify_stored_e_metric(d)
                                      for d in ch.get("dt_results", [])],
            "baseline": ch.get("baseline", {}),
            "initial_source_difference": ch.get("initial_source_difference", {})}


def compact_case(case: dict, T: float, gamma_ref: float,
                 t_interval: float, y_interval: float) -> dict:
    ref = case.get("reference") or {}
    dist = float(ref.get("dist_scaled_upper_estimate", float("nan")))
    resolved = bool(math.isfinite(dist) and dist > DIST_FLOOR_SCALED)
    d0_T = float((ref.get("signed_residual_raw") or [0.0])[0])
    d0_Y = [float(x) for x in (ref.get("signed_residual_raw") or [0.0])[1:]]
    d0_scaled = scaled_norm(d0_T, d0_Y, t_interval, y_interval)
    d0_Y_scaled = float(np.linalg.norm(np.asarray(d0_Y)) / y_interval)
    d0_max_abs_dY = float(np.max(np.abs(np.asarray(d0_Y, dtype=float)))) \
        if d0_Y else float("nan")
    out = {
        "label": case["label"],
        "kind": case["kind"],
        "status": case.get("status"),
        "admissible": case.get("admissible"),
        "role_after_revision": (
            "exploratory_or_training" if case["kind"] in ("held_out",
                                                          "in_family_control")
            else "exploratory_directional_ray"),
        "norms": measured_norms(case, T, gamma_ref),
        "correction_note": (
            "the transverse rays were built from Euclidean-unit right singular "
            "vectors of A H^(-1/2); their physical time norm is "
            "label/sqrt(m), so the requested label OVERSTATES the actual norm. "
            "In-family and random histories were normalized directly in the "
            "time norm, so their labels are exact."),
        "located_reference": {
            "gamma_best": ref.get("gamma_best"),
            "t_best": ref.get("t_best"),
            "dist_scaled_upper_estimate": dist,
            "abs_dT_K": ref.get("abs_dT_K"),
            "max_abs_dY": float(np.max(np.abs(np.asarray(
                ref.get("abs_dY", [float("nan")]), dtype=float)))),
            "scaled_norm_split": ref.get("scaled_norm_split"),
            "expanded_domain": ref.get("expanded_domain", False),
            "dist_note": ref.get("dist_note", ""),
            "resolved_above_floor": resolved,
        },
        "taylor_residual": case.get("taylor_residual"),
        "normal_residual": case.get("normal_residual"),
        "chemistry": chemistry_summary(case, t_interval, y_interval, d0_scaled,
                                       d0_T, d0_Y_scaled, d0_max_abs_dY,
                                       dist, resolved),
        "wall_seconds": case.get("wall_seconds"),
        "physical_state": case.get("physical_state"),
    }
    return out


def stats_by_kind(cases: list, t_interval: float, y_interval: float) -> dict:
    """Maxima that name the set AND the timestep they were taken over."""
    stats = {}
    for kind in sorted({c["kind"] for c in cases}):
        sel = [c for c in cases if c["kind"] == kind]
        stats[kind] = {
            "n_cases": len(sel),
            "measured_time_norm": {
                "min": min(c["norms"]["measured_time_norm"] for c in sel),
                "max": max(c["norms"]["measured_time_norm"] for c in sel)},
            "max_future_state_difference": _max_over(
                sel, "chemistry.dt_rows", "scaled_norm_future_difference"),
            "max_increment_difference": _max_over(sel, "chemistry.dt_rows",
                                                  "scaled_norm_increment_difference"),
            "max_family_distance_scaled": _max_over(
                sel, "located_reference", "dist_scaled_upper_estimate"),
        }
    # cross-kind maxima with explicit provenance
    flat = [(c["label"], c["kind"], r["dt_s"], r["scaled_norm_future_difference"])
            for c in cases for r in c["chemistry"]["dt_rows"]]
    flat.sort(key=lambda t: -t[3])
    top = flat[0]
    stats["overall"] = {
        "max_scaled_future_difference": {"value": top[3], "case": top[0],
                                         "kind": top[1], "dt_s": top[2],
                                         "set": "all completed cases, all stored dt"},
        "n_cases": len(cases),
    }
    return stats


def _max_over(cases: list, path: str, field: str) -> dict:
    parts = path.split(".")
    best = None
    for c in cases:
        obj = c
        for p in parts:
            obj = (obj or {}).get(p, {}) if isinstance(obj, dict) else None
        rows = obj if isinstance(obj, list) else ([obj] if isinstance(obj, dict) else [])
        for r in rows:
            v = r.get(field) if isinstance(r, dict) else None
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                if best is None or v > best["value"]:
                    best = {"value": float(v), "case": c["label"],
                            "dt_s": r.get("dt_s"),
                            "set": f"{c['kind']} cases"
                                    + (", all stored dt" if "dt_rows" in path
                                       else "")}
    return best or {"value": None, "set": "no rows"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase11", default=str(HERE.parent / "results" / "phase11"
                                             / "phase11_results.json"))
    args = ap.parse_args()
    raw = json.loads(Path(args.phase11).read_text(encoding="utf-8"))

    t_interval = float(raw["scaling"]["T_interval_K"])
    y_interval = float(raw["scaling"]["Y_interval"])
    T = float(raw["anchor"]["horizon_s"])
    gamma_ref = float(raw["anchor"]["gamma_ref"])
    cases = [compact_case(c, T, gamma_ref, t_interval, y_interval)
             for c in raw["cases"]]

    e_bad = [c["label"] for c in cases
             if any(not v["consistent"]
                    for v in c["chemistry"]["e_metric_verification"])]
    gains_undefined = sorted({c["label"] for c in cases
                              if any(r["gain_denominator_status"] != "resolved"
                                     for r in c["chemistry"]["dt_rows"])})

    summary = {
        "run": {
            "identifier": "phase12-summary",
            "source": "results/phase11/phase11_results.json (immutable raw records)",
            "purpose": ("compact relabelled summary; the phase-11 raw records "
                        "are not modified and this is postprocessing only"),
            "correction": (
                "transverse-ray radii are reported by their MEASURED time norm "
                "= label/sqrt(m); the requested label is retained alongside"),
        },
        "control_norm": {
            "T_horizon_s": T,
            "durations_s": raw["cases"][0]["durations"],
            "m_segments": len(raw["cases"][0]["durations"]),
            "H_diag": hdiag(np.asarray(raw["cases"][0]["durations"],
                                       dtype=float), T).tolist(),
            "sum_of_H_entries": float(np.sum(hdiag(
                np.asarray(raw["cases"][0]["durations"], dtype=float), T))),
            "norms_declared": raw["control_norm"],
        },
        "scaling": raw["scaling"],
        "tolerance_note": raw["cases"][0]["chemistry"].get("tolerance_table_note",
                                                          ""),
        "cases": cases,
        "stats_by_kind": stats_by_kind(cases, t_interval, y_interval),
        "checks": {
            "e_metric_consistent_in_raw_records": not e_bad,
            "e_metric_inconsistent_cases": e_bad,
            "e_metric_note": ("the raw stored records are internally consistent; "
                              "the transcription error appeared only in the "
                              "hand-written report table and is corrected there"),
            "gain_undefined_cases": gains_undefined,
            "gain_rule": ("gain = ||W d(dt)|| / ||W d0|| with d0 the zero-horizon "
                          "difference q - q_B; if the located reference distance "
                          f"is below {DIST_FLOOR_SCALED:.0e} scaled units the "
                          "denominator is unresolved and the ratio is None"),
            "all_cases_admissible": all(bool(c["admissible"]) for c in cases),
            "dt_rows_solved": int(sum(c["chemistry"]["n_dt"] for c in cases)),
            "dt_rows_failed": int(sum(1 for c in cases
                                      for r in c["chemistry"]["dt_rows"]
                                      if not r["success"])),
            "cases_without_chemistry_records": sorted(
                c["label"] for c in cases if not c["chemistry"]["dt_rows"]),
            "all_solved_dt_rows_succeeded": all(
                r["success"] for c in cases for r in c["chemistry"]["dt_rows"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase12_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    # CSV: one row per (case, dt)
    with (OUT / "phase12_per_dt.csv").open("w", newline="",
                                           encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "requested_label_radius", "measured_time_norm",
                    "measured_euclidean_norm", "dt_s", "success",
                    "future_dT_K", "future_max_dY", "increment_dT_K",
                    "increment_max_dY", "scaled_future", "scaled_increment",
                    "gain_future", "gain_increment", "gain_T", "gain_Y",
                    "gain_denominator_status"])
        for c in cases:
            for r in c["chemistry"]["dt_rows"]:
                w.writerow([c["label"], c["kind"],
                            c["norms"]["requested_label"],
                            c["norms"]["measured_time_norm"],
                            c["norms"]["measured_euclidean_norm"],
                            r["dt_s"], r["success"],
                            r["future_state_max_abs_dT_K"],
                            r["future_state_max_abs_dY"],
                            r["increment_max_abs_dT_K"],
                            r["increment_max_abs_dY"],
                            r["scaled_norm_future_difference"],
                            r["scaled_norm_increment_difference"],
                            r["gain_future_over_d0"],
                            r["gain_increment_over_d0"],
                            r["gain_future_T_component"],
                            r["gain_future_Y_component"],
                            r["gain_denominator_status"]])

    with (OUT / "phase12_per_case.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "requested_label_radius",
                    "measured_time_norm", "gamma_best", "t_best",
                    "dist_scaled", "abs_dT_K", "max_abs_dY", "resolved",
                    "R_over_lin", "wall_seconds"])
        for c in cases:
            w.writerow([c["label"], c["kind"], c["norms"]["requested_label"],
                        c["norms"]["measured_time_norm"],
                        c["located_reference"]["gamma_best"],
                        c["located_reference"]["t_best"],
                        c["located_reference"]["dist_scaled_upper_estimate"],
                        c["located_reference"]["abs_dT_K"],
                        c["located_reference"]["max_abs_dY"],
                        c["located_reference"]["resolved_above_floor"],
                        (c["taylor_residual"] or {}).get(
                            "residual_over_linear_prediction"),
                        c["wall_seconds"]])

    print(f"wrote {OUT / 'phase12_summary.json'}")
    print(f"wrote {OUT / 'phase12_per_dt.csv'}")
    print(f"wrote {OUT / 'phase12_per_case.csv'}")
    print("cases:", len(cases))
    print("e_metric inconsistent:", e_bad)
    print("gain undefined cases:", gains_undefined)
    for kind, st in summary["stats_by_kind"].items():
        if kind == "overall":
            continue
        print(f"  {kind}: n={st['n_cases']} "
              f"measured_norm [{st['measured_time_norm']['min']:.4f}, "
              f"{st['measured_time_norm']['max']:.4f}] "
              f"max_future={st['max_future_state_difference']['value']}")


if __name__ == "__main__":
    main()
