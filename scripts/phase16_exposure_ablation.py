"""Phase 16: the exposure-law coordinate ablation (Revision 8 section C).

A small, server-free ablation on the COMMITTED phase-11 records (which carry both
the actual controls and the located reference coordinates).  It compares three
inductive biases for the reference coordinates, all at UNCHANGED low complexity:

  (i)  FREE            log gamma_B and log t_B regressed on the segment logs;
  (ii) EXPOSURE        the Revision-7 parametrization, log gamma_B = log Gamma
                       - v + u with log Gamma an exact feature;
  (iii EXPOSURE + M1   the same, with the exact first moment M1 added as a
       feature, plus the pure short-time asymptotic baseline
       t_B ~ 2 M1 / Gamma, gamma_B ~ Gamma^2 / (2 M1).

The M1 relations are an ASYMPTOTIC model (an O(t_f^3) expansion along the
exchange direction), never an exact constraint; out-of-domain estimates are
RECORDED, never clamped.

Reported: in-sample and leave-one-out rms of the reference-coordinate errors on
the SAME targets and the SAME split for all three biases, and how often the M1
baseline leaves the allowed domain.  No conclusion is drawn from a differently
transformed rms being smaller.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from thermoreach.generator import (  # noqa: E402
    LinearRegressor, exposure_log_feature, first_moment_integral,
    generator_features, m1_coordinate_estimate)
from thermoreach.io_utils import manifest, utc_now  # noqa: E402

OUT = HERE.parent / "results" / "phase16"
GAMMA_REF = 1e4
HORIZON = 1e-6
M = 3
GAMMA_LO = GAMMA_REF * 1e-3
GAMMA_HI = GAMMA_REF * 1e3
T_MAX = 100.0 * HORIZON


def _features_free(eta, order=1):
    """Genuinely free: the constant and the segment logs, with NO exposure or
    moment feature - the reference-coordinate regression the exposure bias was
    claimed to improve upon."""
    e = np.asarray(eta, dtype=float).ravel()
    cols = [np.ones(1), e]
    if order >= 2:
        cols.append(e ** 2)
        iu = np.triu_indices(e.size, k=1)
        cols.append(e[iu[0]] * e[iu[1]])
    return np.concatenate(cols)


def _features_exposure(eta, order=1):
    return generator_features(eta, np.full(M, HORIZON / M), HORIZON, GAMMA_REF,
                              order)


def _features_exposure_m1(eta, order=1):
    d = np.full(M, HORIZON / M)
    g = GAMMA_REF * np.exp(np.asarray(eta, dtype=float))
    base = generator_features(eta, d, HORIZON, GAMMA_REF, order)
    return np.concatenate([base, [math.log(max(first_moment_integral(
        g, d, HORIZON), 1e-300) / HORIZON ** 2)]])


def _loo_rms(X, Y):
    """Leave-one-out fitted rms: refit on all but one case, predict the held-out
    case.  Both the in-sample and the LOO rms use the SAME targets."""
    n = X.shape[0]
    pred = np.zeros_like(Y, dtype=float)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        A = X[keep].T @ X[keep] + 1e-8 * np.eye(X.shape[1])
        A[0, 0] -= 1e-8
        coef = np.linalg.solve(A, X[keep].T @ Y[keep])
        pred[i] = X[i] @ coef
    return pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((HERE.parent / "results" / "phase11"
                      / "phase11_results.json").read_text(encoding="utf-8"))
    d = np.full(M, HORIZON / M)
    cases = []
    for rec in raw["cases"]:
        ref = rec.get("reference") or {}
        if rec.get("status") != "completed" or ref.get("gamma_best") is None:
            continue
        theta = np.asarray(rec["theta_actual"], dtype=float)
        eta = np.log(theta / GAMMA_REF)
        norm = math.sqrt(float(np.sum((d / HORIZON) * eta ** 2)))
        if not (0.0 < norm <= 0.6 + 1e-9):
            continue
        cases.append({"label": rec["label"], "eta": eta, "norm": norm,
                      "log_gamma_B": math.log(ref["gamma_best"] / GAMMA_REF),
                      "log_t_B": math.log(max(ref["t_best"], 1e-30) / HORIZON)})
    print(f"diagnostic cases from phase 11: {len(cases)}")
    etas = [c["eta"] for c in cases]
    Y = np.array([[c["log_gamma_B"], c["log_t_B"]] for c in cases])

    biases = {
        "free": _features_free,
        "exposure": _features_exposure,
        "exposure_plus_M1": _features_exposure_m1,
    }
    table = {}
    for name, feat in biases.items():
        X = np.stack([feat(e, 1) for e in etas])
        reg = LinearRegressor(feat, order=1, ridge=1e-8).fit(etas, Y)
        in_sample = np.asarray(reg.predict(etas)) if False else X @ reg.coef_
        loo = _loo_rms(X, Y)
        table[name] = {
            "n_features": int(X.shape[1]),
            "in_sample_rms": {
                "log_gamma_B": float(np.sqrt(np.mean((in_sample[:, 0]
                                                      - Y[:, 0]) ** 2))),
                "log_t_B": float(np.sqrt(np.mean((in_sample[:, 1]
                                                  - Y[:, 1]) ** 2)))},
            "leave_one_out_rms": {
                "log_gamma_B": float(np.sqrt(np.mean((loo[:, 0] - Y[:, 0]) ** 2))),
                "log_t_B": float(np.sqrt(np.mean((loo[:, 1] - Y[:, 1]) ** 2)))},
            "cond_XtX": float(np.linalg.cond(X.T @ X)),
        }
        print(f"  {name:>16s} ({X.shape[1]} feats): in-sample log-gamma rms "
              f"{table[name]['in_sample_rms']['log_gamma_B']:.3e}, "
              f"LOO {table[name]['leave_one_out_rms']['log_gamma_B']:.3e}")

    # the pure asymptotic baseline (no fitting at all)
    m1_rows = []
    for c in cases:
        est = m1_coordinate_estimate(GAMMA_REF * np.exp(c["eta"]), d, HORIZON,
                                     gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI,
                                     t_max=T_MAX)
        err = {"label": c["label"],
               "log_gamma_error": math.log(est["gamma_B_estimate"]
                                           / (GAMMA_REF * math.exp(c["log_gamma_B"])))
               if est.get("gamma_B_estimate") else None,
               "log_t_error": math.log(est["t_B_estimate_s"]
                                       / (HORIZON * math.exp(c["log_t_B"])))
               if est.get("t_B_estimate_s") else None,
               "in_domain": est["in_domain"],
               "out_of_domain_reason": est.get("out_of_domain_reason"),
               "gamma_t_product": est.get("gamma_t_product")}
        err["gamma_t_error"] = (err["log_gamma_error"] + err["log_t_error"]
                                if err["log_gamma_error"] is not None
                                and err["log_t_error"] is not None else None)
        m1_rows.append(err)
    m1_stats = {
        "n": len(m1_rows),
        "n_out_of_domain": int(sum(1 for r in m1_rows if not r["in_domain"])),
        "log_gamma_error_rms": float(np.sqrt(np.mean([r["log_gamma_error"] ** 2
                                                      for r in m1_rows
                                                      if r["log_gamma_error"]
                                                      is not None]))),
        "log_t_error_rms": float(np.sqrt(np.mean([r["log_t_error"] ** 2
                                                  for r in m1_rows
                                                  if r["log_t_error"]
                                                  is not None]))),
        "gamma_t_product_error_rms": float(np.sqrt(np.mean(
            [r["gamma_t_error"] ** 2 for r in m1_rows
             if r["gamma_t_error"] is not None]))),
    }
    print(f"  pure M1 asymptotic baseline: gamma rms "
          f"{m1_stats['log_gamma_error_rms']:.3e}, t rms "
          f"{m1_stats['log_t_error_rms']:.3e}, "
          f"{m1_stats['n_out_of_domain']}/{m1_stats['n']} out of domain")

    payload = {
        "run": {"identifier": "phase16", "timestamp_utc": utc_now(),
                "diagnostic_only": True},
        "cases_used": [{"label": c["label"], "measured_time_norm": c["norm"]}
                       for c in cases],
        "biases": table,
        "m1_asymptotic_baseline": m1_stats,
        "m1_per_case": m1_rows,
        "method_note": (
            "all three biases are compared on the SAME targets (log gamma_B and "
            "log t_B, both in anchor units) and the SAME split, at unchanged "
            "order-1 complexity; the leave-one-out rms is reported alongside the "
            "in-sample rms so a wider feature set cannot hide its variance.  The "
            "M1 relations are an asymptotic O(t_f^3) model, not an exact "
            "constraint; out-of-domain estimates are recorded, never clamped."),
    }
    (OUT / "phase16_exposure_ablation.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    manifest("phase16", OUT,
             {"script": "scripts/phase16_exposure_ablation.py",
              "inputs": ["results/phase11/phase11_results.json"],
              "diagnostic_only": True},
             cwd=HERE.parent)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
