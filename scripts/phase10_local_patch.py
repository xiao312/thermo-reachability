"""Phase 10 - a finite local reachable patch at the flagship anchor (Task 3).

Only after the primary anchor survived Section 2 does this run.  It turns a
calibrated DERIVATIVE into an actual finite state-space excursion:

  * finite histories near the base eta0 at admissible radii
    r in {0.003, 0.01, 0.03, 0.1}, perturbing along the leading TOTAL and
    TRANSVERSE right-singular directions, positive and negative;
  * every generated state is witnessed by an actual history from the ORIGINAL q0,
    and the RAW nonlinear trajectory endpoint is evaluated, not a linearized
    displacement;
  * the local model E(eta0 + deta) = E0 + J deta + R is checked by measuring
    ||W R|| at each radius, and a containing shell from a fitted second-derivative
    scale L is labelled EMPIRICAL, not certified;
  * the perturbed endpoint is compared against the CONTINUOUS local
    constant-control family B(gamma, t; q0) with BOTH gamma and elapsed time
    varying over a declared domain, by global grid + local refinement + replay of
    the nearest candidate - labelled as an upper estimate, not a global proof.

One scaled unit is 100 K or 1e-2 mass fraction, so at radius r the first-order
excursion is r times the derivative magnitude.  Significance is judged on the
FINITE state difference, not on the derivative per unit parameter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls  # noqa: E402
from thermoreach.controls import History  # noqa: E402
from thermoreach.io_utils import utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, hot_hp_state  # noqa: E402
from thermoreach.sensitivity import StateScaling, svd_basis  # noqa: E402

RADII = (0.003, 0.01, 0.03, 0.1)
FAMILY_GAMMA_DOMAIN = (10.0, 1e5)          # declared reference domain in gamma
FAMILY_TIME_DOMAIN_FACTOR = 100.0          # t in [0, factor * T]
FAMILY_GRID = (61, 61)                     # coarse global grid (gamma, time)
REFINE_LOOPS = 3


def endpoint(c: CSTR, theta: np.ndarray, durs: np.ndarray, q0: np.ndarray,
             method: str = "Radau", rtol: float = 1e-11, atol_T: float = 1e-10,
             atol_Y: float = 1e-16) -> np.ndarray:
    res = c.integrate(History(theta, durs), q0, method=method,
                      samples_per_segment=2, rtol=rtol, atol_T=atol_T,
                      atol_Y=atol_Y, record_extrema=False)
    if not res["success"]:
        raise RuntimeError(f"integration failed: {res['message']}")
    return res["states"][:, -1].copy()


def family_distance(c: CSTR, q_target: np.ndarray, q0: np.ndarray, scaling: StateScaling,
                    T_ref: float, domain_gamma=FAMILY_GAMMA_DOMAIN,
                    grid=FAMILY_GRID, refine_loops=REFINE_LOOPS) -> dict:
    """Distance from q_target to the continuous constant-control family
    B(gamma, t; q0) = phi_gamma(t; q0), in the scaled metric.

    Global coarse grid over the declared (gamma, t) domain, then local
    refinement, then a REPLAY of the nearest candidate by fresh integration.
    A grid+local-minimizer result can OVERESTIMATE the true minimum, so the
    outcome is labelled an upper estimate of the distance to the continuous
    family, not a certified global exclusion.
    """
    W = scaling.W
    d = W * (q_target - q0)
    best = {"dist": float(np.linalg.norm(d)), "gamma": None, "t": 0.0}

    g_lo, g_hi = domain_gamma
    t_hi = FAMILY_TIME_DOMAIN_FACTOR * T_ref
    gs = np.exp(np.linspace(np.log(g_lo), np.log(g_hi), grid[0]))
    ts = np.linspace(0.0, t_hi, grid[1])

    def dist_at(g, t):
        if t <= 0:
            return float(np.linalg.norm(W * (q_target - q0))), q0.copy()
        q = endpoint(c, np.array([g]), np.array([t]), q0)
        return float(np.linalg.norm(W * (q_target - q))), q

    # coarse global pass (independent of any previous trajectory)
    for g in gs:
        for t in ts:
            if t <= 0:
                continue
            try:
                dd, _ = dist_at(g, t)
            except Exception:                                  # noqa: BLE001
                continue
            if dd < best["dist"]:
                best = {"dist": dd, "gamma": float(g), "t": float(t)}

    # local refinement around the best candidate, plus the boundary cases
    cand = [(best["gamma"], best["t"]), (g_lo, best["t"]), (g_hi, best["t"]),
            (best["gamma"], t_hi)]
    for _ in range(refine_loops):
        newbest = dict(best)
        for g0, t0 in cand:
            if g0 is None:
                continue

            def obj(x):
                g = np.exp(x[0])
                t = float(np.exp(x[1]))
                if not (g_lo < g < g_hi) or t <= 0 or t > t_hi:
                    return 1e9
                try:
                    return dist_at(g, t)[0]
                except Exception:                              # noqa: BLE001
                    return 1e9

            try:
                r = minimize(obj, [np.log(g0), np.log(max(t0, 1e-12))],
                             method="Nelder-Mead",
                             options={"xatol": 1e-6, "fatol": 1e-14,
                                      "maxiter": 200})
                if r.fun < newbest["dist"]:
                    newbest = {"dist": float(r.fun),
                               "gamma": float(np.exp(r.x[0])),
                               "t": float(np.exp(r.x[1]))}
            except Exception:                                  # noqa: BLE001
                continue
        if newbest["dist"] >= best["dist"]:
            break
        best = newbest
        cand = [(best["gamma"], best["t"])]

    # replay the nearest candidate by a fresh integration
    replay_ok, replay_dist = None, None
    if best["gamma"] is not None and best["t"] > 0:
        try:
            dd, _ = dist_at(best["gamma"], best["t"])
            replay_dist, replay_ok = dd, True
        except Exception as exc:                               # noqa: BLE001
            replay_ok, replay_dist = False, repr(exc)
    return {"scaled_distance_upper_estimate": best["dist"],
            "gamma_at_best": best["gamma"], "t_at_best_s": best["t"],
            "domain_gamma": list(domain_gamma),
            "domain_time_s": [0.0, t_hi],
            "grid": list(grid),
            "refine_loops": refine_loops,
            "replay_distance": replay_dist, "replay_ok": replay_ok,
            "label": ("upper estimate of the distance to the CONTINUOUS family "
                      "by global grid + local refinement; a local minimizer can "
                      "overestimate the minimum, so this is not a certified "
                      "global exclusion")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase10"))
    ap.add_argument("--npz", type=Path,
                    default=Path("results/phase9/primary_hot_g1e+04_T1e-05.npz"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    ap.add_argument("--radii", type=float, nargs="+", default=list(RADII))
    args = ap.parse_args()

    t0 = time.perf_counter()
    z = np.load(args.npz, allow_pickle=False)
    A, D, J, V = z["A"], z["D"], z["J"], z["V"]
    q0, theta, durs, W = z["q0"], z["theta"], z["durations"], z["W"]
    sv, U, Vh = z["singular_values"], z["U"], z["Vh"]
    E0 = z["endpoint"]
    T_ref = float(durs.sum())

    cfg = ReactorConfig(mechanism="h2o2.yaml",
                        inlet_temperature=args.inlet_temperature)
    c = CSTR(cfg)
    scaling = StateScaling(n_species=c.gas.n_species)
    assert np.allclose(scaling.W, W), "scaling mismatch between npz and config"

    # reference projector for tangent-plane distances, built by the ONE
    # rank-revealing helper: an unfiltered QR fills a rank-deficient span with
    # rounding noise instead of reporting it.
    _basis = svd_basis(D, rel_tol=1e-12)
    if _basis.vectors is None:
        raise SystemExit("reference span is rank-deficient; cannot build the "
                         "projector - report rather than fill with QR noise")
    Q = _basis.vectors
    P = Q @ Q.T
    # transverse direction from the phase-9 spectra
    B = (np.eye(P.shape[0]) - P) @ A
    _, sv_perp, Vhb = np.linalg.svd(B, full_matrices=False)
    v_perp = Vhb[0]

    # directions: the three total right singular vectors plus the transverse one
    directions = {"v1_total": Vh[0], "v2_total": Vh[1], "v3_total": Vh[2],
                  "v_transverse": v_perp}
    # physical control norm: sum_j (durations_j/T) deta_j^2 keeps the same meaning
    # across segment counts
    wts = durs / T_ref

    def cnorm(v):
        return float(np.sqrt(np.sum(wts * v ** 2)))

    out = {"timestamp_utc": utc_now(), "mechanism": c.mech,
           "config": cfg.to_record(), "base_endpoint": E0.tolist(),
           "singular_values_total": sv.tolist(),
           "singular_values_transverse": sv_perp.tolist(),
           "radii": list(args.radii),
           "control_norm": {"definition": "sqrt(sum_j (duration_j/T) deta_j^2)",
                              "weights": wts.tolist()},
           "directions": {k: v.tolist() for k, v in directions.items()},
           "cases": []}

    for name, v in directions.items():
        for r in args.radii:
            for sign in (+1, -1):
                deta = sign * r * v
                eta = np.log(theta) + deta
                th = np.exp(eta)
                if not (np.all(th > 10.0) and np.all(th < 1e5)):
                    out["cases"].append({"direction": name, "radius": r,
                                         "sign": sign, "admissible": False})
                    continue
                try:
                    q = endpoint(c, th, durs, q0)
                except Exception as exc:                                # noqa: BLE001
                    out["cases"].append({"direction": name, "radius": r,
                                         "sign": sign, "admissible": True,
                                         "status": "failed", "failure": repr(exc)})
                    continue
                d = q - E0
                d_scaled = W * d
                # local model residual: E(eta0+deta) = E0 + J deta + R
                R_scaled = d_scaled - (W[:, None] * J) @ deta
                # tangent-plane distance at the base point
                tp = float(np.linalg.norm((np.eye(P.shape[0]) - P) @ d_scaled))
                rec = {
                    "direction": name, "radius": r, "sign": sign,
                    "admissible": True, "theta": th.tolist(),
                    "control_norm_of_deta": cnorm(deta),
                    "endpoint": q.tolist(),
                    "finite_deviation": {"dT_K": float(d[0]),
                                         "dY_max_abs": float(np.max(np.abs(d[1:])))},
                    "scaled_displacement_norm": float(np.linalg.norm(d_scaled)),
                    "local_model_residual_norm": float(np.linalg.norm(R_scaled)),
                    "tangent_plane_distance_scaled": tp,
                }
                out["cases"].append(rec)
                # checkpoint per case
                write_json(args.results_dir / "phase10_results.json", out)
                print(f"[phase10] {name} r={r:g} sign={sign:+d}: "
                      f"|W d|={rec['scaled_displacement_norm']:.3e} "
                      f"|W R|={rec['local_model_residual_norm']:.3e} "
                      f"tp={tp:.3e} dT={rec['finite_deviation']['dT_K']:+.3e} K")

    # ---- second-derivative scale L from the measured residuals, EMPIRICAL -----
    ok = [r_ for r_ in out["cases"] if r_.get("local_model_residual_norm") is not None]
    L_empirical = None
    if ok:
        # least squares on ||W R|| = (L/2) n^2:  L = 2 * sum(n^2 |R|) / sum(n^4)
        num, den = 0.0, 0.0
        for r_ in ok:
            v_dir = np.asarray(directions[r_["direction"]])
            deta = r_["sign"] * r_["radius"] * v_dir
            nn = cnorm(deta)
            num += nn ** 2 * r_["local_model_residual_norm"]
            den += nn ** 4
        L_empirical = 2.0 * num / den if den > 0 else None
    out["second_derivative_scale"] = {
        "L_empirical": L_empirical,
        "label": ("EMPIRICAL least-squares fit of ||W R|| = (L/2) ||deta||^2 over "
                  "the sampled radii; NOT a certified uniform bound"),
        "shell_note": ("if L were a certified uniform bound then "
                       "||W R|| <= L ||deta||^2/2 would give a containing "
                       "ellipsoid shell; L here is estimated from samples")}

    # ---- curved-family distance for the largest transverse excursions ---------
    fam = []
    for r_ in out["cases"]:
        if not r_.get("admissible"):
            continue
        # only the finite excursions worth testing: largest radius, both signs
        if r_["radius"] != max(args.radii):
            continue
        q = np.asarray(r_["endpoint"])
        fd = family_distance(c, q, q0, scaling, T_ref)
        fd.update({"direction": r_["direction"], "sign": r_["sign"],
                   "radius": r_["radius"],
                   "tangent_plane_distance_scaled": r_["tangent_plane_distance_scaled"],
                   "scaled_displacement_norm": r_["scaled_displacement_norm"]})
        fam.append(fd)
        print(f"[phase10] family dist {r_['direction']} {r_['sign']:+d}: "
              f"upper={fd['scaled_distance_upper_estimate']:.3e} "
              f"(tangent plane {r_['tangent_plane_distance_scaled']:.3e}) "
              f"at gamma={fd['gamma_at_best']:.3g} t={fd['t_at_best_s']:.3e}")
    out["family_distance"] = fam

    out["wall_seconds"] = time.perf_counter() - t0
    out["complete"] = True
    write_json(args.results_dir / "phase10_results.json", out)
    print(f"[phase10] {len(out['cases'])} excursions in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
