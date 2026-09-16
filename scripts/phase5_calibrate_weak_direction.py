"""Phase 5 - calibration of the weak second direction (review Task 2).

The previous Phase 4 study reported sigma_2 ~ 1e-5 at short horizons and called
it a real history-generated direction.  That conclusion needs calibration, not
144 repeated cases:

  * the weak direction must be validated by REPLAYING finite perturbations along
    the calibrated singular pair, not only by reading a singular value;
  * Radau and LSODA/BDF must be used for the ENTIRE weak-direction test, not only
    for a constant-control derivative dominated by sigma_1;
  * refinement discrepancy ||J(h) - J(h/10)|| must be compared against tolerance
    change and cross-integrator change to find a stable window - it is NOT a
    certified noise bound;
  * a second TOTAL singular value is not automatically a direction outside the
    two-parameter constant-control family: the TRANSVERSE spectrum
    (I - Q Q^T) W J is what can support that, and only above the noise scale.

Anchors: equal constant-control base histories first (where the reference-family
identity is exact), 6-8 unique cases - NOT a prefix repeated under four hold
choices.

    fresh/hot  x  gamma = 100, 1000 at T = 1e-3      (4 leading cases)
    gamma = 100 at T = 1e-2                          (1 longer-time control)
    gamma = 10000 at T = 1e-3                        (1 short-time high-rate case)

Replay test.  Let u2, v2 be a calibrated singular pair of W J.  Then

    u2^T W [E(eta + eps v2) - E(eta - eps v2)] / (2 eps)  ~  sigma_2

with eps varied and the integrator varied independently.  The full signed vector
is checked, not only the scalar.  Invalid stencils exclude the whole matrix from
full-rank claims.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls  # noqa: E402
from thermoreach.controls import History  # noqa: E402
from thermoreach.io_utils import manifest, utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402
from thermoreach.sensitivity import (  # noqa: E402
    APPLICATION_THRESHOLD_DEFAULT, StateScaling, classify_ranks, endpoint_jacobian_fd,
    log_control_plan, scaled_tangent_space, transverse_spectrum,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = float(np.sqrt(GAMMA_LO * GAMMA_HI))
M_PREFIX = 3
FD_RELS = (1e-2, 1e-3, 1e-4)          # refinement sweep h, h/10, h/100


def constraint_jacobian(c: CSTR, q: np.ndarray) -> np.ndarray:
    n_sp = c.gas.n_species
    T = float(q[0])
    Y = np.asarray(q[1:])
    G = np.zeros((c.E.shape[0], n_sp + 1))
    G[:, 1:] = c.E
    hk = c.species_enthalpies(T)
    c.gas.TPY = T, c.p, np.maximum(Y, 0.0)
    g_h = np.zeros(n_sp + 1)
    g_h[0] = float(c.gas.cp_mass)
    g_h[1:] = hk
    return np.vstack([G, g_h[None, :]])


def endpoint(c: CSTR, theta: np.ndarray, durs: np.ndarray, q0: np.ndarray,
             method: str, rtol: float, atol_T: float, atol_Y: float) -> np.ndarray:
    res = c.integrate(History(theta, durs), q0, method=method,
                      samples_per_segment=2, rtol=rtol, atol_T=atol_T,
                      atol_Y=atol_Y, record_extrema=False)
    if not res["success"]:
        raise RuntimeError(f"integration failed ({method}): {res['message']}")
    return res["states"][:, -1].copy()


def reference_span(c: CSTR, gamma_star: float, T: float, q0: np.ndarray,
                   method: str, rtol: float, atol_T: float, atol_Y: float,
                   T_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Anchored constant-control tangent span at the endpoint of the
    constant-control trajectory phi_{gamma_star}(T; q0).

    Columns, both in state units per unit dimensionless parameter:
      v_tau = T_ref * F(q_base, gamma_star)     (time parameter tau = t / T_ref)
      v_eta = d phi / d log gamma               (log-gamma parameter)
    Both differentiated from the SAME original initial state q0, and evaluated at
    the SAME endpoint q_base, so the span is the tangent of the two-parameter
    constant-control family at that point.
    """
    q_base = endpoint(c, np.full(1, gamma_star), np.array([T]), q0,
                      method, rtol, atol_T, atol_Y)
    v_tau = T_ref * c.rhs(0.0, q_base, gamma_star)
    # d phi / d log gamma: centred in log-gamma, strictly interior
    eta = np.log(gamma_star / GAMMA_REF)
    room = min(eta - np.log(GAMMA_LO / GAMMA_REF),
               np.log(GAMMA_HI / GAMMA_REF) - eta)
    h = 0.5 * 1e-3 * room
    zp = endpoint(c, np.array([gamma_star * np.exp(h)]), np.array([T]), q0,
                  method, rtol, atol_T, atol_Y)
    zm = endpoint(c, np.array([gamma_star * np.exp(-h)]), np.array([T]), q0,
                  method, rtol, atol_T, atol_Y)
    v_eta = (zp - zm) / (2 * h)
    return np.vstack([v_tau, v_eta]).T, q_base


def calibrate_case(c: CSTR, gamma_star: float, T: float, q0: np.ndarray,
                   scaling: StateScaling, label: str) -> dict:
    """One anchor: full J_eta across step sizes and integrators, then the
    replay validation of the weak singular pair and the transverse spectrum."""
    m = M_PREFIX
    durs = np.full(m, T / m)
    theta = np.full(m, gamma_star)          # EQUAL anchor: identity is valid here
    adm = AdmissibleControls(GAMMA_LO, GAMMA_HI, float(durs.sum()))
    ok, reason = adm.validate(History(theta, durs))
    if not ok:
        raise ValueError(f"{label}: inadmissible base: {reason}")

    out = {"label": label, "gamma_star": gamma_star, "T": T, "m": m,
           "theta": theta.tolist(), "durations": durs.tolist(),
           "horizon": float(durs.sum()),
           "q0": q0.tolist(), "time_reference": {"tau": "t / T_ref", "T_ref": T}}

    W = scaling.W
    rtol, atol_T, atol_Y = 1e-10, 1e-9, 1e-16
    rtol_alt, atol_T_alt, atol_Y_alt = 1e-8, 1e-8, 1e-14

    # ---- J_eta over step sizes, integrators and tolerances ----------------
    runs = {}
    for method in ("Radau", "LSODA"):
        for tol_name, (r_, aT_, aY_) in (("fine", (rtol, atol_T, atol_Y)),
                                         ("coarse", (rtol_alt, atol_T_alt,
                                                     atol_Y_alt))):
            mats = {}
            for rel in FD_RELS:
                fd = endpoint_jacobian_fd(c, theta, durs, q0, rel,
                                          GAMMA_LO, GAMMA_HI, GAMMA_REF,
                                          method=method, rtol=r_, atol_T=aT_,
                                          atol_Y=aY_)
                mats[rel] = fd
            runs[f"{method}_{tol_name}"] = mats
    out["runs"] = {k: {rel: {kk: vv for kk, vv in v.items()}
                       for rel, v in mats.items()} for k, mats in runs.items()}
    # compact stable-window summary: singular values of W J (SCALED) per step
    out["sv_by_step"] = {
        f"{method}_{tol}": {str(rel): np.linalg.svd(
            W[:, None] * runs[f"{method}_{tol}"][rel]["J"],
            compute_uv=False).tolist() for rel in FD_RELS}
        for method in ("Radau", "LSODA")
        for tol in ("fine", "coarse")}

    # primary matrix and its refinement discrepancy ||J(h) - J(h/10)||
    prim = runs["Radau_fine"]
    J = prim[FD_RELS[0]]["J"]
    J_refined = prim[FD_RELS[1]]["J"]
    WJ = W[:, None] * J
    noise = float(np.linalg.norm((W[:, None] * J) - (W[:, None] * J_refined), ord=2))

    # ---- anchored reference span, noise-filtered ---------------------------
    V, q_base = reference_span(c, gamma_star, T, q0, "Radau", rtol, atol_T,
                               atol_Y, T_ref=T)
    Cmat = constraint_jacobian(c, q_base)
    ts = scaled_tangent_space(Cmat, scaling)
    WV = W[:, None] * V
    # noise-filter the reference span itself: never fill with arbitrary vectors
    sv_V = np.linalg.svd(WV, compute_uv=False)
    keep_V = sv_V > 1e-8 * sv_V[0]
    Q = np.linalg.qr(WV[:, keep_V])[0]

    out["reference"] = {
        "q_base": q_base.tolist(),
        "singular_values_V_scaled": sv_V.tolist(),
        "rank_V_kept": int(keep_V.sum()),
        "tangent_space": ts.to_record(),
        "leakage_of_WJ_before_projection": float(np.linalg.norm(
            (Cmat @ np.diag(1.0 / W)) @ WJ)),
    }

    # ---- spectra: total, transverse; rank classification ------------------
    out["classification"] = classify_ranks(J, scaling, noise_scale=noise)
    out["spectra"] = transverse_spectrum(J, V, scaling, noise_scale=noise,
                                         C_of_q=Cmat)

    # ---- direct weak-direction replay validation ---------------------------
    U, sv, Vh = np.linalg.svd(WJ, full_matrices=False)
    replays = []
    if sv.size >= 2 and sv[1] > 0:
        v2 = Vh[1]                            # unit vector in log-control space
        # the perturbed parameters must remain admissible: scale eps to fit
        eta = np.log(theta / GAMMA_REF)
        room = np.minimum(eta - np.log(GAMMA_LO / GAMMA_REF),
                          np.log(GAMMA_HI / GAMMA_REF) - eta)
        eps_max = float(np.min(room / np.maximum(np.abs(v2), 1e-300)))
        for eps in (min(1e-2, 0.5 * eps_max), min(1e-1, 0.5 * eps_max),
                    min(0.3, 0.5 * eps_max)):
            if eps <= 0:
                continue
            tp = theta * np.exp(eps * v2)
            tm = theta * np.exp(-eps * v2)
            if not (np.all(tp > GAMMA_LO) and np.all(tp < GAMMA_HI)
                    and np.all(tm > GAMMA_LO) and np.all(tm < GAMMA_HI)):
                continue
            row = {"epsilon": eps, "valid": True}
            for method in ("Radau", "LSODA"):
                ep = endpoint(c, tp, durs, q0, method, rtol, atol_T, atol_Y)
                em = endpoint(c, tm, durs, q0, method, rtol, atol_T, atol_Y)
                # u2^T W [E+ - E-] / (2 eps)  ~ sigma_2
                dir_deriv = (U[:, 1] @ (W * (ep - em))) / (2 * eps)
                row[f"{method}_sigma2_estimate"] = float(dir_deriv)
                row[f"{method}_endpoint_plus"] = ep.tolist()
                row[f"{method}_endpoint_minus"] = em.tolist()
            row["sigma2_from_SVD"] = float(sv[1])
            replays.append(row)
    out["weak_direction_replay"] = {
        "u2_first_components": U[:, 1][:4].tolist(),
        "v2": v2.tolist(),
        "rows": replays,
        "note": ("u2^T W [E(eta+eps v2) - E(eta-eps v2)]/(2 eps) should track "
                 "sigma_2 across epsilon and across the independent integrator; "
                 "the full signed vector is checked, not only the scalar"),
    }
    out["svd_of_WJ"] = {"U": U.tolist(), "S": sv.tolist(), "Vh": Vh.tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase5"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = ReactorConfig(mechanism="h2o2.yaml",
                        inlet_temperature=args.inlet_temperature)
    c = CSTR(cfg)
    scaling = StateScaling(n_species=c.gas.n_species)

    # 6-8 UNIQUE calibration anchors: equal constant-control base histories.
    # (A prefix repeated under four hold choices is NOT four independent rank
    # findings; holds are a separate experiment.)
    anchors = []
    for init in ("fresh", "hot"):
        for g in (100.0, 1000.0):
            anchors.append((init, g, 1e-3))
    anchors.append(("fresh", 100.0, 1e-2))       # longer-time control
    anchors.append(("fresh", 10000.0, 1e-3))     # short-time high rate
    if args.quick:
        anchors = anchors[:3]

    out = {"timestamp_utc": utc_now(), "mechanism": c.mech, "config": cfg.to_record(),
           "scaling": scaling.to_record(),
           "application_threshold": APPLICATION_THRESHOLD_DEFAULT,
           "admissible_class": {"gamma_lo": GAMMA_LO, "gamma_hi": GAMMA_HI,
                                "gamma_ref": GAMMA_REF, "m_prefix": M_PREFIX},
           "fd_relative_steps": list(FD_RELS),
           "cases": []}

    q0s = {"fresh": fresh_state(c), "hot": hot_hp_state(c)}
    for init, g, T in anchors:
        label = f"{init}_g{g:.0e}_T{T:.0e}"
        rec = calibrate_case(c, g, T, q0s[init], scaling, label)
        rec["init"] = init
        out["cases"].append(rec)
        cl = rec["classification"]
        sp = rec["spectra"]
        print(f"[phase5] {label}: sv_total={[f'{v:.2e}' for v in sp['singular_values_total_scaled']]} "
              f"sv_perp={[f'{v:.2e}' for v in sp['singular_values_transverse_scaled']]} "
              f"noise={cl['noise_scale_refinement_discrepancy']:.2e} "
              f"r_machine={cl['rank_machine']} r_noise={cl['rank_derivative_noise_resolved']} "
              f"r_app={cl['rank_application_effective']} stencil_ok={cl['stencil_complete']}")

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase5_results.json", out)
    manifest("phase5", args.results_dir,
             {"inlet_temperature": args.inlet_temperature,
              "gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF,
              "m_prefix": M_PREFIX, "n_anchors": len(anchors)},
             cwd=Path(__file__).resolve().parents[1])
    print(f"[phase5] {len(anchors)} anchors in {out['wall_seconds']:.1f}s "
          f"-> {args.results_dir}")


if __name__ == "__main__":
    main()
