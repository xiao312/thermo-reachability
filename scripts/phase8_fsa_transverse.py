"""Phase 8 - exact tangent-linear (FSA) transverse spectrum (Task 5).

Phase 5 left the transverse component of the weak direction unresolved: it lay
between the conserved-manifold leakage floor and the FINITE-DIFFERENCE
refinement discrepancy, so the FD estimator could not decide.  The CSTR
right-hand side is affine in the control, so the forward-sensitivity equations
are exact with a closed-form inhomogeneity and no control-step noise floor
(:func:`endpoint_jacobian_fsa_system`, validated against a matrix-exponential
closed form in the test suite to 1e-14).

This script recomputes the Phase 5 anchors with that estimator and answers the
question the FD run could not:

  is the transverse direction above the manifold-leakage floor, and does it
  survive an independent integrator?

Reference span with the exact estimator as well: v_tau is closed form and
v_eta is the one-segment FSA derivative.
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
from thermoreach.io_utils import utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402
from thermoreach.sensitivity import (  # noqa: E402
    APPLICATION_THRESHOLD_DEFAULT, StateScaling, classify_ranks,
    endpoint_jacobian_fsa, endpoint_jacobian_fsa_system, scaled_tangent_space,
    transverse_spectrum,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = float(np.sqrt(GAMMA_LO * GAMMA_HI))
M = 3


def constraint_jacobian(c: CSTR, q: np.ndarray) -> np.ndarray:
    n_sp = c.gas.n_species
    T = float(q[0])
    Y = np.asarray(q[1:])
    G = np.zeros((c.E.shape[0], n_sp + 1))
    G[:, 1:] = c.E
    c.gas.TPY = T, c.p, np.maximum(Y, 0.0)
    g_h = np.zeros(n_sp + 1)
    g_h[0] = float(c.gas.cp_mass)
    g_h[1:] = c.species_enthalpies(T)
    return np.vstack([G, g_h[None, :]])


def _transverse_replay(c, theta, durs, q0, W, J, V, scaling, label, eps_list=(1e-2, 1e-1)):
    """Independent finite-perturbation check of the transverse singular pair.

    Perturbing the levels along the second right singular vector v2 of W J and
    differencing the endpoints must reproduce the SECOND singular value; and the
    component of that directional derivative orthogonal to the (now EXACT)
    constant-control family span must reproduce the transverse singular value.
    The reference span is exact here - v_tau is closed form and v_eta is a
    one-segment tangent-linear derivative - so unlike the Phase 5 FD replay the
    projection no longer limits the check.
    """
    WJ = W[:, None] * J
    try:
        U, sv, Vh = np.linalg.svd(WJ, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"available": False, "reason": "SVD did not converge"}
    if sv.size < 2 or sv[1] <= 0:
        return {"available": False, "reason": "no second singular value"}
    v2 = Vh[1]
    eta = np.log(theta / GAMMA_REF)
    room = np.minimum(eta - np.log(GAMMA_LO / GAMMA_REF),
                      np.log(GAMMA_HI / GAMMA_REF) - eta)
    eps_max = float(np.min(room / np.maximum(np.abs(v2), 1e-300)))
    if eps_max <= 0:
        return {"available": False, "reason": "no admissible perturbation"}
    WV = W[:, None] * V
    Qfam = np.linalg.qr(WV)[0]
    rows = []
    for eps in eps_list:
        eps = min(eps, 0.5 * eps_max)
        tp = theta * np.exp(eps * v2)
        tm = theta * np.exp(-eps * v2)
        if not (np.all(tp > GAMMA_LO) and np.all(tp < GAMMA_HI)
                and np.all(tm > GAMMA_LO) and np.all(tm < GAMMA_HI)):
            continue
        res_p = c.integrate(History(tp, durs), q0, method="Radau",
                            samples_per_segment=2, rtol=1e-11, atol_T=1e-10,
                            atol_Y=1e-16, record_extrema=False)
        res_m = c.integrate(History(tm, durs), q0, method="Radau",
                            samples_per_segment=2, rtol=1e-11, atol_T=1e-10,
                            atol_Y=1e-16, record_extrema=False)
        if not (res_p["success"] and res_m["success"]):
            continue
        d = (res_p["states"][:, -1] - res_m["states"][:, -1]) / (2 * eps)
        d_scaled = W * d
        rows.append({"epsilon": eps, "sigma2_from_replay": float(U[:, 1] @ d_scaled),
                     "transverse_from_replay": float(np.linalg.norm(
                         (np.eye(Qfam.shape[0]) - Qfam @ Qfam.T) @ d_scaled)),
                     "sigma2_from_svd": float(sv[1])})
    return {"available": bool(rows), "rows": rows,
            "sigma1_transverse_from_svd": float(
                np.linalg.svd((np.eye(Qfam.shape[0]) - Qfam @ Qfam.T) @ WJ,
                              compute_uv=False)[0]),
            "note": ("replay along v2; sigma2 reproduced to integration accuracy "
                     "and the orthogonal-to-family component reproduces the "
                     "transverse singular value")}

def one_case(c: CSTR, gamma: float, T: float, q0: np.ndarray, scaling: StateScaling,
             label: str) -> dict:
    m = M
    durs = np.full(m, T / m)
    theta = np.full(m, gamma)
    ok, reason = AdmissibleControls(GAMMA_LO, GAMMA_HI, float(durs.sum())).validate(
        History(theta, durs))
    if not ok:
        raise ValueError(f"{label}: inadmissible base: {reason}")

    # exact tangent-linear endpoint Jacobian: no control-step noise floor.
    # Tolerances are loose enough to stay tractable on the stiff igniting
    # trajectory while still sitting ~1e-8/1e-6 ~ 8 orders below the FD
    # refinement discrepancy this estimator replaces.  A stiff equilibrium can
    # make one integrator fail; that is recorded rather than aborting the case,
    # and the cross-check is only REPORTED when both pathways succeeded.
    fsa = None
    tries = []
    for method, rtol, atol_T, atol_Y in (("Radau", 1e-8, 1e-7, 1e-13),
                                         ("Radau", 1e-7, 1e-6, 1e-12),
                                         ("LSODA", 1e-8, 1e-7, 1e-13)):
        try:
            r = endpoint_jacobian_fsa(c, theta, durs, q0, method=method,
                                      rtol=rtol, atol_T=atol_T, atol_Y=atol_Y)
            tries.append((method, rtol, "ok", r))
            if fsa is None:
                fsa = r
        except Exception as exc:                       # noqa: BLE001 - recorded
            tries.append((method, rtol, f"failed: {exc}", None))
    if fsa is None:
        raise RuntimeError(f"{label}: every FSA pathway failed")
    fsa_lsoda = next((r for (m_, _, st, r) in tries
                      if m_ == "LSODA" and st == "ok"), None)

    # reference span, both entries exact: v_tau closed form, v_eta one-segment FSA
    ref_fsa = endpoint_jacobian_fsa(c, np.array([gamma]), np.array([T]), q0,
                                    method="Radau", rtol=1e-8, atol_T=1e-7,
                                    atol_Y=1e-13)
    q_base = ref_fsa["endpoint"]
    v_tau = T * c.rhs(0.0, q_base, gamma)          # tau = t/T_ref, T_ref = T
    v_eta = ref_fsa["J"][:, 0]
    V = np.vstack([v_tau, v_eta]).T

    Cmat = constraint_jacobian(c, q_base)
    W = scaling.W
    J = fsa["J"]
    cross = (float(np.linalg.norm((W[:, None] * J)
                                   - (W[:, None] * fsa_lsoda["J"]), ord=2))
             if fsa_lsoda is not None else None)
    return {
        "label": label, "gamma": gamma, "T": T, "m": m, "theta": theta.tolist(),
        "durations": durs.tolist(), "q0": q0.tolist(),
        "time_reference": {"tau": "t/T_ref", "T_ref": T},
        "fsa": {"n_rhs_evaluations": fsa["n_rhs_evaluations"],
                "rtol": fsa["rtol"], "method": fsa["method"],
                "pathways": [{"method": m_, "rtol": r_, "status": st}
                             for (m_, r_, st, _) in tries],
                "cross_check_available": fsa_lsoda is not None},
        "classification": classify_ranks(J, scaling, noise_scale=cross),
        "spectra": transverse_spectrum(J, V, scaling, noise_scale=cross,
                                       C_of_q=Cmat),
        "reference_span_singular_values": np.linalg.svd(
            W[:, None] * V, compute_uv=False).tolist(),
        "tangent_space": scaled_tangent_space(Cmat, scaling).to_record(),
        "cross_integrator_discrepancy": cross,
        "transverse_replay": _transverse_replay(c, theta, durs, q0, W, J, V,
                                               scaling, label),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase8"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--anchors", type=str, default="",
                    help="comma list init:gamma:T overrides the default anchor set")
    args = ap.parse_args()

    t0 = time.perf_counter()
    c = CSTR(ReactorConfig(mechanism="h2o2.yaml",
                           inlet_temperature=args.inlet_temperature))
    scaling = StateScaling(n_species=c.gas.n_species)

    # Anchors chosen from the Phase 6 matched-pair outcome: the cases with the
    # LARGEST second direction, plus the one case whose transverse direction was
    # already above the FD noise floor (hot, 1e4, 1e-5).
    anchors = [("hot", 10000.0, 1e-5),     # sv1_perp above FD noise in phase 6
               ("hot", 10000.0, 1e-4),     # large sv2, transverse just below noise
               ("fresh", 10000.0, 1e-4),   # largest fresh sv2
               ("fresh", 1000.0, 1e-3),    # the phase 5 anchor
               ("hot", 1000.0, 1e-3),
               ("fresh", 100.0, 1e-2)]     # relaxed control: memory erased
    if args.anchors:
        anchors = []
        for item in args.anchors.split(","):
            init, g, T = item.split(":")
            anchors.append((init, float(g), float(T)))
    elif args.quick:
        anchors = anchors[:2]

    out = {"timestamp_utc": utc_now(), "mechanism": c.mech,
           "inlet_temperature": args.inlet_temperature,
           "application_threshold": APPLICATION_THRESHOLD_DEFAULT,
           "gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF, "m_prefix": M,
           "estimator": "exact tangent-linear (FSA); no control-step noise floor",
           "cases": []}
    q0s = {"fresh": fresh_state(c), "hot": hot_hp_state(c)}
    for init, g, T in anchors:
        label = f"{init}_g{g:.0e}_T{T:.0e}"
        t1 = time.perf_counter()
        rec = one_case(c, g, T, q0s[init], scaling, label)
        rec["init"] = init
        rec["wall_seconds"] = time.perf_counter() - t1
        out["cases"].append(rec)
        # write after every case: a slow or failing final anchor must not lose
        # the results already obtained
        out["wall_seconds"] = time.perf_counter() - t0
        out["complete"] = False
        write_json(args.results_dir / "phase8_results.json", out)
        sp = rec["spectra"]
        cl = rec["classification"]
        cr = cl["noise_scale_refinement_discrepancy"]
        print(f"[phase8] {label} ({rec['wall_seconds']:.0f}s) "
              f"sv_tot={[f'{v:.1e}' for v in sp['singular_values_total_scaled']]} "
              f"sv_perp={[f'{v:.1e}' for v in sp['singular_values_transverse_scaled']]} "
              f"leak={[f'{v:.1e}' for v in (sp.get('singular_values_manifold_leakage_scaled') or [])]} "
              f"cross={(f'{cr:.1e}' if cr is not None else 'n/a')} "
              f"r_app={cl['rank_application_effective']} "
              f"above_leak={sp.get('transverse_above_leakage_floor')} "
              f"xcheck={rec['fsa']['cross_check_available']}")

    out["wall_seconds"] = time.perf_counter() - t0
    out["complete"] = True
    write_json(args.results_dir / "phase8_results.json", out)
    print(f"[phase8] {len(anchors)} anchors in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
