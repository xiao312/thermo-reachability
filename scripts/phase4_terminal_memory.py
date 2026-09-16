"""Phase 4 - transient history sensitivity versus terminal-memory erasure.

This is the corrected sensitivity study.  It replaces the Phase 3D pipeline,
which could not support its own conclusions (see the module docstring of
``thermoreach.sensitivity`` for the list of defects).

Scientific question.  An m-segment switched history has an endpoint map
E_m(theta).  Its derivative measures how much *earlier* controls are still
visible at the terminal state.  A long final hold at gamma_hold contracts the
state toward the steady branch, and the chain rule is exact:

    E(theta_prefix, gamma_hold, L) = phi^L_{gamma_hold}(q_prefix(theta_prefix))
    dE/dtheta_j = D_q phi^L_{gamma_hold} * dq_prefix/dtheta_j

So sensitivity to the prefix controls is *propagated* through the hold and can
become unresolved while sensitivity to gamma_hold approaches the tangent of the
steady branch.  Exponentially tiny is not zero: the state-transition map of a
smooth flow is locally invertible, so algebraic rank survives while
application-scale effective rank decays.  This experiment separates

  (1) real independent transient history effects,
  (2) decay of earlier-history sensitivity under a long terminal hold,
  (3) directions unresolved at the chosen numerical accuracy,
  (4) directions resolved but below a declared application tolerance.

Protocol (review section D).

* Same verified ideal-gas h2o2 mechanism, 101325 Pa, stoichiometric H2/air,
  T_in = 1200 K, gamma in [10, 1e5]; fresh and HP-hot initial states kept
  separate; 900 K carried as a separately labelled weak-reaction comparison.
* Constant-control anchors gamma = 1e2, 1e3, 1e4 with a small set of early,
  ignition-region and relaxed horizons chosen from an actual time scan, never
  manufactured.
* Reference tangents from the ORIGINAL initial state:
  v_t = F(q_base, gamma_star);  v_log_gamma = d phi_gamma(T; q0)/d log gamma by
  BOTH finite differences and a tangent-linear (forward sensitivity) solve.
* Exact identity control: sum_j dE_m/d(log gamma_j) = d phi/d log gamma at
  constant-control base histories.
* m = 3 prefix; terminal hold at gamma_hold with swept L.

All negative findings are confined to the tested map, histories, time window and
tolerance.  A validated negative result within a stated scope is an acceptable
outcome; no predetermined rank is sought.
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
    StateScaling, constant_control_fsa, endpoint_jacobian_fd, rhs_forcing_gamma,
    scaled_tangent_space, sin_to_span, svd_basis,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = float(np.sqrt(GAMMA_LO * GAMMA_HI))
ANCHORS = (1e2, 1e3, 1e4)
M_PREFIX = 3
SCAN_TIMES = np.concatenate([np.logspace(-6, -2, 25), [0.1]])
# DECLARED application tolerance in scaled state units.  One scaled unit is a
# 100 K temperature interval or a 1% mass-fraction interval (StateScaling), so a
# direction below this threshold moves the endpoint by less than 1e-4 K or
# 1e-6 in mass fraction per unit step in log control - below any measurable or
# model-relevant scale for this study.  It is a DECLARED choice, not measured.
ABS_APPLICATION_THRESHOLD = 1e-6


def constraint_jacobian(c: CSTR, q: np.ndarray) -> np.ndarray:
    """Rows: grad of the 4 elemental constraints and the enthalpy constraint."""
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


def integrate_to(c: CSTR, gamma: float, q0: np.ndarray, T: float, **kw):
    """Constant-control trajectory endpoint at time T."""
    res = c.integrate(History(np.array([gamma]), np.array([T])), q0,
                      samples_per_segment=2, record_extrema=False, **kw)
    if not res["success"]:
        raise RuntimeError(f"integration failed at gamma={gamma}, T={T}: {res['message']}")
    return res


def log_gamma_derivative_fd(c: CSTR, gamma_star: float, T: float, q0: np.ndarray,
                            rtol: float, atol_T: float, atol_Y: float,
                            rel: float = 1e-3, method: str = "Radau") -> np.ndarray:
    """d phi_gamma(T; q0)/d log gamma by central differences in log gamma.

    The stencil stays strictly inside the admissible interval.  ``method`` is
    exposed so two independent stiff integrators can be compared on the same
    problem - a genuine cross-integration check of the reference derivative.
    """
    eta = np.log(gamma_star / GAMMA_REF)
    room = min(eta - np.log(GAMMA_LO / GAMMA_REF),
               np.log(GAMMA_HI / GAMMA_REF) - eta)
    h = 0.5 * rel * room
    zp = integrate_to(c, gamma_star * np.exp(h), q0, T, method=method,
                      rtol=rtol, atol_T=atol_T, atol_Y=atol_Y)["states"][:, -1]
    zm = integrate_to(c, gamma_star * np.exp(-h), q0, T, method=method,
                      rtol=rtol, atol_T=atol_T, atol_Y=atol_Y)["states"][:, -1]
    return (zp - zm) / (2 * h)


def traj_scan(c: CSTR, gamma: float, q0: np.ndarray, times: np.ndarray):
    """Coarse trajectory at the given physical times, for choosing horizons.

    Uses the dense solution so the sampled times are exact evaluation points,
    not a uniform grid."""
    res = c.integrate(History(np.array([gamma]), np.array([times[-1]])), q0,
                      samples_per_segment=2, record_extrema=False)
    return res["evaluate_dense"](times) if res["success"] else None


def reference_tangents(c: CSTR, gamma_star: float, T: float, q0: np.ndarray,
                       rtol: float, atol_T: float, atol_Y: float,
                       fd_rels=(1e-3, 1e-4)) -> dict:
    """v_t and v_log_gamma from the ORIGINAL initial state q0.

    v_t = F(q_base, gamma_star) is exactly the tangent of the constant-control
    trajectory at its own endpoint.  v_log_gamma is computed by central finite
    differences in log gamma from q0.  The independent tangent-linear pathway is
    run separately (and sparingly) by the FSA cross-check stage; doing it for
    every anchor is far too slow to justify.
    """
    res = integrate_to(c, gamma_star, q0, T, rtol=rtol, atol_T=atol_T, atol_Y=atol_Y)
    q_base = res["states"][:, -1].copy()
    v_t = c.rhs(0.0, q_base, gamma_star)

    fd_rows = {}
    for rel in fd_rels:
        fd_rows[rel] = {"h_eta": None, "v_log_gamma":
                        log_gamma_derivative_fd(c, gamma_star, T, q0, rtol,
                                                atol_T, atol_Y, rel=rel).tolist()}
    return {"q_base": q_base, "v_t": v_t,
            "v_log_gamma": np.array(fd_rows[fd_rels[0]]["v_log_gamma"]),
            "v_log_gamma_fd": fd_rows}


def analyze_jacobian(J: np.ndarray, V: np.ndarray, scaling: StateScaling,
                     C: np.ndarray, rel_tol: float = 1e-8,
                     abs_tol: float | None = None,
                     abs_tol_V: float | None = None,
                     J_alt: np.ndarray | None = None) -> dict:
    """Scaled bases, conservation leakage BEFORE projection, alignment.

    Both J and V are rank-filtered by SVD with explicit thresholds.  The
    conserved-manifold tangent space is the null space of C W^-1, computed by a
    rank-revealing SVD rather than from the normal equations.

    Three rank notions are reported separately, per the review:
      * algebraic (every finite column; NaN columns are invalid stencils),
      * relative-threshold numerical rank (ratio-preserving - misleading at the
        noise floor, where ALL singular values are small),
      * absolute-threshold application rank, against a DECLARED tolerance.
    If ``J_alt`` (an independent finite-difference step size) is given, a
    differentiation-noise scale is estimated as ||J - J_alt||_2 and a
    noise-threshold rank is reported; this is an EMPIRICAL estimate, not a
    rigorous bound.
    """
    W = scaling.W
    ts = scaled_tangent_space(C, scaling)
    # element-wise ROW scaling (W is 1-D): W @ J would multiply a vector by J
    Js = W[:, None] * J
    Vs = W[:, None] * V
    bJ = svd_basis(Js, rel_tol=rel_tol, abs_tol=abs_tol)
    bV = svd_basis(Vs, rel_tol=rel_tol, abs_tol=abs_tol_V)
    noise_scale = None
    if J_alt is not None:
        Js_alt = W[:, None] * np.asarray(J_alt, dtype=float)
        noise_scale = float(np.linalg.norm(Js - Js_alt, ord=2))
    rows = []
    for i in range(bJ.singular_values.size):
        u = None
        if bJ.vectors is not None and i < bJ.vectors.shape[1]:
            u = bJ.vectors[:, i]
        rows.append({
            "index": i, "singular_value_scaled": float(bJ.singular_values[i]),
            "retained": bool(bJ.retained[i]),
            "leakage_before_projection": (ts.leakage(Js[:, i])
                                          if i < Js.shape[1] else None),
            "sin_angle_to_const_ctrl_span": (sin_to_span(u, bV) if u is not None
                                             else None),
        })
    sv = bJ.singular_values
    return {"singular_values_J_scaled": sv.tolist(),
            "singular_values_V_scaled": bV.singular_values.tolist(),
            "rank_J": bJ.rank, "rank_V": bV.rank,
            "rank_J_relative_threshold": int((sv > rel_tol * sv[0]).sum())
            if sv.size and sv[0] > 0 else 0,
            "rank_J_absolute_application": int((sv > abs_tol).sum())
            if abs_tol is not None else None,
            "rank_J_above_noise": int((sv > noise_scale).sum())
            if noise_scale is not None else None,
            "noise_scale_from_step_refinement": noise_scale,
            "threshold_J": bJ.threshold_used, "threshold_V": bV.threshold_used,
            "absolute_application_threshold": abs_tol,
            "projector_residual_J": bJ.projector_residual,
            "projector_residual_V": bV.projector_residual,
            "conditioning_CWinverse": ts.conditioning,
            "rank_CWinverse": ts.rank_CWinverse,
            "tangent_dimension": ts.basis.shape[1] if ts.basis is not None else 0,
            "per_direction": rows,
            "scaling": scaling.to_record()}


def prefix_case(c: CSTR, gamma_star: float, T: float, q0: np.ndarray,
                prefix_pattern: str, rel: float, rtol: float, atol_T: float,
                atol_Y: float, gamma_hold: float | None = None,
                hold_L: float = 0.0, ref_cache: dict | None = None,
                steady_cache: dict | None = None,
                init_label: str = "fresh",
                rel_alt: float = 1e-4) -> dict:
    """One sensitivity case with a constant-control base history.

    Base history: gamma_1 = ... = gamma_m = gamma_star over m equal segments, so
    the base endpoint lies EXACTLY on the constant-control family through q0 and
    the reference tangents there are well defined.  ``prefix_pattern`` perturbs
    the levels to introduce history variation while staying strictly interior.
    """
    m = M_PREFIX
    durs = np.full(m, T / m)
    if prefix_pattern == "equal":
        theta = np.full(m, gamma_star)
    elif prefix_pattern == "early_low":
        theta = np.full(m, gamma_star)
        theta[0] = gamma_star / 3.0
    elif prefix_pattern == "early_high":
        theta = np.full(m, gamma_star)
        theta[0] = min(gamma_star * 3.0, GAMMA_HI / 1.05)
    else:
        raise ValueError(prefix_pattern)
    adm = AdmissibleControls(GAMMA_LO, GAMMA_HI, float(durs.sum()))
    ok, reason = adm.validate(History(theta, durs))
    if not ok:
        raise ValueError(f"inadmissible base history: {reason}")

    fd = endpoint_jacobian_fd(c, theta, durs, q0, rel, GAMMA_LO, GAMMA_HI,
                              GAMMA_REF, rtol=rtol, atol_T=atol_T, atol_Y=atol_Y)
    # an independent step size estimates the differentiation noise scale:
    # ||J(h1) - J(h2)||_2 is an EMPIRICAL noise estimate, not a rigorous bound
    fd_alt = endpoint_jacobian_fd(c, theta, durs, q0, rel_alt, GAMMA_LO, GAMMA_HI,
                                  GAMMA_REF, rtol=rtol, atol_T=atol_T,
                                  atol_Y=atol_Y)
    # terminal hold: propagate the SAME prefix sensitivities through phi^L
    full_durs = np.concatenate([durs, [hold_L]]) if hold_L > 0 else durs
    full_theta = np.concatenate([theta, [gamma_hold]]) if hold_L > 0 else theta
    J_full = J_full_alt = None
    if hold_L > 0:
        fd_full = endpoint_jacobian_fd(c, full_theta, full_durs, q0, rel,
                                       GAMMA_LO, GAMMA_HI, GAMMA_REF, rtol=rtol,
                                       atol_T=atol_T, atol_Y=atol_Y)
        fd_full_alt = endpoint_jacobian_fd(c, full_theta, full_durs, q0, rel_alt,
                                           GAMMA_LO, GAMMA_HI, GAMMA_REF,
                                           rtol=rtol, atol_T=atol_T,
                                           atol_Y=atol_Y)
        J_full, J_full_alt = fd_full["J"], fd_full_alt["J"]

    ref = ref_cache[(gamma_star, T, init_label)] if ref_cache is not None \
        else reference_tangents(c, gamma_star, T, q0, rtol, atol_T, atol_Y)
    V = np.vstack([ref["v_t"], ref["v_log_gamma"]]).T
    q_base = np.array(fd["endpoint"])
    Cmat = constraint_jacobian(c, q_base)
    scaling = StateScaling(n_species=c.gas.n_species)
    analysis = analyze_jacobian(fd["J"], V, scaling, Cmat,
                                abs_tol=ABS_APPLICATION_THRESHOLD,
                                J_alt=fd_alt["J"])

    # identity control: sum of prefix columns equals the constant-control deriv.
    colsum = np.nansum(fd["J"], axis=1)
    identity_err = float(np.max(np.abs(colsum - ref["v_log_gamma"])))

    hold = {}
    if hold_L > 0 and J_full is not None:
        q_end = np.array(fd_full["endpoint"])
        # steady reference for gamma_hold by long relaxation (cached by caller)
        q_ss = steady_cache.get((gamma_hold, init_label)) if steady_cache \
            else None
        C2 = constraint_jacobian(c, q_end)
        hold = {"hold_L": hold_L, "gamma_hold": gamma_hold,
                "endpoint": q_end.tolist(),
                "endpoint_rhs_norm": float(np.linalg.norm(
                    c.rhs(0.0, q_end, gamma_hold))),
                "hold_exposure_gamma_times_L": float(gamma_hold * hold_L),
                "prefix_block_analysis": analyze_jacobian(
                    J_full[:, :m], V, scaling, C2,
                    abs_tol=ABS_APPLICATION_THRESHOLD, J_alt=J_full_alt[:, :m]),
                "full_analysis": analyze_jacobian(
                    J_full, V, scaling, C2,
                    abs_tol=ABS_APPLICATION_THRESHOLD, J_alt=J_full_alt),
                "distance_to_steady": (float(np.linalg.norm(q_end - q_ss))
                                       if q_ss is not None else None)}
    return {"gamma_star": gamma_star, "T": T, "prefix_pattern": prefix_pattern,
            "theta": theta.tolist(), "durations": durs.tolist(),
            "horizon": float(durs.sum()),
            "fd": {k: v for k, v in fd.items() if k != "perturbed_endpoints"},
            "perturbed_endpoints": fd["perturbed_endpoints"],
            "reference": {k: v for k, v in ref.items() if k != "v_log_gamma_fd"},
            "reference_fd_check": ref["v_log_gamma_fd"],
            "identity_max_abs_err": identity_err,
            "identity_scale": float(np.max(np.abs(ref["v_log_gamma"]))),
            "analysis": analysis, "hold": hold}


def steady_state(c: CSTR, gamma: float, q0: np.ndarray, T_relax: float = 50.0,
                 **kw):
    """Long-time steady reference for a constant gamma (relaxation)."""
    res = c.integrate(History(np.array([gamma]), np.array([T_relax])), q0,
                      samples_per_segment=2, record_extrema=False, **kw)
    return res["states"][:, -1].copy() if res["success"] else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase4"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    ap.add_argument("--weak-temperature", type=float, default=900.0,
                    help="separately labelled weak-reaction inlet comparison")
    ap.add_argument("--quick", action="store_true",
                    help="reduced budget for smoke tests (not the study of record)")
    ap.add_argument("--fsa", action="store_true",
                    help="also run the tangent-linear cross-check (exact F_gamma; slow)")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = ReactorConfig(mechanism="h2o2.yaml",
                        inlet_temperature=args.inlet_temperature)
    c = CSTR(cfg)
    scaling = StateScaling(n_species=c.gas.n_species)
    rtol, atol_T, atol_Y = 1e-10, 1e-9, 1e-16
    rtol_coarse, atol_T_coarse, atol_Y_coarse = 1e-8, 1e-8, 1e-14

    out: dict = {
        "timestamp_utc": utc_now(), "mechanism": c.mech, "config": cfg.to_record(),
        "admissible_class": {"gamma_lo": GAMMA_LO, "gamma_hi": GAMMA_HI,
                             "gamma_ref": GAMMA_REF, "m_prefix": M_PREFIX},
        "scaling": scaling.to_record(),
        "absolute_application_threshold": ABS_APPLICATION_THRESHOLD,
        "threshold_justification": (
            "one scaled unit = 100 K or 1% mass fraction (StateScaling); below "
            "this threshold a direction moves the endpoint by less than 1e-4 K "
            "or 1e-6 in mass fraction per unit step in log control. DECLARED, "
            "not measured; the empirical differentiation-noise scale is reported "
            "separately per case."),
        "solver": {"rtol": rtol, "atol_T": atol_T, "atol_Y": atol_Y,
                   "rtol_coarse": rtol_coarse, "atol_T_coarse": atol_T_coarse,
                   "atol_Y_coarse": atol_Y_coarse},
        "stages": {},
    }

    # ---- stage 1: horizon selection from an actual time scan -----------------
    scan = {}
    for Tin_label, Tin in (("Tin1200K", args.inlet_temperature),
                           ("Tin900K_weak", args.weak_temperature)):
        cfg_i = ReactorConfig(mechanism="h2o2.yaml", inlet_temperature=Tin)
        ci = CSTR(cfg_i)
        scan[Tin_label] = {}
        for init_label, q0 in (("fresh", fresh_state(ci)), ("hot", hot_hp_state(ci))):
            scan[Tin_label][init_label] = {}
            for g in ANCHORS:
                traj = traj_scan(ci, g, q0, SCAN_TIMES)
                T_traj = traj[0]
                scan[Tin_label][init_label][f"g{g:.0e}"] = {
                    "times": SCAN_TIMES.tolist(),
                    "T": T_traj.tolist(),
                    "T_max": float(T_traj.max()),
                    "T_end": float(T_traj[-1]),
                    "argmax_time": float(SCAN_TIMES[int(np.argmax(T_traj))]),
                }
    out["stages"]["horizon_scan"] = scan
    print(f"[phase4] horizon scan done ({time.perf_counter()-t0:.1f}s)")
    for Tin_label, per_init in scan.items():
        for init_label, per_g in per_init.items():
            for gk, s in per_g.items():
                print(f"  {Tin_label}/{init_label}/{gk}: T_max={s['T_max']:.1f}K "
                      f"at t={s['argmax_time']:.2e}s, T_end={s['T_end']:.1f}K")

    # horizons: early / ignition-region / relaxed, chosen from the scan
    horizons = [1e-3, 1e-2, 0.1] if not args.quick else [1e-3, 0.1]
    anchors = ANCHORS if not args.quick else (1e3,)
    patterns = ("equal", "early_high") if not args.quick else ("equal",)
    hold_Ls = (0.0, 1e-2, 1e-1, 1.0) if not args.quick else (0.0, 1e-2)

    # reference tangents are shared by every case at the same (gamma, T, init)
    ref_cache = {}
    for init_label, q0 in (("fresh", fresh_state(c)), ("hot", hot_hp_state(c))):
        for g in anchors:
            for T_h in horizons:
                ref_cache[(g, T_h, init_label)] = reference_tangents(
                    c, g, T_h, q0, rtol, atol_T, atol_Y)
    print(f"[phase4] reference tangents built ({time.perf_counter()-t0:.1f}s)")

    # ---- stage 2: reference-tangent agreement across step sizes --------------
    # v_log_gamma by FD at two step sizes; their agreement bounds the
    # differentiation error of the reference.  The independent tangent-linear
    # (FSA) pathway is run below on a small subset only - it is exact in F_gamma
    # and independent of the FD stencil, but too slow for every anchor.
    ident = []
    for g in anchors:
        for T_h in horizons:
            ref = ref_cache[(g, T_h, "fresh")]
            fd = ref["v_log_gamma_fd"]
            errs = {rel: float(np.max(np.abs(np.array(r["v_log_gamma"])
                                             - np.array(fd[min(fd)]["v_log_gamma"]))))
                    for rel, r in fd.items()}
            ident.append({"gamma_star": g, "T": T_h,
                          "v_log_gamma_step_agreement": errs,
                          "scale": float(np.max(np.abs(ref["v_log_gamma"])))})
            print(f"[phase4] reference gamma={g:.0e} T={T_h:.0e}: "
                  f"step agreement={ {k: f'{v:.2e}' for k, v in errs.items()} }")
    out["stages"]["reference_cross_check"] = ident

    # independent cross-integration check: the same reference derivative by a
    # DIFFERENT stiff integrator (LSODA vs Radau) at one step size.  This is a
    # genuine second pathway - different error control and different linear
    # algebra - and is cheap.  The tangent-linear (FSA) pathway is exact in
    # F_gamma but far too slow for routine use; it is available behind --fsa.
    cross = []
    for g, T_h in ((anchors[0], horizons[0]), (anchors[-1], horizons[-1])):
        ref = ref_cache[(g, T_h, "fresh")]
        v_lsoda = log_gamma_derivative_fd(c, g, T_h, fresh_state(c),
                                          rtol, atol_T, atol_Y, rel=1e-3,
                                          method="LSODA")
        err = float(np.max(np.abs(v_lsoda - ref["v_log_gamma"])))
        cross.append({"gamma_star": g, "T": T_h,
                      "lsoda_vs_radau_max_abs_err": err,
                      "scale": float(np.max(np.abs(ref["v_log_gamma"])))})
        print(f"[phase4] cross-integration gamma={g:.0e} T={T_h:.0e}: "
              f"LSODA-vs-Radau max err = {err:.3e} "
              f"(scale {cross[-1]['scale']:.3e})")
    out["stages"]["cross_integration_check"] = cross

    if args.fsa:
        fsa_rows = []
        for g, T_h in ((anchors[0], horizons[0]), (anchors[-1], horizons[-1])):
            ref = ref_cache[(g, T_h, "fresh")]
            fsa = constant_control_fsa(c, g, T_h, fresh_state(c),
                                       rtol=rtol_coarse, atol_T=atol_T_coarse,
                                       atol_Y=atol_Y_coarse)
            if fsa.get("success"):
                err = float(np.max(np.abs(np.array(fsa["d_endpoint_d_log_gamma"])
                                          - ref["v_log_gamma"])))
                fsa_rows.append({"gamma_star": g, "T": T_h,
                                 "fsa_vs_fd_max_abs_err": err,
                                 "scale": float(np.max(np.abs(ref["v_log_gamma"]))),
                                 "fsa_nfev": fsa.get("nfev")})
                print(f"[phase4] FSA cross-check gamma={g:.0e} T={T_h:.0e}: "
                      f"max|fsa-fd| = {err:.3e}")
            else:
                fsa_rows.append({"gamma_star": g, "T": T_h, "success": False,
                                 "message": fsa.get("message")})
                print(f"[phase4] FSA FAILED gamma={g:.0e} T={T_h:.0e}: "
                      f"{fsa.get('message')}")
        out["stages"]["fsa_cross_check"] = fsa_rows

    # ---- stage 3: prefix sensitivity and the terminal-hold sweep -------------
    # steady references for each hold level, by relaxation (coarse is enough)
    steady_cache = {}
    for init_label, q0 in (("fresh", fresh_state(c)), ("hot", hot_hp_state(c))):
        for g in anchors:
            steady_cache[(g, init_label)] = steady_state(
                c, g, q0, rtol=rtol_coarse, atol_T=atol_T_coarse,
                atol_Y=atol_Y_coarse)

    cases = []
    for init_label, q0 in (("fresh", fresh_state(c)), ("hot", hot_hp_state(c))):
        for g in anchors:
            for T_h in horizons:
                for pat in patterns:
                    for L in hold_Ls:
                        rec = prefix_case(c, g, T_h, q0, pat, 1e-3, rtol,
                                          atol_T, atol_Y,
                                          gamma_hold=g, hold_L=L,
                                          ref_cache=ref_cache,
                                          steady_cache=steady_cache,
                                          init_label=init_label)
                        rec["init"] = init_label
                        cases.append(rec)
                        rank = rec["analysis"]["rank_J"]
                        sv = rec["analysis"]["singular_values_J_scaled"]
                        msg = (f"[phase4] {init_label}/g{g:.0e}/T{T_h:.0e}/{pat}"
                               f"/L{L:g}: rank_J={rank} sv="
                               + "[" + ", ".join(f"{v:.2e}" for v in sv) + "]")
                        if L > 0:
                            pr = rec["hold"]["prefix_block_analysis"]
                            msg += (f" | hold-prefix rank={pr['rank_J']} sv="
                                    + "[" + ", ".join(f"{v:.2e}" for v in
                                                      pr['singular_values_J_scaled'])
                                    + f"] dist_ss={rec['hold']['distance_to_steady']:.2e}")
                        print(msg)
    out["stages"]["prefix_and_hold"] = cases
    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase4_results.json", out)
    manifest("phase4", args.results_dir,
             {"inlet_temperature": args.inlet_temperature,
              "weak_temperature": args.weak_temperature, "seed": args.seed,
              "gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF,
              "m_prefix": M_PREFIX},
             cwd=Path(__file__).resolve().parents[1])
    print(f"[phase4] {len(cases)} cases in {out['wall_seconds']:.1f}s "
          f"-> {args.results_dir}")


if __name__ == "__main__":
    main()
