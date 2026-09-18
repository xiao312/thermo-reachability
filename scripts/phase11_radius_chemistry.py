"""Phase 11 - bounded radius sweep, chemistry response, and a local coverage
statement at the calibrated secondary anchor.

The rank-three result is an accepted numerical milestone; this study asks the
question that follows it: at permitted finite perturbations, can the canonical
constant-control family represent the generated states accurately enough for the
intended chemistry computation, and how well can a local state generator describe
the extra variation?

Design notes:

* Parameter norm.  Everything is measured in the declared TIME control norm
  ||d eta||_time^2 = sum_j (dur_j/T) d eta_j^2, which is portable across segment
  counts.  The phase-9 matrices A = W J and D = W V were computed in the
  Euclidean parameter norm; they are re-expressed through the whitened map
  A H^(-1/2) here, and both norms plus every actual gamma_j are reported so no
  past result silently changes meaning.  For m = 3 equal segments a Euclidean
  radius r is a time-norm radius r/sqrt(3).

* Radii are never clamped.  The exact admissible interval for
  gamma_j(r) = gamma_ref exp(r v_j) is computed first and the schedule is TRIMMED
  outside it, with the trimmed radius recorded as excluded and its binding
  constraint named.

* The reference is B(gamma, t; q_initial) with the ORIGINAL initial state, and it
  is an evaluation comparator: each library gamma is integrated ONCE to a shared
  maximum reference time with dense output retained.  The located candidate is
  refined by damped Gauss-Newton on the scaled residual, whose Jacobian is the
  family tangent at the located point, and re-evaluated by an independent
  integration at tighter tolerances.  The refined distance is an UPPER ESTIMATE
  of the infimum over the domain, never a certified global exclusion.

* Chemistry is measured, not inferred.  The same chemistry-only map
  Phi_dt (adiabatic, constant pressure, exchange control exactly zero) is applied
  to a generated state and to its nearest found representative; initial source
  differences, finite-time future-state differences and chemistry-increment
  differences are reported separately.  No target application tolerances have been
  supplied for this project, so the E_metric table spans explicitly illustrative
  choices and asserts nothing universal.

* Significance is judged on the finite state difference.  A numerically resolved
  geometric difference and a consequential chemical-map difference are separate
  outcomes; neither implies the other.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls, stable_child_seed  # noqa: E402
from thermoreach.chemresponse import (  # noqa: E402
    ChemistryConfig, ReferenceLibrary, admissible_radius_interval,
    chemistry_response, hdiag, local_family_tangent_at, located_reference,
    normal_residual, radius_from_deta, refine_reference, time_norm,
    unwhiten_svd_direction, unit_time_norm_direction, whitened_map,
)
from thermoreach.controls import History  # noqa: E402
from thermoreach.io_utils import manifest, utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, hot_hp_state  # noqa: E402
from thermoreach.sensitivity import (  # noqa: E402
    APPLICATION_THRESHOLD_DEFAULT, StateScaling, svd_basis,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = 1e4                     # the secondary anchor's base level
HORIZON = 1e-6                      # the calibrated secondary anchor
M = 3
RADII = (0.0, 0.03, 0.1, 0.3, 0.6, 1.0)
HELD_OUT_N = 12
HELD_OUT_RADII = (0.1, 0.3, 0.6)
LIB_GAMMA_POINTS = 21
LIB_GAMMA_HALF_WIDTH_LOG = 1.5
LIB_T_MAX_FACTOR = 100.0            # t_max = 100 x the anchor horizon


# ---------------------------------------------------------------------------
# loading and verification of the calibrated matrices
# ---------------------------------------------------------------------------


def load_verified(npz_path: Path, cstr: CSTR, scaling: StateScaling) -> dict:
    """Load the phase-9 matrices and verify them against the loaded MECHANISM
    CONTENT, not just dimensions and filenames: mechanism hash, species order,
    pressure, feed/initial state and the declared scaling.

    A mismatch means the saved matrices describe a different physical system and
    every downstream number would be meaningless, so it is fatal by design."""
    d = np.load(npz_path)
    A, D, J = d["A"], d["D"], d["J"]
    n, m = J.shape
    import hashlib
    from pathlib import Path as _P
    import cantera as _ct
    mech_path = None
    for dir_ in _ct.get_data_directories():
        cand = _P(dir_) / cstr.cfg.mechanism
        if cand.is_file():
            mech_path = cand
            break
    mech_hash = hashlib.sha256(mech_path.read_bytes()).hexdigest()[:16] \
        if mech_path else None
    q0_hot = hot_hp_state(cstr)
    checks = {
        # matrix structure
        "A_shape": list(A.shape), "D_shape": list(D.shape), "J_shape": list(J.shape),
        "n_states": n, "m_segments": m,
        "n_species_matches": bool(n == cstr.gas.n_species + 1),
        "all_finite": bool(np.all(np.isfinite(A)) and np.all(np.isfinite(D))
                           and np.all(np.isfinite(J))),
        # declared scaling
        "W_matches_declared_scaling": bool(
            np.allclose(d["W"], scaling.W, rtol=1e-12)),
        "A_is_scaling_of_J": bool(np.allclose(A, d["W"][:, None] * J, rtol=1e-12)),
        # physical system identity
        "mechanism_content_hash_matches": bool(
            mech_hash is not None
            and cstr.mech.get("sha256", "")[:16] == mech_hash),
        "mechanism_hash_on_disk": mech_hash,
        "mechanism_hash_in_record": cstr.mech.get("sha256", "")[:16],
        "species_order_matches": bool(
            list(cstr.gas.species_names) == cstr.mech.get("species")),
        "pressure_matches": bool(float(cstr.p) == float(cstr.cfg.pressure)),
        "feed_composition_recorded": {"Y_in_sha256_16": hashlib.sha256(
            cstr.Y_in.tobytes()).hexdigest()[:16],
            "h_in_J_kg": float(cstr.h_in),
            "equivalence_ratio": float(cstr.cfg.equivalence_ratio),
            "fuel": cstr.cfg.fuel, "oxidizer": cstr.cfg.oxidizer,
            "T_in_K": float(cstr.T_in)},
        "initial_state_is_hot_hp_equilibrium": bool(
            np.allclose(d["q0"], q0_hot, rtol=1e-9, atol=1e-12)),
        "theta_all_equal_base": bool(np.allclose(d["theta"], GAMMA_REF)),
        "durations_sum_to_horizon": bool(
            np.isclose(float(d["durations"].sum()), HORIZON)),
        "singular_values_reproduce": bool(
            np.allclose(d["singular_values"],
                        np.linalg.svd(A, compute_uv=False), rtol=1e-6)),
    }
    fatal = [k for k, v in checks.items()
             if k in ("n_species_matches", "all_finite", "W_matches_declared_scaling",
                      "A_is_scaling_of_J", "mechanism_content_hash_matches",
                      "species_order_matches", "pressure_matches",
                      "initial_state_is_hot_hp_equilibrium",
                      "durations_sum_to_horizon", "singular_values_reproduce")
             and v is not True]
    return {"npz": npz_path.name, "checks": checks, "verified": not fatal,
            "failed_checks": fatal,
            "A": A, "D": D, "J": J, "q0": d["q0"], "theta": d["theta"],
            "durations": d["durations"], "W": d["W"],
            "singular_values": d["singular_values"], "Vh": d["Vh"], "U": d["U"],
            "endpoint": d["endpoint"]}


# ---------------------------------------------------------------------------
# one generated state and its full analysis
# ---------------------------------------------------------------------------


def physical_state(cstr: CSTR, q: np.ndarray) -> dict:
    T = float(q[0])
    Y = np.asarray(q[1:], dtype=float)
    _T, _Y, rho = cstr.state(T, Y)
    return {"T_K": float(cstr.gas.T), "pressure_Pa": float(cstr.gas.P),
            "density_kg_m3": rho, "enthalpy_mass_J_kg": float(cstr.gas.enthalpy_mass),
            "sum_Y": float(cstr.gas.Y.sum()), "min_Y": float(cstr.gas.Y.min()),
            "min_Y_species": cstr.gas.species_names[int(np.argmin(cstr.gas.Y))],
            "Y": cstr.gas.Y.tolist()}


def integrate_history(cstr: CSTR, theta: np.ndarray, durations: np.ndarray,
                      q0: np.ndarray, method: str = "Radau", rtol: float = 1e-10,
                      atol_T: float = 1e-9, atol_Y: float = 1e-16) -> dict:
    hist = History(np.asarray(theta, dtype=float), np.asarray(durations, dtype=float))
    res = cstr.integrate(hist, q0, method=method, samples_per_segment=51,
                         rtol=rtol, atol_T=atol_T, atol_Y=atol_Y,
                         record_extrema=False)
    if not res["success"]:
        return {"success": False, "message": res["message"], "history": hist.to_record()}
    out = {"success": True, "history": hist.to_record(),
           "terminal_state": res["terminal_state"].tolist(),
           "times": res["times"].tolist(),
           "states": res["states"].tolist(),
           "solver_stats": res["solver_stats"],
           "raw_state_diagnostics": res["raw_state_diagnostics"],
           "method": method, "rtol": rtol,
           "atol": {"T": atol_T, "Y": atol_Y}}
    out["conservation"] = cstr.residuals(res["times"], res["states"], hist)
    return out


def analyze_case(cstr: CSTR, lib: ReferenceLibrary, q0: np.ndarray, q_base: np.ndarray,
                 J: np.ndarray, A: np.ndarray, W: np.ndarray, v_time: np.ndarray,
                 theta0: np.ndarray, durations: np.ndarray, T: float,
                 radius: float, sign: int, label: str,
                 chem_cfg: ChemistryConfig, adm: AdmissibleControls,
                 cross_check: bool = False) -> dict:
    """One admissible perturbation: generate the state, locate its reference,
    measure the chemistry response, and decompose the residual."""
    t_start = time.perf_counter()
    rec = {"label": label, "kind": "transverse_ray", "sign": sign,
           "radius_time_norm": float(radius), "status": "running"}
    # v_time is a Euclidean-unit RIGHT singular vector of the whitened map
    # A H^(-1/2).  The physical log-control direction of unit TIME norm is
    # H^(-1/2) v_time; using v_time directly would give an actual time norm of
    # r/sqrt(m) for m equal segments.
    v_whitened = np.asarray(v_time, dtype=float)
    v = unwhiten_svd_direction(v_whitened, durations, T)
    r_signed = sign * float(radius)
    deta = r_signed * v
    theta = np.asarray(theta0, dtype=float) * np.exp(deta)
    rec["deta_time_norm_coords"] = deta.tolist()
    rec["theta_actual"] = theta.tolist()
    rec["durations"] = durations.tolist()
    # both radii are derived from deta itself, never from a memorized factor
    rec["norms"] = {"time_norm_of_deta": time_norm(deta, durations, T),
                    "euclidean_norm_of_deta": float(np.linalg.norm(deta)),
                    "radius_time_norm": float(radius),
                    "radius_euclidean": float(np.linalg.norm(deta)),
                    "whitened_direction_euclidean_norm":
                        float(np.linalg.norm(v_whitened)),
                    "unwhitening_note": (
                        "deta = r H^(-1/2) u for u a Euclidean-unit right singular "
                        "vector of A H^(-1/2), so ||deta||_time = r exactly")}
    ok, reason = adm.validate(History(theta, durations))
    rec["admissible"] = bool(ok)
    if not ok:
        rec["status"] = "excluded"
        rec["exclusion_reason"] = reason
        return rec

    # ---- 1. the nonlinear history -------------------------------------
    hist = integrate_history(cstr, theta, durations, q0)
    if not hist["success"]:
        rec["status"] = "failed"
        rec["failure"] = hist["message"]
        return rec
    rec["history_integration"] = hist
    q_target = np.asarray(hist["terminal_state"], dtype=float)
    rec["q_target"] = q_target.tolist()
    rec["physical_state"] = physical_state(cstr, q_target)

    # ---- 2. whole-state Taylor residual --------------------------------
    q_pred = np.asarray(q_base, dtype=float) + J @ deta
    R = q_target - q_pred
    lin = A @ deta                       # scaled linear prediction
    lin_norm = float(np.linalg.norm(lin))
    rec["taylor_residual"] = {
        "linear_prediction_scaled_norm": lin_norm,
        "residual_scaled_norm": float(np.linalg.norm(W * R)),
        "residual_over_linear_prediction":
            (float(np.linalg.norm(W * R) / lin_norm) if lin_norm > 1e-12
             else None),
        "ratio_undefined_reason": (
            "the linear prediction is zero (a zero-radius case), so the "
            "residual-to-prediction ratio is 0/0 and is not reported")
            if lin_norm <= 1e-12 else None,
        "residual_raw": R.tolist(),
        "note": ("the residual is the whole-state Taylor remainder; its "
                 "projection onto the MOVING curved reference is reported "
                 "separately below")}

    # ---- 3. nearest canonical representative ---------------------------
    ref = located_reference(lib, q_target, W, cstr, label=label)
    rec["reference"] = ref
    if not ref.get("refined", {}).get("success"):
        rec["status"] = "reference_unresolved"
        rec["failure"] = ref.get("refined", {}).get("message", "refinement failed")
        return rec

    # ---- 4. chemistry response -----------------------------------------
    rec["chemistry"] = chemistry_response(cstr, q_target,
                                          np.asarray(ref["q_B"], dtype=float),
                                          chem_cfg)

    # ---- 5. normal residual after re-fitting the curved reference ------
    nr = normal_residual(cstr, q0, q_target, np.asarray(ref["q_B"], dtype=float),
                         ref["gamma_best"], ref["t_best"], W, dlog=1e-4,
                         method="Radau", rtol=1e-11, atol_T=1e-10, atol_Y=1e-17)
    rec["normal_residual"] = nr

    # ---- 6. cross-check at tighter tolerances and another integrator ----
    if cross_check:
        hist2 = integrate_history(cstr, theta, durations, q0, method="LSODA",
                                  rtol=1e-11, atol_T=1e-10, atol_Y=1e-17)
        rec["cross_check"] = {
            "history_LSODA": {"success": hist2["success"],
                              "terminal_state": hist2.get("terminal_state")},
            "state_difference_between_integrators":
                (np.asarray(hist2.get("terminal_state", q_target), dtype=float)
                 - q_target).tolist() if hist2["success"] else None,
            "reference_LSODA": located_reference(
                lib, q_target, W, cstr, refine_method="LSODA",
                refine_rtol=1e-12, refine_atol_T=1e-11, refine_atol_Y=1e-18,
                label=label + "_xcheck") if hist2["success"] else None}

    rec["status"] = "completed"
    rec["wall_seconds"] = time.perf_counter() - t_start
    return rec


def in_family_control(cstr: CSTR, lib: ReferenceLibrary, q0: np.ndarray, W: np.ndarray,
                      theta0: np.ndarray, durations: np.ndarray, T: float,
                      radius: float, chem_cfg: ChemistryConfig,
                      adm: AdmissibleControls) -> dict:
    """EXACT in-family positive control: all prefix levels equal and perturbed
    together.  The reference family contains this trajectory at its own gamma and
    at t = T, so the search must recover it within verified numerical accuracy.
    A vector called v1(A) is not an exact in-family finite control and is not a
    substitute."""
    t_start = time.perf_counter()
    gamma_c = float(theta0[0]) * np.exp(radius)
    theta = np.full(len(theta0), gamma_c)
    rec = {"label": f"infamily_r{radius:g}", "kind": "in_family_control",
           "radius_time_norm": float(radius), "gamma_control": gamma_c,
           "theta_actual": theta.tolist(), "durations": durations.tolist(),
           "expected_reference": {"gamma": gamma_c, "t": float(T)}}
    ok, reason = adm.validate(History(theta, durations))
    rec["admissible"] = bool(ok)
    if not ok:
        rec["status"] = "excluded"
        rec["exclusion_reason"] = reason
        return rec
    hist = integrate_history(cstr, theta, durations, q0)
    if not hist["success"]:
        rec["status"] = "failed"
        rec["failure"] = hist["message"]
        return rec
    q_target = np.asarray(hist["terminal_state"], dtype=float)
    rec["q_target"] = q_target.tolist()
    ref = located_reference(lib, q_target, W, cstr)
    rec["reference"] = ref
    if ref.get("refined", {}).get("success"):
        dg = abs(ref["gamma_best"] - gamma_c) / gamma_c
        dt = abs(ref["t_best"] - float(T)) / float(T)
        rec["recovery_error"] = {"relative_gamma": float(dg),
                                 "relative_t": float(dt),
                                 "dist_scaled": ref["dist_scaled_upper_estimate"],
                                 "recovered_within_numerical_accuracy":
                                     bool(dg < 1e-3 and dt < 1e-2
                                          and ref["dist_scaled_upper_estimate"]
                                          < 1e-6 * np.linalg.norm(W))}
    rec["status"] = "completed"
    rec["wall_seconds"] = time.perf_counter() - t_start
    return rec


def held_out_case(cstr: CSTR, lib: ReferenceLibrary, q0: np.ndarray, q_base: np.ndarray,
                  J: np.ndarray, A: np.ndarray, W: np.ndarray, theta0: np.ndarray,
                  durations: np.ndarray, T: float, rng: np.random.Generator,
                  radius: float, chem_cfg: ChemistryConfig,
                  adm: AdmissibleControls, index: int) -> dict:
    """A NEW admissible three-segment history on a random direction at the given
    time-norm radius, kept separate from the four original rays so the fit is
    tested rather than memorized."""
    t_start = time.perf_counter()
    for _trial in range(200):
        v = rng.standard_normal(len(theta0))
        v = v / np.sqrt(np.sum(hdiag(durations, T) * v ** 2))     # unit time norm
        theta = np.asarray(theta0, dtype=float) * np.exp(radius * v)
        ok, reason = adm.validate(History(theta, durations))
        if ok:
            break
    else:
        return {"label": f"heldout_{index}", "kind": "held_out",
                "status": "excluded",
                "exclusion_reason": f"no admissible direction in 200 trials: {reason}"}
    rec = {"label": f"heldout_{index}", "kind": "held_out",
           "radius_time_norm": float(radius), "sign": 0,
           "direction_time_norm_coords": v.tolist(),
           "theta_actual": theta.tolist(), "durations": durations.tolist(),
           "admissible": True}
    hist = integrate_history(cstr, theta, durations, q0)
    if not hist["success"]:
        rec["status"] = "failed"
        rec["failure"] = hist["message"]
        return rec
    q_target = np.asarray(hist["terminal_state"], dtype=float)
    rec["q_target"] = q_target.tolist()
    deta = radius * v
    q_pred = np.asarray(q_base, dtype=float) + J @ deta
    R = q_target - q_pred
    lin_norm = float(np.linalg.norm(A @ deta))
    rec["taylor_residual"] = {
        "linear_prediction_scaled_norm": lin_norm,
        "residual_scaled_norm": float(np.linalg.norm(W * R)),
        "residual_over_linear_prediction":
            (float(np.linalg.norm(W * R) / lin_norm) if lin_norm > 1e-12
             else None)}
    ref = located_reference(lib, q_target, W, cstr)
    rec["reference"] = ref
    if ref.get("refined", {}).get("success"):
        rec["chemistry"] = chemistry_response(cstr, q_target,
                                              np.asarray(ref["q_B"], dtype=float),
                                              chem_cfg)
        rec["normal_residual"] = normal_residual(
            cstr, q0, q_target, np.asarray(ref["q_B"], dtype=float),
            ref["gamma_best"], ref["t_best"], W, dlog=1e-4, method="Radau",
            rtol=1e-11, atol_T=1e-10, atol_Y=1e-17)
    rec["status"] = "completed"
    rec["wall_seconds"] = time.perf_counter() - t_start
    return rec


# ---------------------------------------------------------------------------
# local curvature descriptors and the normal-residual structure
# ---------------------------------------------------------------------------


def curvature_descriptors(cases: list, durations, T: float) -> dict:
    """L_fit (a descriptive least-squares curvature fit) and L_sample_max (the
    worst observed sample), in the declared time norm.

    NEITHER is a containing envelope.  L_sample_max contains the observed samples
    only; it is not a uniform derivative bound over the neighbourhood.  L_fit is a
    least-squares descriptor and individual samples can exceed it - the report
    checks and states by how much."""
    rows = []
    for c in cases:
        if c.get("status") != "completed" or "taylor_residual" not in c:
            continue
        radius = c.get("radius_time_norm")
        if "deta_time_norm_coords" in c:
            deta = np.asarray(c["deta_time_norm_coords"], dtype=float)
        elif "direction_time_norm_coords" in c:
            deta = float(radius) * np.asarray(
                c["direction_time_norm_coords"], dtype=float)
        else:
            continue
        n = time_norm(deta, durations, T)
        rho = c["taylor_residual"]["residual_scaled_norm"]
        if n > 0.0 and np.isfinite(rho):
            rows.append((n, rho))
    if len(rows) < 2:
        return {"available": False, "reason": "fewer than two usable samples",
                "n_samples": len(rows)}
    n_arr = np.array([r[0] for r in rows])
    rho = np.array([r[1] for r in rows])
    # ||W R|| ~ (L/2) ||d eta||_time^2  ->  L = 2 sum(n^2 rho) / sum(n^4)
    L_fit = float(2.0 * np.sum(n_arr ** 2 * rho) / np.sum(n_arr ** 4))
    L_sample_max = float(np.max(2.0 * rho / n_arr ** 2))
    envelope = 0.5 * L_fit * n_arr ** 2
    n_exceed = int(np.sum(rho > envelope))
    return {"available": True, "n_samples": len(rows),
            "L_fit": L_fit,
            "L_sample_max": L_sample_max,
            "L_sample_max_over_L_fit": float(L_sample_max / L_fit) if L_fit > 0 else None,
            "n_samples_exceeding_L_fit_envelope": n_exceed,
            "max_excess_factor": float(np.max(rho / np.maximum(envelope, 1e-300))),
            "note": ("L_fit is a descriptive least-squares fit, not a shell that "
                     "contains all samples; L_sample_max contains the observed "
                     "samples and is not a uniform derivative bound.  Both are in "
                     "the declared time norm and are NOT comparable to the "
                     "Euclidean-norm L reported in phase 10."),
            "samples": [{"radius_time_norm": float(n),
                         "residual_scaled": float(r),
                         "L_fit_envelope": float(e)}
                        for n, r, e in zip(n_arr, rho, envelope)]}


def normal_residual_structure(cases: list) -> dict:
    """Is the resolved normal residual largely described by one or a few local
    directions?  The basis is recomputed at each located point; the anchor normal
    is not assumed to remain correct."""
    vecs, meta = [], []
    for c in cases:
        nr = c.get("normal_residual") or {}
        if not nr.get("available"):
            continue
        v = np.asarray(nr["normal_residual_scaled"], dtype=float)
        if np.all(np.isfinite(v)) and np.linalg.norm(v) > 0:
            vecs.append(v)
            meta.append({"label": c["label"], "norm": float(np.linalg.norm(v)),
                         "tangent_rank": nr.get("tangent_rank")})
    if len(vecs) < 2:
        return {"available": False, "reason": "fewer than two normal residuals",
                "n": len(vecs)}
    M = np.array(vecs).T
    sv = np.linalg.svd(M, compute_uv=False)
    tot = float(np.sum(sv ** 2))
    cum = np.cumsum(sv ** 2) / tot if tot > 0 else np.zeros_like(sv)
    return {"available": True, "n_samples": len(vecs),
            "singular_values": sv.tolist(),
            "cumulative_energy_fraction": cum.tolist(),
            "n_directions_for_90_percent_energy": int(np.searchsorted(cum, 0.90) + 1),
            "n_directions_for_99_percent_energy": int(np.searchsorted(cum, 0.99) + 1),
            "samples": meta,
            "note": ("the normal basis is recomputed at each located reference "
                     "point; concentration in one direction would make a "
                     "one-dimensional normal correction a candidate local model")}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase11"))
    ap.add_argument("--npz", type=Path,
                    default=Path("results/phase9/secondary_hot_g1e+04_T1e-06.npz"))
    ap.add_argument("--radii", type=float, nargs="*", default=list(RADII))
    ap.add_argument("--held-out", type=int, default=HELD_OUT_N)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    cfg = ReactorConfig(mechanism="h2o2.yaml", inlet_temperature=1200.0)
    cstr = CSTR(cfg)
    scaling = StateScaling(n_species=cstr.gas.n_species)
    W = scaling.W
    chem_cfg = ChemistryConfig()
    adm = AdmissibleControls(GAMMA_LO, GAMMA_HI, HORIZON, max_segments=M)

    loaded = load_verified(args.npz, cstr, scaling)
    if not loaded["verified"]:
        raise SystemExit(f"phase-9 matrices failed verification: {loaded['checks']}")
    q0 = np.asarray(loaded["q0"], dtype=float)
    theta0 = np.asarray(loaded["theta"], dtype=float)
    durations = np.asarray(loaded["durations"], dtype=float)
    T = float(durations.sum())
    q_base = np.asarray(loaded["endpoint"], dtype=float)
    J, A = loaded["J"], loaded["A"]

    # ---- time-norm migration: whitened map and the fixed transverse direction --
    # A is the endpoint Jacobian in the m-segment log-control coordinates, so it
    # IS whitened by the segment metric H.  D is NOT: its two columns are the
    # family tangents dB/dtau and dB/deta, i.e. a 2-D subspace of the SCALED STATE
    # space parametrized by the family's own coordinates (t, gamma), not by the
    # m segment controls.  The reference projector P therefore does not change
    # when the input norm changes; only A does.
    A_time = whitened_map(A, durations, T)
    D_time = loaded["D"]
    ref_basis = svd_basis(D_time, rel_tol=1e-10)
    P = ref_basis.vectors @ ref_basis.vectors.T if ref_basis.vectors is not None else None
    B_time = (np.eye(A_time.shape[0]) - P) @ A_time if P is not None else None
    sv_tot_time = np.linalg.svd(A_time, compute_uv=False)
    sv_perp_time = np.linalg.svd(B_time, compute_uv=False) \
        if B_time is not None else np.full(1, np.nan)
    # FIXED leading transverse direction for the first sweep.  The SVD is taken
    # in the whitened coordinates, then UNWHITENED to physical log-control
    # coordinates of unit time norm; the admissible interval is computed on the
    # converted direction, since r is a time-norm radius.
    Vh_t = np.linalg.svd(A_time, full_matrices=False)[2]
    Bsvd = np.linalg.svd(B_time, full_matrices=False) if B_time is not None else None
    u_perp_whitened = Bsvd[2][0] if (Bsvd is not None and Bsvd[2].size) else Vh_t[2]
    v_perp_time = unwhiten_svd_direction(u_perp_whitened, durations, T)
    interval = admissible_radius_interval(theta0, v_perp_time, GAMMA_LO, GAMMA_HI)

    # ---- the cached continuous reference library -------------------------
    g_lo = GAMMA_REF * np.exp(-LIB_GAMMA_HALF_WIDTH_LOG)
    g_hi = GAMMA_REF * np.exp(LIB_GAMMA_HALF_WIDTH_LOG)
    gamma_grid = np.exp(np.linspace(np.log(g_lo), np.log(g_hi), LIB_GAMMA_POINTS))
    t_max = LIB_T_MAX_FACTOR * T
    lib = ReferenceLibrary(cstr, q0, gamma_grid, t_max=t_max, t_anchor=T,
                           method="Radau", rtol=1e-10, atol_T=1e-9, atol_Y=1e-16,
                           t_grid_points=121)

    # trim the radius schedule outside the exact admissible interval
    radii, excluded = [], []
    for r in args.radii:
        if r == 0.0:
            radii.append(r)
            continue
        okp = interval["r_max_positive"] is None or r <= interval["r_max_positive"]
        okn = interval["r_min_negative"] is None or -r >= interval["r_min_negative"]
        if okp and okn:
            radii.append(r)
        else:
            excluded.append({"radius": r, "reason":
                             f"outside the admissible interval "
                             f"{interval['r_min_negative']} .. "
                             f"{interval['r_max_positive']}; binding constraints: "
                             f"{interval['binding_constraint_positive']} / "
                             f"{interval['binding_constraint_negative']}"})
    if args.quick:
        radii = [r for r in radii if r <= 0.1]

    out = {
        "timestamp_utc": utc_now(),
        "mechanism": cstr.mech, "config": cfg.to_record(),
        "scaling": scaling.to_record(),
        "application_threshold": APPLICATION_THRESHOLD_DEFAULT,
        "gamma_bounds": [GAMMA_LO, GAMMA_HI], "anchor": {
            "init": "hot_hp_equilibrium", "gamma_ref": GAMMA_REF,
            "horizon_s": T, "m_segments": M, "npz": args.npz.name},
        "control_norm": {
            "declared": "||d eta||_time^2 = sum_j (duration_j/T) d eta_j^2",
            "H_diag": hdiag(durations, T).tolist(),
            "euclidean_radius_to_time_radius_factor": 1.0 / np.sqrt(len(durations)),
            "note": ("the phase-9 matrices were computed in the Euclidean "
                     "parameter norm and are re-expressed here through the "
                     "whitened map A H^(-1/2); both norms and every actual "
                     "gamma_j are reported so no past result changes meaning")},
        "spectra_time_norm": {
            "singular_values_total": sv_tot_time.tolist(),
            "singular_values_transverse": sv_perp_time.tolist(),
            "reference_basis": ref_basis.to_record(),
            "transverse_direction_time_norm_coords": v_perp_time.tolist(),
            "whitened_svd_vector_euclidean_unit": u_perp_whitened.tolist(),
            "euclidean_norm_of_transverse_direction":
                float(np.linalg.norm(v_perp_time)),
            "time_norm_of_transverse_direction":
                time_norm(v_perp_time, durations, T),
            "fixed_direction_note": ("the leading transverse direction is FIXED "
                                     "for this sweep; it is not claimed to remain "
                                     "globally optimal as the amplitude grows"),
            "phase9_euclidean_singular_values": loaded["singular_values"].tolist()},
        "admissible_radius_interval": interval,
        "radius_schedule": {"requested": list(args.radii), "used": list(radii),
                            "excluded": excluded},
        "reference_library": lib.identity(),
        "chemistry_config": chem_cfg.to_record(),
        "matrix_verification": loaded["checks"],
        "index": [], "cases": [],
        "complete": False,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest("phase11", args.results_dir,
             {"anchor": out["anchor"], "control_norm": out["control_norm"],
              "radii": list(radii), "held_out": args.held_out},
             cwd=Path(__file__).resolve().parents[1], manifest_name="manifest.json")

    def checkpoint(rec: dict, kind: str) -> None:
        out["cases"].append(rec)
        out["index"].append({"label": rec.get("label"), "kind": kind,
                             "status": rec.get("status", "unknown"),
                             "radius": rec.get("radius_time_norm"),
                             "sign": rec.get("sign")})
        out["wall_seconds"] = time.perf_counter() - t0
        write_json(args.results_dir / "phase11_results.json", out)

    # ---- 1. the transverse radius sweep ---------------------------------
    for r in radii:
        signs = (0,) if r == 0.0 else (+1, -1)
        for s in signs:
            cc = (r == max(radii)) or (r == 0.0)      # cross-check the extremes
            rec = analyze_case(cstr, lib, q0, q_base, J, A, W, v_perp_time,
                               theta0, durations, T, r, s,
                               f"transverse_r{r:g}_s{s:+d}", chem_cfg, adm,
                               cross_check=bool(cc))
            checkpoint(rec, "transverse_ray")

    # ---- 2. the exact in-family positive control -------------------------
    for r in [x for x in radii if x > 0.0][:3]:
        rec = in_family_control(cstr, lib, q0, W, theta0, durations, T, r,
                                chem_cfg, adm)
        checkpoint(rec, "in_family_control")

    # ---- 3. the held-out histories ---------------------------------------
    seed = stable_child_seed(20240, "phase11-held-out")
    rng = np.random.default_rng(seed)
    for i in range(args.held_out):
        r = HELD_OUT_RADII[i % len(HELD_OUT_RADII)]
        rec = held_out_case(cstr, lib, q0, q_base, J, A, W, theta0, durations, T,
                            rng, r, chem_cfg, adm, i)
        checkpoint(rec, "held_out")

    # ---- 4. local coverage descriptors -----------------------------------
    out["curvature_descriptors"] = curvature_descriptors(out["cases"], durations, T)
    out["normal_residual_structure"] = normal_residual_structure(out["cases"])
    n_ok = sum(1 for c in out["index"] if c["status"] == "completed")
    out["summary"] = {
        "n_indexed": len(out["index"]),
        "n_completed": n_ok,
        "n_failed": sum(1 for c in out["index"] if c["status"] == "failed"),
        "n_excluded": sum(1 for c in out["index"] if c["status"] == "excluded"),
        "n_reference_unresolved":
            sum(1 for c in out["index"] if c["status"] == "reference_unresolved"),
        "library_integrations": len(lib._sols),
        "library_failures": lib._failures,
    }
    out["wall_seconds"] = time.perf_counter() - t0
    out["complete"] = True
    write_json(args.results_dir / "phase11_results.json", out)
    cd = out["curvature_descriptors"]
    print(f"[phase11] {n_ok}/{len(out['index'])} completed in "
          f"{out['wall_seconds']:.1f}s")
    print(f"[phase11] curvature: L_fit={cd.get('L_fit')} "
          f"L_sample_max={cd.get('L_sample_max')} "
          f"exceeding envelope={cd.get('n_samples_exceeding_L_fit_envelope')}")
    nrs = out["normal_residual_structure"]
    print(f"[phase11] normal structure: {nrs.get('n_directions_for_90_percent_energy')} "
          f"direction(s) for 90% energy over {nrs.get('n_samples')} samples")


if __name__ == "__main__":
    main()
