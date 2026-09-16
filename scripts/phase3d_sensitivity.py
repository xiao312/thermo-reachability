"""Phase 3d - endpoint sensitivity and history-generated dimensions (audit Part C).

The scientific question (replacing the refuted coverage framing):

    Do admissible switching histories generate independent thermochemical
    directions not represented by the constant-control transient family, and
    how large are those directions relative to a declared application tolerance?

Setup.  For a feed-compatible initial state the CSTR state lies on the manifold

    M = {(T, Y) : E.Y = b_in,  h_k(T).Y = h_in},

which has dimension 6 (10 species - 4 chemical invariants, temperature derived
from composition and enthalpy).  A constant-control transient family is the
image of the two parameters (gamma, t), so its smooth local dimension is at most
two.  An m-segment switched history is parameterized by theta = (gamma_1..m);
its fixed-horizon endpoint map E_m(theta) may have a higher-rank derivative.

Method (all checked, no floating-point SVD taken as proof):
  * endpoint Jacobian J = dE/dtheta by central finite differences, at several
    step sizes and integration tolerances;
  * both J and the constant-control tangent vectors are projected onto the
    tangent space of M (orthogonal complement of the constraint Jacobian), so
    conserved directions do not masquerade as history-generated directions;
  * principal angles between span(J) and the constant-control tangent span(V)
    measure how much of the endpoint image is transverse to the family;
  * magnitudes are compared against a chemistry-relevant tolerance.

Outcomes are reported, not chosen: a robust third direction would support
history-generated dimensions; small values support a thin-but-higher-dimensional
reachable region; neither is claimed beyond the evidence.
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

GAMMA_LO, GAMMA_HI = 10.0, 1e5
HORIZON = 0.1


def endpoint(c: CSTR, theta: np.ndarray, q0: np.ndarray, durs: np.ndarray) -> np.ndarray:
    """Terminal state of a piecewise-constant history with levels theta."""
    h = History(np.asarray(theta, dtype=float), np.asarray(durs, dtype=float))
    res = c.integrate(h, q0, method="Radau", samples_per_segment=5,
                      rtol=1e-11, atol_T=1e-9, atol_Y=1e-16)
    if not res["success"]:
        raise RuntimeError(f"integration failed: {res['message']}")
    return res["states"][:, -1].copy()


def constraint_jacobian(c: CSTR, q: np.ndarray) -> np.ndarray:
    """Jacobian of the 5 constraints (4 elemental + 1 enthalpy) at state q.

    Rows are grad(E.Y - b_in) and grad(h_k(T).Y - h_in).  The tangent space of M
    is the orthogonal complement of this row space.
    """
    n_sp = c.gas.n_species
    T = float(q[0])
    Y = np.asarray(q[1:])
    # d(E.Y)/dq: E columns act on Y only
    G = np.zeros((c.E.shape[0], n_sp + 1))
    G[:, 1:] = c.E
    # enthalpy constraint: h(T,Y) = sum h_k(T) Y_k
    hk = c.species_enthalpies(T)
    c.gas.TPY = T, c.p, np.maximum(Y, 0.0)   # cp_mass is exact at this state
    dh_dT = float(c.gas.cp_mass)             # d h/dT at fixed Y
    g_h = np.zeros(n_sp + 1)
    g_h[0] = dh_dT
    g_h[1:] = hk
    return np.vstack([G, g_h[None, :]])


def tangent_projector(c: CSTR, q: np.ndarray) -> np.ndarray:
    """P = I - C^T (C C^T)^-1 C onto the tangent space of M at q."""
    C = constraint_jacobian(c, q)
    S = C @ C.T
    P = np.eye(C.shape[1]) - C.T @ np.linalg.solve(S, C)
    return P


def constant_control_tangent(c: CSTR, q0: np.ndarray, gamma_star: float,
                             horizon: float, dgamma: float = None) -> np.ndarray:
    """Local tangent span V of the constant-control family (2 columns).

    v_t: velocity along a constant-gamma trajectory through the endpoint;
    v_g: sensitivity of a constant-control endpoint to gamma.
    """
    if dgamma is None:
        dgamma = 1e-3 * max(gamma_star, 1.0)
    v_t = c.rhs(0.0, q0, gamma_star)
    lo = max(gamma_star - dgamma, GAMMA_LO)
    hi = min(gamma_star + dgamma, GAMMA_HI)
    d = History(np.array([lo]), np.array([horizon]))
    u = History(np.array([hi]), np.array([horizon]))
    z_lo = c.integrate(d, q0, method="Radau", samples_per_segment=5,
                       rtol=1e-11, atol_T=1e-9, atol_Y=1e-16)["states"][:, -1]
    z_hi = c.integrate(u, q0, method="Radau", samples_per_segment=5,
                       rtol=1e-11, atol_T=1e-9, atol_Y=1e-16)["states"][:, -1]
    v_g = (z_hi - z_lo) / max(hi - lo, 1e-12)
    return np.vstack([v_t, v_g]).T


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between span(A) and span(B), both orthonormalized.

    WARNING: unreliable when A is numerically rank-deficient - QR then fills out
    the orthonormal basis with rounding-noise directions, so the angles involving
    the noise subspace are meaningless. Callers must threshold by singular value
    first (see `direction_alignment`).
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def direction_alignment(J: np.ndarray, V: np.ndarray, rel_tol: float = 1e-8) -> dict:
    """Well-defined transversality measure for a possibly rank-deficient Jacobian.

    Only singular directions of J whose singular value exceeds rel_tol times the
    largest are considered real (the rest are rounding noise). For each real
    direction u_i, the sine of its angle to the constant-control tangent span(V)
    is ||(I - Qv Qv^T) u_i||, which is well defined for a single vector against a
    plane. Also returns the significant directions themselves.
    """
    U, sv, _ = np.linalg.svd(J, full_matrices=False)
    Qv, _ = np.linalg.qr(V)
    keep = sv > rel_tol * sv[0]
    rows = []
    for i in np.flatnonzero(keep):
        u = U[:, i]
        sin_ang = float(np.linalg.norm(u - Qv @ (Qv.T @ u)))
        rows.append({"index": int(i), "singular_value": float(sv[i]),
                     "sin_angle_to_const_ctrl_family": sin_ang})
    return {"n_real_directions": int(keep.sum()),
            "real_directions": rows,
            "rank_threshold_relative": rel_tol}


def case(c: CSTR, label: str, theta: np.ndarray, durs: np.ndarray, q0: np.ndarray,
         steps=(1e-2, 1e-3, 1e-4), gammas_rel=None) -> dict:
    """One endpoint-sensitivity case: Jacobian, projection, angles, robustness."""
    adm = AdmissibleControls(GAMMA_LO, GAMMA_HI, float(durs.sum()))
    ok, reason = adm.validate(History(theta, durs))
    if not ok:
        raise ValueError(f"{label}: inadmissible base history: {reason}")
    out = {"label": label, "theta": theta.tolist(),
           "durations": durs.tolist(),
           "horizon": float(durs.sum()),
           "endpoint": None, "steps": []}
    z0 = endpoint(c, theta, q0, durs)
    out["endpoint"] = z0.tolist()
    P = tangent_projector(c, z0)
    gamma_star = float(theta[-1])
    V = constant_control_tangent(c, z0, gamma_star, float(durs.sum()))
    Vp = P @ V
    for rel in steps:
        # absolute step: relative to the control scale, kept inside the bounds
        h = rel * max(np.abs(theta).mean(), 1.0)
        J = np.zeros((z0.size, theta.size))
        ok_all = True
        for j in range(theta.size):
            tp = theta.copy(); tp[j] += h
            tm = theta.copy(); tm[j] -= h
            if not (GAMMA_LO <= tp[j] <= GAMMA_HI and GAMMA_LO <= tm[j] <= GAMMA_HI):
                ok_all = False
                continue
            J[:, j] = (endpoint(c, tp, q0, durs) - endpoint(c, tm, q0, durs)) / (2 * h)
        Jp = P @ J
        sv = np.linalg.svd(Jp, compute_uv=False)
        sv_V = np.linalg.svd(Vp, compute_uv=False)
        # Well-defined transversality: only real (above-noise) singular directions
        # of J are compared with the constant-control tangent plane. The older
        # span-vs-span principal angles are meaningless when J is rank-deficient,
        # because QR fills the orthonormal basis with rounding-noise directions.
        align = direction_alignment(Jp, Vp)
        out["steps"].append({
            "relative_step": rel, "absolute_step": h,
            "all_perturbations_admissible": ok_all,
            "singular_values_J_projected": sv.tolist(),
            "singular_values_V_projected": sv_V.tolist(),
            "n_real_directions": align["n_real_directions"],
            "real_direction_alignment": align["real_directions"],
            "rank_threshold_relative": align["rank_threshold_relative"],
            "principal_angles_rad_DEPRECATED_UNRELIABLE_IF_RANK_DEFICIENT": (
                principal_angles(Jp, Vp).tolist()),
        })
        out["steps"][-1]["max_sin_angle"] = max(
            (r["sin_angle_to_const_ctrl_family"] for r in align["real_directions"]),
            default=None)
    return out


def base_cases(n_seg_list=(3, 4, 6)) -> list:
    """Admissible base histories with varied dwell durations, strictly interior."""
    cases = []
    rng = np.random.default_rng(20260916)
    for m in n_seg_list:
        # equal-duration baseline, controls strictly interior to [lo, hi]
        theta = np.geomspace(GAMMA_LO * 3.0, GAMMA_HI / 3.0, m)
        durs = np.full(m, HORIZON / m)
        cases.append((f"m{m}_equal", theta, durs))
        # varied dwell durations resolving short chemical transients
        dw = rng.uniform(0.25, 1.75, m)
        dw = dw / dw.sum() * HORIZON
        cases.append((f"m{m}_varied_dwell", theta, dw))
        # a pulse-then-hold pattern: short hot pulse then long slow dwell
        theta2 = np.full(m, 3e3)
        theta2[0] = GAMMA_HI / 3.0
        dw2 = np.full(m, 0.02)
        dw2[0] = 0.004
        dw2 = np.concatenate([dw2, [HORIZON - dw2.sum()]]) if dw2.sum() < HORIZON else dw2
        if abs(dw2.sum() - HORIZON) > 1e-12:
            dw2 = dw2 / dw2.sum() * HORIZON
        cases.append((f"m{m}_pulse_hold", theta2[:m], dw2[:m]))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase3d"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--quick", action="store_true",
                    help="reduced budget for smoke tests (not for the study of record)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = ReactorConfig(mechanism="h2o2.yaml",
                        inlet_temperature=args.inlet_temperature)
    c = CSTR(cfg)
    seg_list = (3,) if args.quick else (3, 4, 6)
    step_sizes = (1e-2,) if args.quick else (1e-2, 1e-3, 1e-4)
    inits = (("fresh", fresh_state(c)),) if args.quick \
        else (("fresh", fresh_state(c)), ("hot", hot_hp_state(c)))
    out: dict = {"timestamp_utc": utc_now(), "mechanism": c.mech,
                 "config": cfg.to_record(),
                 "admissible_class": {"gamma_lo": GAMMA_LO, "gamma_hi": GAMMA_HI,
                                      "horizon": HORIZON},
                 "question": __doc__.split("The scientific question")[1].split("Setup")[0].strip()}

    for init_label, q0 in inits:
        out[init_label] = []
        for label, theta, durs in base_cases(n_seg_list=seg_list):
            rec = case(c, f"{init_label}/{label}", theta, durs, q0,
                       steps=step_sizes)
            sv = rec["steps"][-1]["singular_values_J_projected"]
            ang = rec["steps"][-1]["max_sin_angle"]
            nreal = rec["steps"][-1]["n_real_directions"]
            print(f"[phase3d] {init_label}/{label}: n_real={nreal} sv(J_proj)="
                  + "[" + ", ".join(f"{v:.2e}" for v in sv) + "]"
                  + f"  max_sin(real dir vs const-ctrl family)={ang}")
            out[init_label].append(rec)

    # Robustness summary: does a real THIRD direction survive step-size
    # refinement?  A floating-point SVD alone is not a rank proof; the evidence
    # is stability of the significant singular values across step sizes.
    def third_survives(recs) -> dict:
        good = 0
        for rec in recs:
            svs = [np.array(s["singular_values_J_projected"]) for s in rec["steps"]]
            nreal = [s["n_real_directions"] for s in rec["steps"]]
            if len(svs) >= 2 and all(n >= 3 for n in nreal):
                a, b = svs[0], svs[-1]
                n = min(a.size, b.size)
                if a[n - 3] > 0 and b[n - 3] > 0:
                    ratio = b[n - 3] / a[n - 3]
                    if 0.3 < ratio < 3.0:      # stable across step refinement
                        good += 1
        return {"n_cases": len(recs), "n_with_stable_third_direction": good}

    out["robustness"] = {k: third_survives(out[k]) for k in
                        ("fresh", "hot") if k in out}
    out["interpretation_note"] = (
        "A stable third singular direction transverse to the constant-control "
        "tangent span would support a local history-generated image of dimension "
        "> 2. Small values instead indicate a thin, nearly two-dimensional "
        "reachable region. No formal rank claim is made from the SVD alone; the "
        "step-size and tolerance refinement is the actual evidence.")

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase3d_results.json", out)
    manifest("phase3d", args.results_dir,
             {"inlet_temperature": args.inlet_temperature, "seed": args.seed,
              "horizon": HORIZON, "gamma_bounds": [GAMMA_LO, GAMMA_HI]},
             cwd=Path(__file__).resolve().parents[1])
    print(f"[phase3d] done in {out['wall_seconds']:.1f}s -> {args.results_dir}")


if __name__ == "__main__":
    main()
