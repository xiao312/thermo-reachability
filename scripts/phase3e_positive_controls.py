"""Phase 3e - positive controls for the sensitivity pipeline (review section C).

Before any detailed chemistry, this script proves that the analysis pipeline can
DETECT the geometry it is asked to measure.  A negative result is only
interpretable if the analyzer demonstrably resolves a positive case at the same
scale.  All checks here are exact or near-exact and run without Cantera.

C.1  Exact toy CSTR: a nondegenerate finite-time history with expected rank 2,
     detected by the analyzer; plus a terminal hold showing that a finite-time
     invertible flow preserves the exact prefix rank even after the effective
     rank drops below numerical resolution.
C.2  The three-state linear benchmark of the review specification:
     dz_i/dt = -lambda_i z_i + u(t), lambda = (1,2,4), T = 1, three equal
     segments.  The exact endpoint control Jacobian has rank 3 and singular
     values (0.5007801, 0.08533145, 0.006828632); the analyzer must detect the
     third direction.
C.3  The same-initial-state identity sum_j dE_m/d(log gamma_j)
     = d phi_gamma(T; q0)/d log gamma at constant-control base histories.
C.4  The reference derivative against exact toy formulas.
C.5  Genuine orthogonality and rank-deficient-reference angle tests, including
     V = [e1, 0] versus J = [e2], where the correct sine is ONE, not zero.

Run: python scripts/phase3e_positive_controls.py --results-dir results/phase3e
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.io_utils import manifest, utc_now, write_json  # noqa: E402
from thermoreach.sensitivity import (  # noqa: E402
    Basis, PerturbationPlan, StateScaling, log_control_plan, linear3_endpoint_map,
    linear3_exact_jacobian, sin_to_span, svd_basis, toy_constant_control_loggamma_derivative,
    toy_endpoint, toy_endpoint_sensitivity,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = float(np.sqrt(GAMMA_LO * GAMMA_HI))     # geometric mean = 1000


def log_fd_jacobian(f, theta: np.ndarray, relative: float, lo: float, hi: float,
                    ref: float) -> tuple[np.ndarray, PerturbationPlan]:
    """Generic central-difference Jacobian in log(theta/ref) with per-coordinate
    interior steps and NO fabricated zero columns.  Used for the toy and
    three-state benchmarks, which have exact endpoints and so need no ODE."""
    theta = np.asarray(theta, dtype=float)
    plan = log_control_plan(theta, lo, hi, ref, relative)
    m = theta.size
    base = f(theta)
    J = np.full((base.size, m), np.nan)
    for j in range(m):
        if not plan.column_valid(j):
            continue
        tp = theta.copy(); tp[j] = plan.gamma(j, +1)
        tm = theta.copy(); tm[j] = plan.gamma(j, -1)
        J[:, j] = (f(tp) - f(tm)) / (np.log(tp[j]) - np.log(tm[j]))
    return J, plan


def analyze(J: np.ndarray, scaling: StateScaling, rel_tol: float = 1e-8,
            abs_tol: float | None = None) -> dict:
    """Scaled basis of span(J) with explicit numerical-rank bookkeeping."""
    Js = scaling.scale(J) if J.shape[0] == scaling.W.size else J
    b = svd_basis(Js, rel_tol=rel_tol, abs_tol=abs_tol)
    return {"basis": b.to_record(), "rank": b.rank}


def raw_fd_jacobian(f, theta: np.ndarray, relative: float, lo: float, hi: float
                    ) -> np.ndarray:
    """Central-difference Jacobian in RAW controls with per-coordinate interior
    steps and NaN (never zero) for a column whose stencil cannot be centred.
    Used by benchmarks whose parameter may be signed."""
    theta = np.asarray(theta, dtype=float)
    base = f(theta)
    J = np.full((base.size, theta.size), np.nan)
    for j in range(theta.size):
        room = min(theta[j] - lo, hi - theta[j])
        h = 0.5 * relative * room
        if h <= 0.0:
            continue
        tp = theta.copy(); tp[j] += h
        tm = theta.copy(); tm[j] -= h
        J[:, j] = (f(tp) - f(tm)) / (2.0 * h)
    return J


def check_linear3() -> dict:
    """C.2: the analyzer must detect all three directions of the exact map."""
    lam = np.array([1.0, 2.0, 4.0])
    T, m = 1.0, 3
    G_ex = linear3_exact_jacobian(lam, T, m)
    sv_ex = np.linalg.svd(G_ex, compute_uv=False)
    # finite-difference reconstruction in RAW controls: the map is linear, so
    # the raw-control Jacobian is exactly G, whose singular values are the
    # benchmark.  (A log-control FD would return diag(u) G with different
    # singular values; the log parameterization is for the positive CSTR rates.)
    controls = np.array([0.3, 0.5, 0.7])          # values; the map is linear in them
    lo, hi = -10.0, 10.0                          # raw controls may be signed

    def fmap(u):
        return linear3_endpoint_map(lam, T, m, np.asarray(u, dtype=float))

    rows = {}
    for rel in (1e-2, 1e-3, 1e-4):
        J = raw_fd_jacobian(fmap, controls, rel, lo, hi)
        sv = np.linalg.svd(J, compute_uv=False)
        b = svd_basis(J, rel_tol=1e-8)
        rows[rel] = {"singular_values_fd": sv.tolist(),
                     "singular_values_exact": sv_ex.tolist(),
                     "rank_detected": b.rank,
                     "n_valid_columns": int(np.isfinite(J).all(axis=0).sum()),
                     "max_rel_sv_error":
                         float(np.max(np.abs(sv - sv_ex) / np.maximum(sv_ex, 1e-300)))}
    # third singular value must be at FIXED index 2, independent of m
    idx_rows = []
    for mm in (3, 4, 6):
        G = linear3_exact_jacobian(lam, T, mm)
        s = np.linalg.svd(G, compute_uv=False)
        idx_rows.append({"m": mm, "third_largest_singular_value": float(s[2]),
                         "index_2_is_third": bool(s[2] == np.sort(s)[::-1][2])})
    return {"exact_singular_values": sv_ex.tolist(),
            "expected": [0.5007801, 0.08533145, 0.006828632],
            "rank_exact": int(np.linalg.matrix_rank(G_ex)),
            "steps": rows,
            "third_value_index_check": idx_rows,
            "passed": bool(all(r["rank_detected"] == 3 for r in rows.values())
                           and all(r["index_2_is_third"] for r in idx_rows)
                           and abs(sv_ex[0] - 0.5007801) < 1e-7
                           and abs(sv_ex[2] - 0.006828632) < 1e-8),
            "note": ("the third singular value is at FIXED index 2 for every m; "
                     "the old index n-3 read the smallest value when m = 3")}


def check_toy_rank2() -> dict:
    """C.1: a toy history whose endpoint Jacobian has rank 2, plus a terminal
    hold that preserves the exact prefix rank while the effective rank drops."""
    rng = np.random.default_rng(7)
    q0 = np.array([0.8, 0.02])
    # two distinct controls at two distinct times generically give a rank-2
    # endpoint Jacobian (the two columns are not parallel)
    for trial in range(200):
        m = 2
        gam = np.exp(rng.uniform(np.log(1.0), np.log(50.0), m))
        dur = rng.uniform(0.05, 1.0, m)
        if abs(gam[0] - gam[1]) < 0.5:
            continue
        S = toy_endpoint_sensitivity(gam, dur, q0)
        if np.linalg.matrix_rank(S) == 2:
            break
    else:
        raise RuntimeError("no rank-2 toy history found")
    scaling = StateScaling(n_species=1)      # toy has 2 components, not 11
    scaling = StateScaling(T_interval=1.0, Y_interval=1.0, n_species=1)

    # FD detection of the same rank, at several step sizes
    steps = {}
    for rel in (1e-2, 1e-3, 1e-4):
        J, plan = log_fd_jacobian(
            lambda g: toy_endpoint(g, dur, q0), gam, rel, GAMMA_LO, GAMMA_HI, GAMMA_REF)
        steps[rel] = {"rank_detected": svd_basis(J, rel_tol=1e-8).rank,
                     "n_valid_columns": plan.n_valid,
                     "max_abs_err_vs_exact":
                         float(np.nanmax(np.abs(J - S)))}

    # terminal hold: append gamma_hold for duration L.  The exact prefix
    # sensitivity is propagated by the invertible linear-ish segment map, so the
    # ALGEBRAIC rank is preserved for every L; the EFFECTIVE rank decays once the
    # propagated columns fall below a declared numerical threshold.
    gamma_hold = float(gam[-1])
    hold_rows = []
    ABS_TOL = 1e-10          # declared application-scale threshold in scaled units
    for L in (0.0, 0.1, 1.0, 10.0, 100.0):
        durs_h = np.concatenate([dur, [L]]) if L > 0 else dur
        gam_h = np.concatenate([gam, [gamma_hold]]) if L > 0 else gam
        S_h = toy_endpoint_sensitivity(gam_h, durs_h, q0)
        # the prefix-block of S_h (columns 0..m-1) is the propagated prefix
        # sensitivity; its ALGEBRAIC rank is preserved by the invertible
        # finite-time segment maps while the values remain representable.
        prefix = S_h[:, :m]
        sv_prefix = np.linalg.svd(prefix, compute_uv=False)
        hold_rows.append({
            "hold_L": L, "hold_gamma": gamma_hold,
            "endpoint": toy_endpoint(gam_h, durs_h, q0).tolist(),
            "prefix_singular_values": sv_prefix.tolist(),
            "largest_prefix_singular_value": float(sv_prefix[0]),
            "prefix_rank_algebraic": int(np.linalg.matrix_rank(prefix)),
            # ratio-preserving threshold: keeps the rank ratio, not the scale
            "prefix_rank_effective_relative_1e-8": int(svB_rank(sv_prefix, 1e-8)),
            # absolute application threshold: this is what actually decays
            "prefix_rank_effective_absolute_1e-10": int(
                (sv_prefix > ABS_TOL).sum()),
            "distance_to_steady": float(np.linalg.norm(
                toy_endpoint(gam_h, durs_h, q0) - np.array(
                    [gamma_hold / (1 + gamma_hold), gamma_hold / (1 + gamma_hold)**2]))),
        })
    rep = [r for r in hold_rows if r["largest_prefix_singular_value"] > 0.0]
    eff_abs = [r["prefix_rank_effective_absolute_1e-10"] for r in hold_rows]
    return {"gamma": gam.tolist(), "durations": dur.tolist(), "q0": q0.tolist(),
            "absolute_effective_threshold": ABS_TOL,
            "exact_J": S.tolist(), "exact_rank": int(np.linalg.matrix_rank(S)),
            "fd_steps": steps,
            "terminal_hold": hold_rows,
            "passed": (int(np.linalg.matrix_rank(S)) == 2
                       and all(s["rank_detected"] == 2 for s in steps.values())
                       # algebraic rank preserved while representable
                       and all(r["prefix_rank_algebraic"] == 2 for r in rep)
                       # and the APPLICATION-SCALE rank actually decays
                       and eff_abs[0] == 2 and eff_abs[-1] < 2
                       and all(x >= y for x, y in zip(eff_abs, eff_abs[1:]))),
            "note": ("a finite-time smooth flow has an invertible state-transition "
                     "map locally, so the ALGEBRAIC prefix rank is preserved under a "
                     "long terminal hold (until the propagated sensitivities "
                     "underflow to exactly zero), while the EFFECTIVE rank at a "
                     "declared ABSOLUTE application threshold decays; a "
                     "ratio-preserving relative threshold alone would keep "
                     "reporting rank 2 at any scale, which is the distinction "
                     "this control exists to expose")}


def svB_rank(sv: np.ndarray, rel: float) -> int:
    sv = np.asarray(sv, float)
    return int((sv > rel * sv[0]).sum()) if sv.size and sv[0] > 0 else 0


def check_identity() -> dict:
    """C.3 + C.4: sum_j dE_m/d(log gamma_j) = d phi/d log gamma at constant-control
    base histories, both sides from the exact toy machinery."""
    rows = []
    for gamma_star in (30.0, 300.0, 3000.0, 30000.0):
        for T in (0.3, 1.0, 3.0):
            for m in (1, 3, 5):
                q0 = np.array([0.75, 0.03])
                gam = np.full(m, gamma_star)
                dur = np.full(m, T / m)
                lhs = toy_endpoint_sensitivity(gam, dur, q0).sum(axis=1)
                rhs = toy_constant_control_loggamma_derivative(gamma_star, T, q0)
                rows.append({"gamma_star": gamma_star, "T": T, "m": m,
                             "max_abs_err": float(np.max(np.abs(lhs - rhs))),
                             "scale": float(np.max(np.abs(rhs)))})
    scale = max(r["scale"] for r in rows)
    worst = max(r["max_abs_err"] for r in rows)
    return {"cases": rows, "worst_abs_err": worst, "scale": scale,
            "passed": worst < 1e-12 * max(scale, 1.0),
            "note": ("exact identity: a uniform relative shift of every segment "
                     "level IS a constant-control trajectory, so the column sum of "
                     "the switched Jacobian equals the constant-control derivative")}


def check_angles() -> dict:
    """C.5: genuine orthogonality and rank-deficient-reference angle tests."""
    out = {}
    # V = [e1, 0], J = [e2]: the correct sine to span(V) is ONE, not zero.
    V = np.array([[1.0, 0.0], [0.0, 0.0]])
    J = np.array([[0.0], [1.0]])
    bV, bJ = svd_basis(V), svd_basis(J)
    out["e2_vs_span_e1"] = {"sin": sin_to_span(bJ.vectors[:, 0], bV),
                            "expected": 1.0,
                            "passed": abs(sin_to_span(bJ.vectors[:, 0], bV) - 1.0) < 1e-12}
    out["e1_vs_span_e1"] = {"sin": sin_to_span(np.array([1.0, 0.0]), bV),
                            "expected": 0.0,
                            "passed": sin_to_span(np.array([1.0, 0.0]), bV) < 1e-12}
    # a tiny noisy reference column must NOT create a second retained direction
    Vn = np.array([[1.0, 1e-14], [0.0, 0.0]])
    bVn = svd_basis(Vn, rel_tol=1e-8)
    out["noisy_reference"] = {"retained_rank": bVn.rank,
                              "n_singular_values": bVn.singular_values.size,
                              "passed": bVn.rank == 1}
    # both matrices rank deficient: the sine must be well defined (0 or 1)
    Jz = np.zeros((2, 2))
    bJz = svd_basis(Jz)
    out["zero_jacobian"] = {"rank": bJz.rank, "vectors_is_none": bJz.vectors is None,
                            "sin_to_empty_basis": sin_to_span(np.array([1.0, 0.0]), bJz),
                            "expected_sin": 1.0,
                            "passed": (bJz.vectors is None
                                       and sin_to_span(np.array([1.0, 0.0]), bJz) == 1.0)}
    # genuinely aligned: J direction inside a 2-dim reference plane
    Q = np.linalg.qr(np.random.default_rng(1).normal(size=(5, 2)))[0]
    u = Q[:, 0] + 1e-15 * np.ones(5)
    bQ = svd_basis(Q)
    out["aligned_case"] = {"sin": sin_to_span(u, bQ), "expected_near": 0.0,
                           "passed": sin_to_span(u, bQ) < 1e-10}
    out["passed"] = all(v["passed"] for k, v in out.items() if k != "passed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase3e"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = {"timestamp_utc": utc_now(),
           "gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF,
           "checks": {}}
    for name, fn in (("linear3_rank3", check_linear3),
                     ("toy_rank2_and_terminal_hold", check_toy_rank2),
                     ("same_initial_state_identity", check_identity),
                     ("angle_and_rank_deficient_reference", check_angles)):
        res = fn()
        out["checks"][name] = res
        print(f"[phase3e] {name}: passed={res['passed']}")
    out["all_passed"] = all(c["passed"] for c in out["checks"].values())
    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase3e_results.json", out)
    manifest("phase3e", args.results_dir,
             {"gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF},
             cwd=Path(__file__).resolve().parents[1])
    print(f"[phase3e] all_passed={out['all_passed']} "
          f"in {out['wall_seconds']:.1f}s -> {args.results_dir}")


if __name__ == "__main__":
    main()
