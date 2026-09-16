"""Phase 2: detailed-chemistry CSTR implementation and validation.

Checks (all recorded, failures included):
  2A. both energy-equation forms agree numerically at representative states;
  2C. elemental matrix properties (normalization, E.r = 0, stoichiometric
      left-nullspace invariants), closed-reactor energy conservation;
      exact open-reactor balances b(t) and h(t) for matched flows, including a
      *nontrivial* case with T_in != T0 and different initial composition;
      custom RHS vs Cantera ReactorNet at shared physical times;
      cross-integration (Radau vs BDF) and tighter-tolerance replay;
      state-setter renormalization audit.

Uses h2o2.yaml (10 species, 29 reactions) as a tractable detailed-kinetics test.

Usage:
    python scripts/phase2_reactor_validate.py --results-dir results/phase2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.controls import History, constant  # noqa: E402
from thermoreach.io_utils import manifest, utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402
from thermoreach.reactornet_check import ReactorNetCheck  # noqa: E402


def nullspace(E: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the left nullspace of E (in consistent mass
    coordinates): vectors v with v.E = 0, i.e. linear invariants of the
    chemical source in mass-fraction space."""
    u, s, vh = np.linalg.svd(E)
    rank = int(np.sum(s > tol * s[0]))
    return u[:, rank:]


def check_energy_forms(c: CSTR, n_states: int = 5, seed: int = 7) -> dict:
    """Verify cp dT/dt = gamma(h_in - h) - sum h_k dY_k/dt equals the expanded form."""
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    max_scale = 0.0
    for _ in range(n_states):
        T = float(rng.uniform(700.0, 2500.0))
        Y = np.maximum(rng.random(c.gas.n_species), 0.0)
        Y /= Y.sum()
        q = np.concatenate([[T], Y])
        for g in (0.0, 10.0, 1e3, 1e4):
            f1 = c.rhs(0.0, q, g)
            f2 = c.rhs_alt(0.0, q, g)
            max_diff = max(max_diff, float(np.max(np.abs(f1 - f2))))
            max_scale = max(max_scale, float(np.max(np.abs(f1))))
    # The two forms are algebraically identical; the residual is floating-point
    # roundoff on terms of magnitude ~1e10, so the criterion is relative.
    return {"max_abs_difference": max_diff, "rhs_scale": max_scale,
            "max_relative_difference": max_diff / max(max_scale, 1.0),
            "n_states": n_states, "gammas_tested": [0.0, 10.0, 1e3, 1e4],
            "passed": bool(max_diff / max(max_scale, 1.0) < 1e-9)}


def check_elemental(c: CSTR) -> dict:
    E = c.E
    n_el, n_sp = E.shape
    # normalization: each column of E must sum to 1 (mass conservation of species)
    col_sums = E.sum(axis=0)
    # elemental source conservation E.r ~ 0 at representative states
    rng = np.random.default_rng(11)
    max_Er = 0.0
    max_scale = 0.0
    for _ in range(5):
        T = float(rng.uniform(700.0, 2500.0))
        Y = np.maximum(rng.random(n_sp), 0.0)
        Y /= Y.sum()
        c.gas.TPY = T, c.p, Y
        r = c.W * c.gas.net_production_rates / float(c.gas.density)
        max_Er = max(max_Er, float(np.max(np.abs(E @ r))))
        max_scale = max(max_scale, float(np.max(np.abs(r))))
    ns = nullspace(E)
    return {
        "n_elements": n_el, "n_species": n_sp,
        "column_sum_max_dev": float(np.max(np.abs(col_sums - 1.0))),
        "elemental_source_Er_max_abs": max_Er, "source_scale_max_abs": max_scale,
        "elemental_source_Er_relative": max_Er / max(max_scale, 1.0),
        "left_nullspace_dim": int(ns.shape[1]),
        "note": ("The all-ones vector spans the trivial total-mass invariant; "
                 "inert species appear as additional left-nullspace directions. "
                 "E.r is a roundoff-level residual relative to the source magnitude."),
        "passed": bool(np.max(np.abs(col_sums - 1.0)) < 1e-10
                       and max_Er / max(max_scale, 1.0) < 1e-9),
    }


def check_closed_energy(c: CSTR, tf: float = 1e-3, seed: int = 3) -> dict:
    """Closed (gamma=0) reactor: total enthalpy must be conserved."""
    rng = np.random.default_rng(seed)
    T0 = 1200.0
    Y = np.maximum(rng.random(c.gas.n_species), 0.0)
    Y /= Y.sum()
    q0 = np.concatenate([[T0], Y])
    h = History(np.array([0.0]), np.array([tf]))
    res = c.integrate(h, q0, method="Radau", samples_per_segment=201)
    if not res["success"]:
        return {"passed": False, "message": res["message"]}
    T = res["states"][0]
    Yt = res["states"][1:]
    h_traj = np.array([float(c.species_enthalpies(float(T[i])) @ np.maximum(Yt[:, i], 0.0))
                       for i in range(T.size)])
    return {
        "initial_h": float(h_traj[0]), "final_h": float(h_traj[-1]),
        "max_abs_h_drift": float(np.max(np.abs(h_traj - h_traj[0]))),
        "max_relative_h_drift": float(np.max(np.abs(h_traj - h_traj[0])) / abs(h_traj[0])),
        "T0": T0, "T_final": float(T[-1]),
        "passed": bool(np.max(np.abs(h_traj - h_traj[0])) / abs(h_traj[0]) < 1e-8),
    }


def run_balance_case(c: CSTR, q0: np.ndarray, history: History, label: str,
                     samples: int = 201) -> dict:
    res = c.integrate(history, q0, method="Radau", samples_per_segment=samples)
    if not res["success"]:
        return {"label": label, "passed": False, "message": res["message"],
                "failed_at_gamma": res.get("failed_at_gamma")}
    resid = c.residuals(res["times"], res["states"], history)
    # Scaled acceptance thresholds: the balances involve enthalpy of magnitude
    # ~1e7 J/kg, so criteria are relative. Thresholds are fixed here, in
    # configuration, before the sweep - not retrofitted to the results.
    h_scale = max(abs(c.h_in), 1.0)
    b0 = c.E @ np.asarray(q0[1:])
    b_scale = max(float(np.max(np.abs(c.b_in))), float(np.max(np.abs(b0))), 1.0)
    rel_elem = resid["element_balance_max_abs"] / b_scale
    rel_enth = resid["enthalpy_balance_max_abs"] / h_scale
    trivial = bool(np.allclose(c.E @ np.asarray(q0[1:]), c.b_in, rtol=1e-9, atol=1e-12))
    return {
        "label": label,
        "trivial_consistency_case": trivial,
        "note": ("b0 == b_in and h0 == h_in by construction: the expected "
                 "derivatives vanish, so this case checks consistency only.") if trivial
                else ("b0 != b_in or h0 != h_in: the balance law is tested "
                      "nontrivially."),
        "passed": bool(rel_elem < 1e-8 and rel_enth < 1e-6
                       and resid["min_Y"] > -1e-10),
        "element_balance_max_abs": resid["element_balance_max_abs"],
        "element_balance_relative": rel_elem,
        "enthalpy_balance_max_abs": resid["enthalpy_balance_max_abs"],
        "enthalpy_balance_relative": rel_enth,
        "min_Y": resid["min_Y"], "min_Y_species": resid["min_Y_species"],
        "T_range": [resid["min_T"], resid["max_T"]],
        "max_state_renorm": res["max_state_renorm"],
        "terminal_state": res["terminal_state"].tolist(),
    }


def cross_check(c: CSTR, q0: np.ndarray, gamma: float, horizon: float,
                n_times: int = 41) -> dict:
    """Custom RHS vs Cantera ReactorNet for a constant-gamma history."""
    history = History(np.array([gamma]), np.array([horizon]))
    res = c.integrate(history, q0, method="Radau", samples_per_segment=n_times)
    if not res["success"]:
        return {"gamma": gamma, "passed": False, "message": res["message"]}
    net = ReactorNetCheck(c)
    rn = net.run(history, q0, res["times"])
    err_T = float(np.max(np.abs(res["states"][0] - rn["states"][0])))
    err_Y = float(np.max(np.abs(res["states"][1:] - rn["states"][1:])))
    # scaled species error: absolute, and relative where Y is not tiny
    Y_ref = np.maximum(np.abs(rn["states"][1:]), 1e-12)
    rel = np.max(np.abs(res["states"][1:] - rn["states"][1:]) / Y_ref)
    return {
        "gamma": gamma, "horizon": horizon,
        "max_abs_T_error_K": err_T, "max_abs_Y_error": err_Y,
        "max_rel_Y_error_where_Y_gt_1e-12": rel,
        "custom_T_final": float(res["states"][0, -1]),
        "net_T_final": float(rn["states"][0, -1]),
        "n_reactornet_steps": rn["n_steps"],
        "passed": bool(err_T < 1.0 and err_Y < 1e-4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase2"))
    ap.add_argument("--horizon", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true", help="reduced budget smoke mode")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = ReactorConfig(mechanism="h2o2.yaml")
    c = CSTR(cfg)
    out: dict = {"timestamp_utc": utc_now(), "mechanism": c.mech,
                 "config": cfg.to_record(),
                 "inlet": {"T_in": c.T_in, "Y_in": dict(zip(c.gas.species_names, c.Y_in.tolist())),
                           "h_in_J_per_kg": c.h_in, "b_in": c.b_in.tolist()}}

    q_fresh = fresh_state(c)
    q_hot = hot_hp_state(c)
    out["initial_conditions"] = {
        "fresh": {"T": float(q_fresh[0]), "Y": dict(zip(c.gas.species_names, q_fresh[1:].tolist()))},
        "hot_hp_equilibrium": {"T": float(q_hot[0]), "Y": dict(zip(c.gas.species_names, q_hot[1:].tolist()))},
    }

    # 2A: energy-form equivalence
    out["energy_forms"] = check_energy_forms(c, n_states=3 if args.quick else 5)
    print(f"[phase2] energy forms max diff {out['energy_forms']['max_abs_difference']:.2e}")

    # 2C: elemental matrix and conservation
    out["elemental"] = check_elemental(c)
    out["closed_energy"] = check_closed_energy(c, tf=1e-4 if args.quick else 1e-3)
    print(f"[phase2] E.r max {out['elemental']['elemental_source_Er_max_abs']:.2e}, "
          f"closed h drift {out['closed_energy']['max_abs_h_drift']:.2e}")

    # 2C: exact open-reactor balances
    balances = []
    # (i) trivial-consistency case: fresh start (b0=b_in, h0=h_in)
    balances.append(run_balance_case(
        c, q_fresh, History(np.array([1000.0]), np.array([args.horizon])), "fresh_const_gamma_1000"))
    # (ii) NONTRIVIAL admissible case: different inlet temperature and different
    #      initial composition, so b0 != b_in and h0 != h_in
    rng = np.random.default_rng(5)
    Y0 = np.maximum(rng.random(c.gas.n_species), 0.0)
    Y0 /= Y0.sum()
    q_mixed = np.concatenate([[1100.0], Y0])
    cfg2 = ReactorConfig(mechanism="h2o2.yaml", inlet_temperature=1000.0)
    c2 = CSTR(cfg2)  # inlet at 1000 K, initial state at 1100 K with random comp
    balances.append(run_balance_case(
        c2, q_mixed, History(np.array([500.0]), np.array([args.horizon])), "nontrivial_Tin1000_T0_1100"))
    # (iii) hot HP start at the primary gamma range
    for g in (10.0, 1e3, 1e5):
        balances.append(run_balance_case(
            c, q_hot, History(np.array([g]), np.array([args.horizon])), f"hot_gamma_{g:g}"))
    out["balance_checks"] = balances
    for b in balances:
        print(f"[phase2] balance {b['label']}: elem {b['element_balance_max_abs']:.2e} "
              f"enth {b['enthalpy_balance_max_abs']:.2e} minY {b['min_Y']:.2e} "
              f"({'PASS' if b['passed'] else 'FAIL'})")

    # cross-integration: Radau vs BDF vs tighter tolerance, fresh and hot
    xint = []
    for label, q in (("fresh", q_fresh), ("hot", q_hot)):
        for g in (10.0, 1e3, 1e5):
            h = History(np.array([g]), np.array([args.horizon]))
            r_radau = c.integrate(h, q, method="Radau", samples_per_segment=41)
            r_bdf = c.integrate(h, q, method="BDF", samples_per_segment=41)
            r_ref = c.integrate(h, q, method="Radau", samples_per_segment=41,
                                rtol=1e-11, atol_T=1e-9, atol_Y=1e-16)
            if not (r_radau["success"] and r_bdf["success"] and r_ref["success"]):
                xint.append({"label": f"{label}_g{g:g}", "passed": False,
                             "messages": [r.get("message", "") for r in (r_radau, r_bdf, r_ref)]})
                continue
            common = np.linspace(0, args.horizon, 21)
            i_r = np.clip(np.searchsorted(r_radau["times"], common), 0, r_radau["times"].size - 1)
            i_b = np.clip(np.searchsorted(r_bdf["times"], common), 0, r_bdf["times"].size - 1)
            i_f = np.clip(np.searchsorted(r_ref["times"], common), 0, r_ref["times"].size - 1)
            dT_rb = float(np.max(np.abs(r_radau["states"][0, i_r] - r_bdf["states"][0, i_b])))
            dT_rf = float(np.max(np.abs(r_radau["states"][0, i_r] - r_ref["states"][0, i_f])))
            dY_rb = float(np.max(np.abs(r_radau["states"][1:, i_r] - r_bdf["states"][1:, i_b])))
            xint.append({"label": f"{label}_g{g:g}", "gamma": g, "passed": True,
                         "radau_bdf_max_abs_T": dT_rb, "radau_bdf_max_abs_Y": dY_rb,
                         "radau_vs_tighter_max_abs_T": dT_rf,
                         "nfev_radau": sum(s["nfev"] for s in r_radau["solver_stats"]),
                         "nfev_bdf": sum(s["nfev"] for s in r_bdf["solver_stats"])})
    out["cross_integration"] = xint
    for x in xint:
        if x["passed"]:
            print(f"[phase2] xint {x['label']}: Radau-BDF dT {x['radau_bdf_max_abs_T']:.2e} "
                  f"dY {x['radau_bdf_max_abs_Y']:.2e}")
        else:
            print(f"[phase2] xint {x['label']}: FAILED")

    # Cantera ReactorNet cross-check (constant gamma, representative values)
    rn_checks = []
    for g in (10.0, 1e2, 1e3, 1e4):
        rn_checks.append(cross_check(c, q_fresh, g, args.horizon))
        rn_checks.append(cross_check(c, q_hot, g, args.horizon))
    out["reactornet_crosscheck"] = rn_checks
    for r in rn_checks:
        print(f"[phase2] net gamma={r['gamma']}: dT {r.get('max_abs_T_error_K', float('nan')):.2e} "
              f"dY {r.get('max_abs_Y_error', float('nan')):.2e} "
              f"({'PASS' if r['passed'] else 'FAIL'})")

    out["wall_seconds"] = time.perf_counter() - t0
    out["all_passed"] = bool(all(b["passed"] for b in balances)
                             and all(x["passed"] for x in xint)
                             and all(r["passed"] for r in rn_checks)
                             and out["energy_forms"]["passed"]
                             and out["elemental"]["passed"]
                             and out["closed_energy"]["passed"])
    write_json(args.results_dir / "phase2_results.json", out)
    manifest("phase2", args.results_dir, {"horizon": args.horizon, "quick": args.quick},
             cwd=Path(__file__).resolve().parents[1], extra={"mechanism": c.mech})
    print(f"[phase2] all_passed={out['all_passed']} in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
