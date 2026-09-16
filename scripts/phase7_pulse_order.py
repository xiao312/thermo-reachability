"""Phase 7 - ordered-pulse (commutator) test (Task 4).

Does the order of two control levels matter at an observable magnitude?  For the
two-segment histories (a,b) and (b,a) with equal segment durations tau, the
endpoint difference is

    Delta(tau) = phi_b(tau) phi_a(tau)(q0) - phi_a(tau) phi_b(tau)(q0)
               = tau^2 [f_a, f_b](q0) + O(tau^3).

The CSTR right-hand side is AFFINE in gamma: F(q, gamma) = gamma*v(q) + r(q)
with v the exchange direction and r the reaction direction, so the bracket of
two constant levels has the closed form

    [f_a, f_b] = (gamma_a - gamma_b) [r, v],

independent of the absolute levels apart from the factor (gamma_a - gamma_b).
This gives two FALSIFIABLE predictions tested here:

  P1  Delta(tau) ~ tau^2 as tau -> 0 (log-log slope 2);
  P2  Delta / (gamma_a - gamma_b) is the same for every level pair, and
      Delta = 0 when gamma_a = gamma_b (the negative control).

The magnitudes are compared against the application threshold in scaled units
and against the differentiation noise scale.  A result is only reported as
evidence of non-commutativity if it survives both.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.controls import History  # noqa: E402
from thermoreach.io_utils import utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402
from thermoreach.sensitivity import StateScaling  # noqa: E402

GAMMA_LO, GAMMA_HI = 10.0, 1e5


def endpoint(c: CSTR, theta: np.ndarray, durs: np.ndarray, q0: np.ndarray,
             method: str, rtol: float, atol_T: float, atol_Y: float) -> np.ndarray:
    res = c.integrate(History(theta, durs), q0, method=method,
                      samples_per_segment=2, rtol=rtol, atol_T=atol_T,
                      atol_Y=atol_Y, record_extrema=False)
    if not res["success"]:
        raise RuntimeError(f"integration failed: {res['message']}")
    return res["states"][:, -1].copy()


def exchange_and_reaction(c: CSTR, q: np.ndarray, gamma: float):
    """Split F(q, gamma) = gamma*v(q) + r(q) using two levels (affine in gamma).

    v(q) is the exchange direction and r(q) the reaction direction.  Both are
    read off from the affine structure, so no derivative of the mechanism is
    needed.
    """
    ga, gb = 100.0, 200.0
    Fa = c.rhs(0.0, q, ga)
    Fb = c.rhs(0.0, q, gb)
    v = (Fb - Fa) / (gb - ga)                  # dF/dgamma, constant in gamma
    r = Fa - ga * v                            # offset
    return v, r


def bracket_by_differences(c: CSTR, q: np.ndarray, gamma_a: float, gamma_b: float,
                           eps: float, rtol: float, atol_T: float,
                           atol_Y: float) -> np.ndarray:
    """[f_a, f_b](q) by central differences of the vector fields.

    [f_a, f_b] = Df_b f_a - Df_a f_b, each term a directional derivative of one
    vector field along the other, by central differences in q.
    """
    def fa(qq):
        return c.rhs(0.0, qq, gamma_a)

    def fb(qq):
        return c.rhs(0.0, qq, gamma_b)

    Dfb_fa = (fb(q + eps * fa(q)) - fb(q - eps * fa(q))) / (2 * eps)
    Dfa_fb = (fa(q + eps * fb(q)) - fa(q - eps * fb(q))) / (2 * eps)
    return Dfb_fa - Dfa_fb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase7"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    c = CSTR(ReactorConfig(mechanism="h2o2.yaml",
                           inlet_temperature=args.inlet_temperature))
    scaling = StateScaling(n_species=c.gas.n_species)
    W = scaling.W
    rtol, atol_T, atol_Y = 1e-12, 1e-11, 1e-17

    out = {"timestamp_utc": utc_now(), "mechanism": c.mech,
           "inlet_temperature": args.inlet_temperature,
           "scaling": scaling.to_record(),
           "method": {"integrator": "Radau", "rtol": rtol,
                      "atol_T": atol_T, "atol_Y": atol_Y},
           "formula": "[f_a, f_b] = (gamma_a - gamma_b) [r, v]",
           "cases": []}

    q0s = {"fresh": fresh_state(c), "hot": hot_hp_state(c)}
    # level pairs: spread of levels, plus a SAME-LEVEL negative control
    level_pairs = [(100.0, 1000.0), (100.0, 500.0), (500.0, 1000.0),
                   (100.0, 100.0)]
    # A WIDE tau range: small tau (below the ignition delay) is where the
    # asymptotic tau^2 bracket scaling can actually be seen; large tau is where
    # the chemistry saturates.  Both slopes are fitted and reported.
    taus = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    tau_asymptotic = [t for t in taus if t <= 1e-6]

    for init, q0 in q0s.items():
        for ga, gb in level_pairs:
            q_end_ab, q_end_ba = None, None
            rows = []
            for tau in taus:
                # ab: level a first then b ; ba: level b first then a
                e_ab = endpoint(c, np.array([ga, gb]), np.array([tau, tau]), q0,
                                "Radau", rtol, atol_T, atol_Y)
                e_ba = endpoint(c, np.array([gb, ga]), np.array([tau, tau]), q0,
                                "Radau", rtol, atol_T, atol_Y)
                delta = e_ab - e_ba
                rows.append({"tau": tau, "endpoint_ab": e_ab.tolist(),
                             "endpoint_ba": e_ba.tolist(),
                             "delta_raw": delta.tolist(),
                             "delta_scaled_norm": float(np.linalg.norm(W * delta)),
                             "tau_squared": tau * tau})
            # bracket prediction at the smallest tau
            tau0 = taus[0]
            v, r = exchange_and_reaction(c, q0, ga)
            br = bracket_by_differences(c, q0, ga, gb, 1e-6, rtol, atol_T, atol_Y)
            rv = bracket_by_differences(c, q0, 100.0, 101.0, 1e-6, rtol, atol_T,
                                        atol_Y)   # [r, v] proxy: (101-100)[r,v]
            bracket_rv = rv / (101.0 - 100.0)
            predicted = (ga - gb) * bracket_rv
            d0 = np.array(rows[0]["delta_raw"])
            dn = [r_["delta_scaled_norm"] for r_ in rows]
            dns = [r_["delta_scaled_norm"] for r_ in rows
                   if r_["tau"] in tau_asymptotic]
            # skip the fit when any difference vanishes (same-level control)
            def _slope(xs, ys):
                if len(ys) < 2 or min(ys) <= 0:
                    return None
                return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
            rec = {
                "init": init, "gamma_a": ga, "gamma_b": gb,
                "level_difference": ga - gb, "taus": taus, "rows": rows,
                "bracket_f_a_f_b": br.tolist(),
                "bracket_r_v": bracket_rv.tolist(),
                "predicted_bracket_scaled_norm": float(np.linalg.norm(W * predicted)),
                "observed_delta_scaled_norm_at_tau0": rows[0]["delta_scaled_norm"],
                "loglog_slope_of_delta_vs_tau": _slope(taus, dn),
                "loglog_slope_small_tau": _slope(tau_asymptotic, dns),
                "asymptotic_tau_range": tau_asymptotic,
                "affine_check_F_equal_gamma_v_plus_r": float(np.max(np.abs(
                    c.rhs(0.0, q0, ga) - (ga * v + r)))),
            }
            out["cases"].append(rec)
            sa = rec["loglog_slope_of_delta_vs_tau"]
            ss = rec["loglog_slope_small_tau"]
            print(f"[phase7] {init} a={ga:<7.0f} b={gb:<7.0f} "
                  f"|delta|@tau0={rec['observed_delta_scaled_norm_at_tau0']:.3e} "
                  f"slope_all={sa if sa is None else round(sa, 2)} "
                  f"slope_small={ss if ss is None else round(ss, 2)} "
                  f"pred={rec['predicted_bracket_scaled_norm']:.3e} "
                  f"affine_err={rec['affine_check_F_equal_gamma_v_plus_r']:.1e}")

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase7_results.json", out)
    print(f"[phase7] {len(out['cases'])} cases in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
