"""Phase 6 - separating exchange exposure from chemical relaxation (Task 3).

Phase 4 found the weak history-generated direction governed by the EXCHANGE TIME
gamma*T, but that was confounded: at fixed gamma, changing T changes both the
exchange exposure gamma*T and the amount of chemical relaxation.  This script
disentangles them with MATCHED pairs: cases sharing the same gamma*T but having
different gamma and T, so the exchange exposure is equal and only the chemistry
time scale differs.

  gamma*T = 1   : (100, 1e-2), (1000, 1e-3), (1e4, 1e-4), (1e5, 1e-5)
  gamma*T = 0.1 : (100, 1e-3), (1000, 1e-4)
  gamma*T = 10  : (1000, 1e-2), (1e4, 1e-3)

For each case we record the scaled singular values of the prefix Jacobian and -
the diagnostic that can actually support a claim - the TRANSVERSE spectrum and
the manifold-leakage floor.

Ignition metric is DECLARED up front: t_ign is the first time the temperature
rises 100 K above its initial value along the constant-control trajectory, or
None if it does not occur within the horizon.
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
    APPLICATION_THRESHOLD_DEFAULT, StateScaling, classify_ranks, endpoint_jacobian_fd,
    log_control_plan, scaled_tangent_space, transverse_spectrum,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = float(np.sqrt(GAMMA_LO * GAMMA_HI))
IGNITION_RISE_K = 100.0            # DECLARED metric
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


def constant_family_span(c: CSTR, gamma_star: float, T: float, q0: np.ndarray,
                         scaling: StateScaling, method: str = "Radau",
                         rtol: float = 1e-10, atol_T: float = 1e-9,
                         atol_Y: float = 1e-16) -> tuple[np.ndarray, np.ndarray]:
    res = c.integrate(History(np.array([gamma_star]), np.array([T])), q0,
                      method=method, samples_per_segment=400, rtol=rtol,
                      atol_T=atol_T, atol_Y=atol_Y, record_extrema=True)
    q_base = res["states"][:, -1].copy()
    v_tau = T * c.rhs(0.0, q_base, gamma_star)          # tau = t / T_ref, T_ref = T
    eta = np.log(gamma_star / GAMMA_REF)
    room = min(eta - np.log(GAMMA_LO / GAMMA_REF),
               np.log(GAMMA_HI / GAMMA_REF) - eta)
    h = 0.5e-3 * room
    a = AdmissibleControls(GAMMA_LO, GAMMA_HI, T)
    gp, gm = gamma_star * np.exp(h), gamma_star * np.exp(-h)
    if not (GAMMA_LO < gp < GAMMA_HI and GAMMA_LO < gm < GAMMA_HI):
        raise ValueError("reference gamma stencil not strictly interior")
    rp = c.integrate(History(np.array([gp]), np.array([T])), q0, method=method,
                     samples_per_segment=2, rtol=rtol, atol_T=atol_T,
                     atol_Y=atol_Y, record_extrema=False)
    rm = c.integrate(History(np.array([gm]), np.array([T])), q0, method=method,
                     samples_per_segment=2, rtol=rtol, atol_T=atol_T,
                     atol_Y=atol_Y, record_extrema=False)
    v_eta = (rp["states"][:, -1] - rm["states"][:, -1]) / (2 * h)
    # ignition: first time T rises IGNITION_RISE_K above its initial value
    tt = res["times"]
    TT = res["states"][0, :]
    idx = np.where(TT >= q0[0] + IGNITION_RISE_K)[0]
    t_ign = float(tt[idx[0]]) if idx.size else None
    return np.vstack([v_tau, v_eta]).T, q_base, t_ign, float(TT.max())


def one_case(c: CSTR, gamma: float, T: float, q0: np.ndarray, scaling: StateScaling,
             init: str, hold_levels: tuple[float, ...]) -> dict:
    m = M
    durs = np.full(m, T / m)
    theta = np.full(m, gamma)
    adm = AdmissibleControls(GAMMA_LO, GAMMA_HI, float(durs.sum()))
    ok, reason = adm.validate(History(theta, durs))
    if not ok:
        raise ValueError(f"({gamma},{T}): inadmissible base: {reason}")
    rtol, atol_T, atol_Y = 1e-10, 1e-9, 1e-16
    fd = endpoint_jacobian_fd(c, theta, durs, q0, 1e-3, GAMMA_LO, GAMMA_HI,
                              GAMMA_REF, method="Radau", rtol=rtol,
                              atol_T=atol_T, atol_Y=atol_Y)
    fd_alt = endpoint_jacobian_fd(c, theta, durs, q0, 1e-4, GAMMA_LO, GAMMA_HI,
                                  GAMMA_REF, method="Radau", rtol=rtol,
                                  atol_T=atol_T, atol_Y=atol_Y)
    V, q_base, t_ign, Tmax = constant_family_span(c, gamma, T, q0, scaling,
                                                  rtol=rtol, atol_T=atol_T,
                                                  atol_Y=atol_Y)
    Cmat = constraint_jacobian(c, q_base)
    W = scaling.W
    J, J_alt = fd["J"], fd_alt["J"]
    noise = float(np.linalg.norm((W[:, None] * J) - (W[:, None] * J_alt), ord=2))
    rec = {
        "init": init, "gamma": gamma, "T": T, "gamma_times_T": gamma * T,
        "theta": theta.tolist(), "durations": durs.tolist(),
        "horizon": float(durs.sum()), "q0": q0.tolist(),
        "ignition": {"metric": "first time T exceeds T0 + 100 K",
                     "rise_K": IGNITION_RISE_K,
                     "t_ign": t_ign, "max_temperature_K": Tmax,
                     "ignited": t_ign is not None},
        "classification": classify_ranks(J, scaling, noise_scale=noise),
        "spectra": transverse_spectrum(J, V, scaling, noise_scale=noise,
                                       C_of_q=Cmat),
        "reference_q_base": q_base.tolist(),
        "reference_span_singular_values": np.linalg.svd(
            W[:, None] * V, compute_uv=False).tolist(),
        "tangent_space": scaled_tangent_space(Cmat, scaling).to_record(),
    }
    # hold study at the MATCHED exchange exposure: L measured in residence times
    holds = []
    for g_hold in hold_levels:
        for L_over_tau in (0.0, 0.5, 1.0, 3.0):
            L = L_over_tau / g_hold if g_hold > 0 else 0.0
            if L <= 0:
                continue
            full_durs = np.concatenate([durs, np.array([L])])
            full_theta = np.concatenate([theta, np.array([g_hold])])
            fdp = endpoint_jacobian_fd(c, full_theta, full_durs, q0, 1e-3,
                                       GAMMA_LO, GAMMA_HI, GAMMA_REF,
                                       method="Radau", rtol=rtol,
                                       atol_T=atol_T, atol_Y=atol_Y)
            q_end = fdp["endpoint"]
            # long relaxation to the steady state at g_hold (declared reference)
            q_ss = c.integrate(History(np.array([g_hold]), np.array([50.0 / g_hold])),
                               q0, method="Radau", samples_per_segment=2,
                               rtol=rtol, atol_T=atol_T, atol_Y=atol_Y,
                               record_extrema=False)["states"][:, -1]
            C2 = constraint_jacobian(c, q_end)
            # anchored at TOTAL time T + L only when the hold level equals the
            # prefix level (then the history is still the constant family)
            if abs(g_hold - gamma) < 1e-12:
                Vh, _, _, _ = constant_family_span(c, g_hold, T + L, q0, scaling,
                                                   rtol=rtol, atol_T=atol_T,
                                                   atol_Y=atol_Y)
                anchored = True
            else:
                Vh = V
                anchored = False
            holds.append({
                "gamma_hold": g_hold, "L": L, "total_time": T + L,
                "exchange_gamma_hold_times_L": g_hold * L,
                "reference_anchored_at_T_plus_L": anchored,
                "endpoint": q_end.tolist(),
                "distance_to_steady_scaled": float(np.linalg.norm(
                    (W * (q_end - q_ss)))),
                "prefix_block_spectra": transverse_spectrum(
                    fdp["J"][:, :m], Vh, scaling, noise_scale=noise,
                    C_of_q=C2),
            })
    rec["hold_study"] = holds
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase6"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    c = CSTR(ReactorConfig(mechanism="h2o2.yaml",
                           inlet_temperature=args.inlet_temperature))
    scaling = StateScaling(n_species=c.gas.n_species)

    # MATCHED PAIRS: for each exchange exposure gamma*T, several (gamma, T)
    # pairs share the exposure while the chemical relaxation time differs.  The
    # pairs are keyed on (gamma, T) - NOT on gamma*T, which is the quantity being
    # tested and would silently delete exactly the cases the design needs.
    cases = []
    for gt in (0.1, 1.0, 10.0):
        for g in (100.0, 1000.0, 1e4):
            T = gt / g
            if 1e-6 <= T <= 1.0:
                cases.append((g, T))
    cases = sorted(set(cases), key=lambda x: (x[0] * x[1], x[0]))

    out = {"timestamp_utc": utc_now(), "mechanism": c.mech,
           "inlet_temperature": args.inlet_temperature,
           "application_threshold": APPLICATION_THRESHOLD_DEFAULT,
           "gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF,
           "m_prefix": M,
           "ignition_metric": {"definition": "first time T exceeds T0 + 100 K",
                               "rise_K": IGNITION_RISE_K},
           "hold_levels": [100.0, 1000.0],
           "hold_L_in_residence_times": [0.5, 1.0, 3.0],
           "cases": []}
    q0s = {"fresh": fresh_state(c), "hot": hot_hp_state(c)}

    for init in ("fresh", "hot"):
        for g, T in cases:
            rec = one_case(c, g, T, q0s[init], scaling, init, (100.0, 1000.0))
            out["cases"].append(rec)
            sp = rec["spectra"]
            print(f"[phase6] {init} g*T={g*T:<6.1f} g={g:<8.0f} T={T:.1e} "
                  f"sv_tot={[f'{v:.1e}' for v in sp['singular_values_total_scaled']]} "
                  f"sv_perp={[f'{v:.1e}' for v in sp['singular_values_transverse_scaled']]} "
                  f"noise={rec['classification']['noise_scale_refinement_discrepancy']:.1e} "
                  f"t_ign={rec['ignition']['t_ign']}")

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase6_results.json", out)
    print(f"[phase6] {len(out['cases'])} cases in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
