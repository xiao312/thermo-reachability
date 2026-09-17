"""Phase 9 - flagship validation of the third direction (review Task 2).

The single most consequential result of the previous revision is the hot-start,
gamma = 1e4 1/s, horizon = 1e-5 s, three-equal-segment endpoint map, whose scaled
Jacobian had total singular values (0.9484, 0.1189, 5.96e-5) and a leading
TRANSVERSE singular value 1.234e-4 above both the manifold-leakage floor and the
Radau/LSODA discrepancy.  This script validates it properly instead of repeating
the sweep:

  * the state Jacobian F_q is itself numerical, so it is varied independently
    (an eps ladder) of the ODE tolerance and of the replay;
  * A = W J and D = W V are PERSISTED per case as NPZ, with singular vectors,
    projectors and endpoints, so the claimed direction can be reconstructed;
  * delta_A (derivative error) and delta_P (reference-plane error) are measured
    SEPARATELY, and combined through ||Bhat - B|| <= ||Ahat - A|| + ||Phat - P||
    ||Ahat||;
  * the replay perturbs along the leading transverse RIGHT singular vector of
    B = (I - P) A - not along v2(A), which maximizes neither ||B v|| nor
    sigma1(B) - and compares the SIGNED projection and the full vector;
  * ||(I - P) A||_2 >= sigma3(A) for ANY rank <= 2 projector, so a stable
    sigma3(A) is reported as independent evidence.

Primary anchor: HP-equilibrium hot start of stoichiometric H2/air, Tin = 1200 K,
p = 101325 Pa, gamma = 1e4, horizon = 1e-5 s, m = 3 equal segments and equal base
controls.  Secondary (only after primary calibration): the same at horizon 1e-6 s.
Negative/resolution comparison: fresh gamma = 1e4, horizon = 1e-4 s, reduced
budget.  The two expensive gamma = 1000 / horizon = 1e-3 cases are deliberately
NOT revisited here: their reference-span condition numbers (~1.94e7 and 1.56e6)
make projector-based inference substantially harder than the flagship (~20), and
the review directs that they wait for reference-uncertainty machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    endpoint_jacobian_fsa, replay_signed, scaled_tangent_space, svd_basis,
    transverse_replay, transverse_spectrum,
)

GAMMA_LO, GAMMA_HI = 10.0, 1e5
GAMMA_REF = float(np.sqrt(GAMMA_LO * GAMMA_HI))
M = 3
RTOL = 1e-8
ATOL_T, ATOL_Y = 1e-7, 1e-13
EPS_LADDER = (1e-5, 1e-6, 1e-7)        # state-Jacobian perturbations
REPLAY_EPS = (0.1, 0.03, 0.01, 0.003, 0.001)


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


def mechanism_hash(mechanism: str) -> str:
    import cantera as ct
    for d in ct.get_data_directories():
        cand = Path(d) / mechanism
        if cand.is_file():
            return hashlib.sha256(cand.read_bytes()).hexdigest()[:16]
    return "unresolved"


def reference_span(c: CSTR, gamma: float, T: float, q0: np.ndarray, method: str,
                   eps: float, scaling: StateScaling) -> tuple[np.ndarray, np.ndarray]:
    """Anchored constant-control tangent span with the SAME estimator as the
    Jacobian: v_tau closed form, v_eta a one-segment tangent-linear derivative.
    Both differentiated from the ORIGINAL q0 and evaluated at the same endpoint."""
    ref = endpoint_jacobian_fsa(c, np.array([gamma]), np.array([T]), q0,
                                method=method, rtol=RTOL, atol_T=ATOL_T,
                                atol_Y=ATOL_Y, eps=eps)
    q_base = ref["endpoint"]
    v_tau = T * c.rhs(0.0, q_base, gamma)          # tau = t / T_ref, T_ref = T
    v_eta = ref["J"][:, 0]
    return np.vstack([v_tau, v_eta]).T, q_base


def flagship_case(c: CSTR, gamma: float, T: float, q0: np.ndarray, scaling: StateScaling,
                  label: str, methods=("Radau", "LSODA"),
                  eps_ladder=EPS_LADDER, budget_note: str = "") -> dict:
    m = M
    durs = np.full(m, T / m)
    theta = np.full(m, gamma)
    ok, reason = AdmissibleControls(GAMMA_LO, GAMMA_HI, float(durs.sum())).validate(
        History(theta, durs))
    if not ok:
        raise ValueError(f"{label}: inadmissible base: {reason}")

    W = scaling.W
    variants = {}          # (method, eps) -> {J, endpoint, A}
    failures = []
    for method in methods:
        for eps in eps_ladder:
            key = f"{method}_eps{eps:g}"
            try:
                r = endpoint_jacobian_fsa(c, theta, durs, q0, method=method,
                                          rtol=RTOL, atol_T=ATOL_T,
                                          atol_Y=ATOL_Y, eps=eps)
                variants[key] = {"J": r["J"], "endpoint": r["endpoint"],
                                 "A": W[:, None] * r["J"],
                                 "nfev": r["n_rhs_evaluations"]}
            except Exception as exc:                       # noqa: BLE001
                failures.append({"setting": key, "failure": repr(exc)})

    if not variants:
        return {"label": label, "status": "failed",
                "failures": failures, "reason": "every FSA setting failed"}

    # primary variant: Radau at the middle of the eps ladder
    primary_key = next((f"Radau_eps{e:g}" for e in eps_ladder
                        if f"Radau_eps{e:g}" in variants), next(iter(variants)))
    J = variants[primary_key]["J"]
    A = variants[primary_key]["A"]
    q_base3 = variants[primary_key]["endpoint"]

    # ---- independent reference-plane error: build D under every setting -------
    refs = {}
    for key, v in variants.items():
        method, _e = key.split("_eps")
        e = float(key.split("_eps")[1])
        try:
            V, qref = reference_span(c, gamma, T, q0, method, e, scaling)
            refs[key] = {"V": V, "D": W[:, None] * V, "q_ref": qref}
        except Exception as exc:                       # noqa: BLE001
            failures.append({"setting": f"ref:{key}", "failure": repr(exc)})
    ref_key = primary_key if primary_key in refs else next(iter(refs))
    V = refs[ref_key]["V"]
    D = refs[ref_key]["D"]
    q_base_ref = refs[ref_key]["q_ref"]

    # ---- delta_A and delta_P measured SEPARATELY ------------------------------
    keys = sorted(variants)
    dA = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dA = max(dA, float(np.linalg.norm(variants[keys[i]]["A"]
                                              - variants[keys[j]]["A"], ord=2)))
    # projector spread: ||Phat - P|| * ||Ahat|| over all reference spans.
    # The ONE rank-revealing helper builds every projector: unfiltered QR
    # silently fills a rank-deficient span with rounding noise, whereas a
    # rank-deficient reference span must be reported, not smoothed over.  At full
    # rank the two agree exactly (verified in tests against the committed NPZ).
    dP = 0.0
    rkeys = sorted(refs)
    ref_bases = {rk: svd_basis(refs[rk]["D"], rel_tol=1e-12) for rk in rkeys}
    for i in range(len(rkeys)):
        for j in range(i + 1, len(rkeys)):
            bi, bj = ref_bases[rkeys[i]], ref_bases[rkeys[j]]
            if bi.vectors is None or bj.vectors is None:
                continue
            Qi, Qj = bi.vectors, bj.vectors
            # align ranks: use the smaller
            k = min(Qi.shape[1], Qj.shape[1])
            dP = max(dP, float(np.linalg.norm(Qi[:, :k] @ Qi[:, :k].T
                                               - Qj[:, :k] @ Qj[:, :k].T)
                               * np.linalg.norm(A, ord=2)))
    ref_rank_deficient = [rk for rk, b in ref_bases.items() if b.rank < 2]
    n_variants = len(variants)
    cross_available = n_variants >= 2 and len({k.split("_")[0] for k in keys}) >= 2

    # ---- spectra, replay, classification -------------------------------------
    Cmat = constraint_jacobian(c, q_base3)
    spectra = transverse_spectrum(J, V, scaling, noise_scale=(dA if cross_available
                                                              else None),
                                  C_of_q=Cmat)
    # correct replay: along the leading transverse RIGHT singular vector of B
    v_perp = np.asarray(spectra.get("transverse_right_singular_vector") or [0.0] * m,
                        dtype=float)
    if np.any(np.abs(v_perp) > 0) and np.all(np.isfinite(v_perp)):
        rep = replay_signed(
            lambda th, du: c.integrate(History(th, du), q0, method="Radau",
                                       samples_per_segment=2, rtol=1e-11,
                                       atol_T=1e-10, atol_Y=1e-16,
                                       record_extrema=False)["states"][:, -1],
            theta, durs, v_perp, scaling, eps_list=REPLAY_EPS,
            gamma_bounds=(GAMMA_LO, GAMMA_HI))
        replay_report = transverse_replay(A, (svd_basis(D, rel_tol=1e-12).vectors
                                              if D.size else None), rep)
    else:
        rep = {"rows": [], "n_admissible": 0}
        replay_report = {"available": False,
                         "reason": "no transverse singular vector"}
    # separate total-sigma3 replay along v3(A)
    Vh = np.asarray(spectra.get("total_right_singular_vectors") or [], dtype=float)
    total3_replay = None
    if Vh.size and Vh.shape[0] >= 3:
        rep3 = replay_signed(
            lambda th, du: c.integrate(History(th, du), q0, method="Radau",
                                       samples_per_segment=2, rtol=1e-11,
                                       atol_T=1e-10, atol_Y=1e-16,
                                       record_extrema=False)["states"][:, -1],
            theta, durs, Vh[2], scaling, eps_list=REPLAY_EPS,
            gamma_bounds=(GAMMA_LO, GAMMA_HI))
        total3_replay = {"direction": "v3(A)",
                         "rows": [{"epsilon": r["epsilon"], "norm": r["norm"]}
                                  for r in rep3["rows"] if r.get("admissible")]}

    return {
        "label": label, "status": "completed", "gamma": gamma, "T": T, "m": m,
        "theta": theta.tolist(), "durations": durs.tolist(),
        "horizon": float(durs.sum()), "q0": q0.tolist(),
        "time_reference": {"tau": "t/T_ref", "T_ref": T},
        "primary_setting": primary_key, "reference_setting": ref_key,
        "settings_run": keys, "settings_failed": failures,
        "cross_check_available": cross_available,
        "cross_check_methods": sorted({k.split("_")[0] for k in keys}),
        "delta_A_norm": dA, "delta_P_norm_times_A": dP,
        "operator_error_bound_note": ("||Bhat-B|| <= ||Ahat-A|| + ||Phat-P||*||Ahat||; "
                                      "delta_A and delta_P are empirical spreads, "
                                      "not certified bounds"),
        "endpoint_3segment": q_base3.tolist(),
        "endpoint_reference_1segment": q_base_ref.tolist(),
        "endpoint_3_vs_reference_max_abs_diff": float(np.max(np.abs(q_base3
                                                                    - q_base_ref))),
        "classification": classify_ranks(J, scaling, noise_scale=(dA if cross_available
                                                                  else None)),
        "spectra": spectra,
        "transverse_replay": replay_report,
        "total_sigma3_replay": total3_replay,
        "replay_admissible_count": rep["n_admissible"],
        "tangent_space": scaled_tangent_space(Cmat, scaling).to_record(),
        "budget_note": budget_note,
        "variants": {k: {"nfev": v["nfev"],
                         "endpoint": v["endpoint"].tolist()} for k, v in variants.items()},
    }


def persist_npz(results_dir: Path, rec: dict, c: CSTR) -> Path:
    """Persist A = W J and D = W V plus endpoints and singular vectors, so the
    claimed direction can be reconstructed from the raw matrices."""
    label = rec["label"]
    npz = results_dir / f"{label}.npz"
    # recompute the primary matrices (they are not kept in the JSON record)
    q0 = np.asarray(rec["q0"])
    theta = np.asarray(rec["theta"])
    durs = np.asarray(rec["durations"])
    eps = float(rec["primary_setting"].split("_eps")[1])
    method = rec["primary_setting"].split("_")[0]
    r = endpoint_jacobian_fsa(c, theta, durs, q0, method=method, rtol=RTOL,
                              atol_T=ATOL_T, atol_Y=ATOL_Y, eps=eps)
    V, _q = reference_span(c, float(rec["gamma"]), float(rec["T"]), q0, method,
                           eps, StateScaling(n_species=c.gas.n_species))
    scaling = StateScaling(n_species=c.gas.n_species)
    A = scaling.W[:, None] * r["J"]
    D = scaling.W[:, None] * V
    U, sv, Vh = np.linalg.svd(A, full_matrices=False)
    np.savez_compressed(npz, A=A, D=D, J=r["J"], V=V,
                        endpoint=r["endpoint"], q0=q0, theta=theta, durations=durs,
                        W=scaling.W, U=U, singular_values=sv, Vh=Vh)
    return npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase9"))
    ap.add_argument("--inlet-temperature", type=float, default=1200.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = ReactorConfig(mechanism="h2o2.yaml",
                        inlet_temperature=args.inlet_temperature)
    c = CSTR(cfg)
    scaling = StateScaling(n_species=c.gas.n_species)
    mech_hash = mechanism_hash(cfg.mechanism)
    q_hot = hot_hp_state(c)
    q_fresh = fresh_state(c)

    # requested set is EXPLICIT; a deliberate exclusion must be visible as such
    requested = [("primary", "hot", 1e4, 1e-5, ""),
                 ("secondary", "hot", 1e4, 1e-6, ""),
                 ("negative", "fresh", 1e4, 1e-4, "reduced budget")]
    if args.quick:
        requested = requested[:1]

    out = {"timestamp_utc": utc_now(), "mechanism": c.mech,
           "mechanism_sha256_16": mech_hash,
           "config": cfg.to_record(), "scaling": scaling.to_record(),
           "application_threshold": APPLICATION_THRESHOLD_DEFAULT,
           "gamma_bounds": [GAMMA_LO, GAMMA_HI], "gamma_ref": GAMMA_REF, "m_prefix": M,
           "settings": {"rtol": RTOL, "atol_T": ATOL_T, "atol_Y": ATOL_Y,
                        "state_jacobian_eps_ladder": list(EPS_LADDER),
                        "replay_eps": list(REPLAY_EPS)},
           "deliberately_excluded": [
               {"case": "fresh gamma=100 horizon=1e-2",
                "reason": "did not complete within the compute budget in phase 8; "
                          "phase 6 established it as the memory-erased control"},
               {"case": "hot/fresh gamma=1000 horizon=1e-3",
                "reason": "reference-span condition ~1.94e7 and ~1.56e6 versus ~20 "
                          "for the flagship; projector-based inference deferred "
                          "until reference uncertainty is implemented"}],
           "index": [], "cases": []}

    q0s = {"hot": q_hot, "fresh": q_fresh}
    for role, init, gamma, T, note in requested:
        label = f"{role}_{init}_g{gamma:.0e}_T{T:.0e}"
        t1 = time.perf_counter()
        try:
            methods = ("Radau", "LSODA")
            eps_ladder = EPS_LADDER
            if role == "negative":
                # limited budget: one method, full eps ladder
                methods = ("Radau",)
            rec = flagship_case(c, gamma, T, q0s[init], scaling, label,
                                methods=methods, eps_ladder=eps_ladder,
                                budget_note=note or ("reduced budget" if role == "negative"
                                                     else ""))
            rec["role"] = role
            rec["init"] = init
            rec["wall_seconds"] = time.perf_counter() - t1
            try:
                npz = persist_npz(args.results_dir, rec, c)
                rec["matrices_npz"] = npz.name
            except Exception as exc:                    # noqa: BLE001
                rec["matrices_npz_error"] = repr(exc)
        except Exception as exc:                        # noqa: BLE001
            rec = {"label": label, "role": role, "init": init, "status": "failed",
                   "failure": repr(exc)}
        out["cases"].append(rec)
        out["index"].append({"label": label, "role": role, "init": init,
                             "gamma": gamma, "T": T,
                             "status": rec.get("status", "failed")})
        # atomic checkpoint after every case
        out["wall_seconds"] = time.perf_counter() - t0
        out["complete"] = False
        write_json(args.results_dir / "phase9_results.json", out)
        sp = rec.get("spectra", {})
        print(f"[phase9] {label}: {rec.get('status')} "
              f"({rec.get('wall_seconds', 0):.0f}s) "
              f"sv_tot={[f'{v:.2e}' for v in sp.get('singular_values_total_scaled', [])]} "
              f"sv_perp={[f'{v:.2e}' for v in (sp.get('singular_values_transverse_scaled') or [])]} "
              f"dA={rec.get('delta_A_norm')} dP={rec.get('delta_P_norm_times_A')} "
              f"xcheck={rec.get('cross_check_available')}")

    out["wall_seconds"] = time.perf_counter() - t0
    out["complete"] = True
    write_json(args.results_dir / "phase9_results.json", out)
    n_ok = sum(1 for r in out["index"] if r["status"] == "completed")
    print(f"[phase9] {n_ok}/{len(out['index'])} completed in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
