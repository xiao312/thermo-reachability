"""Phase 3: empirical detailed-reactor envelopes and active counterexample search.

At one frozen operating condition (h2o2.yaml, 1 atm, T_in = 900 K, phi = 1
H2/O2/N2), three separately labelled reference sets are built per
initial-condition study (fresh feed, and HP-equilibrium hot start):

  A  steady solutions across constant residence time (time-marched to steady
     state from cold and hot initializations)
  B  finite-time trajectories under constant residence time (horizon-limited)
  C  finite-time trajectories under admissible switching histories
     (randomized + structured, with whole-history held-out set)

Questions:
  1. Do transients extend beyond the steady library (A)?
  2. Do switching histories reach states not well represented even by
     constant-control transients (B) from the same initial condition and horizon?
  3. How robust is any apparent excursion against solver tolerance, temporal
     extrema resolution, reference-library resolution and metric sensitivity?

Plus a conservation-based LP outer comparison, and a bounded optimization search
for histories maximizing a selected radical peak.

Usage:
    python scripts/phase3_envelopes.py --results-dir results/phase3 --pilot
    python scripts/phase3_envelopes.py --results-dir results/phase3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls, stable_child_seed  # noqa: E402
from thermoreach.controls import (History, bang_bang, constant, pulse,  # noqa: E402
                                  ramp, random_history)
from thermoreach.io_utils import manifest, utc_now, write_json  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402
from thermoreach.envelopes import (StateMetric, lp_species_bound,  # noqa: E402
                                   lp_temperature_bound, run_trajectory,
                                   steady_accepted, steady_family,
                                   trajectory_record, trace_steady)

GAMMA_LO, GAMMA_HI = 10.0, 1e5       # s^-1, i.e. residence time in [1e-5, 1e-1] s
HORIZON = 0.1                        # s, primary finite horizon


def structured_histories(gamma_lo: float, gamma_hi: float, n_seg: int, horizon: float):
    out = []
    # The zero-control history was removed (audit A07): gamma=0 is outside the
    # declared admissible class [gamma_lo, gamma_hi]. A quench to zero exchange
    # is not permitted under this study's control bound.
    out.append(("const_lo_seg", History(np.full(n_seg, gamma_lo),
                                        np.full(n_seg, horizon / n_seg))))
    out.append(("bangbang_hi_start", bang_bang(gamma_lo, gamma_hi, n_seg, horizon, start_high=True)))
    out.append(("bangbang_lo_start", bang_bang(gamma_lo, gamma_hi, n_seg, horizon, start_high=False)))
    out.append(("pulse_first", pulse(gamma_hi, n_seg, horizon, pulse_segment=0, base=gamma_lo)))
    out.append(("pulse_mid", pulse(gamma_hi, n_seg, horizon, pulse_segment=n_seg // 2, base=gamma_lo)))
    out.append(("pulse_last", pulse(gamma_hi, n_seg, horizon, pulse_segment=n_seg - 1, base=gamma_lo)))
    out.append(("ramp_up", ramp(gamma_lo, gamma_hi, n_seg, horizon, increasing=True)))
    out.append(("ramp_down", ramp(gamma_lo, gamma_hi, n_seg, horizon, increasing=False)))
    out.append(("slow_to_fast", History(np.geomspace(gamma_lo, gamma_hi, n_seg),
                                        np.full(n_seg, horizon / n_seg))))
    out.append(("fast_to_slow", History(np.geomspace(gamma_hi, gamma_lo, n_seg),
                                        np.full(n_seg, horizon / n_seg))))
    # alternating dwell: two long slow segments then fast
    vals = np.full(n_seg, gamma_lo)
    vals[n_seg // 2:] = gamma_hi
    out.append(("slow_then_fast_block", History(vals, np.full(n_seg, horizon / n_seg))))
    vals2 = np.full(n_seg, gamma_hi)
    vals2[n_seg // 2:] = gamma_lo
    out.append(("fast_then_slow_block", History(vals2, np.full(n_seg, horizon / n_seg))))
    return out


def collect(c: CSTR, histories, q0: np.ndarray, init_label: str, spp: int = 41):
    trajs, states, recs = [], [], []
    for name, h in histories:
        tr = run_trajectory(c, h, q0, init_label, name, samples_per_segment=spp)
        trajs.append(tr)
        recs.append(trajectory_record(tr, c.gas.species_names))
        if tr.success and tr.states.size:
            # analysis states: trajectory grid union with the resolved extrema,
            # so narrow peaks are represented in the geometry.
            S = tr.states
            if tr.extrema_states is not None and tr.extrema_states.size:
                S = np.concatenate([S, tr.extrema_states], axis=1)
            states.append(S)
    if states:
        S = np.concatenate(states, axis=1)
    else:
        S = np.zeros((q0.size, 0))
    return trajs, S, recs


def study(c: CSTR, init_label: str, q0: np.ndarray, gammas: np.ndarray,
          n_random: int, n_struct: int, n_heldout: int, seed: int,
          horizon: float = HORIZON, spp: int = 41) -> dict:
    n_seg = 8
    rng = np.random.default_rng(seed)
    rec: dict = {"init": init_label, "initial_state": {"T": float(q0[0])}, "horizon": horizon}

    # --- A: steady family ----------------------------------------------------
    # The steady family is a property of the reactor (feed + gamma), traced
    # always from the cold (fresh) and hot (HP-equilibrium) initializations of
    # THIS configuration - not from the study's own initial state.
    t_a = time.perf_counter()
    fam = steady_family(c, gammas, fresh_state(c), hot_hp_state(c))
    rec["steady_family"] = fam
    rec["wall_steady"] = time.perf_counter() - t_a
    # Reference library A uses only residual-accepted steady points; unresolved
    # points are retained separately and excluded from the geometry (audit A10).
    sel = steady_accepted(fam)
    rec["steady_selection"] = {"n_accepted": sel["n_accepted"],
                              "n_unresolved": sel["n_unresolved"],
                              "unresolved_gammas": [p["gamma"] for p in sel["unresolved"]]}
    A_states = np.array([[p["T"]] + p["Y"] for p in sel["accepted"]]).T \
        if sel["accepted"] else np.zeros((q0.size, 0))
    rec["steady_residual_max"] = float(np.nanmax(
        [p["steady_residual"] for p in fam["cold"] + fam["hot"]]))
    rec["steady_nonconverged"] = [p["gamma"] for p in fam["cold"] + fam["hot"]
                                  if not p["converged"]]
    print(f"[phase3:{init_label}] steady family: {len(gammas)} gammas x2 inits, "
          f"residual_max={rec['steady_residual_max']:.2e}, "
          f"wall={rec['wall_steady']:.1f}s")

    # --- B: constant-control transients --------------------------------------
    t_b = time.perf_counter()
    b_hist = [(f"const_g{g:.4g}", History(np.array([float(g)]), np.array([horizon])))
              for g in gammas]
    b_trajs, B_states, b_recs = collect(c, b_hist, q0, init_label, spp=spp)
    rec["constant_transients"] = b_recs
    rec["wall_B"] = time.perf_counter() - t_b
    print(f"[phase3:{init_label}] constant transients: {len(b_hist)}, "
          f"failures={sum(1 for t in b_trajs if not t.success)}, wall={rec['wall_B']:.1f}s")

    # --- C: switching histories ----------------------------------------------
    t_c = time.perf_counter()
    struct = structured_histories(GAMMA_LO, GAMMA_HI, n_seg, horizon)
    # Declared sampling law (audit A07/A1): log-uniform over the FULL admissible
    # range [GAMMA_LO, GAMMA_HI], not just the top three decades.
    log_low = float(np.log10(GAMMA_LO / GAMMA_HI))
    rand = [(f"rand_{i:03d}",
             random_history(rng, GAMMA_HI, n_seg, horizon, zero_prob=0.0,
                            log_low=log_low, log_high=0.0))
            for i in range(n_random)]
    struct = struct[:n_struct]
    c_hist = struct + rand
    c_trajs, C_states, c_recs = collect(c, c_hist, q0, init_label, spp=spp)

    rng_ho = np.random.default_rng(seed + 999_983)
    ho = [(f"heldout_{i:03d}",
           random_history(rng_ho, GAMMA_HI, n_seg, horizon, zero_prob=0.0,
                          log_low=log_low, log_high=0.0))
          for i in range(n_heldout)]
    ho_trajs, HO_states, ho_recs = collect(c, ho, q0, init_label, spp=spp)
    rec["switching_transients"] = c_recs
    rec["heldout_transients"] = ho_recs

    # Admissibility gate (audit A07/A1): every executed history - structured,
    # randomized, held-out, constant - must lie in the declared class.
    adm = AdmissibleControls(GAMMA_LO, GAMMA_HI, horizon,
                             label=f"phase3:{init_label}")
    all_hist = dict(b_hist + [(n, h) for n, h in c_hist] + ho)
    report = adm.check_all(all_hist)
    inadmissible = [k for k, v in report.items() if not v["admissible"]]
    rec["admissible_class"] = adm.to_record()
    rec["admissibility"] = {"n_histories": len(report),
                            "n_inadmissible": len(inadmissible),
                            "sampling_law": ("random segments are log-uniform over "
                                             "[GAMMA_LO, GAMMA_HI] with zero_prob=0; "
                                             "the earlier default sampled only "
                                             "[100, 1e5]"),
                            "inadmissible": inadmissible}
    if inadmissible:
        raise ValueError(f"inadmissible histories in study {init_label}: {inadmissible}")
    rec["wall_C"] = time.perf_counter() - t_c
    print(f"[phase3:{init_label}] switching: {len(c_hist)} + {n_heldout} heldout, "
          f"failures={sum(1 for t in c_trajs + ho_trajs if not t.success)}, "
          f"wall={rec['wall_C']:.1f}s")

    # --- geometry: extents and projections -----------------------------------
    ref = np.concatenate([B_states, A_states], axis=1)
    metric = StateMetric(c.gas.species_names, ref)
    # Both representations are now actually evaluated (audit A08): the earlier
    # `C_to_B_max_trace_sensitive` field was computed with kind="engineering",
    # i.e. it repeated the first metric.
    dist_B = metric.distance(C_states, B_states) if C_states.size and B_states.size else np.array([])
    dist_B_ts = metric.distance(C_states, B_states, kind="trace_sensitive") \
        if C_states.size and B_states.size else np.array([])
    dist_A = metric.distance(C_states, A_states) if C_states.size and A_states.size else np.array([])
    dist_B_ho = metric.distance(HO_states, B_states) if HO_states.size and B_states.size else np.array([])
    dist_B_eng = metric.distance(C_states, B_states, kind="engineering") \
        if C_states.size and B_states.size else np.array([])
    rec["metric_scales"] = {
        "T_scale": metric.T_scale, "y_floor": metric.y_floor,
        "Y_scale": {k: float(v) for k, v in zip(metric.species, metric.Y_scale)},
        "representation": "engineering: T/T_scale, Y/Y_scale; "
                           "trace_sensitive: T/T_scale, log10(Y/y_floor)",
    }

    sp = {n: i for i, n in enumerate(c.gas.species_names)}
    rec["extents"] = {
        "steady_T_range": [float(A_states[0].min()), float(A_states[0].max())],
        "B_T_range": [float(B_states[0].min()), float(B_states[0].max())],
        "C_T_range": [float(C_states[0].min()), float(C_states[0].max())] if C_states.size else None,
        "steady_Y_OH_max": float(A_states[1 + sp["OH"]].max()) if "OH" in sp else None,
        "B_Y_OH_max": float(B_states[1 + sp["OH"]].max()) if "OH" in sp and B_states.size else None,
        "C_Y_OH_max": float(C_states[1 + sp["OH"]].max()) if "OH" in sp and C_states.size else None,
        "steady_Y_HO2_max": float(A_states[1 + sp["HO2"]].max()) if "HO2" in sp else None,
        "B_Y_HO2_max": float(B_states[1 + sp["HO2"]].max()) if "HO2" in sp and B_states.size else None,
        "C_Y_HO2_max": float(C_states[1 + sp["HO2"]].max()) if "HO2" in sp and C_states.size else None,
    }
    rec["distances"] = {
        "C_to_B_max": float(dist_B.max()) if dist_B.size else None,
        "C_to_B_mean": float(dist_B.mean()) if dist_B.size else None,
        "C_to_A_max": float(dist_A.max()) if dist_A.size else None,
        "heldout_to_B_max": float(dist_B_ho.max()) if dist_B_ho.size else None,
        "C_to_B_max_trace_sensitive": float(dist_B_ts.max()) if dist_B_ts.size else None,
    }
    rec["counts"] = {
        "steady_points": int(A_states.shape[1]),
        "B_sampled_states": int(B_states.shape[1]),
        "C_sampled_states": int(C_states.shape[1]),
        "heldout_sampled_states": int(HO_states.shape[1]),
        "C_failures": sum(1 for t in c_trajs if not t.success),
        "heldout_failures": sum(1 for t in ho_trajs if not t.success),
    }
    # raw arrays saved for figures
    rec["_arrays"] = {"A": A_states, "B": B_states, "C": C_states, "HO": HO_states}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase3"))
    ap.add_argument("--pilot", action="store_true",
                    help="pilot mode: 8 constant controls, 16 switching histories")
    ap.add_argument("--n-constant", type=int, default=24)
    ap.add_argument("--n-random", type=int, default=32)
    ap.add_argument("--n-structured", type=int, default=16)
    ap.add_argument("--n-heldout", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.pilot:
        args.n_constant, args.n_random, args.n_structured, args.n_heldout = 8, 16, 8, 8
        suffix = "_pilot"
    else:
        suffix = ""

    t0 = time.perf_counter()
    cfg = ReactorConfig(mechanism="h2o2.yaml")
    c = CSTR(cfg)
    gammas = np.geomspace(GAMMA_LO, GAMMA_HI, args.n_constant)

    out: dict = {"timestamp_utc": utc_now(), "mechanism": c.mech,
                 "config": cfg.to_record(), "gamma_bounds": [GAMMA_LO, GAMMA_HI],
                 "horizon": HORIZON, "pilot": args.pilot,
                 "budgets": {"n_constant": args.n_constant, "n_random": args.n_random,
                             "n_structured": args.n_structured, "n_heldout": args.n_heldout,
                             "seed": args.seed}}

    q_fresh = fresh_state(c)
    q_hot = hot_hp_state(c)

    # The brief's defaults (T_in = 900 K) produce only weak reaction from a
    # fresh start within the 0.1 s horizon (see the fresh study). A separately
    # named pilot-adjusted condition at a higher inlet temperature is added so
    # that ignition behavior is observable; the original result is preserved.
    cfg_adj = ReactorConfig(mechanism="h2o2.yaml", inlet_temperature=1200.0)
    c_adj = CSTR(cfg_adj)

    study_specs = [
        ("fresh", c, fresh_state(c), gammas),
        ("hot", c, q_hot, gammas),
        ("fresh_adjusted_Tin1200K", c_adj, fresh_state(c_adj),
         np.geomspace(GAMMA_LO, GAMMA_HI, args.n_constant)),
    ]
    studies = {}
    for label, cc, q0, gg in study_specs:
        studies[label] = study(cc, label, q0, gg, args.n_random, args.n_structured,
                               args.n_heldout, stable_child_seed(args.seed, label))
    out["study_labels_note"] = (
        "fresh: fresh feed at the brief's default T_in=900 K. "
        "hot: HP-equilibrium start of the same feed (a separate permitted initial "
        "condition, not evidence of reachability from fresh feed). "
        "fresh_adjusted_Tin1200K: pilot-adjusted condition (T_in=1200 K) added "
        "because the default fresh study shows only weak reaction in 0.1 s.")
    out["studies"] = {k: {kk: vv for kk, vv in v.items() if kk != "_arrays"}
                      for k, v in studies.items()}
    # Persist the analysis clouds (audit A13): A/B/C/HO state matrices are what
    # the distances are computed from. Saved as compressed NPZ so the saved
    # results are sufficient to recompute every distance without rerunning.
    clouds = {k: {kk: np.asarray(vv) for kk, vv in v["_arrays"].items()}
              for k, v in studies.items()}
    np.savez_compressed(args.results_dir / f"state_clouds{suffix}.npz",
                        **{f"{k}__{kk}": arr for k, d in clouds.items()
                           for kk, arr in d.items()})

    # --- conservation-based LP outer comparison -------------------------------
    # Computed for both operating points (default and pilot-adjusted inlet T).
    out["lp_bounds"] = {}
    for lbl, cc in (("default_Tin900K", c), ("adjusted_Tin1200K", c_adj)):
        out["lp_bounds"][lbl] = {
            "species": {cc.gas.species_names[i]: lp_species_bound(cc, i)
                        for i in range(cc.gas.n_species)},
            "temperature": lp_temperature_bound(cc),
        }
    print(f"[phase3] LP bounds: feasible-T bracket (900K feed)="
          f"{out['lp_bounds']['default_Tin900K']['temperature']['feasible_T_bracket']}, "
          f"(1200K feed)="
          f"{out['lp_bounds']['adjusted_Tin1200K']['temperature']['feasible_T_bracket']}")

    # --- figures --------------------------------------------------------------
    fig_paths = []
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sp = {n: i for i, n in enumerate(c.gas.species_names)}
        figdir = args.results_dir / "figures"
        figdir.mkdir(parents=True, exist_ok=True)

        for label, st in studies.items():
            A, B, C = st["_arrays"]["A"], st["_arrays"]["B"], st["_arrays"]["C"]
            for pair, (sx, sy) in {"T_YOH": ("T", "OH"), "T_YHO2": ("T", "HO2")}.items():
                if sy not in sp:
                    continue
                iy = 1 + sp[sy]
                fig, ax = plt.subplots(figsize=(6, 4.5))
                if A.size:
                    ax.scatter(A[0], A[iy], s=8, c="k", alpha=0.6, label="A steady")
                if B.size:
                    ax.scatter(B[0], B[iy], s=4, c="tab:blue", alpha=0.35, label="B const-ctrl transients")
                if C.size:
                    ax.scatter(C[0], C[iy], s=4, c="tab:red", alpha=0.35, label="C switching")
                ax.set_xlabel("T [K]")
                ax.set_ylabel(f"Y_{sy}")
                ax.set_yscale("log")
                ax.set_title(f"{label} start: {pair} (h2o2, 1 atm, $t_f$={HORIZON} s, "
                             f"$\\gamma\\in$[{GAMMA_LO:g},{GAMMA_HI:g}])")
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = figdir / f"{label}_{pair}{suffix}.png"
                fig.savefig(p, dpi=140)
                plt.close(fig)
                fig_paths.append(str(p))
        out["figures"] = fig_paths
        print(f"[phase3] figures: {len(fig_paths)}")

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / f"phase3_results{suffix}.json", out)
    # Pilot and main runs get distinct immutable manifest paths (audit A12).
    manifest(f"phase3{suffix}", args.results_dir, out["budgets"],
             cwd=Path(__file__).resolve().parents[1], extra={"mechanism": c.mech},
             manifest_name=f"manifest{suffix}.json")
    print(f"[phase3] done in {out['wall_seconds']:.1f}s -> {args.results_dir}")


if __name__ == "__main__":
    main()
