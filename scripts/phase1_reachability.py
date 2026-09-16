"""Phase 1: bounded-control toy reachability study (a genuinely new experiment
beyond the Phase 0 smoke tests).

Design (research defaults, not physical combustion timescales):
  * gamma in [0, G], G in {0.1, 1, 10}
  * nondimensional horizons t_f in {1, 5, 20}
  * 8 equal-duration control segments (nested 4/8/16 comparison in selected cases)
  * initial state (x0, y0) = (1, 0) (pure A)

History classes:
  * constant controls (baseline)
  * bang-bang / isolated pulses (structured)
  * reproducibly randomized histories (declared law)
  * direct-shooting optimization of support-function extrema

Reference sets compared quantitatively:
  A. the steady CSTR locus y = x(1-x) (asymptotic reference)
  B. bounded finite-time witnessed states under admissible histories
  C. the unrestricted analytical enclosure 0 <= x <= 1, 0 <= y <= -x log x

Diagnostics: support functions in several directions in (x,y); a projected-area
diagnostic for this 2-D toy only; monotone-inclusion checks when G, horizon, or
segment count is enlarged; separate search/held-out history sets.

Usage:
    python scripts/phase1_reachability.py --results-dir results/phase1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach import toy  # noqa: E402
from thermoreach.controls import (History, bang_bang, constant, pulse, ramp,  # noqa: E402
                                  random_dwell, random_history, from_record)
from thermoreach.io_utils import manifest, save_trajectory, utc_now, write_json  # noqa: E402
from thermoreach.search import (interior_coverage, projected_area,  # noqa: E402
                               propagate_exact, support,
                                support_analytical_envelope, shoot_optimize, constant_grid)

Q0 = np.array([1.0, 0.0])

# Directions for the support function (unit vectors plus axis directions)
DIRECTIONS = np.array([
    [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
    [np.cos(np.pi / 4), np.sin(np.pi / 4)],
    [np.cos(3 * np.pi / 4), np.sin(3 * np.pi / 4)],
    [np.cos(np.pi / 6), np.sin(np.pi / 6)],
    [np.cos(np.pi / 3), np.sin(np.pi / 3)],
])


def structured_family(gamma_max: float, n_segments: int, horizon: float) -> list[tuple[str, History]]:
    out: list[tuple[str, History]] = []
    out.append(("const_max", constant(gamma_max, n_segments, horizon)))
    out.append(("const_zero", constant(0.0, n_segments, horizon)))
    out.append(("bangbang_hi_start", bang_bang(0.0, gamma_max, n_segments, horizon, start_high=True)))
    out.append(("bangbang_lo_start", bang_bang(0.0, gamma_max, n_segments, horizon, start_high=False)))
    out.append(("pulse_seg0", pulse(gamma_max, n_segments, horizon, pulse_segment=0)))
    out.append(("pulse_mid", pulse(gamma_max, n_segments, horizon, pulse_segment=n_segments // 2)))
    out.append(("pulse_last", pulse(gamma_max, n_segments, horizon, pulse_segment=n_segments - 1)))
    out.append(("ramp_up", ramp(0.0, gamma_max, n_segments, horizon, increasing=True)))
    out.append(("ramp_down", ramp(0.0, gamma_max, n_segments, horizon, increasing=False)))
    # multi-pulse alternating dwell
    half = max(1, n_segments // 2)
    vals = np.zeros(n_segments)
    vals[0::2] = gamma_max
    out.append(("alternating", History(vals, np.full(n_segments, horizon / n_segments))))
    return out


def collect_states(histories, samples_per_segment: int = 33):
    """Return (visited_states, terminal_states, records).

    Visited states are all sampled states on [0, t_f]; terminal states are the
    states at exactly t = t_f (the set R(t_f)). The two must be reported
    separately: R(t_f) is control-sensitive, while R_[0,t_f] for this toy is
    dominated along its upper boundary by the gamma = 0 (batch) trajectory.
    """
    all_states, term_states, recs = [], [], []
    for name, h in histories:
        ts, ys = propagate_exact(h, Q0, samples_per_segment=samples_per_segment)
        all_states.append(ys)
        term_states.append(ys[:, -1:])
        recs.append({"name": name, "history": h.to_record(),
                     "terminal_state": ys[:, -1].tolist()})
    return (np.concatenate(all_states, axis=1),
            np.concatenate(term_states, axis=1), recs)


def analytical_area(n_grid: int = 200001) -> float:
    """Area of the unrestricted enclosure {0<=x<=1, 0<=y<=-x log x}."""
    x = np.linspace(0.0, 1.0, n_grid)
    return float(np.trapezoid(-x * np.log(np.maximum(x, 1e-300)), x))


def run_case(gamma_max: float, horizon: float, n_segments: int, seed: int,
             n_random: int = 32, n_heldout: int = 16, optimize: bool = True,
             opt_directions: int = 4, maxiter: int = 150) -> dict:
    rng = np.random.default_rng(seed)
    rec: dict = {
        "gamma_max": gamma_max, "horizon": horizon, "n_segments": n_segments, "seed": seed,
        "initial_state": Q0.tolist(),
    }

    # ---- reference A: steady locus sampled by constant controls -------------
    gammas = np.concatenate([[0.0], gamma_max * np.power(10.0, np.linspace(-3.0, 0.0, 23))])
    consts = [("const_g%.6g" % g, constant(float(g), 1, horizon)) for g in gammas]
    const_states, const_term, const_recs = collect_states(consts, samples_per_segment=65)
    # terminal points of constant-control trajectories lie on/near the steady
    # locus only asymptotically; record the steady points exactly too.
    steady_pts = np.array([toy.steady_point(float(g)) for g in gammas]).T

    # ---- reference B: structured + randomized switching histories -----------
    struct = structured_family(gamma_max, n_segments, horizon)
    struct_states, struct_term, struct_recs = collect_states(struct)
    rand = [("rand_%03d" % i, random_history(rng, gamma_max, n_segments, horizon))
            for i in range(n_random)]
    rand_states, rand_term, rand_recs = collect_states(rand)
    dwell = [("dwell_%03d" % i, random_dwell(rng, gamma_max, n_segments, horizon))
             for i in range(n_random // 4)]
    dwell_states, dwell_term, dwell_recs = collect_states(dwell)

    # ---- held-out set (whole histories, separate seed) ----------------------
    rng_ho = np.random.default_rng(seed + 999_983)
    ho = [("heldout_%03d" % i, random_history(rng_ho, gamma_max, n_segments, horizon))
          for i in range(n_heldout)]
    ho_states, ho_term, ho_recs = collect_states(ho)

    search_states = np.concatenate([struct_states, rand_states, dwell_states], axis=1)
    all_states = np.concatenate([search_states, ho_states, const_states], axis=1)
    search_term = np.concatenate([struct_term, rand_term, dwell_term], axis=1)
    all_term = np.concatenate([search_term, ho_term, const_term], axis=1)

    # ---- diagnostics --------------------------------------------------------
    rec["support"] = {}
    for i in range(DIRECTIONS.shape[0]):
        c = DIRECTIONS[i]
        rec["support"][f"c{i}"] = {
            "direction": c.tolist(),
            "search_max": support(search_states, c),
            "terminal_max": support(search_term, c),
            "all_max": support(all_states, c),
            "steady_locus_max": float(np.max(steady_pts.T @ c)),
            "analytical_envelope_max": support_analytical_envelope(c),
        }
    rec["extents"] = {
        # visited set R_[0,t_f]
        "visited_x_min": float(search_states[0].min()), "visited_x_max": float(search_states[0].max()),
        "visited_y_min": float(search_states[1].min()), "visited_y_max": float(search_states[1].max()),
        # terminal set R(t_f): control-sensitive
        "terminal_x_min": float(search_term[0].min()), "terminal_x_max": float(search_term[0].max()),
        "terminal_y_min": float(search_term[1].min()), "terminal_y_max": float(search_term[1].max()),
        "steady_x_range": [float(steady_pts[0].min()), float(steady_pts[0].max())],
        "steady_y_range": [float(steady_pts[1].min()), float(steady_pts[1].max())],
    }
    rec["projected_area_search"] = projected_area(search_states)
    rec["projected_area_all"] = projected_area(all_states)
    rec["projected_area_terminal_search"] = projected_area(search_term)
    rec["projected_area_steady_hull"] = projected_area(steady_pts)
    rec["analytical_envelope_area"] = analytical_area()
    rec["envelope_area_fraction_visited"] = (rec["projected_area_search"]
                                             / rec["analytical_envelope_area"])
    # Corrected coverage diagnostic (audit section 5): the hull fraction above is
    # degenerate - a single 1-D curve fills it - so report the one-sided fill
    # distance of the interior grid as well.
    rec["interior_coverage_search"] = interior_coverage(search_states, n_grid=61,
                                                         tol=2e-3)
    rec["area_diagnostic_note"] = (
        "envelope_area_fraction_visited is a convex-hull quantity and is "
        "DEPRECATED as a coverage measure: the zero-control batch curve alone "
        "reproduces 0.129/0.933/1.000 at horizons 1/5/20 while having zero area. "
        "See results/phase1b and interior_coverage_search for the corrected "
        "one-sided fill-distance diagnostic.")
    rec["max_barrier_search"] = float(np.max(toy.barrier(search_states)))
    rec["min_fraction_search"] = float(np.min(np.array([
        search_states[0].min(), search_states[1].min(),
        (1.0 - search_states[0] - search_states[1]).min()])))
    rec["counts"] = {
        "constant_controls": len(consts),
        "structured_histories": len(struct),
        "random_histories": n_random,
        "dwell_histories": len(dwell),
        "heldout_histories": n_heldout,
        "sampled_states_search": int(search_states.shape[1]),
        "sampled_states_all": int(all_states.shape[1]),
        "terminal_states_search": int(search_term.shape[1]),
    }
    rec["histories"] = {
        "structured": struct_recs, "random": rand_recs, "dwell": dwell_recs,
        "heldout": ho_recs, "constant": const_recs,
    }

    # ---- direct-shooting optimization of selected extrema -------------------
    if optimize:
        opt_summary = []
        for i in range(opt_directions):
            c = DIRECTIONS[i]
            for kind in ("max_over_time", "terminal"):
                cands = shoot_optimize(c, Q0, gamma_max, n_segments, horizon,
                                       objective_kind=kind, rng=np.random.default_rng(seed + i),
                                       maxiter=maxiter)
                best = cands[0]
                opt_summary.append({
                    "direction": c.tolist(), "objective_kind": kind,
                    "best_origin": best["origin"], "best_objective": best["objective"],
                    "n_candidates": len(cands), "n_failures": sum(1 for r in cands if r["objective"] is None),
                    "best_values": best["values"], "best_terminal_state": best["terminal_state"],
                })
        rec["optimization"] = opt_summary

    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase1"))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    ap.add_argument("--segments", type=int, nargs="+", default=[8])
    ap.add_argument("--n-random", type=int, default=32)
    ap.add_argument("--n-heldout", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    t0 = time.perf_counter()
    out: dict = {"timestamp_utc": utc_now(), "q0": Q0.tolist(),
                 "directions": DIRECTIONS.tolist(), "cases": []}

    for G in args.gammas:
        for tf in args.horizons:
            for nseg in args.segments:
                rec = run_case(G, tf, nseg, args.seed,
                               n_random=args.n_random, n_heldout=args.n_heldout)
                out["cases"].append(rec)
                print(f"[phase1] G={G} tf={tf} nseg={nseg}: "
                      f"visited_y_max={rec['extents']['visited_y_max']:.4f} "
                      f"term_y_max={rec['extents']['terminal_y_max']:.4f} "
                      f"term_x_min={rec['extents']['terminal_x_min']:.4f} "
                      f"area={rec['projected_area_search']:.4f} "
                      f"env_frac={rec['envelope_area_fraction_visited']:.3f} "
                      f"barrier_max={rec['max_barrier_search']:.2e}")

    # ---- nested segment-count comparison on selected cases ------------------
    G0, tf0 = 1.0, 5.0
    if G0 in args.gammas and tf0 in args.horizons:
        seg_study = []
        for nseg in (4, 8, 16, 32):
            rec = run_case(G0, tf0, nseg, args.seed, n_random=args.n_random,
                           n_heldout=args.n_heldout, optimize=False)
            seg_study.append({"n_segments": nseg,
                              "visited_y_max": rec["extents"]["visited_y_max"],
                              "visited_x_max": rec["extents"]["visited_x_max"],
                              "terminal_y_max": rec["extents"]["terminal_y_max"],
                              "projected_area_search": rec["projected_area_search"],
                              "projected_area_terminal_search": rec["projected_area_terminal_search"],
                              "max_barrier_search": rec["max_barrier_search"],
                              "support_y": rec["support"]["c1"]["search_max"],
                              "counts": rec["counts"]})
            print(f"[phase1] nested nseg={nseg}: "
                  f"area={seg_study[-1]['projected_area_search']:.4f} "
                  f"term_area={seg_study[-1]['projected_area_terminal_search']:.4f}")
        out["segment_resolution_study"] = {"gamma_max": G0, "horizon": tf0, "cases": seg_study}

    # ---- monotone-inclusion checks ------------------------------------------
    by_gh = {(c["gamma_max"], c["horizon"]): c for c in out["cases"]}
    incl = []
    for (G, tf), c in sorted(by_gh.items()):
        row = {"gamma_max": G, "horizon": tf,
               "visited_area": c["projected_area_search"],
               "terminal_area": c["projected_area_terminal_search"],
               "visited_y_max": c["extents"]["visited_y_max"],
               "terminal_y_max": c["extents"]["terminal_y_max"]}
        for (G2, tf2), c2 in by_gh.items():
            if G2 >= G and tf2 >= tf and (G2, tf2) != (G, tf):
                # Expected inclusion of admissible classes: enlarging G and/or
                # horizon cannot shrink the reachable set.
                row[f"in_G{G2}_tf{tf2}_ok"] = bool(
                    c2["extents"]["visited_y_max"] >= c["extents"]["visited_y_max"] - 1e-9
                    and c2["projected_area_search"] >= c["projected_area_search"] - 1e-9)
        incl.append(row)
    out["inclusion_checks"] = incl

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase1_results.json", out)
    manifest("phase1", args.results_dir,
             {"gammas": args.gammas, "horizons": args.horizons,
              "segments": args.segments, "n_random": args.n_random,
              "n_heldout": args.n_heldout, "seed": args.seed},
             cwd=Path(__file__).resolve().parents[1])
    print(f"[phase1] done in {out['wall_seconds']:.1f}s -> {args.results_dir}")


if __name__ == "__main__":
    main()
