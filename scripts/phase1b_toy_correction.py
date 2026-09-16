"""Phase 1b - corrected toy reachability study (audit Parts B and 4-5).

Supersedes the visited-set conclusion of report_01.md (claim C7), which is
REFUTED: the reachable visited set DOES depend on the control bound G.

What was wrong:
  * The reported `envelope_area_fraction_visited` (0.129 / 0.933 / 1.000 at
    horizons 1 / 5 / 20) is the convex-hull area of a SINGLE one-dimensional
    curve - the zero-control batch trajectory y = -x log x.  A curve has zero
    two-dimensional area, yet its convex hull fills most of the envelope, so the
    diagnostic measures nothing about interior coverage.
  * Because gamma = 0 is admissible for every G, that curve is common to all
    bounds, which made the visited SET look G-independent.  It is not.

What this script establishes:
  1. An explicit witness: gamma = 0 for 4 units, then gamma = 10 for
     delta = 0.0707413427911034 reaches (x, y) = (0.5, 0.04939624612802243)
     before t = 5 - reachable with G = 10.
  2. A conditional lower barrier: for W = y - x(1-x), dW/dt|_{W=0} =
     (1-x)[(1+gamma)x - gamma] >= 0 whenever x >= x_G = G/(1+G), and x cannot
     cross x_G upward, so any reachable state with x >= x_G has y >= x(1-x).
     For G = 0.1 at x = 0.5 this forces y >= 0.25, excluding the witness at
     every time - not merely before the horizon.
  3. The hull-area degeneracy, reproduced exactly.
  4. A NON-degenerate interior-coverage diagnostic (one-sided fill distance)
     applied to G-bounded clouds, which is G-sensitive.
  5. A corrected figure: batch curve, its hull, the witness, the G=0.1 exclusion
     region, and the sampled reachable clouds for G = 0.1 and G = 10.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls  # noqa: E402
from thermoreach.controls import History, random_history  # noqa: E402
from thermoreach.io_utils import manifest, utc_now, write_json  # noqa: E402
from thermoreach.search import (interior_coverage, projected_area,  # noqa: E402
                               propagate_exact)
from thermoreach.toy import advance, analytical_upper_curve, steady_locus  # noqa: E402

Q0 = np.array([1.0, 0.0])
HORIZON = 5.0


def lower_barrier_derivative(x: float, gamma: float) -> float:
    """dW/dt on W = 0 with W = y - x(1-x)."""
    return (1.0 - x) * ((1.0 + gamma) * x - gamma)


def witness() -> dict:
    """The audit's G=10 witness, verified with the repository's own propagator."""
    delta = float(np.log((10.0 / 11.0 - np.exp(-4.0)) / (10.0 / 11.0 - 0.5)) / 11.0)
    end = advance(advance(Q0, 0.0, 4.0), 10.0, delta)
    # cross-check by dense integration of the two-segment history
    h = History(np.array([0.0, 10.0]), np.array([4.0, delta]))
    ts, ys = propagate_exact(h, Q0, samples_per_segment=20001)
    return {"gamma_history": [0.0, 10.0], "durations": [4.0, delta],
            "endpoint_exact": end.tolist(),
            "endpoint_dense_integration": ys[:, -1].tolist(),
            "t_final": 4.0 + delta,
            "closed_form_delta": delta}


def exclusion(G: float, n: int = 401) -> dict:
    """Verify the conditional lower barrier for a given bound G.

    The barrier applies only to states with x >= x_G = G/(1+G); for larger G the
    threshold moves right until the witness's x = 0.5 lies below it, at which
    point the barrier says nothing and the witness IS reachable.
    """
    xG = G / (1.0 + G)
    xs = np.linspace(xG, 1.0, n)
    gs = np.linspace(0.0, G, n)
    worst = min(lower_barrier_derivative(x, g) for x in xs for g in gs)
    # the required floor y >= x(1-x) at the witness's x = 0.5
    witness_x, witness_y = 0.5, 0.04939624612802243
    floor_at_half = witness_x * (1 - witness_x)
    barrier_applies = witness_x >= xG
    return {"G": G, "x_G": xG,
            "min_dW_dt_on_boundary_for_x_ge_xG": float(worst),
            "barrier_applies_to_witness": bool(barrier_applies),
            "required_y_at_x_0.5": floor_at_half,
            "witness_y": witness_y,
            "witness_excluded": bool(barrier_applies and witness_y < floor_at_half)}


def hull_degeneracy(n: int = 400_001) -> dict:
    """The batch curve's convex-hull fractions alone reproduce 0.129/0.933/1.000."""
    out = []
    for T, reported in ((1.0, 0.129), (5.0, 0.933), (20.0, 1.000)):
        t = np.linspace(0.0, T, n)
        pts = np.stack([np.exp(-t), t * np.exp(-t)], axis=1)
        from scipy.spatial import ConvexHull
        frac = ConvexHull(pts).volume / 0.25
        out.append({"horizon": T, "batch_curve_hull_area_fraction": frac,
                    "closed_form": 1.0 - np.exp(-2 * T) - 2 * T * np.exp(-T),
                    "value_reported_in_report_01": reported})
    return {"cases": out,
            "note": "a single 1-D curve; its own two-dimensional area is zero"}


def sample_cloud(G: float, n_hist: int, n_seg: int, seed: int,
                 horizon: float = HORIZON) -> tuple[np.ndarray, list]:
    """Visited states (grid + extrema) of admissible switching histories for bound G."""
    rng = np.random.default_rng(seed)
    adm = AdmissibleControls(0.0, G, horizon)
    states, recs = [], []
    log_low = float(np.log10(1e-4))      # sample the full [1e-4, G] range
    for i in range(n_hist):
        h = random_history(rng, G, n_seg, horizon, zero_prob=0.2, log_low=log_low)
        ok, reason = adm.validate(h)
        if not ok:
            raise ValueError(f"inadmissible sample for G={G}: {reason}")
        ts, ys = propagate_exact(h, Q0, samples_per_segment=201)
        states.append(ys)
        recs.append({"label": f"rand_{i:03d}", "values": h.values.tolist(),
                     "terminal": ys[:, -1].tolist()})
    # constant controls complete the declared class representation
    for g in np.geomspace(max(G * 1e-3, 1e-4), G, 8):
        h = History(np.array([float(g)]), np.array([horizon]))
        ts, ys = propagate_exact(h, Q0, samples_per_segment=201)
        states.append(ys)
    return np.concatenate(states, axis=1), recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/phase1b"))
    ap.add_argument("--n-hist", type=int, default=400)
    ap.add_argument("--n-seg", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    t0 = time.perf_counter()
    out: dict = {"timestamp_utc": utc_now(), "initial_state": Q0.tolist(),
                 "horizon": HORIZON, "seed": args.seed}

    out["witness"] = witness()
    print(f"[phase1b] witness endpoint = {out['witness']['endpoint_exact']} "
          f"at t = {out['witness']['t_final']:.6f}")

    out["exclusion"] = {f"G={g:g}": exclusion(g) for g in (0.1, 1.0, 10.0)}
    for k, v in out["exclusion"].items():
        print(f"[phase1b] {k}: x_G={v['x_G']:.6f}, min dW/dt={v['min_dW_dt_on_boundary_for_x_ge_xG']:+.2e}, "
              f"witness excluded = {v['witness_excluded']}")

    out["hull_degeneracy"] = hull_degeneracy()
    print("[phase1b] hull fractions of the single batch curve: "
          + ", ".join(f"{c['horizon']:g}->{c['batch_curve_hull_area_fraction']:.6f}"
                      for c in out["hull_degeneracy"]["cases"]))

    # G-bounded clouds with the corrected coverage diagnostic
    out["bounds"] = {}
    for G in (0.1, 10.0):
        cloud, recs = sample_cloud(G, args.n_hist, args.n_seg,
                                   args.seed + (0 if G < 1 else 1))
        cov = interior_coverage(cloud, n_grid=61, tol=2e-3)
        hull = projected_area(cloud)
        out["bounds"][f"G={G:g}"] = {
            "n_states": int(cloud.shape[1]),
            "interior_coverage": cov,
            "hull_area_fraction_DEPRECATED": hull / 0.25,
            "terminal_y_max": float(cloud[1].max()),
            "terminal_states_sample": recs[:3],
        }
        print(f"[phase1b] G={G:g}: fill_distance={cov['fill_distance']:.4f}, "
              f"covered_fraction={cov['covered_fraction']:.4f}, "
              f"hull_frac(deprecated)={hull / 0.25:.4f}")

    # the witness belongs to G=10's cloud extent but is excluded for G=0.1
    w = np.array(out["witness"]["endpoint_exact"])
    out["witness_membership"] = {
        "reachable_with_G10_before_t5": bool(w[0] == 0.5 and 4.0 + out["witness"]["closed_form_delta"] < 5.0),
        "excluded_for_G0.1_at_every_time": bool(out["exclusion"]["G=0.1"]["witness_excluded"]),
    }
    out["conclusion"] = (
        "The reachable visited set depends on the control bound G (claim C7 of "
        "report_01.md is refuted). The earlier G-insensitivity was an artifact of "
        "the convex-hull area diagnostic, which a single one-dimensional curve can "
        "fill. The corrected diagnostic (one-sided fill distance) is G-sensitive."
    )

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.2))
        x = np.linspace(1e-4, 1.0, 2001)
        ax.plot(x, analytical_upper_curve(x), "k-", lw=1.5, label="upper barrier $y=-x\\log x$")
        ax.plot(x, steady_locus(x), ":", color="0.4", lw=1.2, label="steady locus $y=x(1-x)$")
        # batch curve and its hull
        tb = np.linspace(0.0, HORIZON, 4001)
        xb, yb = np.exp(-tb), tb * np.exp(-tb)
        ax.plot(xb, yb, "g-", lw=1.5, label="$\\gamma=0$ batch curve (hull fills the envelope)")
        ax.fill(xb, yb, color="0.85", zorder=0)
        # G=0.1 exclusion region: x >= x_G and y < x(1-x)
        xG = 0.1 / 1.1
        xm = np.linspace(xG, 1.0, 401)
        ax.fill_between(xm, 0.0, steady_locus(xm), color="red", alpha=0.18,
                        label=f"excluded for $G=0.1$ ($x\\ge x_G$, $y<x(1-x)$)")
        ax.axvline(xG, color="red", lw=0.8, ls="--")
        # clouds
        for G, color in ((0.1, "tab:orange"), (10.0, "tab:blue")):
            cloud, _ = sample_cloud(G, 60, args.n_seg, args.seed + 5)
            ax.scatter(cloud[0], cloud[1], s=3, c=color, alpha=0.35,
                       label=f"sampled visited, $G={G:g}$")
        ax.scatter([w[0]], [w[1]], marker="*", s=220, c="red", edgecolors="k",
                   zorder=5, label="witness $(0.5,\\,0.0494)$, $G=10$ only")
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.37)
        ax.set_xlabel("x  (Y_A)"); ax.set_ylabel("y  (Y_B)")
        ax.set_title("Toy visited sets depend on the control bound (C7 refuted)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        args.results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.results_dir / "toy_visited_sets_corrected.png", dpi=150)
        out["figure"] = str(args.results_dir / "toy_visited_sets_corrected.png")
        plt.close(fig)
        print(f"[phase1b] figure -> {out['figure']}")
    except Exception as exc:                      # pragma: no cover
        out["figure_error"] = str(exc)

    out["wall_seconds"] = time.perf_counter() - t0
    write_json(args.results_dir / "phase1b_results.json", out)
    manifest("phase1b", args.results_dir,
             {"horizon": HORIZON, "n_hist": args.n_hist, "n_seg": args.n_seg,
              "seed": args.seed},
             cwd=Path(__file__).resolve().parents[1])
    print(f"[phase1b] done in {out['wall_seconds']:.1f}s -> {args.results_dir}")


if __name__ == "__main__":
    main()
