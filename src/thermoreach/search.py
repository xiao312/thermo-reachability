"""Trajectory propagation and search for the toy bounded-control study.

Because the constant-control segment solution is exact (see ``toy.exact_segment``),
histories can be propagated with no ODE error, and the direct-shooting
optimization of extrema is inexpensive. Within-segment extrema are resolved on a
dense exact grid rather than at segment endpoints only.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .controls import History, constant, random_history, bang_bang, pulse, ramp
from . import toy


def propagate_exact(history: History, q0: np.ndarray, samples_per_segment: int = 33,
                    include_endpoints: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a piecewise-constant history with the exact segment solution.

    Returns (times, states) with states shape (2, N); times are absolute physical
    times, so accumulated physical time is preserved across segments.
    """
    q = np.asarray(q0, dtype=float).copy()
    ts, ys = [], []
    t_accum = 0.0
    for val, dur in zip(history.values, history.durations):
        n = samples_per_segment
        local = np.linspace(0.0, dur, n) if include_endpoints else np.linspace(0.0, dur, n, endpoint=False)
        seg = toy.exact_segment(q, float(val), local)
        ts.append(t_accum + local)
        ys.append(seg)
        q = seg[:, -1].copy()
        t_accum += dur
    return np.concatenate(ts), np.concatenate(ys, axis=1)


def terminal_state(history: History, q0: np.ndarray) -> np.ndarray:
    q = np.asarray(q0, dtype=float).copy()
    for val, dur in zip(history.values, history.durations):
        q = toy.advance(q, float(val), float(dur))
    return q


# ---------------------------------------------------------------------------
# Support functions and analytical envelope bounds
# ---------------------------------------------------------------------------


def support(states: np.ndarray, c: np.ndarray) -> float:
    """max_t c . q(t) over sampled states (a lower estimate of the true sup)."""
    return float(np.max(states.T @ c))


def support_analytical_envelope(c: np.ndarray, n_grid: int = 200001) -> float:
    """sup of c.(x,y) over the unrestricted analytical enclosure
    {0 <= x <= 1, 0 <= y <= -x log x}.

    This is the claimed closure for gamma >= 0, unlimited duration, and
    arbitrarily rapid idealized feed replacement (to be audited separately for
    attainability). It is a containing-bound quantity, not a finite-time,
    bounded-control quantity.
    """
    cx, cy = float(c[0]), float(c[1])
    if cy >= 0.0:
        # y at its upper curve; maximize cx*x + cy*(-x log x)
        x = np.linspace(0.0, 1.0, n_grid)
        f = cx * x + cy * (-x * np.log(np.maximum(x, 1e-300)))
    else:
        # y = 0 is best; maximize cx*x (x in [0,1])
        x = np.linspace(0.0, 1.0, n_grid)
        f = cx * x
    return float(np.max(f))


# ---------------------------------------------------------------------------
# Direct shooting optimization of a selected extremum
# ---------------------------------------------------------------------------


def _pack_objective(params, q0, c, horizon, objective_kind):
    h = History(np.abs(params), np.full(params.size, horizon / params.size))
    ts, ys = propagate_exact(h, q0, samples_per_segment=9)
    if objective_kind == "terminal":
        val = float(c @ ys[:, -1])
    else:  # "max_over_time"
        val = float(np.max(ys.T @ c))
    return -val  # minimize negative


def shoot_optimize(c: np.ndarray, q0: np.ndarray, gamma_max: float, n_segments: int,
                   horizon: float, objective_kind: str = "max_over_time",
                   seed_history_fns=(), rng: np.random.Generator | None = None,
                   maxiter: int = 200) -> list[dict]:
    """Direct-shooting search for histories maximizing c.q.

    ``objective_kind`` is 'terminal' (value at t = horizon) or 'max_over_time'.
    Returns a list of candidate records: every candidate's controls, objective,
    feasibility flag and trajectory summary. Failed evaluations are recorded too.
    """
    q0 = np.asarray(q0, dtype=float)
    candidates: list[dict] = []

    def evaluate(vals: np.ndarray, origin: str) -> dict:
        vals = np.clip(np.abs(np.asarray(vals, dtype=float)), 0.0, gamma_max)
        h = History(vals, np.full(vals.size, horizon / vals.size))
        rec: dict = {
            "origin": origin,
            "objective_kind": objective_kind,
            "direction": np.asarray(c).tolist(),
            "gamma_max": gamma_max,
            "horizon": horizon,
            "n_segments": n_segments,
            "values": vals.tolist(),
            "durations": h.durations.tolist(),
            "feasible": bool(h.bounds_check(gamma_max) and toy.in_triangle(q0)),
        }
        try:
            ts, ys = propagate_exact(h, q0, samples_per_segment=33)
            rec["objective"] = (float(c @ ys[:, -1]) if objective_kind == "terminal"
                                else float(np.max(ys.T @ c)))
            rec["terminal_state"] = ys[:, -1].tolist()
            rec["max_state"] = ys[:, int(np.argmax(ys.T @ c))].tolist()
            rec["t_at_max"] = float(ts[int(np.argmax(ys.T @ c))])
            rec["status"] = "ok"
        except Exception as exc:  # record the failure, do not raise
            rec["status"] = f"failed: {type(exc).__name__}: {exc}"
            rec["objective"] = None
        candidates.append(rec)
        return rec

    # Seed set: structured + randomized + a zero and max baseline
    seeds = [np.zeros(n_segments), np.full(n_segments, gamma_max)]
    for fn in seed_history_fns:
        seeds.append(np.asarray(fn().values, dtype=float))
    if rng is not None:
        for _ in range(8):
            seeds.append(np.asarray(random_history(rng, gamma_max, n_segments, horizon).values))

    bounds = [(0.0, gamma_max)] * n_segments
    for s in seeds:
        res = minimize(_pack_objective, np.clip(s, 0.0, gamma_max), args=(q0, c, horizon, objective_kind),
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        evaluate(res.x, f"lbfgsb-from-{np.round(s[:3], 3).tolist()}")
        # Also record the unoptimized seed itself for provenance
        evaluate(s, "seed")

    # Sort by achieved objective (failures last)
    candidates.sort(key=lambda r: (r["objective"] is None, -(r["objective"] or -np.inf)))
    return candidates


def constant_grid(gamma_max: float, n: int = 24, horizon: float = 1.0,
                    zero_prob_zero: bool = True) -> list[History]:
    """Constant-control baselines on a log-uniform grid in (0, G], plus gamma=0."""
    vals = gamma_max * np.power(10.0, np.linspace(-3.0, 0.0, n - 1))
    if zero_prob_zero:
        vals = np.concatenate([[0.0], vals])
    return [constant(float(v), 1, horizon) for v in vals]
