"""Strang splitting of the validated CSTR ODE (Phase 3C).

Let R_dt be adiabatic fixed-pressure *reaction* (no flow), and X_dt be the exact
*fixed-feed exchange* at constant gamma:

    Y_new = Y_in + exp(-gamma*dt) * (Y_old - Y_in)
    h_new = h_in + exp(-gamma*dt) * (h_old - h_in)

where h = sum_k Y_k h_k(T). X_dt therefore mixes composition exactly and
preserves the exact mixing invariant h(t) = h_in + exp(-Gamma)(h0 - h_in); the
new temperature is recovered uniquely from h(T, Y_new) = h_new because cp > 0.

The explicitly named Strang sequence is

    X_(dt/2) -> R_dt -> X_(dt/2).

This is a solver-interface study for this CSTR splitting only. It is not a
claim about OpenFOAM or a turbulent flame, and the chemistry-substep states it
produces are not exact unsplit physical states.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .controls import History
from .reactor import CSTR


def mixture_enthalpy(c: CSTR, T: float, Y: np.ndarray) -> float:
    hk = c.species_enthalpies(T)
    return float(hk @ np.maximum(Y, 0.0))


def temperature_from_enthalpy(c: CSTR, h_target: float, Y: np.ndarray,
                              lo: float = 200.0, hi: float = 6000.0) -> float:
    """Unique T with h(T, Y) = h_target (h is strictly increasing in T)."""
    f = lambda T: mixture_enthalpy(c, T, Y) - h_target
    if f(lo) > 0.0:
        return lo
    if f(hi) < 0.0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


def exchange_step(c: CSTR, q: np.ndarray, gamma: float, dt: float) -> np.ndarray:
    """X_dt: exact fixed-feed exchange at constant gamma."""
    T_old, Y_old = float(q[0]), np.asarray(q[1:])
    dec = np.exp(-gamma * dt)
    Y_new = c.Y_in + dec * (Y_old - c.Y_in)
    h_old = mixture_enthalpy(c, T_old, Y_old)
    h_new = c.h_in + dec * (h_old - c.h_in)
    T_new = temperature_from_enthalpy(c, h_new, Y_new)
    return np.concatenate([[T_new], Y_new])


def reaction_step(c: CSTR, q: np.ndarray, dt: float, method: str = "Radau",
                  rtol: float = 1e-10, atol_T: float = 1e-9, atol_Y: float = 1e-16) -> np.ndarray:
    """R_dt: adiabatic fixed-pressure reaction with no flow (gamma = 0)."""
    atol = np.concatenate([[atol_T], np.full(c.gas.n_species, atol_Y)])
    sol = solve_ivp(lambda t, s: c.rhs_alt(t, s, 0.0), (0.0, dt), np.asarray(q, dtype=float),
                    method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"reaction substep failed: {sol.message}")
    return sol.y[:, -1]


def strang_step(c: CSTR, q: np.ndarray, gamma: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X_(dt/2) -> R_dt -> X_(dt/2). Returns (after_first_half, after_R, after_step)."""
    q1 = exchange_step(c, q, gamma, dt / 2.0)
    q2 = reaction_step(c, q1, dt)
    q3 = exchange_step(c, q2, gamma, dt / 2.0)
    return q1, q2, q3


def strang_integrate(c: CSTR, history: History, q0: np.ndarray, steps_per_segment: int,
                     record_substeps: bool = True) -> dict:
    """Integrate with Strang splitting, with steps aligned to control segments.

    ``steps_per_segment`` Strang steps per constant-control segment. Returns
    times/states at step boundaries, plus the recorded pre-R / post-R substep
    states (the chemistry-substep inputs and outputs).
    """
    q = np.asarray(q0, dtype=float).copy()
    ts, ys = [], []
    pre_R, post_R, sub_t = [], [], []
    t_acc = 0.0
    for val, dur in zip(history.values, history.durations):
        g = float(val)
        n = steps_per_segment
        dt = dur / n
        for i in range(n):
            q1, q2, q3 = strang_step(c, q, g, dt)
            if record_substeps:
                pre_R.append(q1)
                post_R.append(q2)
                sub_t.append(t_acc + dt * (i + 0.5))
            t_acc_step = t_acc + dt * (i + 1)
            ts.append(t_acc_step)
            ys.append(q3)
            q = q3
        t_acc += dur
    return {
        "times": np.asarray(ts), "states": np.concatenate([s[:, None] for s in ys], axis=1),
        "pre_R_states": np.concatenate([s[:, None] for s in pre_R], axis=1) if pre_R else None,
        "post_R_states": np.concatenate([s[:, None] for s in post_R], axis=1) if post_R else None,
        "substep_times": np.asarray(sub_t) if sub_t else None,
        "steps_per_segment": steps_per_segment,
    }
