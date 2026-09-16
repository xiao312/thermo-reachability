"""Isothermal A -> B -> C toy reactor with equal molecular weights and equal
nondimensional rate constants k1 = k2 = 1, pure-A feed.

State:  x = Y_A,  y = Y_B,  Y_C = 1 - x - y.
Control: gamma(t) = mass_inflow / reactor_mass >= 0 (nondimensional feed rate).

    dx/dt = -x + gamma(t) * (1 - x)
    dy/dt =  x - (1 + gamma(t)) * y

This is the Phase 1A analytical reference of the research brief. The exact
constant-control propagator below makes the bounded-control reachability search
inexpensive and independent of ODE error.

Units: nondimensional. These are research defaults, not physical combustion
timescales.
"""

from __future__ import annotations

import numpy as np


def rhs(q: np.ndarray, gamma: float) -> np.ndarray:
    """Right-hand side of the toy CSTR at constant control gamma."""
    x, y = q
    return np.array([-x + gamma * (1.0 - x), x - (1.0 + gamma) * y])


def jacobian(gamma: float) -> np.ndarray:
    """Constant Jacobian of the toy CSTR at constant control gamma."""
    return np.array([[-1.0 - gamma, 0.0], [1.0, -1.0 - gamma]])


def steady_point(gamma: float) -> np.ndarray:
    """Steady state for constant gamma: x* = gamma/(1+gamma), y* = gamma/(1+gamma)^2."""
    a = 1.0 + gamma
    return np.array([gamma / a, gamma / a**2])


def exact_segment(q0: np.ndarray, gamma: float, t: np.ndarray) -> np.ndarray:
    """Exact solution of the toy CSTR with constant gamma over times ``t``.

    With a = 1 + gamma, x* = gamma/a, y* = gamma/a^2:

        x(t) = x* + (x0 - x*) exp(-a t)
        y(t) = y* + ((y0 - y*) + (x0 - x*) t) exp(-a t)

    Returns shape (2, len(t)). Vectorised over t.
    """
    q0 = np.asarray(q0, dtype=float)
    t = np.asarray(t, dtype=float)
    a = 1.0 + gamma
    xs = gamma / a
    ys = gamma / a**2
    e = np.exp(-a * t)
    x = xs + (q0[0] - xs) * e
    y = ys + ((q0[1] - ys) + (q0[0] - xs) * t) * e
    return np.vstack([x, y])


def advance(q0: np.ndarray, gamma: float, duration: float) -> np.ndarray:
    """Exact terminal state after one constant-control segment."""
    return exact_segment(q0, gamma, np.array([duration]))[:, -1]


# ---------------------------------------------------------------------------
# Invariant barrier and analytical envelopes
# ---------------------------------------------------------------------------


def barrier(q: np.ndarray) -> float:
    """V(x, y) = y + x log x, evaluated by continuous extension at x = 0.

    The continuous extension of x log x at x = 0 is 0, so V(0, y) = y.
    Never evaluates log(0) directly.
    """
    x, y = q[0], q[1]
    return y + np.where(x > 0.0, x * np.log(np.maximum(x, 1e-300)), 0.0)


def barrier_boundary_derivative(gamma: float, x: float) -> float:
    """dV/dt restricted to the curve V = 0, i.e. y = -x log x.

    Symbolically (verified with SymPy in the Phase 0 smoke tests):

        dV/dt|_(V=0) = gamma * (log(x) + 1 - x) <= 0  for gamma >= 0, 0 < x <= 1

    by the standard inequality log(x) <= x - 1.
    """
    return gamma * (np.log(x) + 1.0 - x)


def analytical_upper_curve(x: np.ndarray) -> np.ndarray:
    """y = -x log(x): candidate upper boundary of the unrestricted-duration,
    unbounded-feed-rate reachable closure. Valid for 0 <= x <= 1."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x > 0.0, -x * np.log(np.maximum(x, 1e-300)), 0.0)


def steady_locus(x: np.ndarray) -> np.ndarray:
    """Steady CSTR locus y = x (1 - x) obtained by eliminating gamma between the
    steady-state equations (see tests: at steady state, x = gamma/(1+gamma) and
    y = x/(1+gamma) = x(1-x))."""
    x = np.asarray(x, dtype=float)
    return x * (1.0 - x)


def batch_curve(x: np.ndarray) -> np.ndarray:
    """Batch (gamma = 0) trajectory y = -x log(x) from (1, 0).

    With gamma = 0: dx/dt = -x => x = exp(-t); dy/dt = x - y => the solution
    through (1, 0) is y = -x log x. Note this coincides with the analytical
    upper curve, which is one of the facts the audit must state carefully.
    """
    return analytical_upper_curve(x)


def in_triangle(q: np.ndarray, tol: float = 1e-12) -> bool:
    """Conservation domain check: 0 <= x, 0 <= y, x + y <= 1 (Y_C = 1-x-y >= 0)."""
    x, y = q[0], q[1]
    return bool(x >= -tol and y >= -tol and (1.0 - x - y) >= -tol)
