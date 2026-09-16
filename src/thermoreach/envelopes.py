"""Reference-set construction and empirical envelope comparison (Phase 3).

Reference sets, kept strictly separate by label:
  A  steady family        - long-time limits of constant-residence-time reactors,
                            traced from hot and cold initializations
  B  constant-control     - finite-time trajectories under constant residence
     transients             time, horizon-limited
  C  switching            - finite-time trajectories under admissible
     transients             piecewise-constant switching histories

Metrics: two dimensionless representations are provided (engineering-scaled and
trace-sensitive), because distance conclusions depend on the species scale and
on the treatment of trace species.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

from .controls import History
from .reactor import CSTR


# ---------------------------------------------------------------------------
# Steady family A
# ---------------------------------------------------------------------------


@dataclass
class SteadyPoint:
    gamma: float
    init_label: str
    T: float
    Y: np.ndarray
    residual: float
    t_end: float
    converged: bool


def steady_residual(c: CSTR, T: float, Y: np.ndarray, gamma: float) -> float:
    """Scaled steady residual: max|dY/dt| and |dT/dt| / (1000 K/s), scaled by
    gamma where relevant so slow-exchange cases are not spuriously 'steady'."""
    q = np.concatenate([[T], Y])
    f = c.rhs(0.0, q, gamma)
    dY_scale = max(gamma, 1.0)
    return float(max(np.max(np.abs(f[1:])) / dY_scale, abs(f[0]) / 1e3))


def trace_steady(c: CSTR, gamma: float, q0: np.ndarray, init_label: str,
                 t_end: float | None = None, rtol: float = 1e-9,
                 atol_T: float = 1e-7, atol_Y: float = 1e-15) -> SteadyPoint:
    """Time-march a constant-gamma reactor toward its steady state."""
    if t_end is None:
        # many residence times, within a bounded window
        t_end = float(np.clip(200.0 / max(gamma, 1e-2), 1e-3, 20.0))
    h = History(np.array([gamma]), np.array([t_end]))
    res = c.integrate(h, q0, method="Radau", samples_per_segment=21,
                      rtol=rtol, atol_T=atol_T, atol_Y=atol_Y)
    if not res["success"]:
        return SteadyPoint(gamma, init_label, float(q0[0]), np.asarray(q0[1:]),
                           np.nan, t_end, False)
    T, Y = float(res["states"][0, -1]), res["states"][1:, -1]
    return SteadyPoint(gamma, init_label, T, Y, steady_residual(c, T, Y, gamma),
                       t_end, True)


def steady_family(c: CSTR, gammas: np.ndarray, cold_q0: np.ndarray, hot_q0: np.ndarray,
                  rtol: float = 1e-9) -> dict:
    """Trace the steady family from cold (fresh) and hot initializations."""
    out = {"gamma": [], "cold": [], "hot": []}
    for g in gammas:
        out["gamma"].append(float(g))
        out["cold"].append(_sp_record(trace_steady(c, float(g), cold_q0, "cold")))
        out["hot"].append(_sp_record(trace_steady(c, float(g), hot_q0, "hot")))
    return out


def _sp_record(sp: SteadyPoint) -> dict:
    return {"gamma": sp.gamma, "init": sp.init_label, "T": sp.T,
            "Y": np.asarray(sp.Y).tolist(), "steady_residual": sp.residual,
            "t_end": sp.t_end, "converged": sp.converged}


# ---------------------------------------------------------------------------
# Transient families B and C
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    label: str
    init_label: str
    history: History
    times: np.ndarray
    states: np.ndarray
    success: bool
    message: str = "ok"
    nfev: int = 0
    min_Y: float = 0.0
    max_T: float = 0.0
    extrema_states: np.ndarray = field(default=None)


def run_trajectory(c: CSTR, history: History, q0: np.ndarray, init_label: str,
                   label: str, samples_per_segment: int = 41,
                   method: str = "Radau", **kw) -> Trajectory:
    res = c.integrate(history, q0, method=method, samples_per_segment=samples_per_segment,
                      **kw)
    if not res["success"]:
        return Trajectory(label, init_label, history,
                          res.get("partial_times", np.array([])),
                          res.get("partial_states", np.zeros((q0.size, 0))),
                          False, res["message"])
    ext = res.get("extrema_states", np.zeros((q0.size, 0)))
    return Trajectory(label, init_label, history, res["times"], res["states"], True,
                      "ok", sum(s["nfev"] for s in res["solver_stats"]),
                      float(res["states"][1:].min()), float(res["states"][0].max()),
                      extrema_states=ext)


def trajectory_record(tr: Trajectory, species: list[str]) -> dict:
    return {
        "label": tr.label, "init": tr.init_label, "history": tr.history.to_record(),
        "success": tr.success, "message": tr.message, "nfev": tr.nfev,
        "min_Y": tr.min_Y, "max_T": tr.max_T,
        "times": tr.times.tolist() if tr.success else [],
        "T": tr.states[0].tolist() if tr.success else [],
        "Y": {k: tr.states[1 + i].tolist() for i, k in enumerate(species)} if tr.success else {},
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class StateMetric:
    """Dimensionless state metric for (T, Y).

    Two representations:
      * engineering: T / T_scale, Y_k / Y_scale_k with Y_scale_k from the
        reference extent (never zero), linear.
      * trace-sensitive: T / T_scale, log10(Y_k / y_floor) with a declared floor.
    """

    def __init__(self, species: list[str], reference_states: np.ndarray,
                 T_scale: float = 1000.0, y_floor: float = 1e-12):
        self.species = list(species)
        self.T_scale = T_scale
        self.y_floor = y_floor
        Yref = np.maximum(reference_states[1:], 0.0)
        self.Y_scale = np.maximum(Yref.max(axis=1), 1e-9)
        self.name = "engineering+trace"

    def engineering(self, states: np.ndarray) -> np.ndarray:
        out = np.zeros_like(states)
        out[0] = states[0] / self.T_scale
        out[1:] = states[1:] / self.Y_scale[:, None]
        return out

    def trace_sensitive(self, states: np.ndarray) -> np.ndarray:
        out = np.zeros_like(states)
        out[0] = states[0] / self.T_scale
        out[1:] = np.log10(np.maximum(np.maximum(states[1:], 0.0), self.y_floor) / self.y_floor)
        return out

    def distance(self, states: np.ndarray, reference: np.ndarray, kind: str = "engineering") -> np.ndarray:
        """For each column of ``states``, the distance to the nearest reference
        point, in the chosen representation."""
        f = self.engineering if kind == "engineering" else self.trace_sensitive
        a = f(states)
        b = f(reference)
        d = np.sqrt(((a[:, :, None] - b[:, None, :]) ** 2).sum(axis=0))
        return d.min(axis=1)


# ---------------------------------------------------------------------------
# Conservation-based LP outer comparison
# ---------------------------------------------------------------------------


def lp_species_bound(c: CSTR, species_index: int) -> dict:
    """max Y_k by linear programming under nonnegativity, normalization, and
    elemental constraints. Ignores kinetics and energy feasibility by
    construction, so the bound is a relaxation and is labelled as such."""
    n_sp = c.gas.n_species
    # variables: Y (n_sp); objective: maximize Y_k
    c_obj = np.zeros(n_sp)
    c_obj[species_index] = -1.0  # linprog minimizes
    # equality: sum Y = 1 ; E.Y = b_in
    A_eq = np.vstack([np.ones((1, n_sp)), c.E])
    b_eq = np.concatenate([[1.0], c.b_in])
    bounds = [(0.0, 1.0)] * n_sp
    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    name = c.gas.species_names[species_index]
    if not res.success:
        return {"species": name, "status": res.message, "bound": None}
    return {"species": name, "lp_bound": float(-res.fun), "status": "ok",
            "note": "elemental+normalization relaxation; ignores kinetics and energy"}


def lp_temperature_bound(c: CSTR, n_grid: int = 4001) -> dict:
    """Energy-conserving temperature/composition bound: maximize T subject to
    h(T, Y) = h_in with Y in the elemental simplex, by scanning T and testing
    feasibility of the LP  max sum_k lambda_k ... Instead we directly search for
    the highest T at which a composition with h(T,Y)=h_in, E.Y=b_in, Y>=0 exists.
    Feasibility is tested by an LP at each candidate T (a linear interpolation of
    h_k(T))."""
    # For each T, h(T,Y) = sum_k Y_k h_k(T) is linear in Y; feasibility of
    # h = h_in under E.Y = b_in, sum Y = 1, Y >= 0 is an LP feasibility problem.
    Ts = np.linspace(300.0, 6000.0, n_grid)
    best_T = None
    for T in Ts:
        hk = c.species_enthalpies(float(T))
        A_eq = np.vstack([np.ones((1, c.gas.n_species)), c.E, hk[None, :]])
        b_eq = np.concatenate([[1.0], c.b_in, [c.h_in]])
        r = linprog(np.zeros(c.gas.n_species), A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0.0, 1.0)] * c.gas.n_species, method="highs")
        if r.status == 0:
            best_T = float(T)
    return {"max_T_energy_conserving": best_T,
            "note": ("h(T,Y)=h_in, E.Y=b_in, sum Y=1, Y>=0; linearized by evaluating "
                     "h_k at the candidate T. Ignores kinetics; a relaxation."),
            "mechanism_thermo_range_K": [200.0, 5000.0]}
