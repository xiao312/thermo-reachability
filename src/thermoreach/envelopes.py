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
    resid = steady_residual(c, T, Y, gamma)
    # Steady acceptance is residual-based (audit A10): a successful ODE
    # integration is NOT a steady state. The residual is compared against
    # physical scales - source magnitude and exchange rate - so a slow-burning
    # but genuinely unsteady endpoint is not accepted as steady.
    if not np.isfinite(resid):
        converged = False
    else:
        c.gas.TPY = T, c.p, np.maximum(Y, 0.0)
        rho = float(c.gas.density)
        r_scale = float(np.max(np.abs(c.W * c.gas.net_production_rates / rho)))
        tol = max(1e-9 * max(gamma, 1.0), 1e-8 * r_scale, 1e-12)
        converged = bool(resid < tol)
    return SteadyPoint(gamma, init_label, T, Y, resid, t_end, converged)


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
            "t_end": sp.t_end, "converged": sp.converged,
            "accepted": sp.converged}


def steady_accepted(fam: dict) -> dict:
    """Split a steady family into accepted (residual-verified) and unresolved
    points (audit A10). Unresolved points are kept for auditability but must not
    enter the steady reference library A."""
    acc, unres = [], []
    for key in ("cold", "hot"):
        for p in fam[key]:
            (acc if p.get("accepted") else unres).append(p)
    return {"accepted": acc, "unresolved": unres,
            "n_accepted": len(acc), "n_unresolved": len(unres)}


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
    extrema: list = field(default_factory=list)


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
                      extrema_states=ext, extrema=res.get("extrema", []))


def trajectory_record(tr: Trajectory, species: list[str]) -> dict:
    return {
        "label": tr.label, "init": tr.init_label, "history": tr.history.to_record(),
        "success": tr.success, "message": tr.message, "nfev": tr.nfev,
        "min_Y": tr.min_Y, "max_T": tr.max_T,
        "times": tr.times.tolist() if tr.success else [],
        "T": tr.states[0].tolist() if tr.success else [],
        "Y": {k: tr.states[1 + i].tolist() for i, k in enumerate(species)} if tr.success else {},
        # Persist the extrema used by the geometry (audit A13): the analysis cloud
        # is the trajectory grid UNION the resolved extrema, so the saved record
        # must contain them to be reconstructible without rerunning.
        "extrema": ([{"variable": e["variable"], "kind": e["kind"],
                       "t_abs": e["t_abs"], "gamma": e["gamma"],
                       "state": {"T": float(e["state"][0]),
                                  **{k: float(e["state"][1 + i])
                                     for i, k in enumerate(species)}}}
                      for e in tr.extrema] if tr.extrema else []),
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


def thermo_valid_range(c: CSTR) -> tuple[float, float]:
    """Common NASA-interval overlap across all species of the loaded mechanism.

    The pinned h2o2 mechanism's all-species overlap is [300, 3500] K, not
    [200, 5000] K (audit A11): the 200 K and 5000 K figures are the union of
    per-species interval endpoints, not the common interval.
    """
    lo, hi = -np.inf, np.inf
    for k in range(c.gas.n_species):
        # Cantera exposes the two NASA7 interval bounds per species.
        tl = c.gas.species(k).thermo.min_temp
        th = c.gas.species(k).thermo.max_temp
        lo, hi = max(lo, float(tl)), min(hi, float(th))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = 300.0, 3500.0
    return float(lo), float(hi)


def h_min_over_polytope(c: CSTR, T: float) -> tuple[float, np.ndarray | None, float]:
    """min_Y h_k(T).Y subject to sum Y = 1, E.Y = b_in, Y >= 0.

    Returns (h_min, argmin Y, status). h_min(T) is the lowest enthalpy any
    feed-element-compatible composition can have at temperature T.
    """
    hk = c.species_enthalpies(float(T))
    n_sp = c.gas.n_species
    A_eq = np.vstack([np.ones((1, n_sp)), c.E])
    b_eq = np.concatenate([[1.0], c.b_in])
    res = linprog(hk, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, 1.0)] * n_sp, method="highs")
    if res.status != 0:
        return float("nan"), None, float(res.status)
    return float(res.fun), np.asarray(res.x), 0.0


def lp_temperature_bound(c: CSTR, n_grid: int = 4001) -> dict:
    """Energy-conserving temperature bound via BRACKETING, not a grid maximum.

    For each T, the minimum possible enthalpy over the elemental polytope is
    h_min(T) = min_Y h_k(T).Y. Any feed-compatible composition at temperature T
    has enthalpy >= h_min(T). Since h_in is fixed, a state with h = h_in at
    temperature T can exist only if h_min(T) <= h_in. Under the physically
    justified assumption that h_min(T) increases with T (mixture heat capacity
    is positive), the set of feasible T is an interval and its upper edge is
    bracketed by the sign change of h_min(T) - h_in.

    The earlier implementation recorded the highest FEASIBLE GRID POINT, which is
    an attained value in the relaxed problem, not a containing upper bound on the
    continuous domain (audit A11). It is retained below as a historical estimate.
    """
    lo, hi = thermo_valid_range(c)
    # Scan within the valid thermodynamic range (no extrapolation to 6000 K).
    Ts = np.linspace(lo, hi, n_grid)
    best_grid = None
    hmin = np.full(Ts.size, np.nan)
    for i, T in enumerate(Ts):
        hm, _, status = h_min_over_polytope(c, float(T))
        hmin[i] = hm
        if status == 0 and hm <= c.h_in:
            best_grid = float(T)             # historical grid estimate
    # Bracket the sign change of h_min(T) - h_in by bisection on the LP values
    feasible = np.where(np.isfinite(hmin) & (hmin <= c.h_in))[0]
    bracket = None
    if feasible.size:
        i_hi = int(feasible[-1])
        if i_hi + 1 < Ts.size and np.isfinite(hmin[i_hi + 1]):
            # last feasible grid point to first infeasible one
            bracket = [float(Ts[i_hi]), float(Ts[i_hi + 1])]
        else:
            bracket = [float(Ts[max(i_hi - 1, 0)]), float(Ts[i_hi])]
    return {
        "grid_feasible_max_T_HISTORICAL": best_grid,
        "feasible_T_bracket": bracket,
        "h_min_at_bracket_hi": (float(hmin[np.searchsorted(Ts, bracket[1])])
                                 if bracket is not None else None),
        "h_in": c.h_in,
        "n_grid": n_grid,
        "note": ("A state with h=h_in at temperature T requires h_min(T) <= h_in, "
                 "where h_min(T) = min_Y h_k(T).Y over the elemental polytope. "
                 "Under the positive-heat-capacity assumption the feasible set is "
                 "an interval; `feasible_T_bracket` contains its upper edge. The "
                 "true continuous maximum lies inside the bracket. This is an "
                 "ordinary numerical bracket, NOT an interval-certified bound. "
                 "Kinetics and transport are ignored (a relaxation)."),
        "mechanism_thermo_range_K": [lo, hi],
        "thermo_range_note": ("common NASA7 interval overlap across all species; "
                              "the earlier [200, 5000] K was the union of "
                              "per-species endpoints, not the overlap"),
    }
