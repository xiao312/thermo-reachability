"""Chemistry response, declared control norms, and a cached continuous reference.

This module serves the question that follows the rank result: at permitted finite
perturbations, can the canonical constant-control family represent the generated
states well enough for the intended chemistry computation, and how well can a
local generator describe the extra variation?

Three pieces of machinery:

1. The declared *time* control norm, which is portable across segment counts::

       ||d eta||_time^2 = sum_j (duration_j / T) d eta_j^2,   H = diag(duration_j/T)

   and the input-whitened endpoint map ``A H^(-1/2)``.  Singular values of that
   map are the ones comparable across segment counts; the Euclidean-normalized
   singular values saved by phase 9 are NOT, because their parameter norm changes
   with m.  A Euclidean unit direction v corresponds to sqrt(H)-scaled directions
   here; for m equal segments a Euclidean radius r is a time-norm radius r/sqrt(m).

2. The chemistry-only map ``Phi_dt(q)`` under an adiabatic constant-pressure
   constraint - the CSTR exchange control is exactly zero in these diagnostics, so
   this is a chemical-map comparison between two states, not a new coupled CSTR
   history.  Each state's own total specific enthalpy (formation enthalpies
   included) and elemental inventory are preserved by the map; baseline mismatches
   are reported, never projected away.

3. A cached continuous reference evaluator ``B(gamma, t; q_initial)`` with the
   ORIGINAL initial state.  Each library gamma is integrated ONCE to a shared
   maximum reference time and its dense output is retained, so the repeated
   fresh-integration grid of phase 10 is not reproduced.  The library is an
   evaluation comparator for the reference search, not a source of reachable-state
   candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar

# --------------------------------------------------------------------------
# 1. declared control norms and the whitened map
# --------------------------------------------------------------------------


def hdiag(durations, T: float) -> np.ndarray:
    """H = diag(duration_j / T), the declared time-norm metric on log controls."""
    return np.asarray(durations, dtype=float) / float(T)


def time_norm(v, durations, T: float) -> float:
    """||v||_time = sqrt(sum_j (dur_j/T) v_j^2)."""
    v = np.asarray(v, dtype=float).ravel()
    return float(np.sqrt(float(np.sum(hdiag(durations, T) * v ** 2))))


def unit_time_norm_direction(v, durations, T: float) -> np.ndarray:
    """Rescale v to unit TIME norm.  For m equal segments a Euclidean unit vector
    becomes a vector of time norm sqrt(m), so a Euclidean radius r maps to the
    time-norm radius r/sqrt(m) - the meaning of past results is not changed, only
    re-expressed."""
    v = np.asarray(v, dtype=float).ravel()
    n = time_norm(v, durations, T)
    return v / n if n > 0 else v


def whitened_map(A, durations, T: float) -> np.ndarray:
    """A H^(-1/2).  Singular values of THIS map are comparable across segment
    counts because its input is measured in the declared time norm."""
    A = np.asarray(A, dtype=float)
    h = hdiag(durations, T)
    return A / np.sqrt(h)[None, :]


def radius_norm_report(theta, v_euclid, durations, T: float) -> dict:
    """Both norms and the actual segment rates for a saved Euclidean direction."""
    v_t = unit_time_norm_direction(v_euclid, durations, T)
    return {"euclidean_norm_of_saved_direction": float(np.linalg.norm(v_euclid)),
            "time_norm_of_saved_direction": time_norm(v_euclid, durations, T),
            "unit_time_norm_direction": v_t.tolist(),
            "euclidean_radius_to_time_radius_factor": 1.0 / np.sqrt(float(len(durations))),
            "actual_gamma_at_unit_time_radius": (np.asarray(theta, dtype=float)
                                                 * np.exp(v_t)).tolist(),
            "H_diag": hdiag(durations, T).tolist()}


def admissible_radius_interval(theta, v, gamma_lo: float, gamma_hi: float) -> dict:
    """Exact admissible interval of the TIME-norm radius r for gamma_j(r) =
    theta_j exp(r v_j).

    Levels are NEVER clamped: a radius outside this interval is inadmissible and
    must be trimmed from the schedule, not silently corrected.  For a MIXED-SIGN
    direction every segment contributes both an upper and a lower bound (a
    negative-v segment bounds r from ABOVE through the lower level bound and from
    BELOW through the upper level bound), so all four cases must be handled:

        v_j > 0:  r < log(gamma_hi/theta_j)/v_j   and   r > log(gamma_lo/theta_j)/v_j
        v_j < 0:  r < log(gamma_lo/theta_j)/v_j   and   r > log(gamma_hi/theta_j)/v_j
    """
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)
    r_pos = np.inf            # largest r keeping every level inside the bounds
    r_neg = -np.inf           # smallest r keeping every level inside the bounds
    bind_pos = bind_neg = None
    for j in range(theta.size):
        if v[j] > 0.0:
            upper = np.log(gamma_hi / theta[j]) / v[j]
            lower = np.log(gamma_lo / theta[j]) / v[j]
        elif v[j] < 0.0:
            # dividing by a negative level flips the inequality
            upper = np.log(gamma_lo / theta[j]) / v[j]
            lower = np.log(gamma_hi / theta[j]) / v[j]
        else:
            continue
        if upper < r_pos:
            r_pos, bind_pos = upper, (f"segment {j}: v={v[j]:g}, gamma={theta[j]:g} "
                                      f"reaches {gamma_hi if v[j] > 0 else gamma_lo:g}")
        if lower > r_neg:
            r_neg, bind_neg = lower, (f"segment {j}: v={v[j]:g}, gamma={theta[j]:g} "
                                      f"reaches {gamma_hi if v[j] < 0 else gamma_lo:g}")
    if not np.isfinite(r_pos):
        r_pos = None
    if not np.isfinite(r_neg):
        r_neg = None
    return {"r_max_positive": r_pos, "r_min_negative": r_neg,
            "binding_constraint_positive": bind_pos,
            "binding_constraint_negative": bind_neg,
            "note": ("inadmissible radii are trimmed, never clamped; a trimmed "
                     "radius is recorded as excluded with its reason")}


# --------------------------------------------------------------------------
# 2. chemistry-only map Phi_dt and its diagnostics
# --------------------------------------------------------------------------


@dataclass
class ChemistryConfig:
    method: str = "Radau"
    rtol: float = 1e-9
    atol_T: float = 1e-9
    atol_Y: float = 1e-15
    # illustrative diagnostic timesteps; NOT a verified solver requirement
    diagnostic_dt: tuple = (1e-7, 1e-6, 1e-5, 1e-4)

    def to_record(self) -> dict:
        return {"method": self.method, "rtol": self.rtol, "atol_T": self.atol_T,
                "atol_Y": self.atol_Y,
                "diagnostic_dt_s": list(self.diagnostic_dt),
                "dt_label": ("illustrative diagnostic timesteps; no target "
                             "application timestep range has been specified for "
                             "this project")}


def chemistry_source(cstr, q: np.ndarray) -> dict:
    """Homogeneous rates at a state with the exchange control OFF (gamma = 0).

    The adiabatic constant-pressure constraint makes cp dT/dt = -sum_k h_k r_k,
    which is the (negative of the) heat-release rate per unit mass.
    """
    T = float(q[0])
    Y = np.asarray(q[1:], dtype=float)
    _T, _Y, rho = cstr.state(T, Y)          # instrumented state-setting path
    omega = cstr.gas.net_production_rates           # kmol/m^3/s
    r = cstr.W * omega / rho                        # 1/s
    hk = cstr.gas.partial_molar_enthalpies / cstr.W  # J/kg at T
    cp = float(cstr.gas.cp_mass)
    heat_release = float(hk @ r)                    # J/kg/s; < 0 = exothermic
    return {"T": float(cstr.gas.T), "Y": cstr.gas.Y.tolist(),
            "density": rho, "pressure": float(cstr.gas.P),
            "cp_mass": cp,
            "net_production_rates_kmol_m3_s": omega.tolist(),
            "species_rates_1_s": r.tolist(),
            "dTdt_chemistry_K_s": -heat_release / cp,
            "heat_release_rate_J_kg_s": heat_release,
            "enthalpy_mass_J_kg": float(cstr.gas.enthalpy_mass),
            "element_mass_fractions": (cstr.E @ np.maximum(cstr.gas.Y, 0.0)).tolist()}


def chemistry_map(cstr, q: np.ndarray, dt: float, cfg: ChemistryConfig | None = None):
    """Phi_dt(q): homogeneous chemistry only, adiabatic, constant pressure.

    The CSTR exchange control is exactly zero, so this is the SAME chemistry-only
    map applied to two different states.  Preserves the state's own total specific
    enthalpy and elemental inventory; the residual of that preservation is
    reported rather than projected away.
    """
    cfg = cfg or ChemistryConfig()
    q = np.asarray(q, dtype=float)
    n = q.size
    atol = np.concatenate([[cfg.atol_T], np.full(n - 1, cfg.atol_Y)])
    src0 = chemistry_source(cstr, q)
    if dt <= 0.0:
        return {"dt": float(dt), "q_in": q.tolist(), "q_out": q.tolist(),
                "success": True, "message": "zero-step identity",
                "initial_source": src0,
                "enthalpy_mismatch_J_kg": 0.0,
                "element_mismatch_max": 0.0,
                "nfev": 0, "method": cfg.method}
    res = solve_ivp(lambda t, y: cstr.rhs(t, y, 0.0), (0.0, float(dt)), q,
                    method=cfg.method, t_eval=np.atleast_1d(float(dt)),
                    rtol=cfg.rtol,
                    atol=atol, dense_output=False)
    if not res.success:
        return {"dt": float(dt), "q_in": q.tolist(), "success": False,
                "message": res.message, "initial_source": src0,
                "nfev": int(res.nfev), "method": cfg.method}
    q_out = res.y[:, -1]
    src1 = chemistry_source(cstr, q_out)
    h0 = src0["enthalpy_mass_J_kg"]
    h1 = src1["enthalpy_mass_J_kg"]
    b0 = np.asarray(src0["element_mass_fractions"], dtype=float)
    b1 = np.asarray(src1["element_mass_fractions"], dtype=float)
    return {"dt": float(dt), "q_in": q.tolist(), "q_out": q_out.tolist(),
            "success": True, "message": res.message,
            "initial_source": src0, "final_source": src1,
            "enthalpy_mismatch_J_kg": float(h1 - h0),
            "element_mismatch_max": float(np.max(np.abs(b1 - b0))),
            "nfev": int(res.nfev), "method": cfg.method}


def chemistry_response(cstr, q: np.ndarray, q_B: np.ndarray,
                       cfg: ChemistryConfig | None = None) -> dict:
    """Full response comparison between a generated state and its nearest found
    canonical representative, under the SAME chemistry-only map.

    Reports initial source differences, finite-time future-state differences and
    chemistry-increment differences SEPARATELY (a geometric difference and a
    consequential chemical-map difference are distinct outcomes), plus the
    declared-output error metric

        E(dt, q, q_B) = max_l |Q_l(Phi_dt(q)) - Q_l(Phi_dt(q_B))| / eps_l

    with explicitly declared outputs Q_l and tolerances eps_l.  No target
    application tolerances have been supplied for this project, so a table across
    illustrative tolerance choices is returned and no universal combustion
    tolerance is invented.  Near-zero rates use an absolute-plus-relative
    denominator, never an unstable pure relative error.
    """
    cfg = cfg or ChemistryConfig()
    s0 = chemistry_source(cstr, q)
    sB = chemistry_source(cstr, q_B)

    def arr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return a - b

    dTdt = arr(s0["dTdt_chemistry_K_s"], sB["dTdt_chemistry_K_s"])
    qrr = arr(s0["heat_release_rate_J_kg_s"], sB["heat_release_rate_J_kg_s"])
    dr = arr(s0["species_rates_1_s"], sB["species_rates_1_s"])
    # absolute-plus-relative denominator for near-zero rates
    scale = np.maximum(np.abs(sB["species_rates_1_s"]), 1e-12)
    out = {"initial_source_difference": {
               "dTdt_K_s": float(dTdt),
               "heat_release_rate_J_kg_s": float(qrr),
               "species_rates_1_s": dr.tolist(),
               "species_rates_relative_to_abs_plus_rel": (dr / scale).tolist(),
               "net_production_rates_kmol_m3_s":
                   arr(s0["net_production_rates_kmol_m3_s"],
                       sB["net_production_rates_kmol_m3_s"]).tolist()},
           "state_difference": {
               "dT_K": float(q[0] - q_B[0]),
               "dY": arr(q[1:], q_B[1:]).tolist()},
           "baseline": {"h_q_J_kg": s0["enthalpy_mass_J_kg"],
                        "h_qB_J_kg": sB["enthalpy_mass_J_kg"],
                        "elemental_inventory_mismatch_max": float(np.max(np.abs(
                            arr(s0["element_mass_fractions"],
                                sB["element_mass_fractions"])))),
                        "note": ("baseline enthalpy/inventory mismatch between the "
                                 "two states is reported, not projected away")},
           "dt_results": []}
    # tolerance table: no application tolerances supplied -> illustrative choices
    tol_table = [{"label": "T 1 K / Y 1e-3", "eps_T": 1.0, "eps_Y": 1e-3},
                 {"label": "T 1 K / Y 1e-4", "eps_T": 1.0, "eps_Y": 1e-4},
                 {"label": "T 10 K / Y 1e-3", "eps_T": 10.0, "eps_Y": 1e-3}]
    for dt in cfg.diagnostic_dt:
        m_q = chemistry_map(cstr, q, dt, cfg)
        m_B = chemistry_map(cstr, q_B, dt, cfg)
        if not (m_q["success"] and m_B["success"]):
            out["dt_results"].append({"dt": float(dt), "success": False,
                                      "failures": [m_q.get("message"),
                                                   m_B.get("message")]})
            continue
        a = np.asarray(m_q["q_out"], dtype=float)
        b = np.asarray(m_B["q_out"], dtype=float)
        d = a - b
        # chemistry increments: Phi_dt(q) - q and Phi_dt(q_B) - q_B, separately
        inc_a = a - np.asarray(q, dtype=float)
        inc_b = b - np.asarray(q_B, dtype=float)
        inc = inc_a - inc_b
        row = {"dt": float(dt), "success": True,
               "future_state_dT_K": float(d[0]),
               "future_state_dY": d[1:].tolist(),
               "increment_dT_K": float(inc[0]),
               "increment_dY": inc[1:].tolist(),
               "nfev": [m_q["nfev"], m_B["nfev"]],
               "enthalpy_mismatch_J_kg": [m_q["enthalpy_mismatch_J_kg"],
                                          m_B["enthalpy_mismatch_J_kg"]],
               "E_metric": []}
        for tt in tol_table:
            eps = np.concatenate([[tt["eps_T"]], np.full(d.size - 1, tt["eps_Y"])])
            num = np.abs(d)
            # absolute-plus-relative denominator: never divide by a tiny eps alone
            # when the value itself is near zero
            den = np.maximum(eps, 1e-300)
            rel = num / den
            row["E_metric"].append({"tolerance_label": tt["label"],
                                    "E_max": float(np.max(rel)),
                                    "E_T": float(rel[0]),
                                    "E_Y_max": float(np.max(rel[1:])),
                                    "max_abs_dT_K": float(np.abs(d[0])),
                                    "max_abs_dY": float(np.max(np.abs(d[1:])))})
        out["dt_results"].append(row)
    out["tolerance_table_note"] = (
        "no target application output tolerances have been supplied for this "
        "project; the E_metric table spans explicitly illustrative choices and "
        "does not assert a universally acceptable combustion tolerance")
    return out


# --------------------------------------------------------------------------
# 3. cached continuous reference B(gamma, t; q_initial)
# --------------------------------------------------------------------------


@dataclass
class ReferenceLibrary:
    """Cached dense trajectories of the constant-control family.

    Each library gamma is integrated ONCE from q0 over [0, t_max] with dense
    output retained; every later evaluation at that gamma and any t in the
    interval is a dense-output call, not a fresh integration.  Cache identity is
    explicit (mechanism, thermodynamics, q0, gamma, tolerances, method, domain)
    so a stale cache cannot be silently reused.
    """

    cstr: object
    q0: np.ndarray
    gamma_grid: np.ndarray
    t_max: float
    t_anchor: float
    method: str = "Radau"
    rtol: float = 1e-10
    atol_T: float = 1e-9
    atol_Y: float = 1e-16
    t_grid_points: int = 161
    _sols: dict = field(default_factory=dict)
    _failures: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.q0 = np.asarray(self.q0, dtype=float)
        self.gamma_grid = np.asarray(self.gamma_grid, dtype=float)
        n = self.q0.size
        atol = np.concatenate([[self.atol_T], np.full(n - 1, self.atol_Y)])
        for g in self.gamma_grid:
            g = float(g)
            try:
                res = solve_ivp(lambda t, y: self.cstr.rhs(t, y, g),
                                (0.0, float(self.t_max)), self.q0, method=self.method,
                                t_eval=None, rtol=self.rtol, atol=atol,
                                dense_output=True)
                if res.success:
                    self._sols[g] = res.sol
                else:
                    self._failures.append({"gamma": g, "message": res.message})
            except Exception as exc:                    # noqa: BLE001
                self._failures.append({"gamma": g, "failure": repr(exc)})


# ---------------------------------------------------------------------------
# bounded local refinement of a located reference candidate
# ---------------------------------------------------------------------------


    def identity(self) -> dict:
        """What this cache is valid for - so a stale cache is never silently used."""
        return {"mechanism": self.cstr.cfg.mechanism,
                "phase": self.cstr.cfg.phase,
                "pressure": self.cstr.cfg.pressure,
                "q0_sha256_16": __import__("hashlib").sha256(
                    self.q0.tobytes()).hexdigest()[:16],
                "gamma_grid": self.gamma_grid.tolist(),
                "t_max": float(self.t_max), "t_anchor": float(self.t_anchor),
                "method": self.method, "rtol": self.rtol,
                "atol_T": self.atol_T, "atol_Y": self.atol_Y,
                "n_integrated": len(self._sols),
                "n_failed": len(self._failures),
                "failures": self._failures}

    def t_grid(self) -> np.ndarray:
        """Declared time grid: zero, the anchor time, and log-distributed
        positive times, so early times are adequately covered."""
        lo = max(self.t_max * 1e-4, 1e-12)
        log_pts = np.logspace(np.log10(lo), np.log10(self.t_max),
                              max(self.t_grid_points - 2, 1))
        return np.unique(np.concatenate([[0.0, float(self.t_anchor)], log_pts]))

    def evaluate(self, gamma: float, t: float) -> np.ndarray:
        """B(gamma, t; q0).  Exact dense output at a library gamma; log-linear
        interpolation between bracketing library gammas otherwise.  t is clamped
        to the integrated interval, and out-of-range gamma returns NaN."""
        t = float(np.clip(t, 0.0, float(self.t_max)))
        if gamma in self._sols:
            return np.asarray(self._sols[gamma](t), dtype=float)
        gs = np.array(sorted(self._sols.keys()), dtype=float)
        if gamma < gs[0] or gamma > gs[-1]:
            return np.full(self.q0.size, np.nan)
        j = int(np.searchsorted(gs, gamma)) - 1
        j = max(0, min(j, gs.size - 2))
        ga, gb = float(gs[j]), float(gs[j + 1])
        w = np.log(gamma / ga) / np.log(gb / ga)
        return (1.0 - w) * self._sols[ga](t) + w * self._sols[gb](t)

    def evaluate_fresh(self, gamma: float, t: float, method: str | None = None,
                       rtol: float | None = None, atol_T: float | None = None,
                       atol_Y: float | None = None):
        """Independent recomputation of B(gamma, t; q0) by a fresh integration,
        for verifying a located candidate at tighter tolerances or another
        method."""
        n = self.q0.size
        atol = np.concatenate([[atol_T or self.atol_T],
                               np.full(n - 1, atol_Y or self.atol_Y)])
        res = solve_ivp(lambda tt, y: self.cstr.rhs(tt, y, float(gamma)),
                        (0.0, float(t)), self.q0, method=method or self.method,
                        t_eval=[float(t)], rtol=rtol or self.rtol, atol=atol,
                        dense_output=False)
        if not res.success:
            return {"success": False, "message": res.message, "nfev": int(res.nfev)}
        return {"success": True, "q_B": res.y[:, -1].tolist(),
                "nfev": int(res.nfev), "method": method or self.method,
                "rtol": rtol or self.rtol}

    # ------------------------------------------------------------------
    def coarse_min(self, q_target: np.ndarray, W: np.ndarray) -> dict:
        """Coarse minimum of ||W(q_target - B(gamma, t))|| over the library gamma
        grid and the declared time grid, then a continuous t refinement at the
        best gammas.  Multiple starts are retained, including the anchor-near
        start and library-boundary cases."""
        q_target = np.asarray(q_target, dtype=float)
        W = np.asarray(W, dtype=float)
        tg = self.t_grid()
        starts = []
        for g in sorted(self._sols.keys()):
            Bt = np.stack([self.evaluate(g, t) for t in tg], axis=1)   # (n, n_t)
            d = W[:, None] * (q_target[:, None] - Bt)
            n = np.linalg.norm(d, axis=0)
            i = int(np.argmin(n))
            # continuous t refinement on the dense output of this gamma
            lo = tg[max(i - 1, 0)] if i > 0 else 0.0
            hi = tg[min(i + 1, tg.size - 1)]
            r = minimize_scalar(lambda t: np.linalg.norm(W * (q_target - self.evaluate(g, t))),
                                bracket=None, bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-16})
            t_best = float(r.x) if r.success else float(tg[i])
            d_best = float(r.fun) if r.success else float(n[i])
            starts.append({"gamma": g, "t": t_best, "dist_scaled": d_best,
                           "t_grid_argmin": float(tg[i]),
                           "dist_at_grid_argmin": float(n[i]),
                           "t_refinement_success": bool(r.success)})
        starts.sort(key=lambda s: s["dist_scaled"])
        best = starts[0]
        gamma_lo_edge = float(sorted(self._sols.keys())[0])
        gamma_hi_edge = float(sorted(self._sols.keys())[-1])
        return {"best_gamma": best["gamma"], "best_t": best["t"],
                "dist_scaled_upper_estimate": best["dist_scaled"],
                "n_starts": len(starts), "starts_kept": starts[:8],
                "all_starts_sorted_by_distance": [s["gamma"] for s in starts],
                "gamma_domain": [gamma_lo_edge, gamma_hi_edge],
                "t_domain": [0.0, float(self.t_max)],
                "best_at_gamma_edge": bool(np.isclose(best["gamma"], gamma_lo_edge)
                                           or np.isclose(best["gamma"], gamma_hi_edge)),
                "best_at_t_edge": bool(np.isclose(best["t"], 0.0)
                                       or np.isclose(best["t"], float(self.t_max))),
                "t_grid": {"points": tg.tolist(),
                           "contains_zero": bool(tg[0] == 0.0),
                           "contains_anchor": bool(np.any(np.isclose(tg, self.t_anchor))),
                           "log_distributed": True}}


def integrate_constant(cstr, q0, gamma: float, t: float, method: str = "Radau",
                       rtol: float = 1e-11, atol_T: float = 1e-10,
                       atol_Y: float = 1e-17, dense: bool = True):
    """B(gamma, t; q0) by a single fresh integration of the constant control.

    A zero horizon is the identity: B(gamma, 0; q0) = q0.  It is returned
    directly rather than handed to the solver, because a zero-length integration
    interval is a degenerate case for the stepper."""
    q0 = np.asarray(q0, dtype=float)
    if float(t) <= 0.0:
        out = {"success": True, "q": q0.copy(), "nfev": 0, "method": method,
               "rtol": rtol}
        if dense:
            out["sol"] = lambda tt, _q=q0: _q.copy()
        return out
    n = q0.size
    atol = np.concatenate([[atol_T], np.full(n - 1, atol_Y)])
    res = solve_ivp(lambda tt, y: cstr.rhs(tt, y, float(gamma)),
                    (0.0, float(t)), q0, method=method,
                    t_eval=np.atleast_1d(float(t)),
                    rtol=rtol, atol=atol, dense_output=dense)
    if not res.success:
        return {"success": False, "message": res.message, "nfev": int(res.nfev)}
    out = {"success": True, "q": res.y[:, -1].copy(), "nfev": int(res.nfev),
           "method": method, "rtol": rtol}
    if dense:
        out["sol"] = res.sol
    return out


def refine_reference(library: "ReferenceLibrary", q_target: np.ndarray, W: np.ndarray,
                     coarse: dict, cstr, *, refine_rtol: float = 1e-11,
                     refine_atol_T: float = 1e-10, refine_atol_Y: float = 1e-17,
                     refine_method: str = "Radau",
                     gamma_half_width_log: float | None = None,
                     t_max_refine: float | None = None,
                     max_iterations: int = 20,
                     initial_damping: float = 1e-6,
                     dist_tolerance: float = 1e-14) -> dict:
    """Bounded local refinement of a coarse reference candidate by FRESH
    integration at tighter tolerances.

    LEVENBERG-MARQUARDT (damped Gauss-Newton) on the scaled residual

        r(gamma, t) = W (q_target - B(gamma, t; q0)),      r in R^n

    whose Jacobian is available almost exactly:  dr/dt = -W F(q_B, gamma) is the
    ODE right-hand side at the located state, and dr/dlog(gamma) is a central
    difference with two fresh integrations.  Each iteration therefore costs three
    integrations, and the method converges quadratically in the valley, which is
    what a coupled 2-D problem needs: a coordinate-wise sweep stalls when the
    coarse candidate is wrong in BOTH coordinates at once (a wrong gamma shifts
    the optimal t and vice versa, because the family direction changes with
    both).  Nelder-Mead and a deterministic zoom grid were both tried first and
    stalled at 1e-4 / 1e-2 respectively inside a bounded evaluation budget.

    Steps are clipped to the declared reference domain, and a step that leaves
    the domain or fails to reduce the distance is rejected with increased
    damping; a boundary minimizer is DETECTED and flagged rather than silently
    penalized.  Every objective evaluation is a fresh integration, so the refined
    distance is an independent recomputation, not a reuse of the library
    interpolation.  The refined distance is an UPPER ESTIMATE of the infimum: a
    local method can overestimate the minimum over the domain, so global
    nonmembership is not claimed.
    """
    q_target = np.asarray(q_target, dtype=float)
    W = np.asarray(W, dtype=float)
    g0 = float(coarse["best_gamma"])
    t0 = float(coarse["best_t"])
    gs = np.array(sorted(library._sols.keys()), dtype=float)
    log_spacing = float(np.median(np.diff(np.log(gs)))) if gs.size > 1 else 0.1
    half = gamma_half_width_log if gamma_half_width_log is not None \
        else 1.5 * log_spacing
    t_cap = float(t_max_refine if t_max_refine is not None else library.t_max)
    log_g0 = float(np.clip(np.log(g0), 0.0, None))
    log_lo, log_hi = log_g0 - half, log_g0 + half
    box_lo = np.array([log_lo, 0.0])
    box_hi = np.array([log_hi, t_cap])

    n_eval = 0

    def evaluate(log_g: float, t: float):
        nonlocal n_eval
        n_eval += 1
        r = integrate_constant(cstr, library.q0, float(np.exp(log_g)), t,
                               method=refine_method, rtol=refine_rtol,
                               atol_T=refine_atol_T, atol_Y=refine_atol_Y,
                               dense=False)
        if not r["success"]:
            return None, np.inf
        q = np.asarray(r["q"], dtype=float)
        return q, float(np.linalg.norm(W * (q_target - q)))

    x = np.array([float(np.clip(log_g0, log_lo, log_hi)),
                  float(np.clip(t0, 0.0, t_cap))])
    q_B, dist = evaluate(float(x[0]), float(x[1]))
    if q_B is None:
        return {"success": False, "message": "initial point failed to integrate",
                "coarse": coarse, "n_fresh_integrations": n_eval}

    lam = float(initial_damping)
    iterations = []
    for _ in range(max(int(max_iterations), 1)):
        # residual Jacobian: columns are -W dB/dt and -W dB/dlog(gamma)
        tan = local_family_tangent_at(cstr, library.q0, q_B, float(np.exp(x[0])),
                                      float(x[1]), W, dlog=1e-4,
                                      method=refine_method, rtol=refine_rtol,
                                      atol_T=refine_atol_T, atol_Y=refine_atol_Y)
        n_eval += 2                  # the two finite-difference integrations
        if not tan["success"]:
            iterations.append({"x": x.tolist(), "dist_scaled": dist,
                               "tangent_failure": tan.get("failures")})
            break
        M = np.asarray(tan["tangent_matrix_scaled"], dtype=float)   # (n, 2)
        if M.shape[1] != 2 or not np.all(np.isfinite(M)):
            iterations.append({"x": x.tolist(), "dist_scaled": dist,
                               "note": "tangent matrix not (n, 2) or non-finite"})
            break
        # local_family_tangent_at returns columns (dB/dt, dB/dlog gamma), but the
        # parameter order here is x = (log gamma, t): the Jacobian of B with
        # respect to x must have its columns in the parameter order.
        J = M[:, [1, 0]]
        r_vec = W * (q_target - q_B)
        accepted = False
        for _trial in range(12):
            try:
                delta = np.linalg.solve(J.T @ J + lam * np.eye(2), J.T @ r_vec)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            x_new = np.clip(x + delta, box_lo, box_hi)
            q_new, d_new = evaluate(float(x_new[0]), float(x_new[1]))
            if q_new is not None and d_new <= dist:
                step = float(np.linalg.norm(x_new - x))
                iterations.append({
                    "x": x_new.tolist(), "dist_scaled": d_new,
                    "previous_dist_scaled": dist,
                    "damping": lam, "step_norm": step,
                    "accepted": True,
                    "clipped_to_domain": bool(np.any(x_new != x + delta))})
                x, q_B, dist = x_new, q_new, d_new
                lam = max(lam / 10.0, 1e-14)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            iterations.append({"x": x.tolist(), "dist_scaled": dist,
                               "accepted": False,
                               "note": "no damping reduced the distance; stopped"})
            break
        if dist <= float(dist_tolerance):
            break
        # convergence: the accepted step has become smaller than the solver's
        # own resolution, so further iterations cannot move the located point
        if iterations and float(iterations[-1].get("step_norm", 1.0)) < 1e-14:
            break

    gamma_best = float(np.exp(float(x[0])))
    t_best = float(x[1])
    # final independent evaluation of the best located point
    final = integrate_constant(cstr, library.q0, gamma_best, t_best,
                               method=refine_method, rtol=refine_rtol,
                               atol_T=refine_atol_T, atol_Y=refine_atol_Y,
                               dense=False)
    n_eval += 1
    if not final["success"]:
        return {"success": False, "message": final["message"],
                "coarse": coarse, "n_fresh_integrations": n_eval,
                "iterations": iterations}
    q_B = np.asarray(final["q"], dtype=float)
    dist = float(np.linalg.norm(W * (q_target - q_B)))
    return {"success": True, "gamma_best": gamma_best, "t_best": t_best,
            "q_B": q_B.tolist(), "dist_scaled_upper_estimate": dist,
            "n_fresh_integrations": n_eval,
            "refine_settings": {"method": refine_method, "rtol": refine_rtol,
                                 "atol_T": refine_atol_T, "atol_Y": refine_atol_Y,
                                 "gamma_bracket_log": [log_lo, log_hi],
                                 "t_bracket": [0.0, t_cap],
                                 "max_iterations": max_iterations,
                                 "initial_damping": initial_damping,
                                 "dist_tolerance": dist_tolerance},
            "boundary_activity": {
                "gamma_at_bracket_edge": bool(np.isclose(np.log(gamma_best), log_lo)
                                                or np.isclose(np.log(gamma_best), log_hi)),
                "t_at_domain_edge": bool(np.isclose(t_best, 0.0)
                                          or np.isclose(t_best, t_cap)),
                "note": ("a bracket/domain edge is flagged so the caller can "
                         "expand the reference domain once before interpreting "
                         "the result")},
            "iterations": iterations,
            "coarse": {k: v for k, v in coarse.items() if k != "starts_kept"}}


def local_family_tangent_at(cstr, q_ref0: np.ndarray, q_B: np.ndarray, gamma: float,
                            t: float, W: np.ndarray, *, dlog: float = 1e-3,
                            method: str = "Radau", rtol: float = 1e-11,
                            atol_T: float = 1e-10, atol_Y: float = 1e-17) -> dict:
    """Scaled tangent space of the constant-control family at (gamma, t).

    dB/dt = F(q_B, gamma) is exact.  dB/dlog(gamma) is a central difference with
    fresh integration from ``q_ref0`` (the ORIGINAL initial state).  The basis uses
    the single rank-revealing helper.
    """
    from .sensitivity import svd_basis

    q_B = np.asarray(q_B, dtype=float)
    W = np.asarray(W, dtype=float)
    v_t = np.asarray(cstr.rhs(0.0, q_B, float(gamma)), dtype=float)   # exact dB/dt
    Bp = integrate_constant(cstr, q_ref0, float(gamma) * np.exp(dlog), t,
                            method=method, rtol=rtol, atol_T=atol_T,
                            atol_Y=atol_Y, dense=False)
    Bm = integrate_constant(cstr, q_ref0, float(gamma) * np.exp(-dlog), t,
                            method=method, rtol=rtol, atol_T=atol_T,
                            atol_Y=atol_Y, dense=False)
    if not (Bp["success"] and Bm["success"]):
        return {"success": False,
                "failures": [Bp.get("message"), Bm.get("message")]}
    v_g = (np.asarray(Bp["q"], dtype=float)
           - np.asarray(Bm["q"], dtype=float)) / (2.0 * dlog)
    M = W[:, None] * np.vstack([v_t, v_g]).T
    basis = svd_basis(M, rel_tol=1e-10)
    return {"success": True, "basis": basis.to_record(),
            "dBdt": v_t.tolist(),
            "dBdloggamma": v_g.tolist(),
            "tangent_matrix_scaled": M.tolist(),
            "rank": basis.rank,
            "conditioning": basis.conditioning,
            "dlog": dlog}


def normal_residual(cstr, q_ref0: np.ndarray, q_target: np.ndarray, q_B: np.ndarray,
                    gamma: float, t: float, W: np.ndarray, **kw) -> dict:
    """Decompose e = q_target - q_B into the local family tangent plane and its
    NORMAL complement, re-fitting the curved reference AT the located point
    instead of assuming the anchor normal basis stays correct."""
    from .sensitivity import svd_basis

    q_target = np.asarray(q_target, dtype=float)
    q_B = np.asarray(q_B, dtype=float)
    W = np.asarray(W, dtype=float)
    tan = local_family_tangent_at(cstr, q_ref0, q_B, gamma, t, W, **kw)
    if not tan["success"]:
        return {"available": False, "reason": "local tangent unavailable",
                "tangent_failure": tan.get("failures")}
    Q = svd_basis(np.asarray(tan["tangent_matrix_scaled"], dtype=float),
                  rel_tol=1e-10).vectors
    e = W * (q_target - q_B)
    if Q is None:
        return {"available": True, "tangent_rank": 0,
                "normal_residual_scaled": e.tolist(),
                "normal_residual_norm": float(np.linalg.norm(e)),
                "tangent_residual_norm": 0.0, "basis": tan}
    P = Q @ Q.T
    nrm = (np.eye(P.shape[0]) - P) @ e
    return {"available": True, "tangent_rank": int(Q.shape[1]),
            "normal_residual_scaled": nrm.tolist(),
            "normal_residual_norm": float(np.linalg.norm(nrm)),
            "tangent_residual_norm": float(np.linalg.norm(P @ e)),
            "total_residual_norm": float(np.linalg.norm(e)),
            "basis": tan}
