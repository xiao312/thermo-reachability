"""Detailed-chemistry adiabatic fixed-pressure CSTR (well-stirred reactor).

Model (single-phase ideal gas, one temperature, prescribed pressure):

    dY_k/dt = r_k(T,p,Y) + gamma(t) * (Y_in,k - Y_k)
    cp * dT/dt = -sum_k h_k(T) * r_k + gamma(t) * (h_in - sum_k h_k(T) * Y_in,k)

with the chemical mass source  r_k = W_k * omega_dot_k / rho  [1/s], h_k the
species-specific (J/kg) enthalpy, and h_in = sum_k h_k(T_in) * Y_in,k.

Equivalent form verified numerically in tests:

    cp * dT/dt = gamma(t) * (h_in - h) - sum_k h_k(T) * dY_k/dt,
    h = sum_k Y_k h_k(T).

Reactor realization: constant pressure, constant mass, variable volume, with
*matched* inlet and outlet mass flows mdot_in = mdot_out = gamma(t) * m. Fixed
pressure, fixed mass and fixed volume cannot all be imposed independently
during arbitrary reacting transients, so this realization is used for both the
custom RHS and the Cantera ReactorNet cross-check.

Exact open-reactor balances (derived from the elemental and enthalpy balances;
verified in tests):

    Gamma(t) = int_0^t gamma(s) ds
    b(t)     = b_in + exp(-Gamma) * (b0 - b_in),   b = E . Y
    h(t)     = h_in + exp(-Gamma) * (h0 - h_in)

These hold regardless of the chemical source, because reaction conserves
elements and the reactor is adiabatic with no shaft work at constant pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

import cantera as ct

from .controls import History
from .io_utils import sha256_file


@dataclass
class ReactorConfig:
    mechanism: str = "h2o2.yaml"
    phase: str | None = None
    pressure: float = 101325.0
    inlet_temperature: float = 900.0
    fuel: str = "H2"
    oxidizer: str = "O2:1,N2:3.76"
    equivalence_ratio: float = 1.0
    # Reactor realization: constant mass m0 (kg) with matched in/out flows.
    reactor_mass: float = 1.0
    # Integration defaults; acceptance thresholds set in config, not retrofitted.
    rtol: float = 1e-8
    atol_T: float = 1e-6
    atol_Y: float = 1e-14
    cgroup: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "mechanism": self.mechanism, "pressure": self.pressure,
            "inlet_temperature": self.inlet_temperature, "fuel": self.fuel,
            "oxidizer": self.oxidizer, "equivalence_ratio": self.equivalence_ratio,
            "reactor_mass": self.reactor_mass, "rtol": self.rtol,
            "atol_T": self.atol_T, "atol_Y": self.atol_Y, "phase": self.phase,
        }


def mechanism_record(mechanism: str, gas: ct.Solution) -> dict:
    """Provenance for the loaded mechanism: source path, hash, species, reactions."""
    from pathlib import Path
    path = None
    for d in ct.get_data_directories():
        cand = Path(d) / mechanism
        if cand.exists():
            path = cand
            break
    rec = {
        "mechanism": mechanism,
        "path": str(path) if path else None,
        "sha256": sha256_file(path) if path else None,
        "phase": gas.name,
        "species": list(gas.species_names),
        "elements": list(gas.element_names),
        "n_reactions": int(gas.n_reactions),
    }
    if hasattr(gas, "input_header"):
        try:
            rec["input_header"] = dict(gas.input_header)
        except Exception:
            rec["input_header"] = None
    return rec


def elemental_matrix(gas: ct.Solution) -> np.ndarray:
    """E[e,k] = atom_count(e,k) * atomic_weight(e) / molecular_weight(k).

    With Y in mass fractions, b = E.Y is the element mass fraction vector.
    """
    n_el, n_sp = gas.n_elements, gas.n_species
    E = np.zeros((n_el, n_sp))
    for k in range(n_sp):
        for e in range(n_el):
            n = gas.n_atoms(k, gas.element_name(e))
            if n:
                E[e, k] = n * gas.atomic_weight(e) / gas.molecular_weights[k]
    return E


class CSTR:
    """Custom-RHS adiabatic fixed-pressure CSTR with Cantera thermodynamics."""

    def __init__(self, cfg: ReactorConfig):
        self.cfg = cfg
        self.gas = ct.Solution(cfg.mechanism, cfg.phase) if cfg.phase else ct.Solution(cfg.mechanism)
        # Fresh feed composition from the library's composition utilities.
        # Temperature/pressure first, then set_equivalence_ratio resolves and
        # normalizes Y_in in place.
        self.gas.TP = cfg.inlet_temperature, cfg.pressure
        self.gas.set_equivalence_ratio(cfg.equivalence_ratio, cfg.fuel, cfg.oxidizer)
        self.Y_in = self.gas.Y.copy()
        self.T_in = cfg.inlet_temperature
        self.p = cfg.pressure
        self.E = elemental_matrix(self.gas)
        self.b_in = self.E @ self.Y_in
        self.h_in = float(self.gas.enthalpy_mass)  # at (T_in, Y_in)
        self.hk_in = self.species_enthalpies(self.T_in)  # h_k(T_in)
        self.W = self.gas.molecular_weights
        self.mech = mechanism_record(cfg.mechanism, self.gas)
        self._set_calls = 0
        self._max_y_renorm = 0.0  # state-setter audit: largest Y renormalization

    # ------------------------------------------------------------------
    # thermodynamic helpers
    # ------------------------------------------------------------------
    def species_enthalpies(self, T: float) -> np.ndarray:
        """h_k(T) in J/kg for all species (formation enthalpy included)."""
        self.gas.TP = T, self.p
        return self.gas.partial_molar_enthalpies / self.gas.molecular_weights

    def state(self, T: float, Y: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Set an (T, p, Y) state. Records the renormalization applied by the
        setter so the numerical policy is auditable; the raw accepted state is
        preserved by the caller."""
        Y = np.asarray(Y, dtype=float)
        s = Y.sum()
        self._max_y_renorm = max(self._max_y_renorm, float(abs(s - 1.0)))
        self.gas.TPY = T, self.p, np.maximum(Y, 0.0)
        return float(self.gas.T), self.gas.Y.copy(), float(self.gas.density)

    # ------------------------------------------------------------------
    # right-hand side
    # ------------------------------------------------------------------
    def rhs(self, t: float, q: np.ndarray, gamma: float) -> np.ndarray:
        """q = [T, Y_1..Y_K]. Uses the expanded energy form; the equivalent form
        is checked numerically by ``check_energy_forms``."""
        T = float(q[0])
        Y = np.asarray(q[1:])
        self.gas.TPY = T, self.p, np.maximum(Y, 0.0)
        rho = float(self.gas.density)
        omega = self.gas.net_production_rates            # kmol/m^3/s
        r = self.W * omega / rho                          # 1/s
        hk = self.gas.partial_molar_enthalpies / self.W   # J/kg at T
        cp = float(self.gas.cp_mass)
        h_mix_in_at_T = float(hk @ self.Y_in)             # sum h_k(T) Y_in,k
        dT = (-float(hk @ r) + gamma * (self.h_in - h_mix_in_at_T)) / cp
        dY = r + gamma * (self.Y_in - Y)
        return np.concatenate([[dT], dY])

    def rhs_alt(self, t: float, q: np.ndarray, gamma: float) -> np.ndarray:
        """Alternative energy form: cp dT/dt = gamma (h_in - h) - sum h_k dY_k/dt."""
        T = float(q[0])
        Y = np.asarray(q[1:])
        self.gas.TPY = T, self.p, np.maximum(Y, 0.0)
        rho = float(self.gas.density)
        omega = self.gas.net_production_rates
        r = self.W * omega / rho
        hk = self.gas.partial_molar_enthalpies / self.W
        cp = float(self.gas.cp_mass)
        dY = r + gamma * (self.Y_in - Y)
        h = float(hk @ Y)
        dT = (gamma * (self.h_in - h) - float(hk @ dY)) / cp
        return np.concatenate([[dT], dY])

    # ------------------------------------------------------------------
    # exact balances
    # ------------------------------------------------------------------
    def gamma_integral(self, history: History, times: np.ndarray) -> np.ndarray:
        """Gamma(t) = int_0^t gamma(s) ds for a piecewise-constant history."""
        times = np.asarray(times, dtype=float)
        out = np.zeros_like(times)
        t_acc, g_acc = 0.0, 0.0
        for val, dur in zip(history.values, history.durations):
            seg_end = t_acc + dur
            mask = times > t_acc
            out[mask] += g_acc + float(val) * np.minimum(times[mask] - t_acc, dur)
            g_acc += float(val) * dur
            t_acc = seg_end
        return out

    def exact_balances(self, q0: np.ndarray, history: History, times: np.ndarray) -> dict:
        """Predicted b(t), h(t), and T(t) from the exponential mixing law."""
        times = np.asarray(times, dtype=float)
        Gamma = self.gamma_integral(history, times)
        dec = np.exp(-Gamma)
        T0 = float(q0[0])
        Y0 = np.asarray(q0[1:])
        b0 = self.E @ Y0
        hk0 = self.species_enthalpies(T0)
        h0 = float(hk0 @ Y0)
        b_pred = self.b_in[:, None] + dec[None, :] * (b0[:, None] - self.b_in[:, None])
        h_pred = self.h_in + dec * (h0 - self.h_in)
        return {"Gamma": Gamma, "b_pred": b_pred, "h_pred": h_pred,
                "b0": b0, "h0": h0, "h_in": self.h_in, "b_in": self.b_in}

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def integrate(self, history: History, q0: np.ndarray, method: str = "Radau",
                  samples_per_segment: int = 101, rtol: float | None = None,
                  atol_T: float | None = None, atol_Y: float | None = None,
                  restart_at_switches: bool = True,
                  record_extrema: bool = True) -> dict:
        """Integrate segment by segment, restarting the solver at each control
        discontinuity so the switch is never smoothed by output interpolation.
        Accumulated physical time is preserved across restarts.

        ``record_extrema`` evaluates the dense solution on a grid that resolves
        the exchange timescale 1/gamma, so narrow peaks (e.g. a rapid cooling dip
        at large gamma) are captured even when the stored trajectory grid is
        coarser. Extremal states are returned separately from the grid states."""
        rtol = rtol or self.cfg.rtol
        atol_T = atol_T or self.cfg.atol_T
        atol_Y = atol_Y or self.cfg.atol_Y
        atol = np.concatenate([[atol_T], np.full(self.gas.n_species, atol_Y)])
        q = np.asarray(q0, dtype=float).copy()
        ts, ys, stats, extrema = [], [], [], []
        t_accum = 0.0
        for val, dur in zip(history.values, history.durations):
            g = float(val)
            local = np.linspace(0.0, dur, samples_per_segment)
            sol = solve_ivp(self.rhs, (0.0, dur), q, method=method, args=(g,),
                            t_eval=local, rtol=rtol, atol=atol, jac=None,
                            dense_output=record_extrema)
            if not sol.success:
                return {"success": False, "message": sol.message,
                        "partial_times": np.concatenate(ts) if ts else np.array([]),
                        "partial_states": np.concatenate(ys, axis=1) if ys else np.zeros((q.size, 0)),
                        "failed_at_gamma": g, "failed_segment_start": t_accum,
                        "last_state": q.tolist()}
            ts.append(t_accum + sol.t)
            ys.append(sol.y)
            if record_extrema:
                # resolve the exchange timescale, bounded to keep cost sane
                dt_ext = max(min(dur / 200.0, 1.0 / (100.0 * max(g, 1.0))), 1e-7)
                n_ext = int(min(max(dur / dt_ext, 2), 200_000))
                t_ext = np.linspace(0.0, dur, n_ext)
                y_ext = sol.sol(t_ext)
                for j in range(q.size):
                    i_max, i_min = int(np.argmax(y_ext[j])), int(np.argmin(y_ext[j]))
                    extrema.append({"variable": "T" if j == 0 else self.gas.species_names[j - 1],
                                    "kind": "max", "t_local": float(t_ext[i_max]),
                                    "t_abs": float(t_accum + t_ext[i_max]),
                                    "gamma": g,
                                    "state": y_ext[:, i_max].tolist()})
                    extrema.append({"variable": "T" if j == 0 else self.gas.species_names[j - 1],
                                    "kind": "min", "t_local": float(t_ext[i_min]),
                                    "t_abs": float(t_accum + t_ext[i_min]),
                                    "gamma": g,
                                    "state": y_ext[:, i_min].tolist()})
            stats.append({"gamma": g, "duration": dur, "nfev": sol.nfev,
                          "njev": sol.njev, "nlu": sol.nlu, "steps": sol.t.size})
            q = sol.y[:, -1].copy()
            t_accum += dur
        times = np.concatenate(ts)
        states = np.concatenate(ys, axis=1)
        ext_states = np.array([e["state"] for e in extrema]).T if extrema else np.zeros((q.size, 0))
        return {
            "success": True, "message": "ok", "times": times, "states": states,
            "terminal_state": states[:, -1], "solver_stats": stats,
            "max_state_renorm": self._max_y_renorm, "method": method,
            "rtol": rtol, "atol": atol.tolist(),
            "extrema": extrema, "extrema_states": ext_states,
        }

    # ------------------------------------------------------------------
    # conservation diagnostics on a computed trajectory
    # ------------------------------------------------------------------
    def residuals(self, times: np.ndarray, states: np.ndarray, history: History) -> dict:
        T = states[0]
        Y = states[1:]
        bal = self.exact_balances(states[:, 0], history, times)
        # measured b(t), h(t) at each saved state
        b_meas = np.zeros_like(bal["b_pred"])
        h_meas = np.zeros_like(bal["h_pred"])
        for i in range(times.size):
            hk = self.species_enthalpies(float(T[i]))
            b_meas[:, i] = self.E @ np.maximum(Y[:, i], 0.0)
            h_meas[i] = float(hk @ np.maximum(Y[:, i], 0.0))
        res = {
            "element_balance_max_abs": float(np.max(np.abs(b_meas - bal["b_pred"]))),
            "enthalpy_balance_max_abs": float(np.max(np.abs(h_meas - bal["h_pred"]))),
            "min_Y": float(Y.min()),
            "min_Y_species": self.gas.species_names[int(np.argmin(Y.min(axis=1)))],
            "sum_Y_deviation_max_abs": float(np.max(np.abs(Y.sum(axis=0) - 1.0))),
            "min_T": float(T.min()), "max_T": float(T.max()),
        }
        return res


# ---------------------------------------------------------------------------
# Initial conditions: fresh feed and HP-equilibrium hot start
# ---------------------------------------------------------------------------


def fresh_state(cstr: CSTR) -> np.ndarray:
    """Fresh inlet mixture at inlet temperature."""
    return np.concatenate([[cstr.T_in], cstr.Y_in])


def hot_hp_state(cstr: CSTR) -> np.ndarray:
    """HP-equilibrium composition of the same feed enthalpy, pressure and
    elemental inventory. A separately labelled initial-condition study; not
    evidence of reachability from fresh feed within the horizon."""
    g = ct.Solution(cstr.cfg.mechanism, cstr.cfg.phase) if cstr.cfg.phase else ct.Solution(cstr.cfg.mechanism)
    g.TPY = (cstr.T_in, cstr.p, cstr.Y_in)
    g.equilibrate("HP")
    return np.concatenate([[float(g.T)], g.Y.copy()])
