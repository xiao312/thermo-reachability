"""Cantera ReactorNet cross-check for the custom CSTR RHS.

Builds the *same* realization as the custom RHS: constant pressure (variable
volume), constant mass, matched inlet/outlet mass flows mdot_in = mdot_out =
gamma * m, using an inlet reservoir at (T_in, Y_in, p), an
IdealGasConstPressureReactor, and two MassFlowControllers.

This is an independent integration pathway: Cantera integrates its own reactor
equations with its own solver, whereas the custom RHS integrates the brief's
equations with SciPy. Agreement validates the implementation of both, not the
chemical mechanism.
"""

from __future__ import annotations

import numpy as np
import cantera as ct

from .controls import History
from .reactor import CSTR


class ReactorNetCheck:
    def __init__(self, cstr: CSTR):
        self.c = cstr
        self.gas_in = ct.Solution(cstr.cfg.mechanism, cstr.cfg.phase) if cstr.cfg.phase \
            else ct.Solution(cstr.cfg.mechanism)
        self.gas_in.TPY = (cstr.T_in, cstr.p, cstr.Y_in)
        self.inlet = ct.Reservoir(self.gas_in)
        self.outlet = ct.Reservoir(self.gas_in)  # state irrelevant for a sink

    def run(self, history: History, q0: np.ndarray, times: np.ndarray,
            max_step: float | None = None) -> dict:
        """Integrate with piecewise-constant gamma; mdot is updated at each
        segment boundary and the same ReactorNet keeps integrating."""
        c = self.c
        gas = ct.Solution(c.cfg.mechanism, c.cfg.phase) if c.cfg.phase else ct.Solution(c.cfg.mechanism)
        T0, Y0 = float(q0[0]), np.asarray(q0[1:])
        gas.TPY = (T0, c.p, np.maximum(Y0, 0.0))
        m0 = c.cfg.reactor_mass
        reactor = ct.IdealGasConstPressureReactor(gas, energy="on")
        reactor.volume = m0 / gas.density
        mfc_in = ct.MassFlowController(self.inlet, reactor)
        mfc_in.mass_flow_rate = float(history.values[0]) * m0
        mfc_out = ct.MassFlowController(reactor, self.outlet)
        mfc_out.mass_flow_rate = float(history.values[0]) * m0
        net = ct.ReactorNet([reactor])
        if max_step is not None:
            net.max_step = max_step

        times = np.asarray(times, dtype=float)
        states = np.zeros((q0.size, times.size))
        t_acc = 0.0
        seg_idx = 0
        for i, t in enumerate(times):
            # advance to absolute time t, switching mdot at segment boundaries
            while t_acc < t - 1e-15 and seg_idx < len(history.values):
                seg_end = t_acc + float(history.durations[seg_idx])
                target = min(t, seg_end)
                net.advance(target)
                t_acc = target
                if t_acc >= seg_end - 1e-15 and seg_idx + 1 < len(history.values):
                    seg_idx += 1
                    mdot = float(history.values[seg_idx]) * m0
                    mfc_in.mass_flow_rate = mdot
                    mfc_out.mass_flow_rate = mdot
            states[0, i] = reactor.T
            states[1:, i] = reactor.thermo.Y
        return {"times": times, "states": states, "success": True,
                "n_steps": int(net.n_steps) if hasattr(net, "n_steps") else None}
