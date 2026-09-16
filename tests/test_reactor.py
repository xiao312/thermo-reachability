"""Cantera-dependent regression tests.

The module is skipped entirely when Cantera is unavailable - the toy and
bookkeeping tests live in ``test_toy.py`` and must NOT be skipped for that
reason (audit A15).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cantera")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.controls import History  # noqa: E402
from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402


@pytest.fixture(scope="module")
def cstr():
    return CSTR(ReactorConfig(mechanism="h2o2.yaml"))


# ---------------------------------------------------------------------------
# A02/A03: switched exposure and multi-segment exact balances
# ---------------------------------------------------------------------------

def test_elemental_matrix_normalization(cstr):
    E = cstr.E
    assert E.shape == (cstr.gas.n_elements, cstr.gas.n_species)
    assert np.max(np.abs(E.sum(axis=0) - 1.0)) < 1e-10


def test_elemental_source_conservation(cstr):
    """E . r = 0 (reactions conserve elements) to roundoff, relative to |r|."""
    rng = np.random.default_rng(2)
    worst_rel = 0.0
    for _ in range(4):
        Y = np.maximum(rng.random(cstr.gas.n_species), 0.0)
        Y /= Y.sum()
        cstr.gas.TPY = 1200.0, cstr.p, Y
        r = cstr.W * cstr.gas.net_production_rates / float(cstr.gas.density)
        worst_rel = max(worst_rel, np.max(np.abs(cstr.E @ r))
                        / max(np.max(np.abs(r)), 1.0))
    assert worst_rel < 1e-9


def test_energy_forms_agree(cstr):
    q = np.concatenate([[1500.0], cstr.Y_in])
    for g in (0.0, 10.0, 1e4):
        f1 = cstr.rhs(0.0, q, g)
        f2 = cstr.rhs_alt(0.0, q, g)
        assert np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f1)), 1.0) < 1e-9


def _nontrivial_q0(cstr, seed=4):
    """An initial state with b0 != b_in and h0 != h_in (masks nothing)."""
    rng = np.random.default_rng(seed)
    Y0 = np.maximum(rng.random(cstr.gas.n_species), 0.0)
    Y0 /= Y0.sum()
    return np.concatenate([[1100.0], Y0])


def test_exact_mixing_balances_nontrivial(cstr):
    """b(t), h(t) follow the exponential law with b0 != b_in, h0 != h_in."""
    q0 = _nontrivial_q0(cstr)
    g, tf = 500.0, 0.05
    h = History(np.array([g]), np.array([tf]))
    res = cstr.integrate(h, q0, samples_per_segment=201)
    assert res["success"]
    bal = cstr.exact_balances(q0, h, res["times"])
    b_meas = cstr.E @ res["states"][1:]
    h_meas = np.array([float(cstr.species_enthalpies(float(res["states"][0, i]))
                         @ np.maximum(res["states"][1:, i], 0.0))
                       for i in range(res["times"].size)])
    scale_b = max(np.max(np.abs(cstr.b_in)), np.max(np.abs(cstr.E @ q0[1:])))
    scale_h = max(abs(cstr.h_in), 1.0)
    assert np.max(np.abs(b_meas - bal["b_pred"])) / scale_b < 1e-8
    assert np.max(np.abs(h_meas - bal["h_pred"])) / scale_h < 1e-6


def test_exact_balances_switched_nontrivial(cstr):
    """Multi-segment (switched) balances - the case the double-counting bug hid.

    With b0 != b_in and h0 != h_in the exponential factor multiplies a nonzero
    initial difference, so a wrong Gamma(t) shows up directly (audit A03).
    """
    q0 = _nontrivial_q0(cstr)
    h = History(np.array([300.0, 4000.0, 50.0]), np.array([0.02, 0.03, 0.05]))
    res = cstr.integrate(h, q0, samples_per_segment=101)
    assert res["success"]
    bal = cstr.exact_balances(q0, h, res["times"])
    b_meas = cstr.E @ res["states"][1:]
    h_meas = np.array([float(cstr.species_enthalpies(float(res["states"][0, i]))
                         @ np.maximum(res["states"][1:, i], 0.0))
                       for i in range(res["times"].size)])
    scale_b = max(np.max(np.abs(cstr.b_in)), np.max(np.abs(cstr.E @ q0[1:])))
    scale_h = max(abs(cstr.h_in), 1.0)
    assert np.max(np.abs(b_meas - bal["b_pred"])) / scale_b < 1e-8
    assert np.max(np.abs(h_meas - bal["h_pred"])) / scale_h < 1e-6


# ---------------------------------------------------------------------------
# A04: ReactorNet switched scheduling must not depend on the output grid
# ---------------------------------------------------------------------------

def _switched_endpoints(cstr, n_out):
    """Run the Cantera ReactorNet wrapper on a switched history; return states."""
    from thermoreach.reactornet_check import ReactorNetCheck
    chk = ReactorNetCheck(cstr)
    h = History(np.array([10.0, 1000.0]), np.array([0.05, 0.05]))
    q0 = np.concatenate([[1200.0], cstr.Y_in])
    times = np.linspace(0.0, 0.1, n_out)
    return chk.run(h, q0, times)


def test_reactornet_switch_occurs_on_coarse_and_fine_grids(cstr):
    """Frequent output sampling must not prevent the control switch (audit A04)."""
    coarse = _switched_endpoints(cstr, 11)    # output every 0.01 s
    fine = _switched_endpoints(cstr, 4001)    # output every 2.5e-5 s
    # both must switch: the second segment's control is 100x the first, so a
    # reactor that never switches has a very different endpoint temperature.
    assert abs(coarse["states"][0, -1] - fine["states"][0, -1]) < 5.0
    # and the endpoint must differ strongly from staying on the first control
    from thermoreach.reactornet_check import ReactorNetCheck
    chk = ReactorNetCheck(cstr)
    h_noswitch = History(np.array([10.0, 10.0]), np.array([0.05, 0.05]))
    q0 = np.concatenate([[1200.0], cstr.Y_in])
    nosw = chk.run(h_noswitch, q0, np.linspace(0.0, 0.1, 4001))
    assert abs(coarse["states"][0, -1] - nosw["states"][0, -1]) > 50.0


def test_reactornet_matches_custom_rhs_switched(cstr):
    """The two integration pathways agree on a switched history (A2/A4)."""
    from thermoreach.reactornet_check import ReactorNetCheck
    chk = ReactorNetCheck(cstr)
    h = History(np.array([100.0, 2000.0]), np.array([0.05, 0.05]))
    q0 = np.concatenate([[1200.0], cstr.Y_in])
    times = np.linspace(0.0, 0.1, 201)
    net = chk.run(h, q0, times)
    ref = cstr.integrate(h, q0, samples_per_segment=101, rtol=1e-11,
                         atol_T=1e-9, atol_Y=1e-16)
    # evaluate the custom RHS reference exactly at the ReactorNet output times
    s_ref = ref["evaluate_dense"](times)
    assert np.max(np.abs(s_ref[0] - net["states"][0])) < 1e-3      # K
    assert np.max(np.abs(s_ref[1:] - net["states"][1:])) < 1e-8


# ---------------------------------------------------------------------------
# A06: chemical invariants
# ---------------------------------------------------------------------------

def test_chemical_invariants_dim_and_elemental_span(cstr):
    """rank(nu)=6, chemical left nullity=4, spanned by the rows of E (audit A06)."""
    from scripts.phase2_reactor_validate import chemical_invariants
    res = chemical_invariants(cstr)
    assert res["n_species"] == 10 and res["n_reactions"] == 29
    assert res["rank_nu"] == 6
    assert res["chemical_invariant_dim"] == 4
    assert res["elemental_rows_are_invariants"]
    assert res["rows_of_E_span_invariant_space"]
    assert res["passed"]


# ---------------------------------------------------------------------------
# A10: steady acceptance is residual-based
# ---------------------------------------------------------------------------

def test_steady_acceptance_requires_small_residual(cstr):
    from thermoreach.envelopes import trace_steady
    # a SHORT integration cannot have reached steady state; must be rejected
    sp = trace_steady(cstr, 100.0, np.concatenate([[1200.0], cstr.Y_in]),
                      "fresh", t_end=1e-4)
    assert not sp.converged or sp.residual < 1e-6
    # a long integration of a burning case should be accepted
    sp2 = trace_steady(cstr, 1e4, hot_hp_state(cstr), "hot")
    assert sp2.converged
    assert sp2.residual < 1e-6


# ---------------------------------------------------------------------------
# A11: temperature relaxation brackets rather than taking a grid maximum
# ---------------------------------------------------------------------------

def test_lp_temperature_bound_returns_bracket(cstr):
    from thermoreach.envelopes import lp_temperature_bound, thermo_valid_range
    lo, hi = thermo_valid_range(cstr)
    assert (lo, hi) == (300.0, 3500.0)          # common NASA overlap, not [200,5000]
    res = lp_temperature_bound(cstr, n_grid=201)
    assert res["mechanism_thermo_range_K"] == [300.0, 3500.0]
    b = res["feasible_T_bracket"]
    assert b is not None and b[1] > b[0]
    # the historical grid estimate must lie inside/at the feasible side
    assert res["grid_feasible_max_T_HISTORICAL"] is not None
    assert b[1] >= res["grid_feasible_max_T_HISTORICAL"] - 1e-6
    assert b[1] < hi + 1e-6


# ---------------------------------------------------------------------------
# A05: raw-state diagnostics are actually recorded on the RHS path
# ---------------------------------------------------------------------------

def test_rhs_diagnostics_record_raw_state_quality(cstr):
    q0 = np.concatenate([[1200.0], cstr.Y_in])
    res = cstr.integrate(History(np.array([1000.0]), np.array([0.05])), q0,
                         samples_per_segment=21)
    d = res["raw_state_diagnostics"]
    assert d["n_thermo_evaluations"] > 0
    assert d["max_sum_y_deviation"] >= 0.0
    assert d["min_raw_component"] <= 0.0 or d["max_clipped"] == 0.0


# ---------------------------------------------------------------------------
# A08: the two metrics actually differ
# ---------------------------------------------------------------------------

def test_engineering_and_trace_sensitive_metrics_differ(cstr):
    """Deterministic dispatch test (audit A08): the two representations must give
    clearly different answers, so a future dispatch error is detectable."""
    from thermoreach.envelopes import StateMetric
    j = 1 + cstr.gas.species_index("OH")
    ref = np.concatenate([[1200.0], cstr.Y_in]).copy()
    ref[j] = 1e-6                      # reference has a small but above-floor OH
    q = ref.copy()
    q[j] = 1e-12                       # query OH crushed by a factor 1e6
    m = StateMetric(cstr.gas.species_names, ref[:, None])
    d_eng = m.distance(q[:, None], ref[:, None], kind="engineering")[0]
    d_ts = m.distance(q[:, None], ref[:, None], kind="trace_sensitive")[0]
    # engineering: (1e-6 - 1e-12)/Y_scale_OH with Y_scale_OH = 1e-6  ->  ~1.0
    assert d_eng == pytest.approx(1.0, abs=0.02)
    # trace-sensitive: |log10(1e-12/1e-12) - log10(1e-6/1e-12)|  ->  6.0
    assert d_ts == pytest.approx(6.0, abs=0.02)
    assert d_ts != pytest.approx(d_eng)


# ---------------------------------------------------------------------------
# A14: exchange map must signal leaving the valid thermodynamic domain
# ---------------------------------------------------------------------------

def test_temperature_from_enthalpy_reports_domain_status(cstr):
    """Silently returning a bracket endpoint was the A14 defect; now it reports."""
    from thermoreach.splitting import temperature_from_enthalpy
    T, status = temperature_from_enthalpy(cstr, cstr.h_in, cstr.Y_in)
    assert status == "ok"                        # feed enthalpy is attainable
    T, status = temperature_from_enthalpy(cstr, -1e12, cstr.Y_in)
    assert status == "below_valid_range" and T == pytest.approx(300.0)
    T, status = temperature_from_enthalpy(cstr, 1e12, cstr.Y_in)
    assert status == "above_valid_range" and T == pytest.approx(3500.0)


def test_exchange_step_stays_inside_domain_for_physical_states(cstr):
    """For physical states the exact exchange map stays in the valid domain.

    h_new is the same convex combination of (h_old, h_in) that Y_new is of
    (Y_old, Y_in), so the mixing line between two valid states stays realizable;
    the status mechanism exists to make any departure visible rather than silent.
    """
    from thermoreach.splitting import exchange_step
    for q in (np.concatenate([[1200.0], cstr.Y_in]),
              np.concatenate([[3500.0], cstr.Y_in]),
              np.concatenate([[300.0], cstr.Y_in])):
        for g, dt in ((10.0, 0.01), (1e5, 1e-4), (1e3, 1e-3)):
            qn = exchange_step(cstr, q.copy(), g, dt)
            assert 300.0 - 1e-6 <= qn[0] <= 3500.0 + 1e-6
