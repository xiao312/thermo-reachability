"""Pytest suite for the thermoreach package.

Covers the Phase 1A analytical derivations and the Phase 2 reactor balances.
Run with: python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach import toy  # noqa: E402
from thermoreach.controls import History, constant, random_history  # noqa: E402
from thermoreach.search import propagate_exact, support_analytical_envelope  # noqa: E402


def test_conservation_triangle_invariance():
    """Pure-A feed with gamma >= 0 keeps (x, y, 1-x-y) nonnegative."""
    rng = np.random.default_rng(0)
    q = np.array([1.0, 0.0])
    for _ in range(200):
        g = float(rng.uniform(0.0, 10.0))
        q = toy.advance(q, g, float(rng.uniform(0.01, 2.0)))
    assert q[0] >= -1e-12 and q[1] >= -1e-12 and 1 - q[0] - q[1] >= -1e-12


def test_exact_propagator_matches_ode():
    """The exact constant-control map agrees with independent Radau and BDF runs."""
    for g in (0.0, 1e-3, 1.0, 10.0, 100.0):
        for q0 in ([1.0, 0.0], [0.5, 0.2], [0.01, 0.05], [0.9, 0.05]):
            t = np.linspace(0.0, 3.0, 41)
            ex = toy.exact_segment(q0, g, t)
            for method in ("Radau", "BDF"):
                sol = solve_ivp(lambda tt, s: toy.rhs(s, g), (0, 3), q0, method=method,
                                t_eval=t, rtol=1e-11, atol=1e-13)
                assert sol.success
                assert np.max(np.abs(sol.y - ex)) < 1e-8


def test_steady_locus():
    """Steady states lie on y = x(1-x) and equal gamma/(1+gamma), gamma/(1+gamma)^2."""
    for g in (0.0, 0.1, 1.0, 10.0, 100.0):
        sp = toy.steady_point(g)
        assert abs(sp[1] - sp[0] * (1 - sp[0])) < 1e-14


def test_barrier_invariant_under_admissible_controls():
    """V = y + x log x <= 0 along admissible histories (upper invariant bound)."""
    rng = np.random.default_rng(1)
    worst = -np.inf
    for _ in range(50):
        h = random_history(rng, 10.0, 8, 5.0, zero_prob=0.0)
        _, ys = propagate_exact(h, np.array([1.0, 0.0]), samples_per_segment=25)
        worst = max(worst, float(np.max(toy.barrier(ys))))
    assert worst <= 1e-12


def test_barrier_boundary_derivative_sign():
    """dV/dt|_(V=0) = gamma (log x + 1 - x) <= 0 for gamma >= 0, 0 < x <= 1."""
    for g in (0.0, 1.0, 100.0):
        for x in (1e-6, 0.01, 0.3, 1.0):
            assert toy.barrier_boundary_derivative(g, x) <= 1e-12


def test_batch_curve_is_the_analytical_upper_curve():
    """The gamma=0 trajectory traces y = -x log x, i.e. the envelope boundary."""
    t = np.linspace(0, 12, 200)
    traj = toy.exact_segment([1.0, 0.0], 0.0, t)
    env = toy.analytical_upper_curve(traj[0])
    assert np.max(np.abs(traj[1] - env)) < 1e-12


def test_support_of_analytical_envelope():
    """Support of the enclosure in the +y direction is max(-x log x) = 1/e at x=1/e."""
    s = support_analytical_envelope(np.array([0.0, 1.0]))
    assert abs(s - 1.0 / np.e) < 1e-6


def test_history_bounds_enforced():
    """random_history must respect gamma <= gamma_max (admissible class)."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        h = random_history(rng, 1e5, 8, 0.1, zero_prob=0.0)
        assert h.values.max() <= 1e5 + 1e-9
        assert h.values.min() >= 0.0
        assert abs(sum(h.durations) - 0.1) < 1e-12


# ---------------------------------------------------------------------------
# Reactor (Phase 2). Skipped if Cantera is unavailable.
# ---------------------------------------------------------------------------

cantera = pytest.importorskip("cantera")

from thermoreach.reactor import CSTR, ReactorConfig, fresh_state, hot_hp_state  # noqa: E402


@pytest.fixture(scope="module")
def cstr():
    return CSTR(ReactorConfig(mechanism="h2o2.yaml"))


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
        worst_rel = max(worst_rel, np.max(np.abs(cstr.E @ r)) / max(np.max(np.abs(r)), 1.0))
    assert worst_rel < 1e-9


def test_energy_forms_agree(cstr):
    """The two algebraically equivalent energy forms agree to roundoff."""
    q = np.concatenate([[1500.0], cstr.Y_in])
    for g in (0.0, 10.0, 1e4):
        f1 = cstr.rhs(0.0, q, g)
        f2 = cstr.rhs_alt(0.0, q, g)
        assert np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f1)), 1.0) < 1e-9


def test_exact_mixing_balances_nontrivial(cstr):
    """b(t) and h(t) follow the exponential law with b0 != b_in, h0 != h_in."""
    rng = np.random.default_rng(4)
    Y0 = np.maximum(rng.random(cstr.gas.n_species), 0.0)
    Y0 /= Y0.sum()
    q0 = np.concatenate([[1100.0], Y0])
    g, tf = 500.0, 0.05
    h = History(np.array([g]), np.array([tf]))
    res = cstr.integrate(h, q0, samples_per_segment=201)
    assert res["success"]
    bal = cstr.exact_balances(q0, h, res["times"])
    b_meas = cstr.E @ res["states"][1:]
    h_meas = np.array([float(cstr.species_enthalpies(float(res["states"][0, i]))
                         @ np.maximum(res["states"][1:, i], 0.0))
                       for i in range(res["times"].size)])
    scale_b = max(np.max(np.abs(cstr.b_in)), np.max(np.abs(cstr.E @ Y0)))
    scale_h = max(abs(cstr.h_in), 1.0)
    assert np.max(np.abs(b_meas - bal["b_pred"])) / scale_b < 1e-8
    assert np.max(np.abs(h_meas - bal["h_pred"])) / scale_h < 1e-6


def test_hot_hp_start_has_feed_enthalpy(cstr):
    """The HP-equilibrium start has the same h and elemental inventory as feed."""
    q_hot = hot_hp_state(cstr)
    assert np.allclose(cstr.E @ q_hot[1:], cstr.b_in, rtol=1e-9, atol=1e-12)
    hk = cstr.species_enthalpies(float(q_hot[0]))
    assert abs(float(hk @ q_hot[1:]) - cstr.h_in) / abs(cstr.h_in) < 1e-9
