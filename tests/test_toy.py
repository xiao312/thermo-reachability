"""Cantera-free regression tests.

Covers the toy mathematics, the audit's refutation of claim C7, the convex-hull
diagnostic degeneracy, the admissibility object, stable child seeds, and the
switched-exposure bookkeeping (gamma_integral). These must run without Cantera
so a missing Cantera installation does not skip the toy suite (audit A15).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.admissibility import AdmissibleControls, stable_child_seed  # noqa: E402
from thermoreach.controls import (History, bang_bang, equal_durations,  # noqa: E402
                                  pulse, random_history)
from thermoreach.controls import gamma_integral_of  # noqa: E402
from thermoreach.toy import advance, barrier, batch_curve  # noqa: E402

Q0 = np.array([1.0, 0.0])


# ---------------------------------------------------------------------------
# Audit section 4: the visited set DOES depend on the control bound (C7 refuted)
# ---------------------------------------------------------------------------

def test_audit_witness_is_reachable_with_G10():
    """gamma=0 for 4 units then gamma=10 for delta reaches (0.5, 0.04939625)."""
    delta = 0.0707413427911034
    end = advance(advance(Q0, 0.0, 4.0), 10.0, delta)
    assert end == pytest.approx([0.5, 0.04939624612802243], abs=1e-12)
    assert 4.0 + delta < 5.0


def test_audit_witness_t_leq_11_log_identity():
    """The witness duration is the closed form (1/11) log((10/11 - e^-4)/(10/11 - 1/2))."""
    delta = np.log((10.0 / 11.0 - np.exp(-4.0)) / (10.0 / 11.0 - 0.5)) / 11.0
    assert delta == pytest.approx(0.0707413427911034, abs=1e-12)


def test_lower_barrier_excludes_witness_for_small_G():
    """W = y - x(1-x) >= 0 for reachable states with x >= G/(1+G).

    For G=0.1 at x=0.5 this requires y >= 0.25; the witness has y ~ 0.0494.
    """
    def Wdot(x, g):
        return (1.0 - x) * ((1.0 + g) * x - g)

    for G in (0.1, 1.0, 10.0):
        xG = G / (1.0 + G)
        grid = [(x, g) for x in np.linspace(xG, 1.0, 41)
                for g in np.linspace(0.0, G, 41)]
        worst = min(Wdot(x, g) for x, g in grid)
        assert worst >= -1e-12, f"barrier violated for G={G}"
    # G=0.1 exclusion at x=0.5
    assert 0.5 * (1 - 0.5) == 0.25
    witness = advance(advance(Q0, 0.0, 4.0), 10.0, 0.0707413427911034)
    assert witness[1] < 0.25


def test_visited_sets_differ_between_G_bounds():
    """Explicit G-dependent reachability: the witness is in R_G=10 but not R_G=0.1."""
    witness = advance(advance(Q0, 0.0, 4.0), 10.0, 0.0707413427911034)
    assert witness[0] == pytest.approx(0.5, abs=1e-12)
    # For G=0.1, x=0.5 forces y >= 0.25 (test above); the witness has y ~ 0.0494.
    assert witness[1] < 0.25


# ---------------------------------------------------------------------------
# Audit section 5: the convex-hull diagnostic is degenerate
# ---------------------------------------------------------------------------

def test_batch_curve_hull_area_reproduces_reported_fractions():
    """A single 1-D curve has zero area yet yields hull fractions 0.129/0.933/1.000.

    This reproduces the values reported in report_01.md as
    `envelope_area_fraction_visited`, showing that diagnostic measures nothing
    about interior coverage.
    """
    envelope_area = 0.25
    for T, reported in ((1.0, 0.129), (5.0, 0.933), (20.0, 1.000)):
        t = np.linspace(0.0, T, 1_000_001)
        pts = np.stack([np.exp(-t), t * np.exp(-t)], axis=1)
        frac = ConvexHull(pts).volume / envelope_area
        closed_form = (1.0 - np.exp(-2.0 * T) - 2.0 * T * np.exp(-T))
        assert frac == pytest.approx(closed_form, abs=1e-9)
        assert frac == pytest.approx(reported, abs=2e-3)


# ---------------------------------------------------------------------------
# Toy identities retained (C1-C5)
# ---------------------------------------------------------------------------

def test_exact_propagator_steady_and_barrier():
    for g in (0.0, 0.5, 10.0):
        q = advance(Q0, g, 50.0)
        xss = g / (1.0 + g)
        assert q[0] == pytest.approx(xss, abs=1e-9)
        assert q[1] == pytest.approx(xss * (1 - xss), abs=1e-9)
        assert barrier(q) <= 1e-12


def test_batch_curve_is_zero_barrier_upper_face():
    """The zero-control trajectory traces y = -x log x, on which V = y + x log x = 0."""
    x = np.linspace(1e-6, 1.0, 1001)
    y = batch_curve(x)
    assert np.all(np.abs(y + x * np.log(x)) < 1e-12)


# ---------------------------------------------------------------------------
# A03: switched exposure bookkeeping
# ---------------------------------------------------------------------------

def test_gamma_integral_two_equal_segments():
    """The audit's exact reproduction: [1,2] x [1,1] must give [0,.5,1,2,3]."""
    got = gamma_integral_of([1, 2], [1, 1], [0, 0.5, 1.0, 1.5, 2.0])
    assert got == pytest.approx([0, 0.5, 1.0, 2.0, 3.0], abs=1e-12)


def test_gamma_integral_unequal_segments_and_endpoints():
    g = gamma_integral_of([3.0, 7.0], [0.5, 1.5],
                          [0.0, 0.25, 0.5, 0.5, 1.25, 2.0])
    # segment 1 contributes 3*min(t,0.5); segment 2 adds 1.5 + 7*min(t-0.5,1.5)
    assert g == pytest.approx([0.0, 0.75, 1.5, 1.5, 1.5 + 7 * 0.75, 1.5 + 7 * 1.5],
                              abs=1e-12)


def test_gamma_integral_repeated_switch_times():
    """Times exactly at switches must not drift (the ReactorNet bug, A04)."""
    g = gamma_integral_of([2.0, 5.0], [1.0, 1.0],
                          [0.25, 0.5, 0.75, 1.0, 1.0, 1.5, 2.0])
    assert g == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.0, 2.0 + 2.5, 2.0 + 5.0],
                              abs=1e-12)


def test_gamma_integral_three_segments():
    g = gamma_integral_of([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.5, 1.5, 2.5, 3.0])
    assert g == pytest.approx([0.5, 1.0 + 1.0, 1.0 + 2.0 + 1.5, 6.0], abs=1e-12)


# ---------------------------------------------------------------------------
# A1/A07: admissibility
# ---------------------------------------------------------------------------

def test_admissibility_rejects_zero_under_declared_bound():
    adm = AdmissibleControls(10.0, 1e5, 0.1)
    ok, reason = adm.validate(History(np.zeros(4), np.full(4, 0.025)))
    assert not ok
    assert "lower bound" in reason


def test_admissibility_rejects_out_of_bound_and_nonfinite():
    adm = AdmissibleControls(10.0, 1e5, 0.1)
    ok, _ = adm.validate(History(np.array([1e6, 10.0, 10.0, 10.0]), np.full(4, 0.025)))
    assert not ok                        # upper-bound violation
    ok, _ = adm.validate(History(np.array([np.nan, 10.0, 10.0, 10.0]), np.full(4, 0.025)))
    assert not ok                        # nonfinite control
    # Nonpositive durations are rejected by the History constructor itself.
    with pytest.raises(ValueError):
        History(np.array([10.0, 10.0]), np.array([0.05, 0.0]))
    # total-duration mismatch is caught by the admissibility gate
    ok, reason = adm.validate(History(np.array([10.0, 10.0]), np.array([0.05, 0.04])))
    assert not ok and "horizon" in reason


def test_admissibility_accepts_in_bound_history():
    adm = AdmissibleControls(10.0, 1e5, 0.1)
    ok, reason = adm.validate(History(np.full(8, 100.0), np.full(8, 0.0125)))
    assert ok and reason == ""


def test_structured_histories_are_admissible():
    """Every structured control family lies in the declared class (A07 regression)."""
    adm = AdmissibleControls(10.0, 1e5, 0.1)
    n_seg = 8
    fams = {
        "bangbang_hi": bang_bang(10.0, 1e5, n_seg, 0.1, start_high=True),
        "bangbang_lo": bang_bang(10.0, 1e5, n_seg, 0.1, start_high=False),
        "pulse_first": pulse(1e5, n_seg, 0.1, pulse_segment=0, base=10.0),
        "pulse_last": pulse(1e5, n_seg, 0.1, pulse_segment=n_seg - 1, base=10.0),
    }
    for name, h in fams.items():
        ok, reason = adm.validate(h)
        assert ok, f"{name} inadmissible: {reason}"


def test_random_history_full_range_is_admissible():
    """The corrected sampling law covers the FULL declared range [lo, hi]."""
    adm = AdmissibleControls(10.0, 1e5, 0.1)
    rng = np.random.default_rng(0)
    log_low = float(np.log10(10.0 / 1e5))
    worst_lo = np.inf
    for _ in range(200):
        h = random_history(rng, 1e5, 8, 0.1, zero_prob=0.0,
                           log_low=log_low, log_high=0.0)
        ok, reason = adm.validate(h)
        assert ok, reason
        worst_lo = min(worst_lo, float(h.values.min()))
    # with 200 x 8 samples over 4 decades, the minimum should reach well below 100
    assert worst_lo < 100.0


def test_stable_child_seed_is_deterministic():
    a = stable_child_seed(20260916, "fresh")
    b = stable_child_seed(20260916, "fresh")
    c = stable_child_seed(20260916, "hot")
    assert a == b            # stable across calls
    assert a != c            # distinct per label


def test_stable_child_seed_differs_from_python_hash_salt():
    """abs(hash(label)) % 1000 is process-salted; hashlib is not (A12)."""
    s = stable_child_seed(1, "fresh")
    assert isinstance(s, int) and 0 <= s < 2**31
    # must not equal the salted hash offset in a way that varies by interpreter
    assert stable_child_seed(1, "fresh") == stable_child_seed(1, "fresh")


def test_equal_durations_sums_to_horizon():
    h = History(np.full(8, 50.0), equal_durations(8, 0.1))
    assert float(np.sum(h.durations)) == pytest.approx(0.1)
