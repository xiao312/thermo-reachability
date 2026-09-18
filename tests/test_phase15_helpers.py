"""Local tests for the phase-15 helper logic that does not need Cantera.

These exist so an expensive server run is not wasted on a bug in the summary or
gain arithmetic: the same-pair gain must preserve AMPLIFICATION (a common
equilibrium does not imply monotone error decay) and the quantile aggregation must
count failures rather than filter them out of the denominators.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.summary_utils import quantiles, same_pair_gain  # noqa: E402


def test_quantiles_drop_only_the_missing_not_the_failures():
    xs = [1.0, None, float("nan"), 3.0, 5.0]
    q = quantiles(xs)
    assert q["n"] == 3                      # the two unavailable entries dropped
    assert q["min"] == 1.0 and q["max"] == 5.0
    assert q["median"] == 3.0
    # an empty column reports n = 0, not a fabricated statistic
    assert quantiles([None, None]) == {"n": 0}


def test_the_same_pair_gain_is_relative_to_dt0():
    chem = {"state_difference": {"dT_K": 0.5, "dY": [0.01, -0.02]},
            "dt_results": [{"dt": 1e-6, "success": True,
                            "future_state_dT_K": 0.25,
                            "future_state_dY": [0.005, -0.01]},
                           {"dt": 1e-4, "success": True,
                            "future_state_dT_K": 0.5,
                            "future_state_dY": [0.01, -0.02]}]}
    g = same_pair_gain(chem)["same_pair_gain_relative_to_dt0"]
    assert g[0]["gain_T"] == pytest.approx(0.5)       # contraction
    assert g[0]["gain_Y"] == pytest.approx(0.5)


def test_the_same_pair_gain_preserves_amplification():
    """A difference that GROWS under the chemical map is reported as amplifying,
    never as a decay toward a common equilibrium."""
    chem = {"state_difference": {"dT_K": 0.1, "dY": [0.001]},
            "dt_results": [{"dt": 1e-6, "success": True,
                            "future_state_dT_K": 1.0,
                            "future_state_dY": [0.01]}]}
    g = same_pair_gain(chem)["same_pair_gain_relative_to_dt0"]
    assert g[0]["gain_T"] == pytest.approx(10.0)
    assert g[0]["amplifying"] is True


def test_a_failed_chemistry_step_is_reported_not_hidden():
    chem = {"state_difference": {"dT_K": 0.5, "dY": [0.01]},
            "dt_results": [{"dt": 1e-6, "success": False,
                            "failures": ["did not converge"]}]}
    g = same_pair_gain(chem)["same_pair_gain_relative_to_dt0"]
    assert g[0]["success"] is False
    assert "failures" in g[0] or g[0].get("success") is False


def test_a_zero_dt0_difference_gives_no_spurious_gain():
    chem = {"state_difference": {"dT_K": 0.0, "dY": [0.0, 0.0]},
            "dt_results": [{"dt": 1e-6, "success": True,
                            "future_state_dT_K": 1e-9,
                            "future_state_dY": [1e-9, 1e-9]}]}
    g = same_pair_gain(chem)["same_pair_gain_relative_to_dt0"]
    assert g[0]["gain_T"] is None
    assert g[0]["gain_Y"] is None
    # and no division by zero occurred
    assert math.isfinite(g[0]["future_dT_K"])
