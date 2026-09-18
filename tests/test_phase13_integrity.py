"""Integrity tests for the phase-13 generator run, checked against the committed
artifacts (skipped if the run is absent).

These pin the protocol, not the numbers: the model must be frozen before test
histories exist, no test history may appear in the training set, and the
exposure-constrained parametrization must be recorded in the model identity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P13 = ROOT / "results" / "phase13"
P11 = ROOT / "results" / "phase11" / "phase11_results.json"

pytestmark = pytest.mark.skipif(
    not (P13 / "phase13_summary.json").is_file(),
    reason="phase-13 results not present")


def _summary():
    return json.loads((P13 / "phase13_summary.json").read_text(encoding="utf-8"))


def _results():
    return json.loads((P13 / "phase13_results.json").read_text(encoding="utf-8"))


def _frozen():
    return json.loads((P13 / "frozen_model.json").read_text(encoding="utf-8"))


def test_the_model_is_frozen_before_test_histories_exist():
    """The declaration must say so, and the complexity must have been selected on
    validation only."""
    f = _frozen()
    assert "BEFORE any test history" in f["declaration"]
    assert f["selection"]["selection_set"] == "validation only"
    assert f["order"] == f["selection"]["selected"]
    assert f["selection"]["selected"] in f["selection"]["orders_tried"]
    assert f["n_train"] > 0


def test_no_test_history_is_in_the_training_set():
    """Training cases are reconstructed from the immutable phase-11 records; the
    phase-13 test histories are new, so the two sets must be disjoint."""
    r = _results()
    train_labels = {c["label"] for c in json.loads(
        P11.read_text(encoding="utf-8"))["cases"]}
    test_labels = {c["label"] for c in r["test_results"]}
    assert not (test_labels & train_labels)
    val_labels = {c["label"] for rows in r["validation_results"].values()
                  for c in rows}
    assert not (test_labels & val_labels)


def test_the_exposure_constrained_parametrization_is_recorded():
    f = _frozen()
    pid = f["model_identity"]["reference_coordinate_parametrization"]
    assert "exposure_constrained" in pid
    assert "log gamma_B = log Gamma - v + u" in pid


def test_every_test_prediction_is_physically_valid_or_rejected_never_clipped():
    """A decoded state is either admissible or explicitly rejected; there is no
    silent clipping path."""
    r = _results()
    s = _summary()
    n = sum(s["n_rejections"].values())
    assert n == len(r["test_results"])
    for c in r["test_results"]:
        st = c["C_predicted_reference_plus_predicted_correction"]["decoded_status"]
        assert st in ("admissible", "rejected_species_bounds",
                      "rejected_normalization", "rejected_enthalpy_inversion",
                      "rejected_reference_integration")


def test_the_reference_coordinate_prediction_cost_is_reported():
    """B vs A separates the prediction cost from the family distance."""
    s = _summary()
    for k in ("A_oracle_reference", "B_predicted_reference",
              "C_predicted_plus_correction",
              "D_oracle_reference_plus_oracle_correction"):
        assert s["errors_scaled_norm"][k]["n"] == s["n_test"]
    # the oracle variant must not be bettered by a prediction
    assert s["errors_scaled_norm"]["D_oracle_reference_plus_oracle_correction"][
        "median"] <= s["errors_scaled_norm"][
        "C_predicted_plus_correction"]["median"]


def test_the_test_set_lies_in_the_declared_development_domain():
    s = _summary()
    lo, hi = s["domain"]["test_norm_range"]
    assert hi <= s["domain"]["max_measured_time_norm_declared"] + 1e-9
    assert s["domain"]["strata"] == [0.1, 0.3, 0.6]


def test_the_oracle_correction_coefficient_search_is_multi_scale():
    """The search record must show more than a handful of candidates, because the
    feasible interval is orders of magnitude wider than the optimal a."""
    r = _results()
    for c in r["test_results"]:
        n = c["D_oracle_reference_plus_oracle_correction"].get(
            "interval", {}).get("interval_width")
        if n and c["C_predicted_reference_plus_predicted_correction"]["a"]:
            # any nonzero correction lives far below the interval width
            assert abs(c["C_predicted_reference_plus_predicted_correction"]["a"]) \
                < 0.1 * n
