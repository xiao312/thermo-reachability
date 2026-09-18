"""Tests for the phase-12 compact summary: it must be a pure reconstruction of
the immutable raw phase-11 records, must relabel by MEASURED norm, and must
enforce the E-metric and gain rules.

These run on the committed raw data when it is present and are skipped
otherwise (no Cantera is needed - the summary is postprocessing only).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "phase11" / "phase11_results.json"
OUT = ROOT / "results" / "phase12" / "phase12_summary.json"
SCRIPT = ROOT / "scripts" / "phase12_summary.py"

pytestmark = pytest.mark.skipif(not RAW.is_file(),
                                reason="phase-11 raw results not present")


def _run(tmp_path):
    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--phase11", str(RAW)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_summary_is_a_pure_reconstruction_of_the_raw_records(tmp_path):
    """Running the summary twice on the same raw records must give bit-identical
    output - it is postprocessing, never a mutation of the source run."""
    a = _run(tmp_path)
    b = _run(tmp_path)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # the raw source run is untouched
    raw_before = RAW.read_bytes()
    _run(tmp_path)
    assert RAW.read_bytes() == raw_before


def test_transverse_rays_are_relabelled_to_measured_norm(tmp_path):
    s = _run(tmp_path)
    rays = {c["label"]: c for c in s["cases"] if c["kind"] == "transverse_ray"}
    m = len(s["control_norm"]["durations_s"])
    for c in rays.values():
        lab = c["norms"]["requested_label"]
        if lab == 0.0:
            assert c["norms"]["measured_time_norm"] == 0.0
            continue
        # the rays were built from Euclidean-unit SVD vectors of A H^(-1/2), so
        # their actual time norm is label/sqrt(m)
        assert c["norms"]["measured_time_norm"] == pytest.approx(lab / (m ** 0.5))
    # the extreme requested label of 1.0 is an actual norm of 0.577 for m = 3
    assert rays["transverse_r1_s+1"]["norms"]["measured_time_norm"] \
        == pytest.approx(0.5773502691896258)


def test_in_family_and_held_out_labels_are_already_exact(tmp_path):
    s = _run(tmp_path)
    for c in s["cases"]:
        if c["kind"] in ("in_family_control", "held_out"):
            assert c["norms"]["measured_time_norm"] \
                == pytest.approx(c["norms"]["requested_label"])


def test_development_domain_is_bounded_by_the_measured_norm(tmp_path):
    """Every existing trajectory lies inside the declared development domain
    measured time norm <= 0.6, so the corrected analysis needs no new sweeps."""
    s = _run(tmp_path)
    worst = max(c["norms"]["measured_time_norm"] for c in s["cases"])
    assert worst <= 0.6 + 1e-12
    assert s["control_norm"]["sum_of_H_entries"] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["transverse_ray", "held_out"])
def test_e_metric_components_are_independent_of_the_other_tolerance(tmp_path, kind):
    """E_T must not change with eps_Y and E_Y must not change with eps_T; the
    report-table transcription error divided E_Y by the temperature factor."""
    s = _run(tmp_path)
    for c in s["cases"]:
        if c["kind"] != kind or not c["chemistry"]["dt_rows"]:
            continue
        for row in c["chemistry"]["dt_rows"]:
            tab = row["E_metric_postprocessed"]["rows"]
            by_t = {}
            for r in tab:
                by_t.setdefault(r["eps_T_K"], []).append(r)
                by_t.setdefault(("Y", r["eps_Y"]), []).append(r)
            for eps_t, rows in by_t.items():
                if not isinstance(eps_t, tuple):
                    assert {r["E_T"] for r in rows} == {rows[0]["E_T"]}
            # E_Y constant across eps_T at fixed eps_Y
            for eps_y in (1e-3, 1e-4):
                vals = {r["E_Y"] for r in tab if r["eps_Y"] == eps_y}
                assert len(vals) == 1
            # and E_max is the max of the two components, never a scaled copy
            for r in tab:
                assert r["E_max"] == pytest.approx(max(r["E_T"], r["E_Y"]))


def test_gain_denominator_rule_and_zero_horizon_identity(tmp_path):
    """The gain is undefined when the denominator is unresolved (including the
    zero-radius case, whose zero-horizon difference is exactly zero); otherwise
    the denominator is the stored state difference, since the chemistry map is
    the identity at dt = 0."""
    s = _run(tmp_path)
    zero = [c for c in s["cases"] if c["label"] == "transverse_r0_s+0"][0]
    assert zero["norms"]["measured_time_norm"] == 0.0
    # the locator returns the refinement floor (of order 1e-14), not exactly
    # zero, so the reference is UNRESOLVED and the gain must be undefined
    assert zero["located_reference"]["dist_scaled_upper_estimate"] < 1e-6
    assert zero["located_reference"]["resolved_above_floor"] is False
    assert all(r["gain_future_over_d0"] is None
               for r in zero["chemistry"]["dt_rows"])
    assert zero["chemistry"]["zero_horizon_identity_check"]["d0_scaled"] \
        == pytest.approx(0.0)
    for c in s["cases"]:
        zh = c["chemistry"]["zero_horizon_identity_check"]
        if zh["d0_scaled"] > 1e-6:
            for r in c["chemistry"]["dt_rows"]:
                assert r["gain_denominator_status"] == "resolved"
                assert r["gain_future_over_d0"] is not None
                # the future-state difference must stay within a modest factor
                # of the zero-horizon difference: the two states relax to the
                # same equilibrium, so no runaway amplification is expected
                assert r["gain_future_over_d0"] < 10.0
    assert s["checks"]["all_solved_dt_rows_succeeded"] is True


def test_raw_e_metric_records_are_internally_consistent(tmp_path):
    s = _run(tmp_path)
    assert s["checks"]["e_metric_consistent_in_raw_records"] is True
    assert s["checks"]["e_metric_inconsistent_cases"] == []
