"""Round-trip test for the reloadable frozen model (Revision 8 section D).

A fresh process must load the serialized artifact and reproduce predictions on
fixed inputs WITHOUT refitting and WITHOUT reading any training endpoint.  The
record is also checked for endpoint leakage: it must not contain training target
states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.generator import (  # noqa: E402
    FEATURE_NAME, LinearRegressor, LocalGenerator, active_species_report,
    apply_structural_zeros, feasible_a_interval, generator_features,
    load_local_generator, model_record, project_onto_active_conservation_subspace)

from test_generator import (  # noqa: E402
    E_FACE, FACE_NAMES, _FaceCstr, _GasStub, _DecoderStub, _face_report,
    _affine_features)


class _FakeLibrary:
    """The minimum a LocalGenerator needs from a reference library: the anchor
    state and an identity record."""

    def __init__(self, q0, identity):
        self.q0 = np.asarray(q0, dtype=float)
        self._identity = identity

    def identity(self):
        return self._identity

    def evaluate_fresh(self, gamma, t):
        # a deterministic stand-in family member: the feed, scaled by the
        # exposure, so the reload test has a reproducible reference state
        Y = np.asarray(_FaceCstr().Y_in, dtype=float).copy()
        return {"success": True, "q_B": np.concatenate([[1200.0], Y]).tolist()}


def _build_a_model():
    """A tiny fitted model over the face, with the same field shapes the real
    fitter produces, so the serialization path is exercised exactly."""
    rep, cstr, q0 = _face_report()
    rng = np.random.default_rng(17)
    out = project_onto_active_conservation_subspace(rng.normal(size=6), E_FACE,
                                                   rep["active_mask"])
    n_Y = apply_structural_zeros(np.asarray(out["n_Y"]), rep)
    durs = np.array([1e-6 / 3.0] * 3)
    T, gref = 1e-6, 1e4

    def feat(eta, order):
        return generator_features(np.asarray(eta, dtype=float), durs, T, gref,
                                  order)

    # FIT on synthetic targets, so the coefficient shapes are exactly what the
    # real fitter produces for this feature width and order
    etas = [np.array([0.05, -0.03, 0.02]), np.array([-0.04, 0.06, 0.0]),
            np.array([0.1, 0.1, -0.1]), np.array([0.0, 0.0, 0.0])]
    beta_reg = LinearRegressor(feat, order=1, ridge=1e-8).fit(
        etas, np.array([[0.01, -0.2], [0.03, 0.1], [-0.02, 0.05],
                        [0.005, -0.1]]))
    a_reg = LinearRegressor(feat, order=1, ridge=1e-8).fit(
        etas, np.array([2.0e-4, 1.0e-4, -3.0e-4, 5.0e-5]))
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6, 1e5, 2e5], c=1200.0)
    decoder = _DecoderStub(gas, E=E_FACE)
    lib = _FakeLibrary(q0, {"q0_sha256_16": "deadbeefdeadbeef",
                            "mechanism": "h2o2.yaml"})
    model = LocalGenerator(
        cstr=cstr, library=lib, decoder=decoder, n_Y=n_Y.tolist(),
        species_report=rep, beta_regressor=beta_reg, a_regressor=a_reg,
        gamma_ref=gref, T_horizon=T, durations=list(durs),
        training_domain={"max_measured_time_norm": 0.6, "n_train": 5,
                         "order": 1, "ridge": 1e-8},
        normal_report={"leading_residual_singular_values": [1.0, 0.1],
                       "retained_fraction": out["projection_retained_fraction"],
                       "subspace": out["subspace"],
                       "residual_cloud_rank_revealed": True},
        model_identity={"kind": "canonical_reference_plus_one_correction",
                        "mechanism": "h2o2.yaml",
                        "reference_coordinate_parametrization": "test",
                        "state_scaling": {"T_interval_K": 100.0,
                                          "Y_interval": 0.01, "n_species": 6},
                        "problem_definition": {"initial_state": "test"},
                        "library_identity": lib.identity()})
    return model, cstr, lib, decoder, rep



def test_the_record_carries_the_coefficient_arrays():
    model, _, _, _, _ = _build_a_model()
    rec = model_record(model, training_case_ids=["t0", "t1"],
                       validation_case_ids=["v0"],
                       config={"order": 1, "ridge": 1e-8},
                       selection={"selected": 1, "selection_set": "validation"})
    assert rec["feature_specification"]["feature"] == FEATURE_NAME
    n_feat = model.beta_regressor.coef_.shape[0]
    assert n_feat == 5                                # 1 + exposure + 3 segments
    assert np.asarray(rec["beta_coefficients"]).shape == (n_feat, 2)
    assert np.asarray(rec["a_coefficients"]).shape == (n_feat,)


def test_the_record_carries_no_training_endpoint():
    """The record is reloadable WITHOUT reading any training endpoint, so it must
    not contain one."""
    model, _, _, _, _ = _build_a_model()
    rec = model_record(model, training_case_ids=["t0"], validation_case_ids=[],
                       config={}, selection={})
    blob = repr(rec)
    for forbidden in ("q_target", "Y_target", "endpoint"):
        assert forbidden not in blob, f"the record leaks a {forbidden}"
    assert rec["training_case_ids"] == ["t0"]


def test_a_reloaded_model_reproduces_predictions_without_refitting():
    model, cstr, lib, decoder, _ = _build_a_model()
    rec = model_record(model, training_case_ids=["t0"],
                       validation_case_ids=["v0"], config={},
                       selection={"selected": 1})
    reloaded = load_local_generator(rec, cstr, lib, decoder)
    eta = np.array([0.1, -0.05, 0.2])
    p1 = model.predict(eta)
    p2 = reloaded.predict(eta)
    # the coefficient arrays are the stored ones; no refitting occurred
    assert reloaded.beta_regressor.coef_ is not model.beta_regressor.coef_
    assert np.allclose(np.asarray(reloaded.beta_regressor.coef_, dtype=float),
                       np.asarray(model.beta_regressor.coef_, dtype=float))
    assert np.allclose(p2["predicted_reference"]["log_gamma_B"],
                       p1["predicted_reference"]["log_gamma_B"])
    assert np.allclose(p2["predicted_correction"]["a_raw"],
                       p1["predicted_correction"]["a_raw"])
    assert p2["decoded_state"]["status"] == p1["decoded_state"]["status"]
    if p1["decoded_state"]["status"] == "admissible":
        assert np.allclose(p2["decoded_state"]["q_hat"],
                           p1["decoded_state"]["q_hat"])


def test_a_reloaded_model_reports_identity_mismatches():
    """A wrong mechanism or a wrong initial-state hash is REPORTED, not silently
    accepted."""
    model, cstr, lib, decoder, _ = _build_a_model()
    rec = model_record(model, training_case_ids=[], validation_case_ids=[],
                       config={}, selection={})
    rec["model_identity"]["mechanism"] = "something-else.yaml"
    reloaded = load_local_generator(rec, cstr, lib, decoder)
    assert any("mechanism mismatch" in w for w in reloaded.reload_warnings)
    # prediction still proceeds; the caller decides what to do with the warning
    assert reloaded.predict(np.zeros(3)) is not None
