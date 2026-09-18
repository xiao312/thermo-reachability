"""Tests for the local generator's mathematical core: the conservation-compatible
correction subspace, the feasible interval, the physical decoder's rejection
rules, and the regressor.

Cantera is needed only for the enthalpy inversion, which is tested against an
exactly-integrable stub; the real-thermo paths run on the compute server.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.generator import (  # noqa: E402
    LinearRegressor, active_species_report, apply_structural_zeros,
    conservation_subspace, conservation_subspace_active,
    exposure_integral, exposure_log_feature, feasible_a_interval,
    generator_features, net_stoichiometry, null_space_basis,
    exact_balances_from_controls, oracle_correction_coefficient,
    project_onto_active_conservation_subspace,
    project_onto_conservation_subspace, structural_zero_mask,
)


def _affine_features(eta, order=1):
    """A test feature map: [1, eta] plus, at order 2, squares and products."""
    e = np.asarray(eta, dtype=float).ravel()
    cols = [np.ones(1), e]
    if order >= 2:
        cols.append(e ** 2)
        iu = np.triu_indices(e.size, k=1)
        cols.append(e[iu[0]] * e[iu[1]])
    return np.concatenate(cols)


# ---------------------------------------------------------------------------
# a stoichiometric stub: 4 species, 2 elements, so the conservation-compatible
# subspace has dimension 4 - 2 - 1 = 1
# ---------------------------------------------------------------------------


# H2, O2, H2O, OH: two elements, and the normalization row is NOT in the row
# space of E (each species carries a different total atom count), so the
# conservation-compatible subspace has dimension n_sp - n_el - 1 = 1
E_STUB = np.array([[2.0, 0.0, 2.0, 1.0],
                   [0.0, 2.0, 1.0, 1.0]])


class _GasStub:
    """h(T, Y) = sum_k a_k Y_k + c*T so enthalpy inversion is closed form:
    T = (h - sum a_k Y_k)/c."""

    def __init__(self, a, c, p=1.0):
        self.a = np.asarray(a, dtype=float)
        self.c = float(c)
        self.p = float(p)
        self.T = None
        self.Y = None

    @property
    def species_names(self):
        return [f"S{i}" for i in range(self.a.size)]


class _DecoderStub:
    """Mimics PhysicalDecoder.decode but solves T in closed form, and refuses
    negative compositions exactly like the real decoder."""

    def __init__(self, gas, y_floor=-1e-12, sum_atol=1e-9, E=None):
        self.cstr = type("C", (), {})()
        self.cstr.gas = gas
        self.cstr.E = E_STUB if E is None else np.asarray(E, dtype=float)
        self.cstr.p = gas.p
        self.y_floor = y_floor
        self.sum_atol = sum_atol
        self.n_rejected = {"species_negative": 0, "normalization": 0,
                           "enthalpy_inversion": 0, "other": 0}

    def decode(self, Y_hat, h_J_kg, a=None):
        Y = np.asarray(Y_hat, dtype=float)
        if not np.all(np.isfinite(Y)):
            self.n_rejected["other"] += 1
            return {"status": "rejected_nonfinite"}
        if abs(Y.sum() - 1.0) > self.sum_atol:
            self.n_rejected["normalization"] += 1
            return {"status": "rejected_normalization",
                    "sum_Y_minus_1": float(Y.sum() - 1.0)}
        if Y.min() < self.y_floor:
            self.n_rejected["species_negative"] += 1
            return {"status": "rejected_species_bounds",
                    "min_Y": float(Y.min()), "n_negative": int(np.sum(Y < 0))}
        T = (float(h_J_kg) - float(self.cstr.gas.a @ Y)) / self.cstr.gas.c
        return {"status": "admissible", "T_K": T,
                "Y": Y.tolist(),
                "q_hat": np.concatenate([[T], Y]).tolist(),
                "b_hat": (self.cstr.E @ Y).tolist()}


# ---------------------------------------------------------------------------
# 1. the conservation-compatible subspace
# ---------------------------------------------------------------------------


def test_conservation_subspace_dimension_and_constraints():
    sub = conservation_subspace(E_STUB)
    assert sub["dim"] == 1                      # n_sp - n_el - 1
    Q = np.asarray(sub["basis"])
    assert Q.shape == (4, 1)
    assert np.allclose(Q.T @ Q, np.eye(1))
    n = Q[:, 0]
    assert np.allclose(E_STUB @ n, 0.0)         # elements preserved
    assert np.isclose(n.sum(), 0.0)             # normalization preserved


def test_conservation_subspace_is_row_equilibrated_and_rank_revealed():
    """A badly scaled element matrix must not change the answer, and the rank
    must be revealed against both absolute and relative thresholds."""
    E_bad = E_STUB * np.array([[1e-12], [1e6]])       # badly scaled rows
    sub_bad = conservation_subspace(E_bad)
    assert sub_bad["dim"] == 1
    n = np.asarray(sub_bad["basis"])[:, 0]
    assert np.allclose(E_bad @ n, 0.0, atol=1e-8)
    # a rank-deficient element matrix (two identical rows) is detected
    E_def = np.vstack([E_STUB, E_STUB[0]])
    sub_def = conservation_subspace(E_def)
    assert sub_def["dim"] == 4 - 2 - 1


def test_projection_reports_unresolved_for_a_degenerate_direction():
    """A direction with no component in the conservation subspace is unresolved,
    and the code must say so rather than fabricate a direction."""
    # the zero direction has no projection at all
    rep0 = project_onto_conservation_subspace(np.zeros(4), E_STUB)
    assert rep0["resolved"] is False
    # a direction orthogonal to the (1-D) conservation subspace: project it out
    Q = np.asarray(conservation_subspace(E_STUB)["basis"])
    n0 = Q[:, 0]
    w = np.array([1.0, 1.0, 0.0, 0.0])
    w = w - (w @ n0) * n0
    w = w / np.linalg.norm(w)
    rep = project_onto_conservation_subspace(w, E_STUB)
    assert rep["resolved"] is False
    assert "degenerate" in rep["reason"]


def test_projection_of_a_genuine_residual_direction_is_conservation_compatible():
    rng = np.random.default_rng(5)
    Q = np.asarray(conservation_subspace(E_STUB)["basis"])
    w = rng.normal(size=4) + 3.0 * Q[:, 0]      # mostly along the subspace
    rep = project_onto_conservation_subspace(w, E_STUB)
    assert rep["resolved"] is True
    n = np.asarray(rep["n_Y"])
    assert np.allclose(E_STUB @ n, 0.0, atol=1e-10)
    assert np.isclose(n.sum(), 0.0, atol=1e-12)
    assert np.linalg.norm(n) == pytest.approx(1.0)
    # the retained fraction is a real diagnostic, not a constant
    assert rep["projection_retained_fraction"] > 0.5


# ---------------------------------------------------------------------------
# 2. the feasible interval, from the species bounds themselves
# ---------------------------------------------------------------------------


def test_feasible_interval_from_species_bounds():
    Y_B = np.array([0.4, 0.3, 0.2, 0.1])
    n_Y = np.array([0.1, -0.1, 0.05, -0.05])
    iv = feasible_a_interval(Y_B, n_Y)
    assert iv["bounded"] is True
    # at the endpoints a species is exactly zero
    assert np.any(Y_B + iv["a_lo"] * n_Y <= 1e-12)
    assert np.any(Y_B + iv["a_hi"] * n_Y <= 1e-12)
    # interior points are strictly feasible
    a_mid = 0.5 * (iv["a_lo"] + iv["a_hi"])
    assert np.all(Y_B + a_mid * n_Y > 0.0)


def test_feasible_interval_is_not_the_training_range():
    """The interval is a feasibility statement about the decoder, so it is
    available before any coefficient is fitted."""
    Y_B = np.array([0.5, 0.5, 0.0, 0.0])
    n_Y = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2)
    iv = feasible_a_interval(Y_B, n_Y)
    assert iv["bounded"]
    assert iv["a_lo"] == pytest.approx(-0.5 * np.sqrt(2))
    assert iv["a_hi"] == pytest.approx(0.5 * np.sqrt(2))


def test_feasible_interval_bounded_by_a_zero_species_on_one_side():
    """A species that is already zero bounds the correction from below at 0."""
    Y_B = np.array([0.0, 1.0, 0.0, 0.0])
    n_Y = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2)
    iv = feasible_a_interval(Y_B, n_Y)
    assert iv["bounded"] is True
    assert iv["a_lo"] == pytest.approx(0.0)
    assert iv["a_hi"] == pytest.approx(np.sqrt(2))
    # a negative a is inadmissible: species 0 would go negative
    assert np.any(Y_B - 0.01 * n_Y < 0.0)


# ---------------------------------------------------------------------------
# 3. the decoder: enthalpy inversion, and rejection instead of clipping
# ---------------------------------------------------------------------------


def test_decoder_solves_temperature_by_enthalpy_inversion():
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    Y = np.array([0.1, 0.2, 0.3, 0.4])
    h = float(gas.a @ Y) + 1200.0 * 900.0
    out = dec.decode(Y, h)
    assert out["status"] == "admissible"
    assert out["T_K"] == pytest.approx(900.0)


def test_decoder_rejects_negative_species_without_clipping():
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    Y = np.array([0.35, 0.35, 0.35, -0.05])   # sums to 1, one negative species
    assert Y.min() < 0.0
    out = dec.decode(Y, 1e6)
    assert out["status"] == "rejected_species_bounds"
    assert out["min_Y"] < 0.0
    # and the rejection was counted, not silently repaired
    assert dec.n_rejected["species_negative"] == 1


def test_decoder_rejects_unnormalized_compositions():
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    Y = np.array([0.1, 0.2, 0.3, 0.5])          # sums to 1.1
    out = dec.decode(Y, 1e6)
    assert out["status"] == "rejected_normalization"


# ---------------------------------------------------------------------------
# 4. the oracle correction coefficient
# ---------------------------------------------------------------------------


def test_oracle_correction_recovers_a_known_coefficient():
    """Given a target that is EXACTLY B + a* n with a known a, the oracle must
    recover a, through the decoder (so temperature is inverted, not assumed)."""
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    n_Y = np.array([0.1, -0.1, 0.05, -0.05])
    n_Y = n_Y / np.linalg.norm(n_Y)
    Y_B = np.array([0.4, 0.3, 0.2, 0.1])
    a_true = 0.05      # must stay inside the feasible interval of Y_B
    Y_true = Y_B + a_true * n_Y
    h = float(gas.a @ Y_true) + 1200.0 * 800.0
    T_true = dec.decode(Y_true, h)["T_K"]
    q_target = np.concatenate([[T_true], Y_true])
    out = oracle_correction_coefficient(q_target, Y_B, n_Y, h, dec)
    assert out["resolved"] is True
    assert out["a"] == pytest.approx(a_true, abs=1e-6)


def test_oracle_correction_respects_the_feasible_interval():
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    n_Y = np.array([0.1, -0.1, 0.05, -0.05])
    n_Y = n_Y / np.linalg.norm(n_Y)
    Y_B = np.array([0.4, 0.3, 0.2, 0.1])
    iv = feasible_a_interval(Y_B, n_Y)
    # a target far outside the feasible interval: the oracle must not leave it
    q_target = np.concatenate([[800.0], Y_B + 50.0 * n_Y])
    out = oracle_correction_coefficient(q_target, Y_B, n_Y, 1.5e6, dec)
    assert out["resolved"] is True
    assert iv["a_lo"] <= out["a"] <= iv["a_hi"]


# ---------------------------------------------------------------------------
# 5. the regressor: frozen regularization, documented complexity
# ---------------------------------------------------------------------------


def test_regressor_recovers_an_affine_map_exactly():
    rng = np.random.default_rng(0)
    true = np.array([[0.5, -0.3], [0.1, 0.2], [-0.2, 0.7]])   # (3, 2)
    b = np.array([0.01, -0.02])
    etas = [rng.normal(size=3) for _ in range(20)]
    targets = np.stack([e @ true + b for e in etas])
    reg = LinearRegressor(_affine_features, order=1,
                         ridge=1e-12).fit(etas, targets)
    for e, t in zip(etas, targets):
        np.testing.assert_allclose(reg.predict(e), t, atol=1e-8)


def test_regressor_quadratic_features_have_expected_size():
    rng = np.random.default_rng(1)
    etas = [rng.normal(size=3) for _ in range(12)]
    reg = LinearRegressor(_affine_features, order=2,
                         ridge=1e-10).fit(etas, np.zeros((12, 1)))
    # 1 + 3 + 3 + 3 = 10 features for m = 3
    assert len(reg.feature_names_) == 1 + 3 + 3 + 3
    assert reg.cond_XtX_ > 0


def test_regressor_never_penalizes_the_intercept():
    """A constant target with zero ridge on the intercept must be recovered
    exactly however large the ridge on the slopes is."""
    rng = np.random.default_rng(2)
    etas = [rng.normal(size=3) for _ in range(15)]
    reg = LinearRegressor(_affine_features, order=1,
                         ridge=1.0).fit(etas, np.full((15, 1), 0.37))
    for e in etas:
        assert reg.predict(e)[0] == pytest.approx(0.37, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. the exposure integral
# ---------------------------------------------------------------------------


def test_exposure_integral_is_additive_over_segments():
    g = [10.0, 20.0, 30.0]
    d = [0.1, 0.2, 0.3]
    assert exposure_integral(g, d) == pytest.approx(1.0 + 4.0 + 9.0)


# ---------------------------------------------------------------------------
# 7. the scaled metric: the projection must be metric-consistent
# ---------------------------------------------------------------------------


def test_scaled_projection_returns_a_physical_conservation_direction():
    """The projected direction must satisfy E n_Y = 0 and 1^T n_Y = 0 in PHYSICAL
    mass-fraction units, and the projection must be orthogonal in the scaled
    metric (not a mix of the two metrics)."""
    rng = np.random.default_rng(9)
    y_int = 0.01
    Q = np.asarray(conservation_subspace(E_STUB, y_interval=y_int)["basis"])
    w = rng.normal(size=4) + 5.0 * Q[:, 0]
    rep = project_onto_conservation_subspace(w, E_STUB, y_interval=y_int)
    assert rep["resolved"] is True
    n = np.asarray(rep["n_Y"])
    # physical constraints hold exactly
    np.testing.assert_allclose(E_STUB @ n, 0.0, atol=1e-12)
    assert abs(n.sum()) < 1e-12
    # and the direction is a mass-fraction-scale direction: its norm is of the
    # order of the declared Y interval, not of the scaled unit
    assert np.linalg.norm(n) == pytest.approx(y_int, rel=1e-9)


def test_scaled_and_unscaled_projections_agree_in_direction():
    """Scaling the metric must not change the DIRECTION the projection selects,
    only its normalization - so the two are the same physical direction."""
    rng = np.random.default_rng(10)
    w = rng.normal(size=4)
    r1 = project_onto_conservation_subspace(w, E_STUB, y_interval=1.0)
    r2 = project_onto_conservation_subspace(w, E_STUB, y_interval=0.01)
    assert r1["resolved"] and r2["resolved"]
    n1 = np.asarray(r1["n_Y"])
    n2 = np.asarray(r2["n_Y"])
    # same direction, different normalization by exactly the interval ratio
    cos = abs(float(n1 @ n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    assert cos == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.norm(n2) / np.linalg.norm(n1) == pytest.approx(0.01)


def test_null_space_basis_detects_rank_deficiency():
    C = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 1.0, 0.0]])
    out = null_space_basis(C)
    assert out["rank_constraints"] == 2
    assert out["dim"] == 1
    n = np.asarray(out["basis"])[:, 0]
    np.testing.assert_allclose(C @ n, 0.0, atol=1e-10)



# ---------------------------------------------------------------------------
# 8. the exposure-constrained reference-coordinate parametrization
# ---------------------------------------------------------------------------


def test_exposure_log_feature_is_the_exact_exposure():
    eta = np.log(np.array([0.5, 2.0, 1.0]) / 1e4)
    durs = np.array([1e-7, 3e-7, 6e-7])
    lg = exposure_log_feature(eta, durs, 1e-6, 1e4)
    gammas = 1e4 * np.exp(eta)
    assert lg == pytest.approx(np.log(np.sum(gammas * durs) / (1e4 * 1e-6)))


def test_generator_features_expose_the_exposure_and_the_segment_ratios():
    eta = np.array([0.1, -0.2, 0.3])
    durs = np.full(3, 1e-6 / 3)
    f1 = generator_features(eta, durs, 1e-6, 1e4, order=1)
    # [1, log Gamma, eta_1, eta_2, eta_3]
    assert f1.shape == (5,)
    assert f1[0] == pytest.approx(1.0)
    assert f1[1] == pytest.approx(exposure_log_feature(eta, durs, 1e-6, 1e4))
    np.testing.assert_allclose(f1[2:], eta)
    f2 = generator_features(eta, durs, 1e-6, 1e4, order=2)
    # 1 + 4 base + 4 squares + 6 products = 15
    assert f2.shape == (15,)
    assert np.all(np.isfinite(f2))
    # order 2 strictly contains the order-1 features
    f1b = generator_features(eta, durs, 1e-6, 1e4, order=1)
    np.testing.assert_allclose(f2[:f1b.size], f1b)


def test_exposure_parametrization_reproduces_the_oracle_gamma():
    """The parametrization log gamma_B = log Gamma - v + u must reproduce the
    oracle coordinates exactly when u and v are the oracle residuals: u is tiny
    by the exact balance law, and gamma_B * t_B = Gamma_B holds."""
    eta = np.array([0.05, -0.1, 0.2])
    durs = np.full(3, 1e-6 / 3)
    gref, T = 1e4, 1e-6
    gammas = gref * np.exp(eta)
    Gamma = float(np.sum(gammas * durs))
    # a located reference with t_B = 0.9 T, so Gamma_B = gamma_B t_B
    t_B = 0.9 * T
    gamma_B = Gamma / t_B
    lg = np.log(gamma_B / gref)
    lt = np.log(t_B / T)
    log_gamma_ex = exposure_log_feature(eta, durs, T, gref)
    u = (lg + lt) - log_gamma_ex
    v = lt
    # round trip
    assert np.log(gamma_B / gref) == pytest.approx(log_gamma_ex - v + u)
    # and the exact balance law makes u small in practice (see the report)
    assert abs(u) < 0.05


def test_oracle_search_resolves_an_optimum_far_below_the_interval_width():
    """The feasible interval can be orders of magnitude wider than the optimal a
    (the interval is set by the least abundant species, the optimum by the
    transverse excursion).  A uniform grid would miss it; the multi-scale search
    must resolve a optimum at ~1e-3 of the interval width."""
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    n_Y = np.array([0.1, -0.1, 0.05, -0.05])
    n_Y = n_Y / np.linalg.norm(n_Y)
    Y_B = np.array([0.4, 0.3, 0.2, 0.1])
    iv = feasible_a_interval(Y_B, n_Y)
    a_true = 0.05 * iv["interval_width"]            # far below the width
    Y_true = Y_B + a_true * n_Y
    h = float(gas.a @ Y_true) + 1200.0 * 800.0
    q_target = np.concatenate([[dec.decode(Y_true, h)["T_K"]], Y_true])
    out = oracle_correction_coefficient(q_target, Y_B, n_Y, h, dec)
    assert out["resolved"] is True
    assert out["a"] == pytest.approx(a_true, rel=1e-6)
    assert out["n_candidates"] > 20


def test_oracle_search_finds_zero_when_the_family_is_exact():
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6], c=1200.0)
    dec = _DecoderStub(gas)
    n_Y = np.array([0.1, -0.1, 0.05, -0.05])
    n_Y = n_Y / np.linalg.norm(n_Y)
    Y_B = np.array([0.4, 0.3, 0.2, 0.1])
    h = float(gas.a @ Y_B) + 1200.0 * 800.0
    # the target IS the family member: the best correction is exactly zero
    q_target = np.concatenate([[dec.decode(Y_B, h)["T_K"]], Y_B])
    out = oracle_correction_coefficient(q_target, Y_B, n_Y, h, dec)
    assert out["resolved"] is True
    assert out["a"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Revision 8: the structural face of the species polytope
# ---------------------------------------------------------------------------

# A 5-species, 3-element stub that mimics h2o2.yaml's structure:
#   S0, S1, S2, S3 are reactive and built from elements 0 and 1
#   S4 (the "argon") is built only from element 2, which the feed lacks entirely
#   S5 (the "nitrogen") is inert and present in the feed at its initial value
E_FACE = np.array([
    [2.0, 0.0, 2.0, 1.0, 0.0, 0.0],       # element 0
    [0.0, 2.0, 1.0, 1.0, 0.0, 0.0],       # element 1
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],       # element 2: only in S4 and S5
])
FACE_NAMES = ["H2", "O2", "H2O", "OH", "AR", "N2"]


class _FaceGas:
    """Minimal gas interface for active_species_report: species names, element
    names, stoichiometry, and no thermo (the report needs no thermo)."""

    def __init__(self, n_species=6, n_elements=3, n_reactions=2):
        self.n_species = n_species
        self.n_elements = n_elements
        self.n_reactions = n_reactions
        # S4 and S5 appear in no reaction
        nu = np.zeros((n_species, n_reactions))
        nu[0, 0], nu[1, 0], nu[2, 0] = -1.0, -0.5, 1.0
        nu[0, 1], nu[3, 1], nu[2, 1] = -1.0, 1.0, 1.0
        self.product_stoich_coeffs = np.clip(nu, 0, None)
        self.reactant_stoich_coeffs = -np.clip(nu, None, 0)

    @property
    def species_names(self):
        return FACE_NAMES[: self.n_species]

    def element_name(self, e):
        return ["H", "O", "Ar"][e]


class _FaceCstr:
    """The feed/initial state: no AR at all, N2 inert at its initial value.

    A stoichiometric H2/O2 feed with N2 as the inert diluent (mimicking
    h2o2.yaml's H2 / O2:1,N2:3.76, phi = 1).  Element 2's inventory is carried
    only by N2; AR has no feed material at all."""

    def __init__(self):
        self.gas = _FaceGas()
        self.E = E_FACE
        self.Y_in = np.array([0.0285, 0.2263, 0.0, 0.0, 0.0, 0.7452])
        self.cfg = type("cfg", (), {})()
        self.cfg.mechanism = "h2o2.yaml"
        self.cfg.pressure = 101325.0
        self.cfg.inlet_temperature = 1200.0
        self.cfg.fuel = "H2"
        self.cfg.oxidizer = "O2:1,N2:3.76"
        self.cfg.equivalence_ratio = 1.0
        self.h_in = 1.0e9
        self.b_in = self.E @ self.Y_in
        self._hk = np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6, 1.0e5, 2.0e5])

    def species_enthalpies(self, T):
        return self._hk


def _face_report():
    cstr = _FaceCstr()
    # initial state: the feed, with AR exactly zero and N2 at its feed value
    q0 = np.concatenate([[1200.0], cstr.Y_in.copy()])
    return active_species_report(cstr, q0), cstr, q0


def test_structural_classification_of_the_species_polytope():
    rep, _, _ = _face_report()
    assert rep["absent_indices"] == [4]            # AR: no feed material at all
    assert rep["fixed_indices"] == [5]             # N2: inert, Y0 == Y_in
    assert rep["active_indices"] == [0, 1, 2, 3]
    assert rep["n_active"] == 4
    assert rep["inconsistent"] == []               # nothing surprising
    # the element cross-check: element 2 is carried only by the absent AR and
    # the fixed N2, which is exactly why both are structurally excluded
    assert rep["element_carriers"]["Ar"] == ["AR", "N2"]
    assert rep["b_in"][2] == pytest.approx(0.7452)
    assert rep["b_in"][2] == pytest.approx(
        float(_FaceCstr().Y_in[5]))       # all of element 2 sits in N2


def test_structural_zeros_remove_the_one_sided_interval():
    """Without the mask, SVD noise of 1e-19 in the AR component of the direction
    meets the exactly-zero Y_AR and imposes an artificial a >= 0.  The structural
    mask must restore a TWO-SIDED interval."""
    rep, _, _ = _face_report()
    mask = structural_zero_mask(rep)
    Y_B = np.array([0.030, 0.220, 0.004, 0.0008, 0.0, 0.7452])
    # a conservation-compatible direction with injected numerical noise in AR
    n_clean = np.array([0.6, -0.5, -0.15, 0.05, 0.0, 0.0])
    n_clean = n_clean / np.linalg.norm(n_clean) * 0.01
    n_noisy = n_clean.copy()
    n_noisy[4] = 9.759474952003097e-19       # the exact frozen-model value

    iv_noisy = feasible_a_interval(Y_B, n_noisy, FACE_NAMES)
    assert iv_noisy["a_lo"] == pytest.approx(0.0)      # the defect
    assert iv_noisy["bounded"] is True
    assert any(e["name"] == "AR" for e in iv_noisy["inconsistent"])

    iv_fixed = feasible_a_interval(Y_B, apply_structural_zeros(n_noisy, rep),
                                   FACE_NAMES, fixed_mask=mask)
    assert iv_fixed["bounded"] is True
    assert iv_fixed["a_lo"] < 0.0 < iv_fixed["a_hi"]    # two-sided again
    assert iv_fixed["inconsistent"] == []
    assert iv_fixed["binding_name_low"] != "AR"
    # the masked entries are reported with their original indices and names
    assert any(r["name"] == "AR" for r in iv_fixed["skipped"]["masked_fixed"])


def test_injected_1e18_component_does_not_make_the_interval_one_sided():
    """A regression guard for the exact defect found in the frozen model."""
    rep, _, _ = _face_report()
    mask = structural_zero_mask(rep)
    Y_B = np.array([0.030, 0.220, 0.004, 0.0008, 0.0, 0.7452])
    n = np.array([0.6, -0.5, -0.15, 0.05, 0.0, 0.0])
    n = n / np.linalg.norm(n) * 0.01
    n[4] = 1e-18                      # a numerical artifact, not a physical entry
    iv = feasible_a_interval(Y_B, apply_structural_zeros(n, rep), FACE_NAMES,
                             fixed_mask=mask)
    assert iv["bounded"] is True
    assert iv["a_lo"] < 0.0, "an injected 1e-18 component must not impose a >= 0"
    assert abs(n[4]) > 0.0           # the raw direction was NOT modified


def _face_direction():
    """A conservation-compatible direction for the face, from the real projection
    code, so the sign/conservation tests exercise the actual code path."""
    rep, _, _ = _face_report()
    rng = np.random.default_rng(3)
    out = project_onto_active_conservation_subspace(rng.normal(size=6), E_FACE,
                                                    rep["active_mask"])
    assert out["resolved"] is True
    return np.asarray(out["n_Y"]) * 0.01, rep


def test_sign_equivalent_representations_agree():
    """(n, a) and (-n, -a) are the same correction: the feasible intervals are
    the same set, and the decoded states coincide."""
    Y_B = np.array([0.030, 0.220, 0.004, 0.0008, 0.0, 0.7452])
    n, _ = _face_direction()
    iv_p = feasible_a_interval(Y_B, n, FACE_NAMES)
    iv_m = feasible_a_interval(Y_B, -n, FACE_NAMES)
    assert iv_m["a_lo"] == pytest.approx(-iv_p["a_hi"])
    assert iv_m["a_hi"] == pytest.approx(-iv_p["a_lo"])
    # the binding species swap roles
    assert iv_m["binding_name_low"] == iv_p["binding_name_high"]
    assert iv_m["binding_name_high"] == iv_p["binding_name_low"]
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6, 1e5, 2e5], c=1200.0)
    dec = _DecoderStub(gas, E=E_FACE)
    h = 1.2e9
    a0 = 0.5 * (iv_p["a_lo"] + iv_p["a_hi"])
    d_p = dec.decode(Y_B + a0 * n, h)
    d_m = dec.decode(Y_B + (-a0) * (-n), h)
    assert d_p["status"] == d_m["status"] == "admissible"
    assert d_p["T_K"] == pytest.approx(d_m["T_K"])
    assert np.allclose(d_p["q_hat"], d_m["q_hat"])


def test_small_corrections_of_both_signs_are_admissible_and_conserving():
    """Both signs of a small admissible correction decode to admissible states
    that preserve the inventory, the normalization and the enthalpy."""
    n, rep = _face_direction()
    mask = structural_zero_mask(rep)
    Y_B = np.array([0.030, 0.220, 0.004, 0.0008, 0.0, 0.7452])
    assert E_FACE @ n == pytest.approx(np.zeros(3), abs=1e-12)
    assert n.sum() == pytest.approx(0.0, abs=1e-12)
    assert n[4] == 0.0 and n[5] == 0.0                  # AR and N2 untouched
    iv = feasible_a_interval(Y_B, n, FACE_NAMES, fixed_mask=mask)
    assert iv["bounded"] is True and iv["a_lo"] < 0.0 < iv["a_hi"]
    gas = _GasStub(a=[1e6, 2e6, 3e6, 4e6, 1e5, 2e5], c=1200.0)
    dec = _DecoderStub(gas, E=E_FACE)
    h = float(gas.a @ Y_B) + 1200.0 * 800.0
    b0 = E_FACE @ Y_B
    for a in (iv["a_lo"], 0.5 * iv["a_lo"], -1e-4, 1e-4, 0.5 * iv["a_hi"],
              iv["a_hi"]):
        d = dec.decode(Y_B + a * n, h)
        assert d["status"] == "admissible", (a, d.get("reason"))
        assert np.allclose(E_FACE @ np.asarray(d["Y"]), b0, atol=1e-10)
        assert sum(d["Y"]) == pytest.approx(1.0, abs=1e-9)
        assert d["Y"][4] == pytest.approx(0.0, abs=1e-15)   # AR stays absent
        assert d["Y"][5] == pytest.approx(Y_B[5], abs=1e-15)  # N2 stays fixed
        # enthalpy inversion round-trips
        assert gas.a @ np.asarray(d["Y"]) + gas.c * d["T_K"] \
            == pytest.approx(h, rel=1e-9)

def test_active_conservation_subspace_excludes_the_fixed_species():
    sub = conservation_subspace_active(E_FACE, np.array(
        [True, True, True, True, False, False]), y_interval=1.0)
    Q = np.asarray(sub["basis"])
    assert Q.shape == (4, sub["dim"])
    assert sub["dim"] == 4 - 2 - 1                      # 4 species, 2 elements
    for col in range(Q.shape[1]):
        v = np.zeros(6)
        v[[0, 1, 2, 3]] = Q[:, col]
        assert np.allclose(E_FACE @ v, 0.0, atol=1e-10)
        assert abs(v.sum()) < 1e-10


def test_projection_into_the_active_subspace_embeds_exact_zeros():
    rep, _, _ = _face_report()
    rng = np.random.default_rng(11)
    w = rng.normal(size=6)
    out = project_onto_active_conservation_subspace(w, E_FACE,
                                                   rep["active_mask"])
    assert out["resolved"] is True
    n = np.asarray(out["n_Y"])
    assert n.shape == (6,)
    assert n[4] == 0.0 and n[5] == 0.0                 # exact, not 1e-19
    assert np.allclose(E_FACE @ n, 0.0, atol=1e-10)
    assert abs(n.sum()) < 1e-10
    assert np.linalg.norm(n) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Revision 8 section C: what the exact balance law can and cannot identify
# ---------------------------------------------------------------------------


class _BalanceCstr:
    """Exactly the fields the closed-form mixing law needs - no thermo."""

    def __init__(self, hk_at_T0, b_in, h_in):
        self._hk = np.asarray(hk_at_T0, dtype=float)
        self.E = np.eye(len(hk_at_T0))          # b = Y itself, one "element" each
        self.b_in = np.asarray(b_in, dtype=float)
        self.h_in = float(h_in)

    def species_enthalpies(self, T):
        return self._hk


def test_feed_compatible_invariants_cannot_identify_the_exposure():
    """The hot-HP start of the same feed has h0 = h_in and b0 = b_in, so both
    invariants are CONSTANT for every control: no measurement of h or b can
    recover Gamma, and the exposure relation must be an empirical bias."""
    hk = np.array([1e6, 2e6, 3e6])
    # the initial state is the SAME feed: b0 = b_in and h0 = h_in exactly
    h_in = float(hk @ hk)
    cstr = _BalanceCstr(hk, b_in=hk, h_in=h_in)
    q0 = np.concatenate([[1200.0], hk.copy()])
    assert exact_balances_from_controls(cstr, q0, [1.0], [1.0])["h0_J_kg"]         == pytest.approx(h_in)
    low = exact_balances_from_controls(cstr, q0, [1e3], [1e-6])
    high = exact_balances_from_controls(cstr, q0, [1e5], [1e-6])
    # the exposures differ by 100x ...
    assert high["Gamma"] == pytest.approx(100.0 * low["Gamma"])
    # ... yet the invariants are bit-identical: no exposure information
    assert high["h_J_kg"] == low["h_J_kg"]
    assert high["b"] == low["b"]
    assert low["dec"] == pytest.approx(math.exp(-low["Gamma"]))
    # the report says so explicitly
    assert abs(low["h0_J_kg"] - low["h_in_J_kg"]) <= 0.0
    assert np.allclose(low["b0"], low["b_in"])


def test_a_nonmatching_invariant_identifies_the_exposure_when_conditioned():
    """If b0 != b_in the decaying factor e^-Gamma is observable, and the exposure
    is recoverable - but the conditioning is set by |b0 - b_in|."""
    hk = np.array([1e6, 2e6, 3e6])
    b_in = hk
    cstr = _BalanceCstr(hk, b_in=b_in, h_in=1.0e9)
    # a deliberately nonmatching initial inventory: species 0 is enriched
    Y0 = hk * np.array([1.5, 1.0, 1.0])
    Y0 = Y0 / Y0.sum() * b_in.sum()
    q0 = np.concatenate([[1200.0], Y0])
    assert not np.allclose(cstr.E @ Y0, b_in)       # the law is NOT degenerate
    for Gamma in (1e-3, 0.5, 3.0):
        # choose a single-segment history with exactly this exposure
        bal = exact_balances_from_controls(cstr, q0, [1.0], [Gamma])
        assert bal["Gamma"] == pytest.approx(Gamma)
        assert bal["dec"] == pytest.approx(math.exp(-Gamma))
        # the invariant has actually moved away from b_in
        b = np.asarray(bal["b"], dtype=float)
        assert np.linalg.norm(b - b_in) > 0.0
        # and the exposure is recoverable from it, by inverting the law
        d = np.asarray(bal["b"], dtype=float) - b_in
        d0 = np.asarray(bal["b0"], dtype=float) - b_in
        Gamma_back = -math.log(np.linalg.norm(d) / np.linalg.norm(d0))
        assert Gamma_back == pytest.approx(Gamma, rel=1e-9)


def test_a_nearly_degenerate_invariant_makes_the_exposure_ill_conditioned():
    """As b0 -> b_in the same measurement noise in b maps to a larger error in
    Gamma: the recovery is well conditioned only while |b0 - b_in| is well above
    the measurement floor, so the conditioning is set by the OFFSET."""
    hk = np.array([1e6, 2e6, 3e6])
    cstr = _BalanceCstr(hk, b_in=hk, h_in=float(hk @ hk))
    # a relative measurement floor on the inventory, independent of the offset
    noise = 1e-6 * float(np.linalg.norm(cstr.b_in))
    gamma_true = 0.7
    rel_errs = {}
    for scale in (1e-1, 1e-4, 1e-7):
        Y0 = hk * (1.0 + scale * np.array([1.0, 0.0, 0.0]))
        q0 = np.concatenate([[1200.0], Y0])
        bal = exact_balances_from_controls(cstr, q0, [1.0], [gamma_true])
        d = np.asarray(bal["b"], dtype=float) - cstr.b_in
        d0 = np.asarray(bal["b0"], dtype=float) - cstr.b_in
        offset = float(np.linalg.norm(d0))
        # the exposure recovered from a noise-corrupted invariant
        r = np.linalg.norm(d + noise) / max(offset, 1e-300)
        gamma_back = -math.log(r) if r > 0 else float("inf")
        rel_errs[scale] = (abs(gamma_back - gamma_true) / gamma_true, offset)
    # the largest offset is well conditioned; the smallest is not
    assert rel_errs[1e-1][0] < 1e-3
    assert rel_errs[1e-1][0] < rel_errs[1e-4][0] < rel_errs[1e-7][0]
    assert rel_errs[1e-7][1] < noise      # the offset is below the floor
