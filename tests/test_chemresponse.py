"""Tests for the chemistry-response machinery: declared control norms, the
admissible radius interval, the cached continuous reference, and the
non-terminal ignition event.

The reference and chemistry objects are tested against an exactly-integrable
decoupled stub so the assertions are about the mathematics, not about a solver.
Cantera-dependent paths (the real CSTR, the chemistry map itself) are skipped
where Cantera is unavailable and run on the compute server otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.chemresponse import (  # noqa: E402
    ChemistryConfig, ReferenceLibrary, admissible_radius_interval,
    chemistry_map, chemistry_response, chemistry_source, hdiag,
    local_family_tangent_at, located_reference, normal_residual,
    radius_from_deta, refine_reference, time_norm, unwhiten_svd_direction,
    unit_time_norm_direction, whitened_map,
)
from thermoreach.sensitivity import ignition_time_event, svd_basis  # noqa: E402


# ---------------------------------------------------------------------------
# exactly-integrable stub: dq_k/dt = -k q_k + gamma, k = 1..n
# family point B(gamma, t)[k] = gamma (1 - exp(-k t)) / k
# ---------------------------------------------------------------------------


class _StubCFG:
    mechanism = "stub-decoupled"
    phase = None
    pressure = 1.0


class _StubCSTR:
    def __init__(self, n: int = 3):
        self._n = n
        self.cfg = _StubCFG()

    @property
    def rhs(self):
        k = np.arange(1, self._n + 1, dtype=float)

        def rhs(t, y, gamma):
            return -k * np.asarray(y, dtype=float) + float(gamma)

        return rhs


def stub_family(gamma: float, t: float, n: int = 3) -> np.ndarray:
    k = np.arange(1, n + 1, dtype=float)
    return float(gamma) * (1.0 - np.exp(-k * float(t))) / k


# ---------------------------------------------------------------------------
# 1. declared control norms
# ---------------------------------------------------------------------------


def test_time_norm_equal_segments_matches_euclidean_rescaling():
    v = np.array([0.3, -0.5, 0.8])
    durs = np.full(3, 1.0 / 3)                     # m = 3 equal segments, T = 1
    # ||v||_time = ||v||_e / sqrt(m) for equal segments
    assert time_norm(v, durs, 1.0) == pytest.approx(np.linalg.norm(v) / np.sqrt(3))
    # a Euclidean unit vector has time norm 1/sqrt(m)
    u = v / np.linalg.norm(v)
    assert time_norm(u, durs, 1.0) == pytest.approx(1.0 / np.sqrt(3))
    # rescaling to unit time norm preserves neither the Euclidean norm nor the
    # direction's meaning silently: the Euclidean radius r maps to r/sqrt(m)
    w = unit_time_norm_direction(u, durs, 1.0)
    assert time_norm(w, durs, 1.0) == pytest.approx(1.0)
    assert np.linalg.norm(w) == pytest.approx(np.sqrt(3))


def test_whitened_map_singular_values_scale_with_sqrt_m():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    durs = np.full(3, 0.4)                          # T = 1.2, so H = diag(1/3)
    Aw = whitened_map(A, durs, 1.2)
    sv_a = np.linalg.svd(A, compute_uv=False)
    sv_w = np.linalg.svd(Aw, compute_uv=False)
    # H = (1/3) I  ->  A H^{-1/2} = sqrt(3) A
    np.testing.assert_allclose(sv_w, np.sqrt(3) * sv_a, rtol=1e-12)


def test_whitened_map_input_is_measured_in_time_norm():
    # a unit-Euclidean right singular vector of A H^{-1/2} corresponds to a
    # unit-TIME-norm direction in eta coordinates
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 2))
    durs = np.array([0.25, 0.75])                   # T = 1
    Aw = whitened_map(A, durs, 1.0)
    _, _, Vh = np.linalg.svd(Aw, full_matrices=False)
    v = Vh[0]
    eta_dir = v / np.sqrt(hdiag(durs, 1.0))         # invert H^{-1/2}
    assert time_norm(eta_dir, durs, 1.0) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. exact admissible radius interval
# ---------------------------------------------------------------------------


def test_admissible_radius_interval_mixed_signs():
    """Every segment bounds r from above AND below when the direction is mixed in
    sign: a negative-v segment reaches the UPPER level bound at a positive r, and
    the LOWER level bound at a negative r."""
    theta = np.array([1e4, 1e4, 1e4])
    v = np.array([0.5, -0.3, 0.1])
    r = admissible_radius_interval(theta, v, 10.0, 1e5)
    # upper bound: min over j of the per-segment upper bounds
    #   j=0 (v=0.5):  log(1e5/1e4)/0.5 = 4.605
    #   j=1 (v=-0.3): log(10/1e4)/-0.3 = 23.03
    #   j=2 (v=0.1):  log(1e5/1e4)/0.1 = 23.03
    assert r["r_max_positive"] == pytest.approx(np.log(10.0) / 0.5)
    assert r["binding_constraint_positive"].startswith("segment 0")
    # lower bound: max over j of the per-segment lower bounds
    #   j=0 (v=0.5):  log(10/1e4)/0.5  = -13.82
    #   j=1 (v=-0.3): log(1e5/1e4)/-0.3 = -7.676   <- binding
    #   j=2 (v=0.1):  log(10/1e4)/0.1  = -69.08
    assert r["r_min_negative"] == pytest.approx(np.log(10.0) / -0.3)
    assert r["binding_constraint_negative"].startswith("segment 1")


def test_admissible_radius_interval_all_positive():
    """Positive-v segments bound r from above through the upper level bound and
    from below through the lower level bound."""
    theta = np.array([1e4, 1e4])
    v = np.array([0.2, 0.7])
    r = admissible_radius_interval(theta, v, 10.0, 1e5)
    assert r["r_max_positive"] == pytest.approx(np.log(10.0) / 0.7)     # 3.289
    assert r["r_min_negative"] == pytest.approx(np.log(1e-3) / 0.7)     # -9.868


def test_admissible_radius_interval_all_negative():
    theta = np.array([1e4, 1e4])
    v = np.array([-0.2, -0.7])
    r = admissible_radius_interval(theta, v, 10.0, 1e5)
    # signs flip: the upper level bound now binds at a POSITIVE radius
    assert r["r_max_positive"] == pytest.approx(np.log(1e-3) / -0.7)    # 9.868
    assert r["r_min_negative"] == pytest.approx(np.log(10.0) / -0.7)    # -3.289


def test_admissible_radius_interval_symmetric_bounds_hold_at_both_ends():
    theta = np.array([1e4, 1e4])
    v = np.array([-0.2, 0.7])
    r = admissible_radius_interval(theta, v, 10.0, 1e5)
    for rr in (r["r_max_positive"], r["r_min_negative"]):
        assert rr is not None
        g = theta * np.exp(rr * v)
        assert float(np.max(g)) <= 1e5 * (1.0 + 1e-12)
        assert float(np.min(g)) >= 10.0 * (1.0 - 1e-12)


def test_admissible_radius_levels_respect_bounds_exactly():
    theta = np.array([1e4, 1e4, 1e4])
    v = np.array([0.5, -0.3, 0.1])
    r = admissible_radius_interval(theta, v, 10.0, 1e5)
    for rr, hi, lo in ((r["r_max_positive"], 1e5, 10.0),
                      (r["r_min_negative"], 1e5, 10.0)):
        g = theta * np.exp(rr * v)
        # the binding segment reaches its bound exactly; the others stay inside
        assert float(np.max(g)) <= hi * (1.0 + 1e-12)
        assert float(np.min(g)) >= lo * (1.0 - 1e-12)


# ---------------------------------------------------------------------------
# 3. cached continuous reference against the exact stub family
# ---------------------------------------------------------------------------


def _library(t_max=2.0, t_anchor=0.5, n_gamma=9):
    cstr = _StubCSTR(3)
    q0 = np.zeros(3)
    grid = np.exp(np.linspace(np.log(1.0), np.log(20.0), n_gamma))
    return ReferenceLibrary(cstr, q0, grid, t_max=t_max, t_anchor=t_anchor,
                            method="LSODA", rtol=1e-11, atol_T=1e-12,
                            atol_Y=1e-12, t_grid_points=65)


def test_reference_library_recovers_exact_family_point():
    lib = _library()
    q = stub_family(7.5, 0.37)
    coarse = lib.coarse_min(q, np.ones(3))
    # the library is a COARSE comparator: log-gamma interpolation between grid
    # points is not exact, so the coarse distance is small but not at solver
    # accuracy.  Exactness is the refinement stage's job.
    assert coarse["dist_scaled_upper_estimate"] < 1e-1
    assert 5.0 < coarse["best_gamma"] < 11.0
    # the coarse t is only approximate: at a wrong gamma the optimal t is not the
    # target's t, because the family direction changes with gamma
    assert 0.1 < coarse["best_t"] < 1.0
    # t grid contains zero and the anchor, as declared
    assert coarse["t_grid"]["contains_zero"]
    assert coarse["t_grid"]["contains_anchor"]


def test_reference_library_identity_and_cache_validity():
    lib = _library()
    ident = lib.identity()
    assert ident["mechanism"] == "stub-decoupled"
    assert ident["n_integrated"] == 9 and ident["n_failed"] == 0
    assert ident["t_max"] == 2.0 and ident["t_anchor"] == 0.5
    # a state outside the reference domain cannot be evaluated (no NaN silently)
    q = lib.evaluate(1e6, 0.1)
    assert np.all(np.isnan(q))


def test_refine_reference_recovers_exact_family_point():
    lib = _library()
    cstr = _StubCSTR(3)
    q = stub_family(7.5, 0.37)
    coarse = lib.coarse_min(q, np.ones(3))
    ref = refine_reference(lib, q, np.ones(3), coarse, cstr,
                           refine_method="LSODA", refine_rtol=1e-12,
                           refine_atol_T=1e-13, refine_atol_Y=1e-13)
    assert ref["success"]
    assert ref["dist_scaled_upper_estimate"] < 1e-6
    assert ref["gamma_best"] == pytest.approx(7.5, rel=1e-4)
    assert ref["t_best"] == pytest.approx(0.37, rel=1e-3)
    assert ref["n_fresh_integrations"] >= 1


def test_local_family_tangent_rank_two_and_exact_dbdt():
    cstr = _StubCSTR(3)
    q0 = np.zeros(3)
    gamma, t = 5.0, 0.4
    q_B = stub_family(gamma, t)
    k = np.arange(1, 4, dtype=float)
    tan = local_family_tangent_at(cstr, q0, q_B, gamma, t, np.ones(3),
                                  method="LSODA", rtol=1e-12,
                                  atol_T=1e-13, atol_Y=1e-13, dlog=1e-4)
    assert tan["success"] and tan["rank"] == 2
    # dB/dt is exact: it is the ODE right-hand side at that state
    np.testing.assert_allclose(tan["dBdt"], cstr.rhs(0.0, q_B, gamma), rtol=1e-12)
    # dB/dlog(gamma) exact: gamma * (1 - exp(-k t))/k
    np.testing.assert_allclose(tan["dBdloggamma"], gamma * q_B / gamma,
                               rtol=1e-6)


def test_normal_residual_vanishes_on_the_family():
    cstr = _StubCSTR(3)
    q0 = np.zeros(3)
    gamma, t = 5.0, 0.4
    q_B = stub_family(gamma, t)
    nr = normal_residual(cstr, q0, q_B, q_B, gamma, t, np.ones(3),
                         method="LSODA", rtol=1e-12, atol_T=1e-13,
                         atol_Y=1e-13, dlog=1e-4)
    assert nr["available"]
    assert nr["normal_residual_norm"] == pytest.approx(0.0, abs=1e-9)
    assert nr["tangent_residual_norm"] == pytest.approx(0.0, abs=1e-9)


def test_normal_residual_detects_off_family_displacement():
    cstr = _StubCSTR(3)
    q0 = np.zeros(3)
    gamma, t = 5.0, 0.4
    q_B = stub_family(gamma, t)
    e = np.array([0.0, 0.0, 0.5])                    # purely off the family plane?
    q = q_B + e
    nr = normal_residual(cstr, q0, q, q_B, gamma, t, np.ones(3),
                         method="LSODA", rtol=1e-12, atol_T=1e-13,
                         atol_Y=1e-13, dlog=1e-4)
    # the total residual is e; the normal part is its component off the tangent
    assert nr["total_residual_norm"] == pytest.approx(0.5, rel=1e-9)
    assert nr["normal_residual_norm"] > 0.0
    assert nr["normal_residual_norm"] <= 0.5 + 1e-9


# ---------------------------------------------------------------------------
# 4. non-terminal ignition event: known event, several horizons, censoring
# ---------------------------------------------------------------------------


class _EventStub:
    """q' = 2t  ->  q(t) = t^2; the event q = 0.25 occurs exactly at t = 0.5."""
    class cfg:
        mechanism = "stub-quadratic"
        phase = None
        pressure = 1.0

    def rhs(self, t, y, gamma):
        return np.array([2.0 * float(t)])


def test_ignition_event_located_and_dense_output_valid_over_horizon():
    stub = _EventStub()
    for horizon in (0.3, 1.0, 2.0):
        r = ignition_time_event(stub, 1.0, horizon, np.array([0.0]),
                                rise_K=0.25, method="LSODA", rtol=1e-12,
                                atol_T=1e-14, atol_Y=1e-14, grid_resolution=50)
        assert r["integration_success"] is True
        # dense output must be valid over the whole integrated interval
        assert r["integrated_interval_end_s"] == pytest.approx(horizon)
        assert r["max_temperature_K"] == pytest.approx(horizon ** 2, rel=1e-6)
        if horizon < 0.5:
            # censored: no crossing within the horizon, NOT an integration failure
            assert r["censored"] is True
            assert r["t_ign_event"] is None
        else:
            assert r["censored"] is False
            assert r["t_ign_event"] == pytest.approx(0.5, rel=1e-6)
            assert r["n_upward_crossings"] == 1


def test_ignition_event_at_the_horizon_boundary_is_consistent():
    """An event exactly at the horizon may be detected at the endpoint or censored;
    either way the two flags must stay mutually consistent."""
    stub = _EventStub()
    r = ignition_time_event(stub, 1.0, 0.5, np.array([0.0]), rise_K=0.25,
                            method="LSODA", rtol=1e-12, atol_T=1e-14,
                            atol_Y=1e-14, grid_resolution=50)
    assert r["integration_success"] is True
    assert bool(r["censored"]) == (r["t_ign_event"] is None)
    if r["t_ign_event"] is not None:
        assert r["t_ign_event"] == pytest.approx(0.5, rel=1e-6)


def test_ignition_event_grid_quantization_is_reported():
    stub = _EventStub()
    r = ignition_time_event(stub, 1.0, 1.0, np.array([0.0]), rise_K=0.25,
                            method="LSODA", rtol=1e-12, atol_T=1e-14,
                            atol_Y=1e-14, grid_resolution=11)
    # the event estimate must be the root, not the grid sample
    assert r["t_ign_event"] == pytest.approx(0.5, rel=1e-6)
    assert abs(r["t_ign_grid_estimate"] - 0.5) <= r["grid_spacing_s"] + 1e-12


# ---------------------------------------------------------------------------
# 5. committed matrices: shape, finiteness, and identity of the loaded study
# ---------------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[1]
PHASE9 = REPO / "results" / "phase9"


@pytest.mark.skipif(not PHASE9.is_dir(), reason="phase9 results not present")
@pytest.mark.parametrize("npz", sorted(PHASE9.glob("*.npz")))
def test_phase9_matrices_are_finite_and_consistently_shaped(npz):
    d = np.load(npz)
    A, D, J = d["A"], d["D"], d["J"]
    n, m = J.shape                      # n states, m control segments
    assert A.shape == (n, m)
    # D is the REFERENCE span: two columns (dB/dtau, dB/dlog gamma), not m
    assert D.shape == (n, 2)
    assert d["q0"].shape == (n,) and d["theta"].shape == (m,)
    assert d["durations"].shape == (m,) and d["W"].shape == (n,)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(D))
    assert np.all(np.isfinite(J)) and np.all(np.isfinite(d["singular_values"]))
    # A must be the declared scaling of J
    np.testing.assert_allclose(A, d["W"][:, None] * J, rtol=1e-12)
    # the saved Euclidean singular vectors must diagonalize A
    U, sv, Vh = d["U"], d["singular_values"], d["Vh"]
    np.testing.assert_allclose(np.diag(sv), U.T @ A @ Vh.T, rtol=1e-8,
                               atol=1e-10)
    # the saved singular values must be those of the saved A.  The smallest one
    # (as small as 7.6e-12 against sigma_1 ~ 1.7, a ratio of 4e-12) is an
    # ill-conditioned quantity that double precision only determines to about
    # seven digits; the factorization consistency ||A - U diag(sv) Vh|| ~ 1e-16
    # above is the invariant that actually certifies the saved triple.
    sv_recomp = np.linalg.svd(A, compute_uv=False)
    np.testing.assert_allclose(sv, sv_recomp, rtol=1e-6)
    # every saved direction must be admissible at unit Euclidean radius
    lo, hi = 10.0, 1e5
    for v in Vh:
        g = d["theta"] * np.exp(v)
        assert np.all(g > lo) and np.all(g < hi)


@pytest.mark.skipif(not PHASE9.is_dir(), reason="phase9 results not present")
def test_phase9_projector_rank_revealing_matches_qr_at_full_rank():
    """The committed projector spread used unfiltered QR.  At full rank (2) the
    rank-revealing basis returns the SAME span, so the committed numbers are not
    changed by the migration to the single helper - and the equivalence is
    asserted rather than assumed.

    Two orthonormal bases of the same 2-D subspace are related by an ORTHOGONAL
    2x2 map, which need not be symmetric; therefore only the orthogonal
    PROJECTORS must agree (Q_svd Q_svd^T = Q_qr Q_qr^T).  Asserting
    Q_svd Q_qr^T = Q_qr Q_svd^T would wrongly demand a symmetric relation."""
    import glob
    for npz in sorted(glob.glob(str(PHASE9 / "*.npz"))):
        d = np.load(npz)
        D = d["D"]
        assert D.shape == (d["J"].shape[0], 2)
        b = svd_basis(D, rel_tol=1e-12)
        assert b.rank == 2, ("reference span is rank-deficient; the committed "
                             "projector spread would then be unreliable")
        Q_svd, Q_qr = b.vectors, np.linalg.qr(D)[0]
        # same span: the orthogonal projectors must agree
        np.testing.assert_allclose(Q_svd @ Q_svd.T, Q_qr @ Q_qr.T, atol=1e-10)
        # and every QR column must lie in the SVD span and vice versa
        np.testing.assert_allclose((Q_svd @ Q_svd.T) @ Q_qr, Q_qr, atol=1e-10)
        np.testing.assert_allclose((Q_qr @ Q_qr.T) @ Q_svd, Q_svd, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. chemistry map (Cantera only; everything above is Cantera-free)
# ---------------------------------------------------------------------------


def _has_cantera() -> bool:
    try:
        import cantera  # noqa: F401
    except Exception:
        return False
    return True


_NEEDS_CANTERA = pytest.mark.skipif(not _has_cantera(),
                                    reason="cantera unavailable")


@_NEEDS_CANTERA
def test_chemistry_map_preserves_enthalpy_and_inventory():
    from thermoreach.reactor import CSTR, ReactorConfig, hot_hp_state
    c = CSTR(ReactorConfig(mechanism="h2o2.yaml", inlet_temperature=1200.0))
    q = hot_hp_state(c)
    for dt in (1e-6, 1e-5):
        r = chemistry_map(c, q, dt)
        assert r["success"]
        # adiabatic constant-pressure chemistry preserves enthalpy and elements
        assert abs(r["enthalpy_mismatch_J_kg"]) < 1e-3 * abs(
            r["initial_source"]["enthalpy_mass_J_kg"]) + 1.0
        assert r["element_mismatch_max"] < 1e-8
        # zero-step identity
        r0 = chemistry_map(c, q, 0.0)
        np.testing.assert_allclose(r0["q_out"], q)


@_NEEDS_CANTERA
def test_chemistry_response_reports_differences_separately():
    from thermoreach.reactor import CSTR, ReactorConfig, hot_hp_state
    c = CSTR(ReactorConfig(mechanism="h2o2.yaml", inlet_temperature=1200.0))
    q = hot_hp_state(c)
    q_B = q.copy()
    q_B[0] += 5.0                       # a 5 K difference between the two states
    cfg = ChemistryConfig(diagnostic_dt=(1e-6,))
    r = chemistry_response(c, q, q_B, cfg)
    assert r["state_difference"]["dT_K"] == pytest.approx(-5.0)
    assert "future_state_dT_K" in r["dt_results"][0]
    assert "increment_dT_K" in r["dt_results"][0]
    assert r["dt_results"][0]["E_metric"]          # tolerance table populated
    assert "tolerance_table_note" in r
    # baseline enthalpy mismatch is reported, not projected away
    assert "baseline" in r and "h_q_J_kg" in r["baseline"]


# ---------------------------------------------------------------------------
# 7. end-to-end whitening: the eta-space direction of unit TIME norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("durations,T", [
    (np.full(3, 1.0 / 3), 1.0),                 # equal durations, m = 3
    (np.array([0.1, 0.3, 0.6]), 1.0),           # unequal durations
    (np.array([0.05, 0.15, 0.3, 0.5]), 1.0),    # m = 4, unequal
])
def test_unwhitened_svd_direction_has_unit_time_norm(durations, T):
    """A Euclidean-unit right singular vector of A H^(-1/2) maps, under
    H^(-1/2), to a log-control direction of exactly unit time norm - and back to
    the same linear image.  Applying r*u directly instead would give an actual
    time norm of r/sqrt(m) for m equal segments."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(6, durations.size))
    A_time = whitened_map(A, durations, T)
    u = np.linalg.svd(A_time, full_matrices=False)[2][0]
    deta = unwhiten_svd_direction(u, durations, T)
    assert time_norm(deta, durations, T) == pytest.approx(1.0, abs=1e-12)
    # the linear images agree exactly: A deta = A_time u
    np.testing.assert_allclose(A @ deta, A_time @ u, rtol=1e-12, atol=1e-12)
    # and the old, incorrect construction does NOT have unit time norm
    bad = 1.0 * u
    assert time_norm(bad, durations, T) != pytest.approx(1.0, abs=1e-9)


def test_scalar_rescaling_is_not_a_substitute_for_unwhitening():
    """For unequal durations, rescaling u by a scalar cannot reproduce the unit\n    time-norm direction, because the metric H is not a multiple of the identity."""
    durations = np.array([0.1, 0.3, 0.6])
    rng = np.random.default_rng(11)
    A = rng.normal(size=(5, 3))
    A_time = whitened_map(A, durations, 1.0)
    u = np.linalg.svd(A_time, full_matrices=False)[2][0]
    deta = unwhiten_svd_direction(u, durations, 1.0)
    h = hdiag(durations, 1.0)
    # any scalar multiple c*u has time norm c*||u||_time; matching 1 requires
    # c = 1/||u||_time, and then A(c u) != A_time u in general
    c = 1.0 / time_norm(u, durations, 1.0)
    assert time_norm(c * u, durations, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(A @ (c * u), A_time @ u, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(A @ deta, A_time @ u, rtol=1e-12, atol=1e-12)


def test_admissible_interval_uses_the_unwhitened_direction():
    """The radius r is a TIME-norm radius, so the admissible interval must be
    computed on the converted direction.  On the whitened vector the interval is
    in the wrong units."""
    durations = np.full(3, 1.0 / 3)
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 3))
    A_time = whitened_map(A, durations, 1.0)
    u = np.linalg.svd(A_time, full_matrices=False)[2][0]
    deta = unwhiten_svd_direction(u, durations, 1.0)
    theta = np.full(3, 1e4)
    r_whitened = admissible_radius_interval(theta, u, 10.0, 1e5)
    r_converted = admissible_radius_interval(theta, deta, 10.0, 1e5)
    assert r_converted["r_max_positive"] is not None
    # a unit time-norm step along deta stays inside the bounds
    g = theta * np.exp(1.0 * deta)
    assert np.all(g > 10.0) and np.all(g < 1e5)
    # the two intervals differ (they are in different units)
    assert abs(r_whitened["r_max_positive"] - r_converted["r_max_positive"]) > 1e-6


def test_radius_from_deta_derives_norms_not_from_a_factor():
    durations = np.array([0.2, 0.3, 0.5])
    deta = np.array([0.1, -0.2, 0.05])
    r = radius_from_deta(deta, durations, 1.0)
    assert r["time_norm"] == pytest.approx(np.sqrt(np.sum(durations * deta ** 2)))
    assert r["euclidean_norm"] == pytest.approx(np.linalg.norm(deta))


# ---------------------------------------------------------------------------
# 8. located_reference: coarse scan + refinement, and the one-time domain
#    expansion (previously untested because the expansion branch in the study
#    script was unreachable in practice)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["no-expansion", "expansion"])
def test_located_reference_recovers_the_stub_family(label):
    """The stub family is known in closed form, so the locator must find the
    exact gamma-t pair (up to tolerance), and the expansion branch must run
    without a signature mismatch."""
    cstr = _StubCSTR()
    n = 3
    q0 = np.zeros(n)
    W = np.ones(n)                     # 1-D scaling diagonal, as StateScaling.W is
    if label == "no-expansion":
        grid = np.exp(np.linspace(np.log(8.0), np.log(12.0), 9))
        lib = ReferenceLibrary(cstr, q0, grid, t_max=1.0, t_anchor=0.5,
                               method="Radau", rtol=1e-10, atol_T=1e-12,
                               atol_Y=1e-12, t_grid_points=12)
        gamma_true, t_true = 10.0, 0.7
        kw = {}
    else:
        # deliberately NARROW domain: the minimizer sits on the top gamma edge
        # and the locator must expand the domain once
        grid = np.exp(np.linspace(np.log(0.5), np.log(2.0), 7))
        lib = ReferenceLibrary(cstr, q0, grid, t_max=0.2, t_anchor=0.1,
                               method="Radau", rtol=1e-10, atol_T=1e-12,
                               atol_Y=1e-12, t_grid_points=8)
        gamma_true, t_true = 10.0, 0.7
        kw = {"expand_gamma_half_width_log": 3.0, "expand_gamma_points": 15}

    q_target = stub_family(gamma_true, t_true, n)
    out = located_reference(lib, q_target, W, cstr, **kw)
    assert out["refined"]["success"] is True
    if label == "no-expansion":
        assert out["expanded_domain"] is False
        assert out["gamma_best"] == pytest.approx(gamma_true, rel=1e-6)
        assert out["t_best"] == pytest.approx(t_true, rel=1e-4, abs=1e-8)
        assert out["dist_scaled_upper_estimate"] == pytest.approx(0.0, abs=1e-7)
    else:
        assert out["expanded_domain"] is True
        # the expanded grid must strictly contain the original domain
        lo0, hi0 = out["coarse_before_expansion"]["gamma_domain"]
        lo1, hi1 = out["expansion"]["gamma_domain"]
        assert lo1 < lo0 and hi1 > hi0
        assert out["expansion"]["reason"] == "minimizer on a reference-domain edge"
        # the expanded library must carry its own identity (never a stale cache)
        assert out["expansion"]["library_identity"]["gamma_grid"] is not None
        assert out["expansion"]["library_identity"]["n_integrated"] >= 1


def test_expansion_branch_does_not_use_a_wrong_keyword():
    """Regression for the integrity defect: the expansion call used ``rtol=``
    where the signature is ``refine_rtol=``, so it would raise TypeError whenever
    it fired.  Force it to fire here and assert no exception and no stale-cache
    reuse of the first library's solution object."""
    cstr = _StubCSTR()
    n = 3
    q0 = np.zeros(n)
    W = np.ones(n)                     # 1-D scaling diagonal, as StateScaling.W is
    grid = np.exp(np.linspace(np.log(0.5), np.log(2.0), 5))
    lib = ReferenceLibrary(cstr, q0, grid, t_max=0.1, t_anchor=0.05,
                           method="Radau", rtol=1e-9, atol_T=1e-12,
                           atol_Y=1e-12, t_grid_points=6)
    q_target = stub_family(30.0, 0.3, n)
    out = located_reference(lib, q_target, W, cstr,
                            expand_gamma_half_width_log=2.0, expand_gamma_points=11)
    assert out["expanded_domain"] is True
    assert isinstance(out["expanded_library_failures"], list)
    # the coarse scan on the expanded library must use the NEW domain
    assert out["coarse"]["gamma_domain"][1] > 2.0
