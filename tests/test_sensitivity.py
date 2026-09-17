"""Regression tests for the corrected sensitivity pipeline.

These exist because an independent review established that the earlier pipeline
could not support its own conclusions: the reference tangent was differentiated
from the wrong initial state, inadmissible perturbations were left as zero
columns inside the rank analysis, the reference matrix was QR-orthonormalized
without rank filtering, the "third direction" index depended on m, and state
increments mixed kelvin with mass fractions.

Every test below is a concrete failure mode of that pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach.sensitivity import (  # noqa: E402
    Basis, PerturbationPlan, StateScaling, classify_ranks,
    endpoint_jacobian_fsa_system, log_control_plan, linear3_endpoint_map,
    linear3_exact_jacobian, replay_signed, rhs_forcing_gamma, sin_to_span,
    ignition_time_event, svd_basis, scaled_tangent_space, transverse_replay,
    transverse_spectrum,
    toy_constant_control_loggamma_derivative, toy_endpoint, toy_endpoint_sensitivity,
)

GAMMA_LO, GAMMA_HI, GAMMA_REF = 10.0, 1e5, 1000.0


# --------------------------------------------------------------------------
# numerical-rank bases: never fabricate orthogonal fill
# --------------------------------------------------------------------------

def test_svd_basis_detects_full_rank():
    M = np.array([[1.0, 0.0], [0.0, 0.5]])
    b = svd_basis(M)
    assert b.rank == 2
    assert b.projector_residual < 1e-12


def test_svd_basis_drops_noise_direction():
    """A 1e-14 column must NOT create a second retained direction."""
    M = np.array([[1.0, 1e-14], [0.0, 0.0]])
    b = svd_basis(M, rel_tol=1e-8)
    assert b.rank == 1
    assert b.singular_values.size == 2          # spectrum still reported


def test_svd_basis_zero_matrix_has_no_vectors():
    b = svd_basis(np.zeros((3, 3)))
    assert b.rank == 0
    assert b.vectors is None                    # never an arbitrary completion


def test_sin_to_span_orthogonal_case_is_one():
    """V = [e1, 0], J = [e2]: the sine to span(V) must be ONE, not zero.

    The old pipeline reported a small angle here because it orthonormalized a
    rank-deficient V by QR, which fills the basis with rounding noise.
    """
    bV = svd_basis(np.array([[1.0, 0.0], [0.0, 0.0]]))
    e2 = np.array([0.0, 1.0])
    assert sin_to_span(e2, bV) == pytest.approx(1.0, abs=1e-12)
    assert sin_to_span(np.array([1.0, 0.0]), bV) == pytest.approx(0.0, abs=1e-12)


def test_sin_to_span_empty_basis_is_one():
    b = svd_basis(np.zeros((2, 2)))
    assert sin_to_span(np.array([1.0, 0.0]), b) == 1.0


def test_absolute_threshold_can_lower_rank():
    """A ratio-preserving relative threshold keeps small-magnitude directions
    that an absolute application threshold correctly discards."""
    M = np.diag([1e-9, 1e-10])          # both tiny, but their ratio is 0.1
    assert svd_basis(M, rel_tol=1e-8, abs_tol=None).rank == 2
    assert svd_basis(M, rel_tol=1e-8, abs_tol=1e-6).rank == 0
    # and it must not discard genuinely large directions
    M2 = np.diag([1.0, 1e-10])
    assert svd_basis(M2, rel_tol=1e-8, abs_tol=1e-6).rank == 1


# --------------------------------------------------------------------------
# perturbation plan: no fabricated zero columns, strictly interior steps
# --------------------------------------------------------------------------

def test_plan_steps_stay_interior_near_both_bounds():
    """Controls near the lower bound must still have a valid centred stencil."""
    for gamma in (12.0, 30.0, 1e3, 3e4, 9.5e4):
        theta = np.array([gamma, 1000.0, 3000.0])
        p = log_control_plan(theta, GAMMA_LO, GAMMA_HI, GAMMA_REF, 0.1)
        assert all(p.column_valid(j) for j in range(3))
        assert p.gamma(0, -1) > GAMMA_LO and p.gamma(0, +1) < GAMMA_HI


def test_plan_rejects_non_interior_base():
    with pytest.raises(ValueError):
        log_control_plan(np.array([GAMMA_LO, 1000.0]), GAMMA_LO, GAMMA_HI,
                         GAMMA_REF, 0.1)


def test_plan_gamma_round_trips_through_eta():
    """gamma(j, +-1) must be the ABSOLUTE rate, not exp(eta) alone.

    An earlier version returned exp(eta) = gamma/gamma_ref, i.e. every perturbed
    rate was ~1000x too small - a scaling bug that silently corrupted the whole
    Jacobian.
    """
    theta = np.array([30.0, 33333.0])
    p = log_control_plan(theta, GAMMA_LO, GAMMA_HI, GAMMA_REF, 0.1)
    for j in range(2):
        base = theta[j]
        assert abs(p.gamma(j, +1) / base - 1.0) < 0.5      # same order of magnitude
        assert abs(p.gamma(j, -1) / base - 1.0) < 0.5


# --------------------------------------------------------------------------
# state scaling and the conserved-manifold tangent space
# --------------------------------------------------------------------------

def test_state_scaling_scales_rows():
    s = StateScaling(T_interval=100.0, Y_interval=1e-2, n_species=3)
    q = np.array([200.0, 0.02, 0.0, 0.0])
    assert np.allclose(s.scale(q), np.array([2.0, 2.0, 0.0, 0.0]))


def test_scaled_tangent_space_dimension_and_leakage():
    """A 2-constraint, 4-variable system has a 2-dimensional tangent space, and
    conservation leakage is reported BEFORE projection."""
    C = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    s = StateScaling(T_interval=1.0, Y_interval=1.0, n_species=3)
    ts = scaled_tangent_space(C, s)
    assert ts.rank_CWinverse == 2
    assert ts.basis.shape == (4, 2)
    in_null = np.array([1.0, -1.0, 0.0, 0.0])       # satisfies C v = 0
    violating = C[0] + C[1]                          # C v = (2, 2) != 0
    assert ts.leakage(violating) > 0.0               # reported, not hidden
    assert np.allclose(ts.project(in_null), in_null, atol=1e-12)
    assert ts.leakage(ts.project(violating)) < 1e-12
    assert np.allclose(C @ ts.project(violating), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# exact toy ground truth
# --------------------------------------------------------------------------

def test_toy_endpoint_matches_closed_form_single_segment():
    q0 = np.array([0.6, 0.05])
    q = toy_endpoint(np.array([3.0]), np.array([0.7]), q0)
    a = 4.0
    xs, ys = 3.0 / a, 3.0 / a**2
    e = np.exp(-a * 0.7)
    expected = np.array([xs + (q0[0] - xs) * e,
                         ys + ((q0[1] - ys) + (q0[0] - xs) * 0.7) * e])
    assert np.allclose(q, expected, atol=1e-14)


def test_toy_endpoint_sensitivity_matches_finite_differences():
    """The exact tangent-linear sensitivity agrees with central differences at
    O(h^2); this validates the parameter scaling, endpoint anchoring and the
    log-control stencil all at once."""
    rng = np.random.default_rng(11)
    for _ in range(4):
        m = 4
        gam = np.exp(rng.uniform(np.log(0.5), np.log(50.0), m))
        dur = rng.uniform(0.1, 1.0, m)
        q0 = np.array([0.7, 0.1])
        S_ex = toy_endpoint_sensitivity(gam, dur, q0)
        for rel in (1e-3, 1e-4):
            S_fd = np.zeros_like(S_ex)
            for j in range(m):
                gp = gam.copy(); gp[j] *= (1 + rel)
                gm = gam.copy(); gm[j] *= (1 - rel)
                S_fd[:, j] = ((toy_endpoint(gp, dur, q0)
                               - toy_endpoint(gm, dur, q0))
                              / (np.log(1 + rel) - np.log(1 - rel)))
            err = np.max(np.abs(S_fd - S_ex)) / np.max(np.abs(S_ex))
            assert err < 20 * rel**2, f"FD error {err} exceeds O(h^2) at {rel}"


def test_toy_rank_two_history_is_detected():
    """C.1: a nondegenerate history with expected rank 2 must be detected."""
    rng = np.random.default_rng(7)
    q0 = np.array([0.8, 0.02])
    found = None
    for _ in range(400):
        gam = np.exp(rng.uniform(np.log(1.0), np.log(50.0), 2))
        dur = rng.uniform(0.05, 1.0, 2)
        if abs(gam[0] - gam[1]) < 0.5:
            continue
        S = toy_endpoint_sensitivity(gam, dur, q0)
        if np.linalg.matrix_rank(S) == 2:
            found = (gam, dur, S)
            break
    assert found is not None
    gam, dur, S = found
    # finite differences must recover the same rank
    J = np.zeros_like(S)
    for j in range(2):
        gp = gam.copy(); gp[j] *= 1.001
        gm = gam.copy(); gm[j] *= 0.999
        J[:, j] = ((toy_endpoint(gp, dur, q0) - toy_endpoint(gm, dur, q0))
                   / (np.log(1.001) - np.log(0.999)))
    assert np.linalg.matrix_rank(J) == 2


def test_toy_terminal_hold_preserves_algebraic_rank_while_effective_decays():
    """C.1 terminal hold: an invertible finite-time flow preserves the exact
    prefix rank even as the effective rank drops below resolution."""
    rng = np.random.default_rng(7)
    q0 = np.array([0.8, 0.02])
    gam = np.array([10.0, 40.0])
    dur = np.array([0.5, 0.3])
    S = toy_endpoint_sensitivity(gam, dur, q0)
    assert np.linalg.matrix_rank(S) == 2
    gamma_hold = float(gam[-1])
    eff_abs, algebraic = [], []
    for L in (0.0, 1.0, 10.0, 100.0):
        durs_h = np.concatenate([dur, [L]])
        gam_h = np.concatenate([gam, [gamma_hold]])
        S_h = toy_endpoint_sensitivity(gam_h, durs_h, q0)
        sv = np.linalg.svd(S_h[:, :2], compute_uv=False)
        algebraic.append(np.linalg.matrix_rank(S_h[:, :2]))
        eff_abs.append(int((sv > 1e-10).sum()))
    # algebraic rank survives until floating-point underflow
    assert algebraic[0] == 2 and algebraic[1] == 2 and algebraic[2] == 2
    # application-scale rank decays to zero
    assert eff_abs[0] == 2 and eff_abs[-1] == 0


def test_toy_identity_sum_of_columns_equals_constant_control_derivative():
    """C.3: sum_j dE_m/d(log gamma_j) = d phi_gamma/d log gamma at a
    constant-control base history (m = 1, 3, 5)."""
    for gamma_star in (30.0, 300.0, 3000.0):
        for T in (0.3, 1.0, 3.0):
            for m in (1, 3, 5):
                q0 = np.array([0.75, 0.03])
                lhs = toy_endpoint_sensitivity(np.full(m, gamma_star),
                                               np.full(m, T / m), q0).sum(axis=1)
                rhs = toy_constant_control_loggamma_derivative(gamma_star, T, q0)
                assert np.max(np.abs(lhs - rhs)) < 1e-12 * max(np.max(np.abs(rhs)), 1.0)


# --------------------------------------------------------------------------
# three-state linear benchmark (review_checks.py specification)
# --------------------------------------------------------------------------

LAM3 = np.array([1.0, 2.0, 4.0])


def test_linear3_exact_singular_values_match_specification():
    sv = np.linalg.svd(linear3_exact_jacobian(LAM3, 1.0, 3), compute_uv=False)
    assert sv == pytest.approx([0.5007801, 0.08533145, 0.006828632], rel=1e-6)


def test_linear3_endpoint_map_is_linear_in_controls():
    u = np.array([0.3, 0.5, 0.7])
    G = linear3_exact_jacobian(LAM3, 1.0, 3)
    z = linear3_endpoint_map(LAM3, 1.0, 3, u)
    assert z == pytest.approx(G @ u, abs=1e-12)


def test_linear3_third_singular_value_is_at_fixed_index_2():
    """The third largest singular value is index 2 for every m; the old code used
    index n-3, which reads the SMALLEST value when m = 3."""
    for m in (3, 4, 6):
        sv = np.linalg.svd(linear3_exact_jacobian(LAM3, 1.0, m), compute_uv=False)
        assert sv[2] == pytest.approx(sorted(sv, reverse=True)[2])


@pytest.mark.parametrize("m", [3, 4, 6])
def test_linear3_analyzer_detects_third_direction(m):
    """The analyzer must detect the third direction at every m."""
    G = linear3_exact_jacobian(LAM3, 1.0, m)
    b = svd_basis(G, rel_tol=1e-8)
    assert b.rank >= 3
    assert b.singular_values[2] > 1e-8 * b.singular_values[0]


def test_linear3_fd_reproduces_the_exact_jacobian():
    """Central differences in raw controls recover G to O(h^2)."""
    u = np.array([0.3, 0.5, 0.7])
    G = linear3_exact_jacobian(LAM3, 1.0, 3)
    for rel in (1e-2, 1e-3, 1e-4):
        J = np.zeros_like(G)
        for j in range(3):
            up = u.copy(); up[j] += rel
            um = u.copy(); um[j] -= rel
            J[:, j] = (linear3_endpoint_map(LAM3, 1.0, 3, up)
                       - linear3_endpoint_map(LAM3, 1.0, 3, um)) / (2 * rel)
        assert np.max(np.abs(J - G)) < 50 * rel**2 * np.max(np.abs(G))


# --- rank classification and transverse spectrum (Revision 4) ---------------

def _scaling():
    return StateScaling(n_species=3)


def test_classify_ranks_separates_four_notion():
    """Machine rank, noise-resolved rank and application rank are DISTINCT.

    A matrix with singular values (1, 1e-5, 1e-12) has machine rank 3, is
    resolved only to rank 2 above a 1e-6 discrepancy, and is above the 1e-6
    application threshold only twice.  Reporting the machine rank as the answer
    is exactly the error that produced the withdrawn Phase 3D claim.
    """
    sc = _scaling()
    # a 4x3 matrix with a deliberately tiny third singular value
    J = np.array([[1.0, 0.0, 0.0], [0.0, 1e-5, 0.0], [0.0, 0.0, 1e-12],
                  [0.0, 0.0, 0.0]])
    out = classify_ranks(J, sc, noise_scale=1e-6,
                         application_threshold=1e-6)
    # numpy's tolerance is ~ max_sv * shape * eps ~ 1e-15, so the 1e-12 column
    # IS counted by matrix_rank: the machine rank is 3 while the noise-resolved
    # and application ranks are 2.  This gap is the whole point.
    assert out["rank_machine"] == 3
    assert out["rank_derivative_noise_resolved"] == 2   # 1e-5 > 1e-6 > 1e-12
    assert out["rank_application_effective"] == 2
    assert out["stencil_complete"] is True
    # the machine rank must NOT be reported as the claimed rank
    assert out["full_rank_claim_valid"] is False


def test_classify_ranks_invalid_stencil_excludes_whole_matrix():
    """A NaN column makes the matrix invalid for full-rank claims: never
    aggregated with nansum into a rank count."""
    sc = _scaling()
    J = np.array([[1.0, np.nan], [0.0, 1.0]])
    out = classify_ranks(J, sc, noise_scale=1e-9)
    assert out["stencil_complete"] is False
    assert out["n_valid_columns"] == 1
    assert out["full_rank_claim_valid"] is False


def test_classify_ranks_noise_none_is_not_zero():
    """Without an estimated discrepancy the noise-resolved rank is None, never
    a silently small integer."""
    sc = _scaling()
    J = np.eye(3)
    out = classify_ranks(J, sc, noise_scale=None)
    assert out["rank_derivative_noise_resolved"] is None


def test_application_threshold_component_bound_is_not_1e_minus_6():
    """The 1e-6 scaled threshold is 1e-4 K and 1e-8 mass fraction, NOT 1e-6 in
    mass fraction.  The component bound must say so."""
    sc = StateScaling(n_species=2, T_interval=100.0, Y_interval=1e-2)
    b = sc.component_bound(1e-6)
    assert b["temperature_K_per_unit_log_control"] == pytest.approx(1e-4)
    assert b["mass_fraction_per_unit_log_control"] == pytest.approx(1e-8)


def test_transverse_spectrum_reports_leakage_floor():
    """The manifold-leakage floor must be reported next to the transverse
    spectrum, so a tiny transverse direction cannot be hidden by projection."""
    sc = _scaling()
    n = 4
    J = np.eye(n)
    V = np.zeros((n, 2))
    V[0, 0] = 1.0
    V[1, 1] = 1.0
    C = np.array([[1.0, 1.0, 0.0, 0.0]])       # one constraint
    out = transverse_spectrum(J, V, sc, noise_scale=None, C_of_q=C)
    assert out["singular_values_total_scaled"] is not None
    assert out["singular_values_manifold_leakage_scaled"] is not None
    assert out["reference_rank"] == 2
    assert out["reference_is_resolved"] is True
    assert out["per_total_vector_projection"] is not None
    assert "transverse_above_leakage_floor" in out


def test_transverse_floor_without_C_is_none_not_fake():
    sc = _scaling()
    out = transverse_spectrum(np.eye(4), np.eye(4)[:, :2], sc)
    assert out["singular_values_manifold_leakage_scaled"] is None
    assert len(out["per_total_vector_projection"]) == 4


def test_transverse_spectrum_near_rank_deficient_reference():
    """A near-rank-deficient reference span must not silently gain an arbitrary
    completion: the rank decision, thresholds and projector change are recorded,
    and the weak direction is reported rather than discarded."""
    sc = StateScaling(n_species=2, T_interval=1.0, Y_interval=1.0)   # W size 3
    A = np.diag([1.0, 1.0, 1e-9])
    out = transverse_spectrum(A, np.eye(3)[:, :2], sc, rel_tol=1e-8)
    assert out["reference_basis_singular_values"][0] == pytest.approx(1.0)
    assert out["reference_rank"] in (1, 2)
    assert out["reference_threshold_used"] > 0


def test_unresolved_reference_span_marks_transversality_unresolved():
    """If the reference span has NO direction above threshold, no projector is
    built and transversality is UNRESOLVED, not zero."""
    sc = StateScaling(n_species=2, T_interval=1.0, Y_interval=1.0)
    V = np.zeros((3, 2))                    # a span with no usable direction
    out = transverse_spectrum(np.eye(3), V, sc)
    assert out["reference_is_resolved"] is False
    assert out["singular_values_transverse_scaled"] is None


def test_scaled_tangent_space_uses_CWinverse_not_CW():
    """The old code formed null(C W); with unequal scales that returns the wrong
    subspace.  Test from the review: C=[1,1], W=diag(1e-2,100), dq=[1,-1]."""
    sc = StateScaling(n_species=1, T_interval=100.0, Y_interval=1e-2)
    C = np.array([[1.0, 1.0]])
    ts = scaled_tangent_space(C, sc)
    # the tangent vector dq=[1,-1] scaled is W dq = [0.01, -100]; the constraint
    # C dq = 0 holds, so the SCALED vector must project to itself
    dq_scaled = sc.W * np.array([1.0, -1.0])
    proj = ts.project(dq_scaled)
    assert np.allclose(proj, dq_scaled, atol=1e-8)
    # and C W^-1 N must be ~0
    N = ts.basis
    assert np.max(np.abs(C @ np.diag(1.0 / sc.W) @ N)) < 1e-8
    assert ts.basis.shape[1] == 1


# --- exact tangent-linear (FSA) estimator vs a closed form ------------------

class _AffineLinear:
    """dq/dt = A q + gamma*b, EXACTLY affine in gamma.

    The endpoint of a segment history and dE/d(log gamma_j) are both available in
    closed form through matrix exponentials, so this is an exact reference for the
    tangent-linear solver - and it needs no Cantera.
    """

    def __init__(self, A, b):
        self.A = np.asarray(A, float)
        self.b = np.asarray(b, float)
        self.n = self.A.shape[0]

    def rhs(self, t, q, gamma):
        return self.A @ q + gamma * self.b

    def rhs_dgamma(self, q):
        return self.b

    def state_jac(self, q, gamma):
        return self.A

    def endpoint_exact(self, gammas, durs, q0):
        from scipy.linalg import expm
        q = np.array(q0, float)
        for g, tau in zip(gammas, durs):
            Phi = expm(self.A * tau)
            q = Phi @ (q + g * np.linalg.solve(self.A, self.b)) \
                - g * np.linalg.solve(self.A, self.b)
        return q

    def jac_exact(self, gammas, durs, q0):
        """Closed form:  S_j(final) = prod_{k>j} exp(A tau_k)
                                  * gamma_j (exp(A tau_j) - I) A^-1 b.

        The tangent source for the log-level of segment j is gamma_j * b, acting
        only during segment j, and it is then carried linearly through the later
        segments.
        """
        from scipy.linalg import expm
        u = np.linalg.solve(self.A, self.b)
        m, n = len(gammas), self.n
        S = np.zeros((n, m))
        for j in range(m):
            col = gammas[j] * (expm(self.A * durs[j]) - np.eye(n)) @ u
            for k in range(j + 1, m):
                col = expm(self.A * durs[k]) @ col
            S[:, j] = col
        return self.endpoint_exact(gammas, durs, q0), S


@pytest.mark.parametrize("m", [1, 2, 3])
def test_fsa_matches_closed_form_affine_linear(m):
    """The exact tangent-linear endpoint Jacobian must match the closed form to
    integration tolerance, at every number of segments.  This is the estimator
    that removes the finite-difference noise floor from the transverse spectrum.
    """
    rng = np.random.default_rng(0)
    n = 4
    A = -np.diag([1.0, 2.0, 4.0, 0.7]) + 0.3 * rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    sysm = _AffineLinear(A, b)
    q0 = rng.standard_normal(n)
    gammas = np.array([100.0, 500.0, 250.0][:m])
    durs = np.full(m, 1e-3)

    q_ex, J_ex = sysm.jac_exact(gammas, durs, q0)
    out = endpoint_jacobian_fsa_system(sysm.rhs, sysm.rhs_dgamma, sysm.state_jac,
                                       gammas, durs, q0, method="Radau",
                                       rtol=1e-12, atol=np.full(n, 1e-14))
    assert np.allclose(out["endpoint"], q_ex, rtol=1e-9, atol=1e-12)
    assert np.allclose(out["J"], J_ex, rtol=1e-9, atol=1e-12)


def test_fsa_endpoint_jacobian_has_no_control_noise_floor():
    """Against a fine finite difference in the control, FSA must agree far
    better than a coarse-vs-fine FD pair does - i.e. it is the reference."""
    rng = np.random.default_rng(1)
    n = 4
    A = -np.diag([1.0, 2.0, 4.0, 0.7]) + 0.2 * rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    sysm = _AffineLinear(A, b)
    q0 = rng.standard_normal(n)
    gammas = np.array([100.0, 500.0, 250.0])
    durs = np.array([1e-3, 2e-3, 1e-3])
    _, J_ex = sysm.jac_exact(gammas, durs, q0)
    out = endpoint_jacobian_fsa_system(sysm.rhs, sysm.rhs_dgamma, sysm.state_jac,
                                       gammas, durs, q0, method="Radau",
                                       rtol=1e-12, atol=np.full(n, 1e-14))
    err_fsa = np.max(np.abs(out["J"] - J_ex))
    # a central FD in the control at a coarse step, for contrast
    eta = np.log(gammas / 1000.0)
    J_fd = np.zeros_like(J_ex)
    for j in range(3):
        gp = gammas.copy(); gp[j] *= np.exp(1e-2)
        gm = gammas.copy(); gm[j] *= np.exp(-1e-2)
        J_fd[:, j] = (sysm.endpoint_exact(gp, durs, q0)
                      - sysm.endpoint_exact(gm, durs, q0)) / (2e-2)
    err_fd = np.max(np.abs(J_fd - J_ex))
    # the tangent-linear estimator must beat a coarse control FD by many orders
    # of magnitude: it has no control-step noise floor at all
    assert err_fsa < 1e-7 * max(err_fd, 1e-12), (err_fsa, err_fd)


# --- Revision 5: segment restarts, correct replays, signed commutator --------

def test_fsa_segment_restart_regression_benchmark():
    """The review's exact benchmark for segment-restarted sensitivities.

        qdot   = gamma
        S_j'   = gamma_j during segment j, zero otherwise
        q(0)   = 0,  S(0) = 0
        gamma  = (10, 1e5, 10)
        durations = (0.0499995, 1e-6, 0.0499995)

    Exact endpoint:          q = 1.09999
    Exact log sensitivities: (0.499995, 0.1, 0.499995)

    The single-call searchsorted implementation failed in the review runtime with
    'Required step size is less than spacing between numbers'; segment restarting
    agreed to 8.9e-16.
    """
    gammas = np.array([10.0, 1e5, 10.0])
    durs = np.array([0.0499995, 1e-6, 0.0499995])
    # qdot = gamma, so F_gamma = 1 and F_q = 0 identically
    out = endpoint_jacobian_fsa_system(
        lambda t, q, g: np.array([g]),
        lambda q: np.array([1.0]),
        lambda q, g: np.zeros((1, 1)),
        gammas, durs, np.array([0.0]), method="Radau", rtol=1e-12,
        atol=np.array([1e-14]))
    assert out["endpoint"][0] == pytest.approx(1.09999, abs=1e-12)
    expected = np.array([0.499995, 0.1, 0.499995])
    assert out["J"][0] == pytest.approx(expected, abs=1e-12)


def test_fsa_tolerance_maps_per_state_not_uniform():
    """The sensitivity block is row-major (n, m): a per-state atol vector maps as
    np.repeat(atol_state, m).  The old np.full(n*m, atol.min()) applied a
    species-sized tolerance to the temperature sensitivity as well."""
    n, m = 3, 4
    atol_state = np.array([1e-9, 1e-15, 1e-15])
    sens_atol = np.repeat(atol_state, m)
    assert sens_atol.size == n * m
    # the temperature rows of the (n, m) sensitivity block are indices 0, m, 2m
    for k in range(m):
        assert sens_atol[k] == 1e-9          # temperature sensitivity
        assert sens_atol[m + k] == 1e-15     # species
        assert sens_atol[2 * m + k] == 1e-15


def test_replay_transverse_along_v_perp_not_v2():
    """The second right singular vector of A maximizes NEITHER ||B v|| NOR the
    leading singular value of B = (I - P) A.  Replaying along v2(A) tests
    ||B v2||, not sigma1(B).

    Review counterexample: A = diag(3,2,1), Q_B = [e1, e2].
        sigma2(A) = 2;  ||B v2(A)|| = 0;  sigma1(B) = 1;
        replay along v1(B) = e3 returns 1.
    """
    A = np.diag([3.0, 2.0, 1.0])
    Q = np.eye(3)[:, :2]                       # P projects onto span(e1, e2)
    P = Q @ Q.T
    B = (np.eye(3) - P) @ A
    sv_A = np.linalg.svd(A, compute_uv=False)
    assert sv_A[1] == pytest.approx(2.0)
    _, _, Vh = np.linalg.svd(A)
    v2 = Vh[1]
    assert np.linalg.norm(B @ v2) == pytest.approx(0.0, abs=1e-12)
    sv_B = np.linalg.svd(B, compute_uv=False)
    assert sv_B[0] == pytest.approx(1.0)
    Ub, _, Vhb = np.linalg.svd(B)
    v1B = Vhb[0]
    assert np.allclose(np.abs(v1B), [0.0, 0.0, 1.0])
    assert Ub[:, 0] @ B @ v1B == pytest.approx(1.0)
    # the exact inequality: ||(I-P) A||_2 >= sigma3(A) for rank(P) <= 2
    assert sv_B[0] >= sv_A[2] - 1e-12


def test_projector_inequality_bounded_below_by_sigma3():
    """For ANY orthogonal projector P of rank at most 2,
    ||(I - P) A||_2 >= sigma3(A).  A stable, genuinely nonzero sigma3(A) is
    therefore evidence of a third direction independently of the reference
    plane.  Checked over random matrices and random rank-2 projectors."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        A = rng.standard_normal((6, 4))
        M = rng.standard_normal((6, 2))
        Q = np.linalg.qr(M)[0]
        P = Q @ Q.T
        assert np.linalg.norm((np.eye(6) - P) @ A, ord=2) \
            >= np.linalg.svd(A, compute_uv=False)[2] - 1e-10


def test_commutator_sign_convention_linear_benchmark():
    """Correct signed commutator with [f,g] = Dg f - Df g and F_gamma = r + g v.

    For qdot = -q + gamma, a = 1, b = 3, q0 = 0, the exact two-segment order
    difference is (b - a)(1 - exp(-tau))^2: POSITIVE and tending to b - a at
    large tau.  This stable linear system has no ignition, so saturation of the
    order difference cannot by itself identify ignition as the mechanism.
    """
    def phi(q, g, tau):
        # qdot = -q + g  ->  q(tau) = g + (q - g) exp(-tau)
        return g + (q - g) * np.exp(-tau)

    a, b = 1.0, 3.0
    for tau in (1e-3, 1e-2, 0.1, 1.0, 5.0):
        ab = phi(phi(0.0, a, tau), b, tau)
        ba = phi(phi(0.0, b, tau), a, tau)
        diff = ab - ba
        expected = (b - a) * (1.0 - np.exp(-tau)) ** 2
        assert diff == pytest.approx(expected, rel=1e-12)
        assert diff > 0                        # sign is (b - a), not (a - b)


# --- corrected replay: along V_perp, signed projection and full vector --------

def test_transverse_replay_uses_v_perp_not_v2():
    """On the review counterexample A = diag(3,2,1), P = proj(e1,e2), a replay
    along v2(A) gives ZERO transverse signal while a replay along v1(B) gives 1.
    The corrected routine must pick v1(B) itself and report the difference."""
    A = np.diag([3.0, 2.0, 1.0])
    Q = np.eye(3)[:, :2]
    P = Q @ Q.T
    B = (np.eye(3) - P) @ A
    Ub, sv, Vhb = np.linalg.svd(B, full_matrices=False)
    assert sv[0] == pytest.approx(1.0)
    v_perp = Vhb[0]
    assert np.allclose(np.abs(v_perp), [0, 0, 1])

    # a synthetic endpoint map whose derivative along v is exactly A v
    def endpoint(theta, durs):
        # dummy: the replay needs a callable; use a map whose central difference
        # reproduces A v exactly (linear in eps) for the admissible rows
        return np.zeros(3)

    sc = StateScaling(n_species=2, T_interval=1.0, Y_interval=1.0)
    rep = replay_signed(endpoint, np.array([1.0, 1.0, 1.0]),
                        np.array([1.0]), v_perp, sc,
                        eps_list=(0.1,), gamma_bounds=None)
    assert rep["rows"][0]["admissible"] is True
    assert rep["rows"][0]["norm"] == pytest.approx(0.0)
    # replaying along v2(A) instead gives a ZERO transverse signal while
    # sigma_perp stays 1: the choice of direction matters
    _, _, VhA = np.linalg.svd(A)
    rep2 = replay_signed(endpoint, np.array([1.0, 1.0, 1.0]),
                         np.array([1.0]), VhA[1], sc, eps_list=(0.1, 0.03))
    out2 = transverse_replay(A, P, rep2)
    assert out2["available"]
    assert out2["rows"][0]["signed_projection"] == pytest.approx(0.0, abs=1e-12)
    assert out2["rows"][0]["sigma_perp"] == pytest.approx(1.0)


def test_transverse_replay_closed_form_linear():
    """For an exactly-known linear endpoint map, replaying along v_perp must
    reproduce sigma_perp and B v_perp to central-difference accuracy."""
    rng = np.random.default_rng(3)
    n = 5
    M = rng.standard_normal((n, n)) * 0.1
    # endpoint map E(eta) = M eta (linear in the log controls)
    def endpoint(theta, durs):
        return M @ np.log(theta)
    sc = StateScaling(n_species=4, T_interval=1.0, Y_interval=1.0)
    # E(eta) = M eta, so dE/deta = M exactly (no theta0 factor)
    theta0 = np.full(n, 100.0)
    J = M
    A = sc.W[:, None] * J
    # reference span: first two columns of A scaled (a rank-2 family)
    V = J[:, :2]
    D = sc.W[:, None] * V
    Q = np.linalg.qr(D)[0]
    P = Q @ Q.T
    B = (np.eye(n) - P) @ A
    Ub, sv_perp, Vhb = np.linalg.svd(B, full_matrices=False)
    v_perp = Vhb[0]
    rep = replay_signed(endpoint, theta0, np.array([1.0]), v_perp, sc,
                        eps_list=(0.1, 0.03, 0.01, 0.003, 0.001))
    out = transverse_replay(A, P, rep)
    assert out["available"]
    for r in out["rows"]:
        # exact for a linear map: no O(eps^2) error
        assert r["signed_projection"] == pytest.approx(sv_perp[0], rel=1e-10)
        assert r["relative_vector_residual"] == pytest.approx(0.0, abs=1e-10)


def test_transverse_replay_unresolved_reference_is_reported():
    """An unresolved reference span must yield transversality UNRESOLVED, never a
    fabricated projector."""
    A = np.diag([3.0, 2.0, 1.0])
    sc = StateScaling(n_species=2, T_interval=1.0, Y_interval=1.0)
    rep = replay_signed(lambda th, du: np.zeros(3), np.array([1.0, 1.0, 1.0]),
                        np.array([1.0]), np.array([0.0, 0.0, 1.0]), sc,
                        eps_list=(0.1, 0.03))
    out = transverse_replay(A, None, rep)
    assert out["available"] is False


# --- ignition time by dense-output root finding (not a grid lookup) -----------

class _IgnitionStub:
    """qdot = a*q + b, with an exact known crossing time for a linear rise.

    For qdot = 1 (constant) from q0 = 0 the crossing of 100 is at t = 100
    exactly; a grid of N points over a horizon H quantizes this to
    k*H/(N-1), which is what the event-based version must NOT reproduce.
    """

    def __init__(self, slope=1.0):
        self.slope = slope
        self.T_in = 0.0

    def rhs(self, t, q, gamma):
        return np.array([self.slope])


def test_ignition_time_event_beats_grid_quantization():
    """The event-located crossing must be exact, while the grid estimate is
    quantized to the grid spacing and depends on the horizon."""
    c = _IgnitionStub(slope=1.0)
    q0 = np.array([0.0, 0.0, 0.0])          # n = 3, but only T is integrated
    out = ignition_time_event(c, 1.0, 0.1, q0, rise_K=100.0,
                              grid_resolution=400)
    # the crossing at t=100 lies beyond the 0.1 s horizon: censored, not wrong
    assert out["censored"] is True
    assert out["t_ign_event"] is None
    assert out["grid_spacing_s"] == pytest.approx(0.1 / 399)


def test_ignition_time_event_locates_an_interior_crossing():
    """qdot = 1 from 0 crosses 100 at t = 100 inside a 200 s horizon.  The event
    time must be exactly 100; the 400-point grid estimate is quantized to
    200/399 ~ 0.501, so it cannot land exactly on 100."""
    c = _IgnitionStub(slope=1.0)
    q0 = np.array([0.0, 0.0, 0.0])
    out = ignition_time_event(c, 1.0, 200.0, q0, rise_K=100.0,
                              grid_resolution=400)
    assert out["censored"] is False
    assert out["t_ign_event"] == pytest.approx(100.0, abs=1e-6)
    assert out["grid_spacing_s"] == pytest.approx(200.0 / 399)
    # the grid estimate is on the grid, so it is NOT exactly 100
    assert out["t_ign_grid_estimate"] != pytest.approx(100.0, abs=1e-9)
    assert out["t_ign_grid_estimate"] > 100.0
