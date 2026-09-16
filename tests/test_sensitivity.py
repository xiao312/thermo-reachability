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
    Basis, PerturbationPlan, StateScaling, classify_ranks, endpoint_jacobian_fsa_system,
    log_control_plan, linear3_endpoint_map, linear3_exact_jacobian,
    rhs_forcing_gamma, sin_to_span, svd_basis, scaled_tangent_space,
    transverse_spectrum, toy_constant_control_loggamma_derivative, toy_endpoint,
    toy_endpoint_sensitivity,
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
    assert out["tangent_space_dimension"] == n - 1
    assert "transverse_above_leakage_floor" in out


def test_transverse_floor_without_C_is_none_not_fake():
    sc = _scaling()
    out = transverse_spectrum(np.eye(4), np.eye(4)[:, :2], sc)
    assert out["singular_values_manifold_leakage_scaled"] is None


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
