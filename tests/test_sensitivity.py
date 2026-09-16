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
    Basis, PerturbationPlan, StateScaling, log_control_plan, linear3_endpoint_map,
    linear3_exact_jacobian, rhs_forcing_gamma, sin_to_span, svd_basis,
    scaled_tangent_space, toy_constant_control_loggamma_derivative, toy_endpoint,
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
