"""Sensitivity analysis for history-generated thermochemical directions.

This module replaces the Phase 3D analysis pipeline. It exists because an
independent review established that the earlier pipeline could not support its
own conclusions:

  * the reference tangent span was differentiated from the *switched endpoint*
    rather than the original initial state, so it did not differentiate the
    constant-control family being claimed about;
  * a common absolute perturbation left inadmissible columns as *zeros* inside
    the Jacobian, which then entered the rank analysis as fabricated zero
    sensitivities;
  * the reference matrix was orthonormalized by unfiltered QR, which fills a
    rank-deficient span with rounding-noise directions;
  * the singular-value index used for the "third direction" depended on m;
  * state increments mixed kelvin with mass fractions, so singular values had
    no single physical meaning;
  * horizons did not always equal the declared value.

Design of the corrected pipeline.

1.  Controls are dimensionless ``eta_j = log(gamma_j / gamma_ref)``. A step in
    eta is a relative change in gamma, so every parameter step has a documented
    magnitude. Steps are chosen *per coordinate* to remain strictly inside the
    admissible interval, and a perturbation that cannot be centred is reported
    as invalid and excluded from every rank summary - it is never a zero column.

2.  The state has a *declared, frozen* scaling ``dq_scaled = W dq`` (engineering
    scaling: temperature in kelvin divided by a declared temperature interval,
    mass fractions divided by a declared composition interval). Singular values
    in the scaled space are the magnitudes of physically comparable state
    changes. A separately justified trace-sensitive analysis is kept distinct.

3.  Numerical rank is decided by SVD with *both* relative and absolute
    thresholds, applied to BOTH matrices. Bases are never filled out with
    arbitrary orthogonal vectors. Zero-rank matrices are handled explicitly.

4.  The conserved-manifold tangent space is computed in the scaled space as the
    null space of ``C W^-1`` (C = constraint Jacobian) by a rank-revealing SVD,
    with the conditioning of ``C W^-1`` recorded. Conservation leakage of every
    vector is reported *before* projection, so projection cannot hide errors.

5.  Two independent differentiation pathways are used where it matters:
    central finite differences in eta, and a tangent-linear (forward
    sensitivity) system integrated alongside the state. For log controls the
    sensitivity equation for segment j is

        dS_j/dt = F_q(q, gamma_j) S_j + gamma_j F_γ(q, gamma_j)

    with switching times FIXED, so S is continuous across a switch and only the
    forcing changes segment. The RHS is affine in gamma, so F_γ is exact and
    only F_q is differenced numerically.

6.  Reference tangents are evaluated from the ORIGINAL initial state:
    v_t = F(q_base, gamma_star) and v_log_gamma = d phi_gamma(T; q0)/d log gamma.
    At a constant-control base history the switched endpoint lies exactly on the
    reference family, and the identity

        sum_j dE_m / d(log gamma_j) = d phi_gamma_star / d log gamma

    is an exact positive control of parameter scaling, anchoring and switching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# numerical-rank bases: never fabricate orthogonal fill
# --------------------------------------------------------------------------

@dataclass
class Basis:
    """Orthonormal basis of the numerically significant column space of a matrix."""
    matrix_shape: tuple
    singular_values: np.ndarray
    retained: np.ndarray                     # bool mask over singular values
    threshold_used: float
    projector_residual: float                # ||(I - QQ^T) M||_F after truncation
    conditioning: float                      # sigma_1 / sigma_rank (1 if rank 0)

    @property
    def rank(self) -> int:
        return int(self.retained.sum())

    @property
    def vectors(self) -> np.ndarray | None:
        return self._Q

    _Q: np.ndarray | None = field(default=None, repr=False)

    def to_record(self) -> dict:
        return {
            "matrix_shape": list(self.matrix_shape),
            "singular_values": np.asarray(self.singular_values, dtype=float).tolist(),
            "retained_mask": self.retained.tolist(),
            "rank": self.rank,
            "threshold_used": float(self.threshold_used),
            "projector_residual": float(self.projector_residual),
            "conditioning_sigma1_over_sigma_rank": float(self.conditioning),
        }


def svd_basis(M: np.ndarray, rel_tol: float = 1e-8,
              abs_tol: float | None = None) -> Basis:
    """Numerical-rank basis of span(M) by SVD.

    A singular direction is retained iff ``sigma > rel_tol * sigma_1`` AND
    ``sigma > abs_tol`` (when abs_tol is given). The threshold actually used is
    recorded. If nothing is retained the basis is empty (None) - never an
    arbitrary orthonormal completion.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    if sv.size == 0 or sv[0] == 0.0:
        return Basis(M.shape, sv, np.zeros(sv.size, dtype=bool),
                     float("inf"), float(np.linalg.norm(M)), 1.0, None)
    thr = rel_tol * sv[0]
    if abs_tol is not None:
        thr = max(thr, float(abs_tol))
    keep = sv > thr
    if not keep.any():
        return Basis(M.shape, sv, keep, float(thr),
                     float(np.linalg.norm(M)), float(sv[0] / sv[-1]), None)
    Q = U[:, keep]
    resid = float(np.linalg.norm(M - Q @ (Q.T @ M)))
    return Basis(M.shape, sv, keep, float(thr), resid,
                 float(sv[0] / sv[keep][-1]), Q)


def sin_to_span(u: np.ndarray, basis: Basis) -> float:
    """Sine of the angle between a vector u and span(basis).

    Returns 1.0 for an empty basis, 0.0 for a vector already inside it. This is
    well defined for a single vector against a basis even when some other matrix
    is rank deficient - unlike span-vs-span principal angles, which become
    meaningless when either span is rank deficient (QR then fills the basis with
    rounding noise).
    """
    u = np.asarray(u, dtype=float)
    Q = basis.vectors
    if Q is None:
        return 1.0
    return float(np.linalg.norm(u - Q @ (Q.T @ u)))


# --------------------------------------------------------------------------
# declared state scaling
# --------------------------------------------------------------------------

@dataclass
class StateScaling:
    """Frozen diagonal scaling dq_scaled = W dq.

    An increment of dT = T_INTERVAL kelvin and dY = Y_INTERVAL in every mass
    fraction are declared to be *comparably significant* for the application.
    Singular values in the scaled space are then the number of such increments.
    These are declared analysis choices, not measured quantities; the
    trace-sensitive analysis in this module is kept separate.
    """
    T_interval: float = 100.0        # K: a thermally significant interval
    Y_interval: float = 1e-2         # 1%: a compositionally significant interval

    @property
    def W(self) -> np.ndarray:
        return np.concatenate([[1.0 / self.T_interval],
                               np.full(self.n_species, 1.0 / self.Y_interval)])

    n_species: int = 10

    def scale(self, dq: np.ndarray) -> np.ndarray:
        return self.W * np.asarray(dq, dtype=float)

    def component_bound(self, sigma_scaled: float) -> dict:
        """Per-component state magnitude of a scaled singular value.

        A scaled unit is dT = T_INTERVAL K in temperature or dY = Y_INTERVAL in
        each mass fraction. So a scaled magnitude sigma corresponds to a
        temperature component of sigma * T_INTERVAL kelvin and a composition
        component of sigma * Y_INTERVAL, PER UNIT Euclidean step in the
        log-control parameter.  With the defaults T_INTERVAL = 100 K and
        Y_INTERVAL = 1e-2, the application threshold sigma = 1e-6 means
        1e-4 K and 1e-8 in mass fraction - NOT 1e-6 in mass fraction.
        """
        s = float(sigma_scaled)
        return {"sigma_scaled": s,
                "temperature_K_per_unit_log_control": s * self.T_interval,
                "mass_fraction_per_unit_log_control": s * self.Y_interval}

    def to_record(self) -> dict:
        return {"T_interval_K": self.T_interval,
                "Y_interval": self.Y_interval,
                "meaning": ("a scaled unit is dT = T_INTERVAL K or dY = Y_INTERVAL "
                            "in every mass fraction; singular values count such "
                            "increments per unit Euclidean step in log control"),
                "component_bound_at_application_threshold":
                    self.component_bound(APPLICATION_THRESHOLD_DEFAULT)}


# Declared application tolerance in scaled units. With the default scaling this
# is 1e-4 K and 1e-8 in mass fraction per unit log-control step (see
# StateScaling.component_bound).  It is a DECLARED analysis choice, not an
# experimentally established accuracy target.
APPLICATION_THRESHOLD_DEFAULT = 1e-6


# --------------------------------------------------------------------------
# admissible perturbations in log-control coordinates
# --------------------------------------------------------------------------

@dataclass
class PerturbationPlan:
    """Per-coordinate central-difference steps in eta = log(gamma/gamma_ref)."""
    eta: np.ndarray
    eta_lo: float
    eta_hi: float
    steps: np.ndarray                      # h_j > 0, both eta_j +/- h_j interior
    relative: float
    gamma_ref: float = 1.0                 # eta = log(gamma/gamma_ref)

    def column_valid(self, j: int) -> bool:
        """Both neighbours of coordinate j are strictly interior."""
        e = self.eta[j]
        return bool((self.eta_lo < e - self.steps[j]) and (e + self.steps[j] < self.eta_hi))

    @property
    def n_valid(self) -> int:
        return int(sum(self.column_valid(j) for j in range(self.eta.size)))

    def gamma(self, j: int, sign: int) -> float:
        """Perturbed absolute rate for coordinate j, strictly interior."""
        return float(self.gamma_ref * np.exp(self.eta[j] + sign * self.steps[j]))

    def to_record(self) -> dict:
        return {"eta": self.eta.tolist(), "eta_lo": self.eta_lo,
                "eta_hi": self.eta_hi, "steps": self.steps.tolist(),
                "relative_step": self.relative, "gamma_ref": self.gamma_ref,
                "column_valid": [self.column_valid(j) for j in range(self.eta.size)],
                "n_valid_columns": self.n_valid}


def log_control_plan(gamma: np.ndarray, gamma_lo: float, gamma_hi: float,
                     gamma_ref: float, relative: float,
                     margin: float = 0.5) -> PerturbationPlan:
    """Steps in eta chosen per coordinate to stay strictly inside the bounds.

    ``margin`` keeps the perturbed controls away from the bounds by that
    fraction of the coordinate's own distance to the nearer bound, so the
    finite-difference stencil never produces an inadmissible history. The base
    history itself must be strictly interior.
    """
    gamma = np.asarray(gamma, dtype=float)
    eta = np.log(gamma / gamma_ref)
    eta_lo = np.log(gamma_lo / gamma_ref)
    eta_hi = np.log(gamma_hi / gamma_ref)
    if not np.all((eta > eta_lo) & (eta < eta_hi)):
        raise ValueError("base history is not strictly interior to the bounds")
    steps = np.empty_like(eta)
    for j in range(eta.size):
        room = min(eta[j] - eta_lo, eta_hi - eta[j])
        steps[j] = relative * margin * room
    return PerturbationPlan(eta, eta_lo, eta_hi, steps, relative, float(gamma_ref))


# --------------------------------------------------------------------------
# conserved-manifold tangent space in the scaled state space
# --------------------------------------------------------------------------

@dataclass
class TangentSpace:
    """Orthonormal basis of the tangent space of the conserved manifold.

    The manifold is M = {q : C q = const} with C the constraint Jacobian
    (4 elemental rows + 1 enthalpy row). In scaled coordinates the tangent space
    is the null space of C W^-1. It is computed by a rank-revealing SVD rather
    than from the normal equations C C^T, whose conditioning is squared.
    """
    n_constraints: int
    singular_values_CWinverse: np.ndarray
    rank_CWinverse: int
    conditioning: float
    basis: np.ndarray | None                # (n, n - rank) orthonormal

    def project(self, dq_scaled: np.ndarray) -> np.ndarray:
        if self.basis is None:
            return np.zeros_like(dq_scaled)
        return self.basis @ (self.basis.T @ dq_scaled)

    def leakage(self, dq_scaled: np.ndarray) -> float:
        """Conservation leakage BEFORE projection: ||C W^-1 dq_scaled||_2."""
        raise NotImplementedError                        # set by the constructor

    def to_record(self) -> dict:
        out = {"n_constraints": self.n_constraints,
               "singular_values_CWinverse": self.singular_values_CWinverse.tolist(),
               "rank_CWinverse": self.rank_CWinverse,
               "conditioning": self.conditioning,
               "tangent_dimension": 0 if self.basis is None else self.basis.shape[1]}
        if getattr(self, "conditioning_raw", None) is not None:
            out["conditioning_CWinverse_before_equilibration"] = self.conditioning_raw
            out["row_equilibration_scales"] = list(self.row_scales)
        return out


def scaled_tangent_space(C: np.ndarray, scaling: StateScaling,
                         rel_tol: float = 1e-10,
                         equilibrate: bool = True) -> TangentSpace:
    """Null space of ``C W^-1`` by rank-revealing SVD.

    For the scaled increment dq_scaled = W dq, the constraint C dq = 0 reads
    C W^-1 dq_scaled = 0, so the tangent space of the constraint manifold in the
    SCALED coordinates is null(C W^-1) - NOT null(C W).  An earlier version
    formed A = C W and therefore returned the wrong subspace whenever W is not a
    multiple of the identity (which is always, here: temperature and mass
    fractions carry different scales).

    Row equilibration rescales each constraint row to unit infinity-norm before
    the SVD, so that the numerical rank test is not dominated by one row with a
    large coefficient.  Conditioning is reported both before and after, so the
    effect of equilibration is auditable.
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = C.shape[1]
    Winv = np.diag(1.0 / scaling.W)
    A_raw = C @ Winv
    cond_raw = float(np.linalg.cond(A_raw)) if A_raw.size else float("inf")
    A = A_raw
    row_scales = np.ones(C.shape[0])
    if equilibrate:
        rn = np.maximum(np.abs(A_raw).max(axis=1), 1e-300)
        row_scales = 1.0 / rn
        A = A_raw * row_scales[:, None]
    # full_matrices=True so Vt is square: for a wide A the reduced Vt has only
    # rank(A) rows, and Vt[r:] would then be empty, silently discarding the null
    # space.
    U, sv, Vt = np.linalg.svd(A, full_matrices=True)
    tol = rel_tol * (sv[0] if sv.size and sv[0] > 0 else 1.0)
    r = int((sv > tol).sum()) if sv.size else 0
    cond = float(sv[0] / sv[r - 1]) if r > 0 else float("inf")
    basis = Vt[r:].T if r < n else np.zeros((n, 0))
    ts = TangentSpace(C.shape[0], sv, r, cond, basis if basis.shape[1] else None)
    ts.conditioning_raw = cond_raw
    ts.row_scales = row_scales.tolist()
    CWinv = Winv
    ts.leakage = lambda dq: float(np.linalg.norm(C @ (CWinv @ np.asarray(dq, float))))
    return ts


# --------------------------------------------------------------------------
# exact benchmarks: three-state linear system (review_checks.py specification)
# --------------------------------------------------------------------------

def linear3_endpoint_map(lam: np.ndarray, T: float, m: int,
                         controls: np.ndarray, z0: np.ndarray | None = None
                         ) -> np.ndarray:
    """Exact endpoint of dz_i/dt = -lam_i z_i + u(t), u piecewise constant.

    ``controls[j]`` (a scalar, shared by all states) is applied on segment j,
    which spans [j*T/m, (j+1)*T/m].  Returns z(T).  Linear in (controls, z0),
    so the endpoint map is exact.
    """
    lam = np.asarray(lam, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.shape != (m,):
        raise ValueError(f"controls must have shape ({m},), got {controls.shape}")
    G = linear3_exact_jacobian(lam, T, m)
    z = np.zeros(lam.size) if z0 is None else np.asarray(z0, dtype=float)
    return z * np.exp(-lam * T) + G @ controls


def linear3_exact_jacobian(lam: np.ndarray, T: float, m: int) -> np.ndarray:
    """Exact dz(T)/d(controls): row i, column j is the response of z_i to u_j.

    Independent of the control values (the system is linear) and of z0.
    Sign convention (review 1.1):

        G_ij = [exp(-lam_i (T - t_{j+1})) - exp(-lam_i (T - t_j))] / lam_i

    with t_j = j*T/m.  This is the SIGNED response to a POSITIVE unit control on
    segment j, so G has positive entries and the endpoint with all controls = 1
    and z0 = 0 is (1 - exp(-lam_i T)) / lam_i, all positive.  An earlier version
    of this helper returned -G; that error is invisible in the singular values
    (they are sign-blind) and was caught only by the endpoint sign check.
    """
    lam = np.asarray(lam, dtype=float)
    d = T / m
    G = np.zeros((lam.size, m))
    for i in range(lam.size):
        for j in range(m):
            tj, tjp = j * d, (j + 1) * d
            G[i, j] = (np.exp(-lam[i] * (T - tjp))
                       - np.exp(-lam[i] * (T - tj))) / lam[i]
    return G


def linear3_ode_reference(lam: np.ndarray, T: float, m: int,
                          controls: np.ndarray, z0: np.ndarray | None = None,
                          rtol: float = 1e-12, atol: float = 1e-14) -> np.ndarray:
    """Independently coded ODE solution of the same system, as a cross-check.

    Deliberately does NOT reuse ``linear3_exact_jacobian``: it integrates
    dz_i/dt = -lam_i z_i + u(t) with a standard ODE solver and a piecewise-
    constant right-hand side, then evaluates at t = T.  Used to verify the
    signed closed form and its endpoints, not only the singular values.
    """
    from scipy.integrate import solve_ivp

    lam = np.asarray(lam, dtype=float)
    controls = np.asarray(controls, dtype=float)
    d = T / m
    z0 = np.zeros(lam.size) if z0 is None else np.asarray(z0, dtype=float)
    z = z0.copy()
    t = 0.0
    # Integrate segment by segment, RESTARTING the solver at each control
    # discontinuity. A single solve across the whole horizon can step over a
    # segment entirely when the RHS happens to vanish there, which silently
    # drops that control's contribution.
    for j in range(m):
        def rhs(tt, zz, _u=controls[j]):
            return -lam * zz + _u
        sol = solve_ivp(rhs, (t, t + d), z, method="LSODA",
                        rtol=rtol, atol=atol, t_eval=[t + d], dense_output=False)
        if not sol.success:
            raise RuntimeError(sol.message)
        z = sol.y[:, -1]
        t += d
    return z


# --------------------------------------------------------------------------
# exact toy CSTR: closed-form propagator and segment-map sensitivities
# --------------------------------------------------------------------------

def _toy_segment_map(g: float, dur: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-control segment map Phi(q0) = qs + E (q0 - qs) and its pieces.

    a = 1 + gamma, qs = (gamma/a, gamma/a^2), E = exp(-a dur) [[1,0],[dur,1]].
    Also returns d(qs)/dgamma and dE/dgamma for the exact sensitivity.
    """
    a = 1.0 + float(g)
    e = np.exp(-a * dur)
    E = np.array([[e, 0.0], [dur * e, e]])
    qs = np.array([g / a, g / a**2])
    dqs = np.array([1.0 / a - g / a**2, 1.0 / a**2 - 2.0 * g / a**3])
    # dE/dgamma = -dur * E  (E = exp(-a dur) [[1,0],[dur,1]], the [[1,0],[dur,1]]
    # factor is constant in gamma and only the exponential depends on it)
    dE = -dur * E
    return E, qs, dqs, dE


def toy_endpoint(gamma_segments: np.ndarray, durations: np.ndarray,
                 q0: np.ndarray) -> np.ndarray:
    """Exact endpoint of the toy CSTR under a piecewise-constant history.

    Uses the closed-form constant-control propagator advanced segment by
    segment; no ODE error enters, so this is a ground truth for benchmarks.
    """
    q = np.asarray(q0, dtype=float).copy()
    for g, dur in zip(gamma_segments, durations):
        E, qs, _, _ = _toy_segment_map(g, dur)
        q = qs + E @ (q - qs)
    return q


def toy_endpoint_sensitivity(gamma_segments: np.ndarray, durations: np.ndarray,
                             q0: np.ndarray) -> np.ndarray:
    """Exact d(endpoint)/d(log gamma_j) of the toy CSTR, j = 0..m-1.

    The endpoint is the composition of segment maps
    Phi_j(q) = qs_j + E_j (q - qs_j).  By the chain rule, column j is

        dE_m ... dE_{j+1}  [  gamma_j (dqs_j + dE_j (q_{j-1} - qs_j)) ]

    where the bracket is the direct response of Phi_j to a relative change in
    gamma_j (an exact closed form; the toy RHS is affine in gamma) and the
    leading product propagates it through the later segments only.  This is the
    tangent-linear system for log controls with FIXED switching times:
    sensitivities are continuous across a switch and only the forcing segment
    changes.  No ODE error enters.  Returns (2, m).
    """
    gamma_segments = np.asarray(gamma_segments, dtype=float)
    durations = np.asarray(durations, dtype=float)
    m = gamma_segments.size
    # forward trajectory, segment endpoints q_0..q_m
    q = np.asarray(q0, dtype=float).copy()
    qs_list, E_list = [], []
    traj = [q.copy()]
    for g, dur in zip(gamma_segments, durations):
        E, qs, _, _ = _toy_segment_map(g, dur)
        q = qs + E @ (q - qs)
        traj.append(q.copy())
        E_list.append(E)
        qs_list.append(qs)
    # column j: direct term at segment j, then propagate through j+1..m-1
    S = np.zeros((q0.size, m))
    for j in range(m):
        g = float(gamma_segments[j])
        _, _, dqs, dE = _toy_segment_map(g, float(durations[j]))
        q_prev = traj[j]
        # d Phi_j / d gamma at FIXED input q_prev:
        #   d/dgamma [ qs + E (q_prev - qs) ] = qs' + E'(q_prev - qs) - E qs'
        v = g * (dqs + dE @ (q_prev - qs_list[j]) - E_list[j] @ dqs)
        for k in range(j + 1, m):
            v = E_list[k] @ v
        S[:, j] = v
    return S


def toy_constant_control_loggamma_derivative(gamma: float, T: float,
                                              q0: np.ndarray) -> np.ndarray:
    """Exact d phi_gamma(T; q0) / d log gamma for the toy CSTR.

    Same closed form with a single segment; used as the independently
    differentiated right-hand side of the identity test.
    """
    return toy_endpoint_sensitivity(np.array([gamma]), np.array([T]), q0)[:, 0]


# --------------------------------------------------------------------------
# detailed chemistry: log-control FD Jacobian and a tangent-linear reference
# --------------------------------------------------------------------------

def rhs_forcing_gamma(cstr, q: np.ndarray) -> np.ndarray:
    """Exact F_gamma = dF/dgamma at state q, in closed form.

    The CSTR RHS is affine in gamma, F(q,gamma) = f(q) + gamma * g(q), with

        g(q) = [ (h_in - h_k(T).Y_in) / cp ;  Y_in - Y ]

    because the exchange term is linear in gamma.  This is the exact forcing of
    the log-control sensitivity equation (the factor gamma_j comes from
    d/dlog gamma = gamma d/dgamma); it is NOT finite-differenced.
    """
    T = float(q[0])
    Y = np.asarray(q[1:])
    cstr._set_thermo(T, Y)
    hk = cstr.species_enthalpies(T)
    cp = float(cstr.gas.cp_mass)
    h_mix_in_at_T = float(hk @ cstr.Y_in)
    return np.concatenate([[ (cstr.h_in - h_mix_in_at_T) / cp ],
                           cstr.Y_in - Y])


def rhs_jacobian_fd(cstr, q: np.ndarray, gamma: float, eps: float = 1e-6
                    ) -> np.ndarray:
    """F_q at (q, gamma) by central differences of the RHS in STATE.

    Deliberately avoided: complex-step differentiation through Cantera setters,
    which clip negative mass fractions and would return a meaningless zero
    derivative at a floor.  The state step is a mixed absolute/relative one and
    is recorded by the caller.
    """
    n = q.size
    J = np.zeros((n, n))
    scale = np.maximum(np.abs(q), 1e-3)
    for i in range(n):
        qp = q.copy(); qp[i] += eps * scale[i]
        qm = q.copy(); qm[i] -= eps * scale[i]
        J[:, i] = (cstr.rhs(0.0, qp, gamma) - cstr.rhs(0.0, qm, gamma)) \
                  / (2.0 * eps * scale[i])
    return J


def constant_control_fsa(cstr, gamma: float, T: float, q0: np.ndarray,
                         method: str = "Radau", rtol: float = 1e-10,
                         atol_T: float = 1e-9, atol_Y: float = 1e-16,
                         eps: float = 1e-6) -> dict:
    """Tangent-linear derivative d phi_gamma(T; q0)/d log gamma.

    Integrates the augmented system (q, S) with

        dq/dt = F(q, gamma)
        dS/dt = F_q(q, gamma) S + gamma * F_gamma(q)

    which is the forward-sensitivity equation for a log control with a FIXED
    switch structure.  F_gamma is exact (see ``rhs_forcing_gamma``) and F_q is
    differenced numerically.  This is the independently differentiated pathway
    for the reference tangent; the endpoint is also returned so the anchor state
    is computed by the same integration.

    NOTE: switching times are parameters of the history but NOT of this map, so
    no jump or time-shift terms appear here.  A history with switching times as
    parameters would need the extra terms.
    """
    from scipy.integrate import solve_ivp

    n = q0.size
    atol = np.concatenate([[atol_T], np.full(n - 1, atol_Y),
                           [atol_T], np.full(n - 1, atol_Y)])

    def aug(t, y):
        q = y[:n]
        S = y[n:]
        Fq = rhs_jacobian_fd(cstr, q, gamma, eps)
        Fgam = rhs_forcing_gamma(cstr, q)
        return np.concatenate([cstr.rhs(t, q, gamma),
                               Fq @ S + gamma * Fgam])

    def aug_jac(t, y):
        q = y[:n]
        S = y[n:]
        Fq = rhs_jacobian_fd(cstr, q, gamma, eps)
        # d(aug)/d(q,S):  [[Fq, 0], [B, Fq]] with B = d(Fq S + gamma Fgam)/dq
        # differenced from the sensitivity block at fixed S.
        B = np.zeros((n, n))
        scale = np.maximum(np.abs(q), 1e-3)
        base = Fq @ S + gamma * rhs_forcing_gamma(cstr, q)
        for i in range(n):
            qp = q.copy(); qp[i] += eps * scale[i]
            qm = q.copy(); qm[i] -= eps * scale[i]
            fp = (rhs_jacobian_fd(cstr, qp, gamma, eps) @ S
                  + gamma * rhs_forcing_gamma(cstr, qp))
            fm = (rhs_jacobian_fd(cstr, qm, gamma, eps) @ S
                  + gamma * rhs_forcing_gamma(cstr, qm))
            B[:, i] = (fp - fm) / (2.0 * eps * scale[i])
        J = np.zeros((2 * n, 2 * n))
        J[:n, :n] = Fq
        J[n:, :n] = B
        J[n:, n:] = Fq
        return J

    y0 = np.concatenate([q0, np.zeros(n)])
    # No analytic block Jacobian is supplied: Radau differences the augmented RHS,
    # which costs about the same as differencing the state alone because every
    # augmented evaluation reuses one shared F_q.  Supplying the full block
    # Jacobian was measured to be far slower in practice.
    sol = solve_ivp(aug, (0.0, T), y0, method=method,
                    rtol=rtol, atol=atol, t_eval=[0.0, T], dense_output=False)
    if not sol.success:
        return {"success": False, "message": sol.message}
    return {"success": True, "message": "ok",
            "endpoint": sol.y[:n, -1].copy(),
            "d_endpoint_d_log_gamma": sol.y[n:, -1].copy(),
            "nfev": sol.nfev, "njev": sol.njev, "nlu": sol.nlu}


def endpoint_jacobian_fd(cstr, gamma: np.ndarray, durations: np.ndarray,
                         q0: np.ndarray, relative: float, gamma_lo: float,
                         gamma_hi: float, gamma_ref: float,
                         method: str = "Radau", rtol: float = 1e-10,
                         atol_T: float = 1e-9, atol_Y: float = 1e-16,
                         record_extrema: bool = False) -> dict:
    """dE_m/d(eta) by central differences in log controls, eta = log(gamma/ref).

    Every perturbed history is validated; a column whose stencil cannot be
    centred inside the bounds is marked INVALID and is excluded from all rank
    summaries (it is never silently treated as a zero sensitivity).  The full
    matrix, the perturbed endpoints and the validity flags are all returned.

    ``record_extrema=False`` for endpoint-only work: the dense extrema grid is
    only needed for peak/cloud analysis and multiplies the cost of every
    perturbation.
    """
    from thermoreach.controls import History

    gamma = np.asarray(gamma, dtype=float)
    durations = np.asarray(durations, dtype=float)
    plan = log_control_plan(gamma, gamma_lo, gamma_hi, gamma_ref, relative)
    m = gamma.size

    def end(g: np.ndarray) -> np.ndarray:
        h = History(g, durations)
        res = cstr.integrate(h, q0, method=method, samples_per_segment=2,
                             rtol=rtol, atol_T=atol_T, atol_Y=atol_Y,
                             record_extrema=record_extrema)
        if not res["success"]:
            raise RuntimeError(f"integration failed: {res['message']}")
        return res["states"][:, -1].copy()

    base = end(gamma)
    J = np.full((base.size, m), np.nan)          # NaN marks an invalid column
    endpoints = {}
    for j in range(m):
        if not plan.column_valid(j):
            continue
        gp = gamma.copy(); gp[j] = plan.gamma(j, +1)
        gm = gamma.copy(); gm[j] = plan.gamma(j, -1)
        zp = end(gp); zm = end(gm)
        deta = np.log(gp[j]) - np.log(gm[j])
        J[:, j] = (zp - zm) / deta
        endpoints[j] = {"gamma_plus": gp[j], "gamma_minus": gm[j],
                        "endpoint_plus": zp.tolist(), "endpoint_minus": zm.tolist(),
                        "d_eta": deta}
    return {"endpoint": base, "J": J, "plan": plan.to_record(),
            "perturbed_endpoints": endpoints,
            "n_valid_columns": int(np.isfinite(J).all(axis=0).sum()),
            "all_columns_valid": bool(np.isfinite(J).all()),
            "horizon": float(durations.sum()),
            "n_segments": int(m)}


# --------------------------------------------------------------------------
# rank classification: four distinct notions, complete-stencil gating
# --------------------------------------------------------------------------

def classify_ranks(J: np.ndarray, scaling: "StateScaling",
                   noise_scale: float | None = None,
                   application_threshold: float = APPLICATION_THRESHOLD_DEFAULT,
                   rel_tol: float = 1e-8) -> dict:
    """Separate the four rank notions the review distinguishes.

    1. ``rank_machine`` - np.linalg.matrix_rank of the scaled matrix. This is a
       MACHINE-RANK ESTIMATE at the solver's working precision; it is NOT the
       exact algebraic rank (available only for the closed-form benchmarks).
    2. ``rank_derivative_noise_resolved`` - directions above the estimated
       differentiation noise scale ||J(h) - J(h')||.  When ``noise_scale`` is
       None this is reported as None, never as zero.
    3. ``rank_application_effective`` - directions above the DECLARED
       application tolerance.  A direction above this threshold is resolved AND
       above tolerance; it is NOT an example of 'resolved but below application
       tolerance'.
    4. ``stencil_complete`` - whether every column of J is a valid centred
       stencil. An invalid stencil EXCLUDES THE WHOLE MATRIX from any full-rank
       claim; it is never aggregated with nansum into a rank count.

    ``noise_scale`` is a refinement DISCREPANCY ||J(h) - J(h/10)||, NOT a
    certified noise bound; it is stored and labelled as such.
    """
    J = np.asarray(J, dtype=float)
    valid = np.isfinite(J).all(axis=0)
    # A failed integration yields NaN columns.  numpy's SVD does not converge on
    # such a matrix, so the singular values are computed from the VALID columns
    # only, and the incomplete stencil is recorded: it excludes the matrix from
    # any full-rank claim and is never aggregated with nansum.
    Jv = J[:, valid] if J.ndim == 2 and valid.any() else J
    Js = scaling.W[:, None] * Jv if (Jv.ndim == 2 and Jv.shape[0] == scaling.W.size
                                     and Jv.size) else Jv
    sv = np.linalg.svd(Js, compute_uv=False) if (Js.ndim == 2 and Js.size) else np.array([])
    if sv.size == 0 or sv[0] <= 0:
        return {"stencil_complete": bool(valid.all()), "n_valid_columns": int(valid.sum()),
                "singular_values_scaled": sv.tolist(),
                "rank_machine": 0,
                "rank_derivative_noise_resolved": None if noise_scale is None else 0,
                "rank_application_effective": 0,
                "noise_scale_refinement_discrepancy": noise_scale,
                "noise_scale_is_certified_bound": False,
                "application_threshold": application_threshold,
                "component_bound_at_threshold": scaling.component_bound(application_threshold),
                "full_rank_claim_valid": False}
    rank_machine = int(np.linalg.matrix_rank(Js))
    r_noise = int((sv > noise_scale).sum()) if noise_scale is not None else None
    r_app = int((sv > application_threshold).sum())
    return {"stencil_complete": bool(valid.all()), "n_valid_columns": int(valid.sum()),
            "singular_values_scaled": sv.tolist(),
            "rank_machine": rank_machine,
            "rank_derivative_noise_resolved": r_noise,
            "rank_application_effective": r_app,
            "noise_scale_refinement_discrepancy": noise_scale,
            "noise_scale_is_certified_bound": False,
            "application_threshold": application_threshold,
            "component_bound_at_threshold": scaling.component_bound(application_threshold),
            # a full-rank claim requires a complete stencil AND noise resolution
            "full_rank_claim_valid": bool(valid.all()
                                          and (r_noise is not None and r_noise >= sv.size))}


def transverse_spectrum(J: np.ndarray, V: np.ndarray, scaling: "StateScaling",
                        noise_scale: float | None = None,
                        rel_tol: float = 1e-8,
                        C_of_q: np.ndarray | None = None) -> dict:
    """Total and transverse spectra of the scaled Jacobian W J.

    ``V`` is a correctly anchored, noise-filtered basis of the constant-control
    family tangent span at the SAME endpoint (columns are orthonormal).  The
    transverse part is (I - Q Q^T) W J.  A second TOTAL singular value is NOT
    automatically a direction outside the two-parameter constant family: only
    the transverse spectrum can support that, and only above the noise scale.

    Both raw and projected spectra are returned and labelled, so a disagreement
    between them can be compared against the physical conservation error rather
    than hidden by projection.
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    W = scaling.W
    # NaN columns must not reach the SVD; they are recorded as an incomplete
    # stencil in classify_ranks and excluded here.
    valid = np.isfinite(J).all(axis=0)
    J = J[:, valid] if (J.ndim == 2 and valid.any()) else J
    WJ = W[:, None] * J
    WV = W[:, None] * V
    sv_total = np.linalg.svd(WJ, compute_uv=False)
    Q, _ = np.linalg.qr(WV)
    # Manifold-leakage floor: the scaled distance of the derivative columns from
    # the conserved-manifold tangent space null(C W^-1).  A transverse direction
    # BELOW this floor is indistinguishable from conservation/ integration error
    # and must not be reported as off-family state generation.  Projection must
    # not hide this error, so the floor is reported alongside the transverse
    # spectrum rather than subtracted from it.
    ts = scaled_tangent_space(C_of_q, scaling) if C_of_q is not None else None
    sv_leak = None
    tangent_dim = None
    if ts is not None and ts.basis is not None:
        N = ts.basis
        J_leak = (np.eye(N.shape[0]) - N @ N.T) @ WJ
        sv_leak = np.linalg.svd(J_leak, compute_uv=False)
        tangent_dim = int(N.shape[1])
    J_perp = (np.eye(Q.shape[0]) - Q @ Q.T) @ WJ
    sv_perp = np.linalg.svd(J_perp, compute_uv=False)
    out = {"singular_values_total_scaled": sv_total.tolist(),
           "singular_values_transverse_scaled": sv_perp.tolist(),
           "singular_values_manifold_leakage_scaled": (sv_leak.tolist()
                                                        if sv_leak is not None else None),
           "tangent_space_dimension": tangent_dim,
           "rank_V_basis": int(Q.shape[1]),
           "noise_scale_refinement_discrepancy": noise_scale,
           "note": ("total = SVD of W J; transverse = SVD of (I - Q Q^T) W J with "
                    "Q an orthonormal basis of the scaled constant-control tangent "
                    "span at the SAME endpoint; manifold leakage = SVD of "
                    "(I - N N^T) W J with N the conserved-manifold tangent space "
                    "null(C W^-1). A transverse direction below the leakage floor "
                    "or below the noise discrepancy is NOT evidence of off-family "
                    "state generation.")}
    if sv_leak is not None and sv_perp.size:
        out["transverse_above_leakage_floor"] = bool(
            sv_perp[0] > (sv_leak[0] if sv_leak.size else 0.0))
    return out


def endpoint_jacobian_fsa_system(rhs, rhs_dgamma, state_jac,
                                 gamma_segments: np.ndarray,
                                 durations: np.ndarray, q0: np.ndarray,
                                 method: str = "Radau", rtol: float = 1e-11,
                                 atol: np.ndarray | None = None) -> dict:
    """Exact tangent-linear endpoint Jacobian d E / d eta for a segment history
    of a system AFFINE in the control:  F(q, gamma) = gamma * d(q) + r(q).

    Arguments:
      rhs(t, q, gamma)            - the base right-hand side
      rhs_dgamma(q)               - d F/d gamma, exact by the affine structure
      state_jac(q, gamma)         - d F/d q

    Forward-sensitivity equations (exact inhomogeneity, no FD in the control):

        dq/dt   = F(q, gamma_k)                     (t in segment k)
        dS_j/dt = F_q(q) S_j + gamma_j F_gamma(q)   (t in segment j ONLY)

    Column j of the endpoint Jacobian is S_j at the final time.  Because no step
    in the control is differenced, this estimator has NO control-step noise floor
    and is the reference against which the FD estimator's refinement discrepancy
    is judged.  Switching times are parameters of the history but not of this map,
    so no time-shift terms appear: this is the Jacobian with respect to LEVELS.
    """
    from scipy.integrate import solve_ivp

    gamma_segments = np.asarray(gamma_segments, dtype=float)
    durations = np.asarray(durations, dtype=float)
    n = q0.size
    m = gamma_segments.size
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    if atol is None:
        atol = np.full(n, 1e-12)

    def aug(t, y):
        q = y[:n]
        S = y[n:].reshape(n, m)
        k = int(np.searchsorted(edges, t, side="right")) - 1
        k = min(max(k, 0), m - 1)
        Fq = np.asarray(state_jac(q, gamma_segments[k]), dtype=float)
        dS = Fq @ S
        dS[:, k] += gamma_segments[k] * np.asarray(rhs_dgamma(q), dtype=float)
        return np.concatenate([rhs(t, q, gamma_segments[k]), dS.ravel()])

    y0 = np.concatenate([q0, np.zeros(n * m)])
    res = solve_ivp(aug, (0.0, float(durations.sum())), y0, method=method,
                    t_eval=[float(durations.sum())], rtol=rtol,
                    atol=np.concatenate([atol, np.full(n * m, atol.min())]),
                    dense_output=False)
    if not res.success:
        raise RuntimeError(f"FSA integration failed: {res.message}")
    return {"J": res.y[n:, -1].reshape(n, m).copy(),
            "endpoint": res.y[:n, -1].copy(),
            "success": bool(res.success),
            "n_rhs_evaluations": int(res.nfev),
            "method": method, "rtol": rtol}


def endpoint_jacobian_fsa(cstr, gamma_segments: np.ndarray, durations: np.ndarray,
                          q0: np.ndarray, method: str = "Radau",
                          rtol: float = 1e-11, atol_T: float = 1e-10,
                          atol_Y: float = 1e-17, eps: float = 1e-6) -> dict:
    """CSTR wrapper around :func:`endpoint_jacobian_fsa_system`.

    F_gamma is exact because the CSTR right-hand side is affine in the control
    (see :func:`rhs_forcing_gamma`); F_q is differenced numerically in STATE since
    Cantera exposes no Jacobian, which costs one FD pass per RHS evaluation
    independent of the number of segments.
    """
    atol = np.concatenate([[atol_T], np.full(cstr.gas.n_species, atol_Y)])
    out = endpoint_jacobian_fsa_system(
        lambda t, q, g: cstr.rhs(t, q, g),
        lambda q: rhs_forcing_gamma(cstr, q),
        lambda q, g: rhs_jacobian_fd(cstr, q, g, eps),
        gamma_segments, durations, q0, method=method, rtol=rtol, atol=atol)
    out["atol"] = {"T": atol_T, "Y": atol_Y}
    out["state_jacobian_eps"] = eps
    return out
