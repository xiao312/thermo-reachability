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

    def to_record(self) -> dict:
        return {"T_interval_K": self.T_interval,
                "Y_interval": self.Y_interval,
                "meaning": ("an increment of dT = T_INTERVAL K is declared "
                            "comparable to dY = Y_INTERVAL in every mass fraction; "
                            "singular values count such increments")}


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
        return {"n_constraints": self.n_constraints,
                "singular_values_CWinverse": self.singular_values_CWinverse.tolist(),
                "rank_CWinverse": self.rank_CWinverse,
                "conditioning": self.conditioning,
                "tangent_dimension": 0 if self.basis is None else self.basis.shape[1]}


def scaled_tangent_space(C: np.ndarray, scaling: StateScaling,
                         rel_tol: float = 1e-10) -> TangentSpace:
    """Null space of C W^-1 by SVD; records conditioning of C W^-1."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = C.shape[1]
    A = C @ np.diag(scaling.W)
    # full_matrices=True so Vt is square: for a wide A the reduced Vt has only
    # rank(A) rows, and Vt[r:] would then be empty, silently discarding the null
    # space.
    U, sv, Vt = np.linalg.svd(A, full_matrices=True)
    tol = rel_tol * (sv[0] if sv.size and sv[0] > 0 else 1.0)
    r = int((sv > tol).sum()) if sv.size else 0
    cond = float(sv[0] / sv[r - 1]) if r > 0 else float("inf")
    basis = Vt[r:].T if r < n else np.zeros((n, 0))
    ts = TangentSpace(C.shape[0], sv, r, cond, basis if basis.shape[1] else None)
    CWinv = np.diag(1.0 / scaling.W)
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
    """
    lam = np.asarray(lam, dtype=float)
    d = T / m
    G = np.zeros((lam.size, m))
    for i in range(lam.size):
        for j in range(m):
            ts, te = j * d, (j + 1) * d
            G[i, j] = (np.exp(-lam[i] * (T - ts))
                       - np.exp(-lam[i] * (T - te))) / lam[i]
    return G


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
