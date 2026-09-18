"""The local thermochemical state generator (Revision 7).

The accepted local rank-three result is turned into a *predictive* generator: a
canonical curved reference state plus ONE conservation-compatible scalar
correction,

    q_hat = decoder( Y_B(beta) + a * n_Y ,  h_B(beta) )

where

  * beta = (log gamma_B, t_B) are reference coordinates on the canonical family
    B(gamma, t; q0) of constant-exchange histories;
  * n_Y is a FIXED species direction satisfying E n_Y = 0 and 1^T n_Y = 0, so the
    corrected composition preserves the reference's elemental inventory and
    normalization - the correction is "conservation-compatible" by construction,
    not by clipping invalid species afterwards;
  * the temperature is recovered from the composition by ENTHALPY INVERSION at
    the reference's own enthalpy h_B, so an additive (T, Y) correction is never
    assumed to conserve energy at finite amplitude;
  * h_B and the inventory b_B are EXACT closed-form functions of beta
    (h = h_in + e^-Gamma (h_0 - h_in), b = b_in + e^-Gamma (b_0 - b_in) with
    Gamma = gamma t), so computing them from beta is not leakage of the target
    endpoint - it uses only the input history coordinates.

Two representations are deliberately separated:

  * the ORACLE representation is given the true endpoint q and fits the single
    scalar a (and, for variant A, the reference coordinates) against it; it
    measures what the ansatz is CAPABLE of representing;
  * the PREDICTIVE generator predicts (beta, a) from the control history eta
    alone, with no access to the endpoint at inference; it measures what the
    ansatz can PREDICT.

Model fitting (complexity, regularization, the normal direction) is frozen
before any test history is generated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# 1. exact balances from the controls alone (closed-form mixing law)
# ---------------------------------------------------------------------------


def exposure_integral(gammas, durations) -> float:
    """Gamma = sum_j gamma_j dur_j for a piecewise-constant history."""
    g = np.asarray(gammas, dtype=float)
    d = np.asarray(durations, dtype=float)
    return float(np.sum(g * d))


def exact_balances_from_controls(cstr, q0, gammas, durations) -> dict:
    """h and the elemental inventory b of the state reached by a
    piecewise-constant history, from the exponential mixing law alone.

    For this reactor db/dt = gamma (b_in - b) and dh/dt = gamma (h_in - h)
    exactly (chemistry conserves elements and enthalpy), so

        b = b_in + e^-Gamma (b_0 - b_in),     h = h_in + e^-Gamma (h_0 - h_in)

    with Gamma = sum_j gamma_j dur_j.  These are exact, and they are functions of
    the CONTROLS only - never of the endpoint.
    """
    Gamma = exposure_integral(gammas, durations)
    dec = math.exp(-Gamma)
    Y0 = np.asarray(q0[1:], dtype=float)
    b0 = cstr.E @ Y0
    hk0 = cstr.species_enthalpies(float(q0[0]))
    h0 = float(hk0 @ Y0)
    return {
        "Gamma": Gamma,
        "dec": dec,
        "h_J_kg": cstr.h_in + dec * (h0 - cstr.h_in),
        "b": (cstr.b_in + dec * (b0 - cstr.b_in)).tolist(),
        "h_in_J_kg": cstr.h_in,
        "h0_J_kg": h0,
        "b_in": cstr.b_in.tolist(),
        "b0": b0.tolist(),
        "source": "closed-form exponential mixing law; no endpoint used",
    }


# ---------------------------------------------------------------------------
# 2. the conservation-compatible subspace for the correction direction
# ---------------------------------------------------------------------------

def null_space_basis(C: np.ndarray, tol_rel: float = 1e-10,
                    tol_abs: float = 1e-14) -> dict:
    """An orthonormal basis of null(C), row-equilibrated and rank-revealing.

    E is badly scaled across element rows, and a relative-only threshold would
    report a rank equal to the noise floor - the trap behind the withdrawn
    Phase 3D claim - so both an absolute and a relative threshold are used.
    The null space lives in the TRAILING ROWS of Vh (C is (n_c, n_sp), so Vh is
    (n_sp, n_sp)).
    """
    C = np.asarray(C, dtype=float)
    s = np.maximum(np.abs(C).max(axis=1), 1e-300)
    Ce = C / s[:, None]
    _, S, Vh = np.linalg.svd(Ce, full_matrices=True)
    rank = int(np.sum((S > tol_abs) & (S > tol_rel * (S[0] if S.size else 1.0))))
    k = max(C.shape[1] - rank, 0)
    basis = Vh[rank:, :].T if k else np.zeros((C.shape[1], 0))
    return {"basis": basis, "dim": int(k), "rank_constraints": int(rank),
            "n_constraints": int(C.shape[0]),
            "singular_values_of_equilibrated_rows": S.tolist(),
            "conditioning": float(S[0] / S[rank - 1]) if rank else float("inf")}


def conservation_subspace(E: np.ndarray, y_interval: float = 1.0,
                          tol_rel: float = 1e-10,
                          tol_abs: float = 1e-14) -> dict:
    """An orthonormal basis, in SCALED composition coordinates, of the species
    directions that preserve every element and the normalization.

    A scaled direction z corresponds to the physical composition change
    n = y_interval * z, and the constraints E n = 0, 1^T n = 0 become
    (E y_interval) z = 0 and (1^T y_interval) z = 0.  Working in the scaled
    coordinates is what makes the projection below an orthogonal projection in
    the declared metric; mixing the two metrics would silently change the
    answer.  y_interval = 1 recovers the unscaled subspace.
    """
    E = np.asarray(E, dtype=float)
    yi = float(y_interval)
    C = np.vstack([E * yi, np.full(E.shape[1], yi)])
    out = null_space_basis(C, tol_rel=tol_rel, tol_abs=tol_abs)
    out["y_interval"] = yi
    out["note"] = ("columns are orthonormal in scaled coordinates; multiply by "
                   "y_interval to obtain the physical mass-fraction direction")
    return out


def project_onto_conservation_subspace(direction: np.ndarray, E: np.ndarray,
                                       y_interval: float = 1.0,
                                       tol_rel: float = 1e-10) -> dict:
    """Project a SCALED composition direction onto the conservation-compatible
    subspace, in the scaled metric.  Returns the PHYSICAL (unscaled) unit
    direction n_Y with E n_Y = 0 and 1^T n_Y = 0.

    Rank-revealing, and UNRESOLVED is reported (never silently replaced by a
    fabricated direction) when the projection is degenerate.
    """
    sub = conservation_subspace(E, y_interval=y_interval, tol_rel=tol_rel)
    Q = sub["basis"]
    d = np.asarray(direction, dtype=float)
    if Q.size == 0:
        return {"n_Y": None, "resolved": False,
                "reason": "conservation subspace is trivial", "subspace": sub}
    proj = Q @ (Q.T @ d)
    nrm = float(np.linalg.norm(proj))
    if nrm <= 1e-14 or nrm < 1e-10 * float(np.linalg.norm(d)):
        return {"n_Y": None, "resolved": False,
                "reason": (f"projection of the fitted direction onto the "
                           f"conservation subspace is degenerate "
                           f"(|proj| = {nrm:.3e})"),
                "subspace": sub}
    n_Y = float(y_interval) * (proj / nrm)
    assert np.allclose(E @ n_Y, 0.0, atol=1e-10), "E n_Y must vanish"
    assert abs(n_Y.sum()) < 1e-10, "1^T n_Y must vanish"
    return {"n_Y": n_Y.tolist(), "resolved": True,
            "projection_retained_fraction": nrm / max(float(np.linalg.norm(d)),
                                                      1e-300),
            "subspace": {"dim": sub["dim"],
                         "conditioning": sub["conditioning"],
                         "y_interval": sub["y_interval"]}}


# ---------------------------------------------------------------------------
# 3. the feasible interval of the scalar correction
# ---------------------------------------------------------------------------


def feasible_a_interval(Y_B, n_Y, y_floor: float = 0.0) -> dict:
    """The interval of a for which Y_B + a n_Y stays non-negative.

    This is derived from the species bounds THEMSELVES (a feasibility statement
    about the physical decoder), not from the range of a seen in training.
    """
    Y_B = np.asarray(Y_B, dtype=float)
    n_Y = np.asarray(n_Y, dtype=float)
    if not np.any(np.abs(n_Y) > 0):
        return {"a_lo": None, "a_hi": None, "bounded": False,
                "reason": "the correction direction is zero"}
    lo, hi = [], []
    for yk, nk in zip(Y_B, n_Y):
        if nk > 0:
            lo.append((y_floor - yk) / nk)
        elif nk < 0:
            hi.append((y_floor - yk) / nk)
    a_lo = max(lo) if lo else None
    a_hi = min(hi) if hi else None
    bounded = a_lo is not None and a_hi is not None and a_lo < a_hi
    if not bounded:
        return {"a_lo": a_lo, "a_hi": a_hi, "bounded": False,
                "reason": ("no admissible interval: the direction drives a "
                           "species negative on both sides of the anchor"),
                "Y_B_min": float(Y_B.min()), "n_Y_argmax": int(np.argmax(n_Y))}
    return {"a_lo": float(a_lo), "a_hi": float(a_hi), "bounded": True,
            "binding_species_low": int(np.argmax(lo)),
            "binding_species_high": int(np.argmin(hi)),
            "Y_B_min": float(Y_B.min()),
            "interval_width": float(a_hi - a_lo)}


# ---------------------------------------------------------------------------
# 4. the physical decoder: composition -> temperature by enthalpy inversion
# ---------------------------------------------------------------------------


class PhysicalDecoder:
    """Decode (T, Y) from a composition and an enthalpy.

    The temperature is SOLVED FOR by enthalpy inversion at fixed (h, p, Y); an
    additive (T, Y) correction is never assumed to conserve energy at finite
    amplitude.  Invalid compositions are REJECTED and recorded, never clipped to
    make a prediction look admissible.
    """

    def __init__(self, cstr, y_floor: float = -1e-12,
                 sum_atol: float = 1e-9):
        self.cstr = cstr
        self.y_floor = y_floor
        self.sum_atol = sum_atol
        self.n_rejected = {"species_negative": 0, "normalization": 0,
                           "enthalpy_inversion": 0, "other": 0}

    def decode(self, Y_hat, h_J_kg: float, a: float | None = None) -> dict:
        cstr = self.cstr
        Y = np.asarray(Y_hat, dtype=float)
        out = {"a": a, "h_J_kg": float(h_J_kg), "Y_raw": Y.tolist()}
        s = float(Y.sum())
        if not np.isfinite(s):
            self.n_rejected["other"] += 1
            return {**out, "status": "rejected_nonfinite",
                    "reason": "non-finite composition"}
        if abs(s - 1.0) > self.sum_atol:
            self.n_rejected["normalization"] += 1
            return {**out, "status": "rejected_normalization",
                    "reason": f"sum(Y) - 1 = {s - 1.0:.3e} exceeds "
                              f"{self.sum_atol:.0e}",
                    "sum_Y_minus_1": s - 1.0}
        if Y.min() < self.y_floor:
            self.n_rejected["species_negative"] += 1
            return {**out, "status": "rejected_species_bounds",
                    "reason": f"min(Y) = {Y.min():.3e} < {self.y_floor:.0e}; "
                              f"the state is NOT admissible and is not clipped "
                              f"to look admissible",
                    "min_Y": float(Y.min()),
                    "n_negative": int(np.sum(Y < 0.0))}
        try:
            gas = cstr.gas
            gas.HPY = float(h_J_kg), float(cstr.p), np.maximum(Y, 0.0)
            T = float(gas.T)
            # verify the inversion round-trips
            h_back = float(gas.enthalpy_mass)
            ok_inv = abs(h_back - float(h_J_kg)) <= (
                1e-8 * max(abs(float(h_J_kg)), 1.0))
            if not ok_inv:
                self.n_rejected["enthalpy_inversion"] += 1
                return {**out, "status": "rejected_enthalpy_inversion",
                        "reason": "inversion did not round-trip the enthalpy",
                        "T_solved": T, "h_roundtrip_J_kg": h_back}
        except Exception as exc:                    # noqa: BLE001
            self.n_rejected["enthalpy_inversion"] += 1
            return {**out, "status": "rejected_enthalpy_inversion",
                    "reason": f"enthalpy inversion raised {exc!r}"}
        # E Y_hat must equal the reference inventory exactly (E n_Y = 0)
        b_hat = (cstr.E @ Y).tolist()
        return {**out, "status": "admissible", "T_K": T,
                "Y": np.maximum(Y, 0.0).tolist(),
                "b_hat": b_hat,
                "h_roundtrip_J_kg": float(gas.enthalpy_mass),
                "density_kg_m3": float(gas.density),
                "q_hat": np.concatenate([[T], np.maximum(Y, 0.0)]).tolist()}


# ---------------------------------------------------------------------------
# 5. regression from control history to reference coordinates and correction
# ---------------------------------------------------------------------------


def _design(eta: np.ndarray, order: int) -> np.ndarray:
    """Feature map.  Order 1: eta plus a constant.  Order 2: adds squares and
    pairwise products (no cubic or higher: the development domain is a small
    ball and the training set is small, so the model must stay well conditioned)."""
    e = np.asarray(eta, dtype=float).ravel()
    cols = [np.ones(1), e]
    if order >= 2:
        cols.append(e ** 2)
        iu = np.triu_indices(e.size, k=1)
        cols.append(e[iu[0]] * e[iu[1]])
    return np.concatenate(cols)


@dataclass
class LinearRegressor:
    """Affine or quadratic least squares with documented ridge regularization.

    Ridge is on the non-constant coefficients only, and it is FIXED at fit time
    (never tuned on the test set)."""
    order: int = 1
    ridge: float = 1e-8
    coef_: np.ndarray | None = field(default=None, repr=False)
    feature_names_: list = field(default_factory=list, repr=False)

    def fit(self, X_etas: list[np.ndarray], targets: np.ndarray):
        X = np.stack([_design(e, self.order) for e in X_etas])
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        if targets.shape[0] != X.shape[0]:
            targets = targets.T
        n_feat = X.shape[1]
        reg = np.eye(n_feat)
        reg[0, 0] = 0.0                       # do not penalize the intercept
        A = X.T @ X + self.ridge * reg
        B = X.T @ targets
        self.coef_ = np.linalg.solve(A, B)
        d = int(X_etas[0].size) if X_etas else 0
        iu = list(zip(*np.triu_indices(d, k=1)))
        self.feature_names_ = (
            ["const"] + [f"eta_{i}" for i in range(d)]
            + ([f"eta_{i}^2" for i in range(d)] if self.order >= 2 else [])
            + ([f"eta_{i}*eta_{j}" for i, j in iu] if self.order >= 2 else []))
        # training diagnostics
        pred = X @ self.coef_
        self.train_rms_ = float(np.sqrt(np.mean((pred - targets) ** 2)))
        self.cond_XtX_ = float(np.linalg.cond(X.T @ X + self.ridge * reg))
        return self

    def predict(self, eta: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("the regressor must be fitted before prediction")
        return _design(eta, self.order) @ self.coef_


# ---------------------------------------------------------------------------
# 6. the generator itself
# ---------------------------------------------------------------------------


@dataclass
class LocalGenerator:
    """Canonical curved reference + one conservation-compatible correction.

    Every field needed at inference is frozen at fit time: the reference library,
    the fitted normal direction, the two regressors, the decoder and the declared
    training domain."""
    cstr: object
    library: object
    decoder: PhysicalDecoder
    n_Y: list                       # unit species direction, E n_Y = 0, 1^T n_Y = 0
    beta_regressor: LinearRegressor
    a_regressor: LinearRegressor
    gamma_ref: float
    T_horizon: float
    durations: list
    training_domain: dict
    normal_report: dict
    model_identity: dict

    # ------------------------------------------------------------------
    def reference_state(self, log_gamma: float, t: float) -> dict:
        """The canonical family member, by an independent fresh integration so a
        stale cache can never be used, with the exact balances from beta."""
        gamma = self.gamma_ref * float(np.exp(log_gamma))
        q_B = np.asarray(self.library.evaluate_fresh(gamma, t)["q_B"],
                         dtype=float)
        bal = exact_balances_from_controls(self.cstr, self.library.q0,
                                          [gamma], [float(t)])
        return {"gamma": gamma, "t": t, "q_B": q_B.tolist(),
                "h_B_J_kg": bal["h_J_kg"], "b_B": bal["b"],
                "Gamma_B": bal["Gamma"]}

    # ------------------------------------------------------------------
    def predict(self, eta: np.ndarray, durations=None) -> dict:
        """Predict the state from control history coordinates ALONE.

        eta = log(gamma_j / gamma_ref) is the input; the balances are computed
        from the controls in closed form and the reference coordinates and the
        correction coefficient come from the frozen regressors.
        """
        eta = np.asarray(eta, dtype=float).ravel()
        durs = np.asarray(durations if durations is not None else self.durations,
                          dtype=float)
        gammas = self.gamma_ref * np.exp(eta)
        # exact balances from the controls
        bal = exact_balances_from_controls(self.cstr, self.library.q0,
                                          gammas, durs)
        beta = np.asarray(self.beta_regressor.predict(eta), dtype=float).ravel()
        log_gamma_B, log_t_B = float(beta[0]), float(beta[1])
        t_B = self.T_horizon * float(np.exp(log_t_B))
        ref = self.reference_state(log_gamma_B, t_B)
        a_raw = float(np.asarray(self.a_regressor.predict(eta)).ravel()[0])
        Y_B = np.asarray(ref["q_B"], dtype=float)[1:]
        n_Y = np.asarray(self.n_Y, dtype=float)
        interval = feasible_a_interval(Y_B, n_Y)
        if interval["bounded"]:
            a = float(np.clip(a_raw, interval["a_lo"], interval["a_hi"]))
            clipped = not math.isclose(a, a_raw)
        else:
            a, clipped = a_raw, False
        Y_hat = Y_B + a * n_Y
        dec = self.decoder.decode(Y_hat, ref["h_B_J_kg"], a=a)
        # uncertainty indicator: distance of the input from the training domain
        meas = math.sqrt(float(np.sum((durs / self.T_horizon) * eta ** 2)))
        return {
            "input": {"eta": eta.tolist(), "gammas": gammas.tolist(),
                      "durations_s": durs.tolist(),
                      "measured_time_norm": meas,
                      "Gamma": bal["Gamma"],
                      "exact_balances_from_controls": bal},
            "model_identity": self.model_identity,
            "training_domain": self.training_domain,
            "predicted_reference": {
                "log_gamma_B": log_gamma_B, "log_t_B_over_T": log_t_B,
                "gamma_B": ref["gamma"], "t_B_s": ref["t"]},
            "predicted_correction": {"a": a, "a_raw": a_raw,
                                     "a_clipped_to_feasible": clipped,
                                     "feasible_interval": interval},
            "canonical_state_before_correction": {
                "q_B": ref["q_B"], "status": "simulated",
                "note": ("a family member is a real integrated state, so it is "
                         "admissible by construction and needs no decoder")},
            "decoded_state": dec,
            "provenance": {
                "balances": "computed in closed form from the input controls",
                "reference_state": "simulated (fresh integration of the canonical "
                                   "family at the predicted coordinates)",
                "correction": "reconstructed (a scalar along the frozen fitted "
                              "direction)",
                "temperature": "solved by enthalpy inversion"},
            "uncertainty_indicator": {
                "measured_time_norm": meas,
                "inside_declared_domain":
                    bool(meas <= self.training_domain["max_measured_time_norm"]),
                "regression_leverage": float(_leverage(
                    self.beta_regressor, eta))},
        }


def _leverage(reg: LinearRegressor, eta: np.ndarray) -> float:
    """A simple uncertainty indicator for the regression: the leverage of the
    input in the fitted design.  Returns NaN if unavailable."""
    try:
        X = getattr(reg, "_X_")
        if X is None:
            return float("nan")
        x = _design(eta, reg.order)
        A = X.T @ X + reg.ridge * np.eye(X.shape[1])
        A[0, 0] -= reg.ridge                      # intercept unpenalized
        return float(x @ np.linalg.pinv(A) @ x)
    except Exception:                                   # noqa: BLE001
        return float("nan")


def fit_local_generator(cstr, library, train_cases, durations, gamma_ref,
                        T_horizon, scaling, decoder, order: int = 1,
                        ridge: float = 1e-8, anchor_log_gamma: float = 0.0,
                        anchor_log_t: float = 0.0,
                        normal_rel_tol: float = 1e-10) -> tuple:
    """Fit the generator on the training cases.

    train_cases: dicts with
        eta                 log control perturbation
        q_target            the true endpoint
        located_reference   the oracle reference coordinates (training only)
    The model (complexity, regularization, normal direction) is FROZEN here and
    must not be refit after test data exists.
    """
    etas = [np.asarray(c["eta"], dtype=float) for c in train_cases]
    # targets: log gamma_B / gamma_ref and log(t_B / T)
    Y_beta = np.array([[float(c["located_reference"]["log_gamma_B"]),
                       float(c["located_reference"]["log_t_B_over_T"])]
                      for c in train_cases])
    beta_reg = LinearRegressor(order=order, ridge=ridge).fit(etas, Y_beta)
    beta_reg._X_ = np.stack([_design(e, order) for e in etas])

    # ---- the normal direction, fitted on the SCALED training residuals -----
    # The projection is done in the SCALED metric (the same metric the singular
    # values are measured in), and the returned direction is unscaled so that
    # E n_Y = 0 and 1^T n_Y = 0 hold in physical mass-fraction units.
    W = np.asarray(scaling, dtype=float)
    y_interval = 1.0 / W[1]                    # a scaled Y unit = y_interval
    residuals = []
    for c in train_cases:
        q = np.asarray(c["q_target"], dtype=float)
        qB = np.asarray(c["q_B"], dtype=float)
        residuals.append(W * (q - qB))
    R = np.stack(residuals, axis=1)                  # (n_states, N)
    # rank-revealing leading direction of the residual cloud; the COMPOSITION
    # part of that direction is what the correction acts on
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    w_Y = U[1:, 0]
    rep = project_onto_conservation_subspace(w_Y, cstr.E,
                                             y_interval=y_interval,
                                             tol_rel=normal_rel_tol)
    if not rep["resolved"]:
        raise RuntimeError(f"the normal field is unresolved: {rep['reason']}")
    n_Y = np.asarray(rep["n_Y"], dtype=float)

    # ---- the correction coefficient: fit on oracle a values ---------------
    a_targets = []
    for c in train_cases:
        Y_B = np.asarray(c["q_B"], dtype=float)[1:]
        q = np.asarray(c["q_target"], dtype=float)
        bal = exact_balances_from_controls(
            cstr, library.q0,
            [gamma_ref * math.exp(float(c["located_reference"]["log_gamma_B"]))],
            [T_horizon * math.exp(float(c["located_reference"]["log_t_B_over_T"]))])
        a_o = oracle_correction_coefficient(q, Y_B, n_Y, bal["h_J_kg"], decoder,
                                            scaling=W)
        a_targets.append(a_o["a"])
    a_reg = LinearRegressor(order=order, ridge=ridge).fit(
        etas, np.asarray(a_targets, dtype=float))
    a_reg._X_ = np.stack([_design(e, order) for e in etas])

    model = LocalGenerator(
        cstr=cstr, library=library, decoder=decoder, n_Y=n_Y.tolist(),
        beta_regressor=beta_reg, a_regressor=a_reg,
        gamma_ref=gamma_ref, T_horizon=T_horizon, durations=list(durations),
        training_domain={"max_measured_time_norm": float(max(
            math.sqrt(float(np.sum((np.asarray(durations, dtype=float)
                                    / T_horizon) * e ** 2))) for e in etas)),
            "n_train": len(train_cases),
            "order": order, "ridge": ridge},
        normal_report={"leading_residual_singular_values": S.tolist(),
                       "retained_fraction": rep["projection_retained_fraction"],
                       "subspace": rep["subspace"],
                       "residual_cloud_rank_revealed": True},
        model_identity={"kind": "canonical_reference_plus_one_correction",
                        "mechanism": cstr.cfg.mechanism,
                        "gamma_ref": gamma_ref, "T_horizon_s": T_horizon,
                        "n_segments": int(len(durations)),
                        "order": order, "ridge": ridge,
                        "library_identity": library.identity()})
    return model, {"beta_train_rms": beta_reg.train_rms_,
                   "a_train_rms": a_reg.train_rms_,
                   "cond_XtX": beta_reg.cond_XtX_,
                   "oracle_a": a_targets,
                   "normal_report": model.normal_report}


def oracle_correction_coefficient(q_target, Y_B, n_Y, h_J_kg: float,
                                  decoder: PhysicalDecoder,
                                  scaling=None, n_scan: int = 41) -> dict:
    """The best single scalar a given the TRUE endpoint: a representational
    measurement, not a prediction.  Bounded 1-D minimization over the feasible
    interval, evaluated through the physical decoder (so the temperature is
    enthalpy-inverted, never assumed additive)."""
    interval = feasible_a_interval(Y_B, n_Y)
    if not interval["bounded"]:
        return {"a": None, "resolved": False,
                "reason": "no feasible correction interval", "interval": interval}
    lo, hi = interval["a_lo"], interval["a_hi"]
    W = np.ones(q_target.size) if scaling is None else np.asarray(scaling,
                                                                 dtype=float)

    def obj(a):
        dec = decoder.decode(Y_B + a * np.asarray(n_Y, dtype=float), h_J_kg)
        if dec["status"] != "admissible":
            return float("inf")
        q_hat = np.asarray(dec["q_hat"], dtype=float)
        return float(np.linalg.norm(W * (q_target - q_hat)))

    grid = np.linspace(lo, hi, n_scan)
    vals = np.array([obj(a) for a in grid])
    finite = np.isfinite(vals)
    if not finite.any():
        return {"a": None, "resolved": False,
                "reason": "no admissible correction on the feasible interval",
                "interval": interval}
    i = int(np.argmin(np.where(finite, vals, np.inf)))
    r = minimize_scalar(obj, bounds=(grid[max(i - 1, 0)],
                                     grid[min(i + 1, n_scan - 1)]),
                        method="bounded",
                        options={"xatol": 1e-14})
    a_best = float(r.x) if r.success and np.isfinite(r.fun) else float(grid[i])
    d_best = float(r.fun) if r.success and np.isfinite(r.fun) else float(vals[i])
    return {"a": a_best, "resolved": True, "dist_scaled_raw": d_best,
            "interval": interval, "at_grid": float(vals[i])}
