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

    INTERPRETATION (Revision 8).  When the initial state is the hot
    HP-equilibrium of the SAME feed, h0 = h_in and b0 = b_in identically, so both
    invariants are CONSTANT for every control and CANNOT identify Gamma: the
    balance law is then degenerate, and any observed gamma_B * t_B ~ Gamma is an
    EMPIRICAL relationship at that anchor, not a consequence of invariant
    matching.  Only when b0 != b_in (or h0 != h_in) does the decaying factor
    e^-Gamma carry exposure information, and then only as well as |b0 - b_in| is
    conditioned.  The caller must therefore check the reported h0/h_in and
    b0/b_in rather than assume the law identifies anything.
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


# ---------------------------------------------------------------------------
# 2a. the structural face of the species polytope
#
# A species is STRUCTURALLY ABSENT when the feed carries none of the elements it
# is built from: no admissible trajectory can ever create it, so its mass
# fraction is exactly zero for all time and all controls.  A species is FIXED
# when it takes no part in any reaction AND its initial value equals its feed
# value, so the exchange term gamma (Y_in - Y) vanishes and its mass fraction is
# exactly constant.  Both are excluded from the correction direction (exact
# zeros embedded) and from the feasible interval; only the ACTIVE species move.
#
# This matters because a conservation-compatible direction already forces the
# absent/fixed components to zero, but only to SVD numerical noise (~1e-19).
# Against an exactly zero Y_k that noise becomes an ACTIVE bound a >= 0 through
# (0 - 0)/1e-19, silently making the feasible interval ONE-SIDED and truncating
# every negative correction.  The structural mask replaces the noise with exact
# zeros; that is not clipping a physical species, it is removing a numerical
# artifact of a constraint that holds exactly.
# ---------------------------------------------------------------------------


def net_stoichiometry(gas) -> np.ndarray:
    """nu[k, r] = products minus reactants for species k and reaction r.

    A zero ROW means the species takes part in no reaction.  Cantera returns
    (n_reactions, n_species) for Kinetics objects; both orientations are handled.
    """
    try:
        pr = np.asarray(gas.product_stoich_coeffs, dtype=float)
        re = np.asarray(gas.reactant_stoich_coeffs, dtype=float)
    except AttributeError:
        return np.zeros((int(gas.n_species), 0))
    if pr.shape != re.shape:
        raise ValueError(f"stoichiometry shapes disagree: {pr.shape} {re.shape}")
    if pr.ndim != 2:
        raise ValueError(f"stoichiometry is not 2-D: shape {pr.shape}")
    # orient as (n_species, n_reactions)
    if pr.shape[0] == gas.n_reactions and pr.shape[1] == gas.n_species:
        pr, re = pr.T, re.T
    elif not (pr.shape[0] == gas.n_species
              and pr.shape[1] == gas.n_reactions):
        raise ValueError(f"cannot orient stoichiometry of shape {pr.shape} "
                         f"for {gas.n_species} species / "
                         f"{gas.n_reactions} reactions")
    return pr - re


def active_species_report(cstr, q0, drift_tol: float = 1e-15) -> dict:
    """Classify every species as structurally absent, fixed, or active.

    Requires only the mechanism, the feed and the initial state - no endpoint
    information.  A report, not a mutation: callers embed the resulting zeros.
    """
    gas = cstr.gas
    E = np.asarray(cstr.E, dtype=float)
    Y_in = np.asarray(cstr.Y_in, dtype=float)
    b_in = E @ Y_in
    names = [str(s) for s in gas.species_names]
    n_sp = int(gas.n_species)
    nu = net_stoichiometry(gas)
    inert = [bool(np.all(nu[k, :] == 0.0)) for k in range(n_sp)]

    Y0 = np.asarray(q0, dtype=float).ravel()
    if Y0.size == n_sp + 1:
        Y0 = Y0[1:]                      # q0 = [T, Y...]
    elif Y0.size != n_sp:
        raise ValueError(f"q0 has {Y0.size} entries for {n_sp} species")

    absent, fixed, active, inconsistent = [], [], [], []
    for k in range(n_sp):
        elems = np.flatnonzero(E[:, k] > 0.0)
        # THREE independent rigorous routes to a structurally constant species:
        #  (a) the feed carries none of ANY element the species is built from, so
        #      no admissible trajectory has the material to form it from;
        #  (b) the species takes part in no reaction AND the feed carries none of
        #      it, so both the chemistry and the exchange term vanish exactly;
        #  (c) the species is the UNIQUE carrier of some element e: then E Y = b
        #      forces Y_k = b_e / E[e,k] for EVERY state with that inventory, so
        #      the mass fraction is determined by the inventory alone (zero when
        #      b_e = 0, constant otherwise).  Route (c) uses no stoichiometry.
        no_feed_material = bool(elems.size and np.all(b_in[elems] == 0.0))
        no_feed_species = bool(inert[k] and Y_in[k] == 0.0)
        unique_carrier = any(
            int(np.flatnonzero(E[e, :] > 0.0).size) == 1
            for e in elems)
        if no_feed_material or no_feed_species:
            absent.append(k)
            which = ("no_feed_material" if no_feed_material else
                     "inert_and_absent_from_feed")
            if Y_in[k] != 0.0 or Y0[k] != 0.0:
                inconsistent.append({
                    "index": k, "name": names[k], "kind": which,
                    "Y_in": float(Y_in[k]), "Y0": float(Y0[k])})
            continue
        if unique_carrier or (inert[k] and abs(float(Y_in[k])
                                              - float(Y0[k])) <= drift_tol):
            fixed.append(k)
            continue
        active.append(k)

    mask = np.zeros(n_sp, dtype=bool)
    mask[active] = True
    # for each element, the species that carry it: a singleton element makes its
    # species fixed by inventory conservation alone (a cross-check on the
    # mechanism-based classification above)
    element_carriers = {
        str(gas.element_name(e)): [names[k] for k in range(n_sp)
                                   if E[e, k] > 0.0]
        for e in range(int(gas.n_elements))}
    return {
        "species_names": names,
        "active_indices": active,
        "fixed_indices": fixed,
        "absent_indices": absent,
        "active_mask": mask.tolist(),
        "structurally_zero_mask": np.logical_not(mask).tolist(),
        "n_active": len(active),
        "inert": inert,
        "b_in": b_in.tolist(),
        "element_carriers": element_carriers,
        "inconsistent": inconsistent,
        "note": ("absent species have zero feed inventory for every element "
                 "they contain; fixed species are inert with Y0 == Y_in; the "
                 "remaining active species are what the correction can move"),
    }


def structural_zero_mask(report: dict) -> np.ndarray:
    """Boolean mask, True where the correction direction must be exactly zero."""
    return np.asarray(report["structurally_zero_mask"], dtype=bool)


def conservation_subspace_active(E: np.ndarray, active_mask,
                                 y_interval: float = 1.0,
                                 tol_rel: float = 1e-10,
                                 tol_abs: float = 1e-14) -> dict:
    """An orthonormal basis, in SCALED coordinates, of the ACTIVE-species
    directions preserving every element inventory and the active normalization.

    The absent/fixed species keep exactly their mass fractions, so the active
    ones must sum to 1 - sum(absent and fixed); that constraint is included
    explicitly (it follows from element conservation only when the active
    species carry every element the fixed ones do).
    """
    E = np.asarray(E, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)
    yi = float(y_interval)
    Ea = E[:, mask]
    C = np.vstack([Ea * yi, np.full(int(mask.sum()), yi)])
    out = null_space_basis(C, tol_rel=tol_rel, tol_abs=tol_abs)
    out["y_interval"] = yi
    out["active_count"] = int(mask.sum())
    out["note"] = ("columns are orthonormal in scaled coordinates over the "
                   "ACTIVE species; multiply by y_interval for the physical "
                   "mass-fraction direction and embed exact zeros elsewhere")
    return out


def apply_structural_zeros(n_Y, report: dict) -> np.ndarray:
    """Embed exact zeros for structurally absent and fixed species."""
    n = np.asarray(n_Y, dtype=float).copy()
    n[structural_zero_mask(report)] = 0.0
    return n


def project_onto_active_conservation_subspace(direction, E: np.ndarray,
                                               active_mask,
                                               y_interval: float = 1.0,
                                               tol_rel: float = 1e-10) -> dict:
    """Project a SCALED full-length composition direction onto the
    conservation-compatible subspace of the ACTIVE species, and embed exact
    zeros for the structurally absent and fixed ones.

    Returns the PHYSICAL (unscaled) unit direction n_Y, full length, with
    E n_Y = 0 over all species and 1^T n_Y = 0.
    """
    mask = np.asarray(active_mask, dtype=bool)
    sub = conservation_subspace_active(E, mask, y_interval=y_interval,
                                       tol_rel=tol_rel)
    Q = sub["basis"]
    d = np.asarray(direction, dtype=float).ravel()
    if d.size != mask.size:
        raise ValueError(f"direction has {d.size} entries for {mask.size} "
                         f"species")
    if Q.size == 0:
        return {"n_Y": None, "resolved": False,
                "reason": "active conservation subspace is trivial",
                "subspace": sub}
    proj_a = Q @ (Q.T @ d[mask])
    nrm = float(np.linalg.norm(proj_a))
    d_a = float(np.linalg.norm(d[mask]))
    if nrm <= 1e-14 or nrm < 1e-10 * max(d_a, 1e-300):
        return {"n_Y": None, "resolved": False,
                "reason": (f"projection of the fitted direction onto the active "
                           f"conservation subspace is degenerate "
                           f"(|proj| = {nrm:.3e})"),
                "subspace": sub}
    n_active = float(y_interval) * (proj_a / nrm)
    n_Y = np.zeros(mask.size)
    n_Y[mask] = n_active
    assert np.allclose(E @ n_Y, 0.0, atol=1e-10), "E n_Y must vanish"
    assert abs(n_Y.sum()) < 1e-10, "1^T n_Y must vanish"
    return {"n_Y": n_Y.tolist(), "resolved": True,
            "projection_retained_fraction": nrm / max(d_a, 1e-300),
            "subspace": {"dim": sub["dim"],
                         "conditioning": sub["conditioning"],
                         "y_interval": sub["y_interval"],
                         "active_count": sub["active_count"]}}


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


def feasible_a_interval(Y_B, n_Y, species_names=None,
                        fixed_mask=None, y_floor: float = 0.0) -> dict:
    """The interval of a for which Y_B + a n_Y stays non-negative.

    This is derived from the species bounds THEMSELVES (a feasibility statement
    about the physical decoder), not from the range of a seen in training.

    Components whose direction entry is exactly zero (the structurally absent
    and fixed species, after `apply_structural_zeros`) impose no bound and are
    skipped, with their ORIGINAL indices and names reported.  A component with an
    exactly (or nearly) zero Y_B but a NONZERO direction entry is a structural
    inconsistency - it makes the interval one-sided at 0 through
    (0 - 0)/|n_k| - and is REPORTED, never silently accepted as a bound.
    """
    Y_B = np.asarray(Y_B, dtype=float)
    n_Y = np.asarray(n_Y, dtype=float)
    names = ([str(s) for s in species_names] if species_names is not None
             else [f"species_{k}" for k in range(Y_B.size)])
    fixed = (np.zeros(Y_B.size, dtype=bool) if fixed_mask is None
             else np.asarray(fixed_mask, dtype=bool))
    skipped = {"exact_zero_direction": [], "masked_fixed": []}
    inconsistent = []
    lo, hi, lo_idx, hi_idx = [], [], [], []
    for k, (yk, nk) in enumerate(zip(Y_B, n_Y)):
        if nk == 0.0:
            rec = {"index": int(k), "name": names[k],
                   "Y_B": float(yk), "n_Y": float(nk)}
            (skipped["masked_fixed"] if bool(fixed[k])
             else skipped["exact_zero_direction"]).append(rec)
            continue
        if yk == 0.0 or (abs(yk) < 1e-14 and not bool(fixed[k])):
            inconsistent.append({"index": int(k), "name": names[k],
                                 "Y_B": float(yk), "n_Y": float(nk),
                                 "bound_imposed": (0.0 if nk > 0 else None),
                                 "note": ("a zero species mass fraction with a "
                                           "nonzero direction entry makes the "
                                           "interval one-sided; the structural "
                                           "mask should have zeroed this "
                                           "direction entry")})
        if nk > 0:
            lo.append((y_floor - yk) / nk)
            lo_idx.append(k)
        elif nk < 0:
            hi.append((y_floor - yk) / nk)
            hi_idx.append(k)
    a_lo = max(lo) if lo else None
    a_hi = min(hi) if hi else None
    bounded = a_lo is not None and a_hi is not None and a_lo < a_hi
    if not bounded:
        return {"a_lo": a_lo, "a_hi": a_hi, "bounded": False,
                "reason": ("no admissible interval: the direction drives a "
                           "species negative on both sides of the anchor"),
                "Y_B_min": float(Y_B.min()), "n_Y_argmax": int(np.argmax(n_Y)),
                "binding_species_low": None, "binding_species_high": None,
                "binding_name_low": None, "binding_name_high": None,
                "skipped": skipped, "inconsistent": inconsistent,
                "interval_width": None}
    i_lo, i_hi = int(lo_idx[int(np.argmax(lo))]), int(hi_idx[int(np.argmin(hi))])
    return {"a_lo": float(a_lo), "a_hi": float(a_hi), "bounded": True,
            "binding_species_low": i_lo, "binding_species_high": i_hi,
            "binding_name_low": names[i_lo], "binding_name_high": names[i_hi],
            "Y_B_min": float(Y_B.min()),
            "interval_width": float(a_hi - a_lo),
            "skipped": skipped, "inconsistent": inconsistent}


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

    # ------------------------------------------------------------------
    def sensitivity(self, Y_hat, n_Y, h_J_kg: float) -> dict:
        """The EXACT sensitivity of the decoded state to the scalar correction,
        at fixed (h, p):

            h = sum_k h_k(T) Y_k  is constant along the correction, so
            0 = cp dT/da + sum_k h_k(T) n_k
            dT/da = -sum_k h_k(T) n_k / cp(T, Y)

        and the full state Jacobian column is j = (dT/da, n_Y).  This is the
        linear estimate of the optimal coefficient:

            a_lin = j^T W^T W (q_true - q_B) / (j^T W^T W j)

        evaluated here without the scaling, which the caller applies.
        """
        Y = np.maximum(np.asarray(Y_hat, dtype=float), 0.0)
        n = np.asarray(n_Y, dtype=float)
        gas = self.cstr.gas
        gas.HPY = float(h_J_kg), float(self.cstr.p), Y
        T = float(gas.T)
        Wm = np.asarray(gas.molecular_weights, dtype=float)
        hk = np.asarray(gas.partial_molar_enthalpies, dtype=float) / Wm
        cp = float(gas.cp_mass)
        if not np.isfinite(cp) or cp <= 0.0:
            return {"resolved": False, "reason": f"bad cp = {cp}"}
        dT_over_da = -float(hk @ n) / cp
        return {"resolved": True, "T_K": T, "cp_J_kg_K": cp,
                "h_k_J_kg": hk.tolist(), "dT_over_da": dT_over_da,
                "j": np.concatenate([[dT_over_da], n]).tolist()}


# ---------------------------------------------------------------------------
# 5. regression from control history to reference coordinates and correction
# ---------------------------------------------------------------------------

def first_moment_integral(gammas, durations, t_final: float) -> float:
    """M1 = int_0^t_f (t_f - s) gamma(s) ds for a piecewise-constant history,

        M1 = sum_j gamma_j [ t_f (t_{j+1} - t_j) - (t_{j+1}^2 - t_j^2)/2 ].

    It is the first time-moment of the exposure and is EXACT; no approximation is
    involved in this integral.
    """
    g = np.asarray(gammas, dtype=float)
    d = np.asarray(durations, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(d)])
    tj, tp = edges[:-1], edges[1:]
    return float(np.sum(g * (t_final * (tp - tj) - 0.5 * (tp ** 2 - tj ** 2))))


def m1_coordinate_estimate(gammas, durations, t_final: float,
                           gamma_lo: float | None = None,
                           gamma_hi: float | None = None,
                           t_max: float | None = None) -> dict:
    """The SHORT-TIME expansion baseline for the reference coordinates.

    For a history with exposure Gamma and first moment M1, expanding the
    composition to O(t_f^3) along the exchange direction suggests

        t_B     ~ 2 M1 / Gamma          gamma_B ~ Gamma^2 / (2 M1)

    This is an ASYMPTOTIC MODEL, not an exact constraint: it is reported alongside
    the exposure-constrained regression as an inductive bias of a different
    provenance.  Estimates that leave the allowed domain are RECORDED as out-of-
    domain, never silently clamped.
    """
    g = np.asarray(gammas, dtype=float)
    d = np.asarray(durations, dtype=float)
    Gamma = exposure_integral(g, d)
    M1 = first_moment_integral(g, d, t_final)
    out = {"Gamma": Gamma, "M1": M1, "asymptotic": True}
    if Gamma > 0.0 and M1 > 0.0:
        t_est = 2.0 * M1 / Gamma
        g_est = Gamma ** 2 / (2.0 * M1)
        out["t_B_estimate_s"] = float(t_est)
        out["gamma_B_estimate"] = float(g_est)
        out["gamma_t_product"] = float(g_est * t_est)
        out["in_domain"] = True
        if gamma_lo is not None and g_est < gamma_lo:
            out["in_domain"] = False
            out["out_of_domain_reason"] = f"gamma estimate {g_est:.3e} < {gamma_lo:.3e}"
        if gamma_hi is not None and g_est > gamma_hi:
            out["in_domain"] = False
            out["out_of_domain_reason"] = f"gamma estimate {g_est:.3e} > {gamma_hi:.3e}"
        if t_max is not None and t_est > t_max:
            out["in_domain"] = False
            out["out_of_domain_reason"] = f"t estimate {t_est:.3e} > {t_max:.3e}"
    else:
        out["in_domain"] = False
        out["out_of_domain_reason"] = f"degenerate Gamma={Gamma:.3e} M1={M1:.3e}"
    return out


def exposure_log_feature(eta, durations, T, gamma_ref) -> float:
    """log(Gamma / (gamma_ref T)) with Gamma = sum_j gamma_j dur_j - the exact
    exposure of the history, in units of the anchor exposure.  The closed-form
    balance law makes h and the elemental inventory functions of Gamma ALONE, so
    this feature carries most of the information the reference coordinates need.
    """
    e = np.asarray(eta, dtype=float).ravel()
    d = np.asarray(durations, dtype=float).ravel()
    gammas = gamma_ref * np.exp(e)
    return float(math.log(max(np.sum(gammas * d), 1e-300) / (gamma_ref * T)))


def generator_features(eta, durations, T, gamma_ref, order: int = 1) -> np.ndarray:
    """Feature map for the reference-coordinate regression.

    Order 1: a constant, the exact exposure log feature, and the segment log
    ratios eta.  Order 2 adds squares and pairwise products of those (no cubic
    or higher: the development domain is a small ball and the training set is
    small, so the model must stay well conditioned).

    Every feature is an exact function of the input history; no endpoint
    information enters.
    """
    e = np.asarray(eta, dtype=float).ravel()
    d = np.asarray(durations, dtype=float).ravel()         if durations is not None else np.full(e.size, 1.0)
    base = np.concatenate([[exposure_log_feature(e, d, T, gamma_ref)], e])
    cols = [np.ones(1), base]
    if order >= 2:
        cols.append(base ** 2)
        iu = np.triu_indices(base.size, k=1)
        cols.append(base[iu[0]] * base[iu[1]])
    return np.concatenate(cols)


class LinearRegressor:
    """Affine or quadratic least squares with documented ridge regularization.

    Ridge is on the non-constant coefficients only, and it is FIXED at fit time
    (never tuned on the test set).  The feature function is part of the frozen
    model identity."""

    def __init__(self, feature_fn, order: int = 1, ridge: float = 1e-8):
        self.feature_fn = feature_fn
        self.order = order
        self.ridge = ridge
        self.coef_ = None
        self.feature_names_ = []

    def _design(self, eta):
        return self.feature_fn(eta, self.order)

    def fit(self, X_etas: list[np.ndarray], targets: np.ndarray):
        X = np.stack([self._design(e) for e in X_etas])
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        if targets.shape[0] != X.shape[0]:
            targets = targets.T
        n_feat = X.shape[1]
        reg = np.eye(n_feat)
        reg[0, 0] = 0.0                       # do not penalize the intercept
        A = X.T @ X + self.ridge * reg
        B = X.T @ targets
        self.coef_ = np.linalg.solve(A, B)
        # generic names tied to the ACTUAL design width; the descriptive mapping
        # is recorded by the model identity, not assumed here
        self.feature_names_ = ["const"] + [f"feat_{i}"
                                          for i in range(1, n_feat)]
        pred = X @ self.coef_
        self.train_rms_ = float(np.sqrt(np.mean((pred - targets) ** 2)))
        self.cond_XtX_ = float(np.linalg.cond(X.T @ X + self.ridge * reg))
        self._X_ = X
        return self

    def predict(self, eta: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("the regressor must be fitted before prediction")
        return self._design(eta) @ self.coef_


def _leverage(reg: LinearRegressor, eta: np.ndarray) -> float:
    """A simple uncertainty indicator for the regression: the leverage of the
    input in the fitted design.  Returns NaN if unavailable."""
    try:
        X = getattr(reg, "_X_")
        if X is None:
            return float("nan")
        x = reg._design(eta)
        A = X.T @ X + reg.ridge * np.eye(X.shape[1])
        A[0, 0] -= reg.ridge                      # intercept unpenalized
        return float(x @ np.linalg.pinv(A) @ x)
    except Exception:                                   # noqa: BLE001
        return float("nan")

@dataclass
class LocalGenerator:
    """Canonical curved reference + one conservation-compatible correction.

    Every field needed at inference is frozen at fit time: the reference library,
    the fitted normal direction, the regressors, the decoder and the declared
    training domain."""
    cstr: object
    library: object
    decoder: PhysicalDecoder
    n_Y: list                       # unit species direction, E n_Y = 0, 1^T n_Y = 0
    species_report: dict            # structural absence/fixed/active classification
    beta_regressor: LinearRegressor
    a_regressor: LinearRegressor
    gamma_ref: float
    T_horizon: float
    durations: list
    training_domain: dict
    normal_report: dict
    model_identity: dict
    # a running count of predictions whose RAW coefficient was outside the
    # feasible interval and was projected back in (reported, never hidden)
    n_a_raw_projected: int = 0
    n_a_raw_infeasible_rejected: int = 0

    # ------------------------------------------------------------------
    def reference_state(self, log_gamma: float, t: float) -> dict:
        """The canonical family member, by an independent fresh integration so a
        stale cache can never be used, with the exact balances from beta."""
        gamma = self.gamma_ref * float(np.exp(log_gamma))
        fresh = self.library.evaluate_fresh(gamma, t)
        if not fresh.get("success"):
            return {"gamma": gamma, "t": t, "q_B": None,
                    "h_B_J_kg": None, "b_B": None, "Gamma_B": None,
                    "integration_failure": fresh.get("message")}
        q_B = np.asarray(fresh["q_B"], dtype=float)
        bal = exact_balances_from_controls(self.cstr, self.library.q0,
                                          [gamma], [float(t)])
        return {"gamma": gamma, "t": t, "q_B": q_B.tolist(),
                "h_B_J_kg": bal["h_J_kg"], "b_B": bal["b"],
                "Gamma_B": bal["Gamma"]}

    # ------------------------------------------------------------------
    def predict(self, eta: np.ndarray, durations=None) -> dict:
        """Predict the state from control history coordinates ALONE.

        eta = log(gamma_j / gamma_ref) is the input.  The exposure Gamma and the
        balances are computed from the controls in closed form; the reference
        coordinates are predicted by the frozen regressors in the
        exposure-constrained parametrization

            log gamma_B = log(Gamma/(gamma_ref T)) - v + u
            log t_B     = v                  (t in units of the anchor horizon)

        where u = log(Gamma_B/Gamma) and v = log(t_B/T) are BOTH REGRESSED - u is
        not an enforced exact zero.  At this anchor the closed-form balance law is
        degenerate (the initial state is the HP equilibrium of the same feed, so
        h0 = h_in and b0 = b_in and the invariants cannot identify Gamma), so the
        exposure relation is carried here as an INDUCTIVE BIAS, not as an
        invariant-matching constraint; v is the split between gamma and t that the
        regression must actually learn.
        """
        eta = np.asarray(eta, dtype=float).ravel()
        durs = np.asarray(durations if durations is not None else self.durations,
                          dtype=float)
        gammas = self.gamma_ref * np.exp(eta)
        # exact exposure and balances from the controls
        bal = exact_balances_from_controls(self.cstr, self.library.q0,
                                          gammas, durs)
        log_gamma_exposure = exposure_log_feature(eta, durs, self.T_horizon,
                                                  self.gamma_ref)
        beta = np.asarray(self.beta_regressor.predict(eta), dtype=float).ravel()
        u, v = float(beta[0]), float(beta[1])
        log_gamma_B = log_gamma_exposure - v + u
        t_B = self.T_horizon * float(np.exp(v))
        ref = self.reference_state(log_gamma_B, t_B)
        a_raw = float(np.asarray(self.a_regressor.predict(eta)).ravel()[0])
        if ref.get("q_B") is None:
            return {"input": {"eta": eta.tolist(), "gammas": gammas.tolist(),
                              "durations_s": durs.tolist(),
                              "measured_time_norm":
                                  math.sqrt(float(np.sum(
                                      (durs / self.T_horizon) * eta ** 2))),
                              "Gamma": bal["Gamma"],
                              "exact_balances_from_controls": bal},
                    "model_identity": self.model_identity,
                    "training_domain": self.training_domain,
                    "predicted_reference": {
                        "log_gamma_B": log_gamma_B,
                        "log_t_B_over_T": v, "gamma_B": ref["gamma"],
                        "t_B_s": ref["t"],
                        "exposure_constrained": True, "u": u, "v": v,
                        "log_gamma_exposure": log_gamma_exposure,
                        "integration_failure": ref.get("integration_failure")},
                    "predicted_correction": {"a": None, "a_raw": a_raw,
                                             "feasible_interval": None,
                                             "a_clipped_to_feasible": False},
                    "decoded_state": {"status": "rejected_reference_integration",
                                      "reason": ref.get("integration_failure")},
                    "canonical_state_before_correction": {"q_B": None,
                                                          "status": "failed"},
                    "uncertainty_indicator": {"measured_time_norm":
                                              math.sqrt(float(np.sum(
                                                  (durs / self.T_horizon)
                                                  * eta ** 2)))},
                    "provenance": {}}
        Y_B = np.asarray(ref["q_B"], dtype=float)[1:]
        n_Y = np.asarray(self.n_Y, dtype=float)      # exact structural zeros
        interval = feasible_a_interval(
            Y_B, n_Y, species_names=self.species_report["species_names"],
            fixed_mask=self.species_report["structurally_zero_mask"])
        if interval["bounded"]:
            a = float(np.clip(a_raw, interval["a_lo"], interval["a_hi"]))
            clipped = not math.isclose(a, a_raw, rel_tol=0.0, abs_tol=0.0)
            if clipped:
                self.n_a_raw_projected += 1
        else:
            a, clipped = a_raw, False
            self.n_a_raw_infeasible_rejected += 1
        Y_hat = Y_B + a * n_Y
        dec = self.decoder.decode(Y_hat, ref["h_B_J_kg"], a=a)
        meas = math.sqrt(float(np.sum((durs / self.T_horizon) * eta ** 2)))
        return {
            "input": {"eta": eta.tolist(), "gammas": gammas.tolist(),
                      "durations_s": durs.tolist(),
                      "measured_time_norm": meas, "Gamma": bal["Gamma"],
                      "exact_balances_from_controls": bal},
            "model_identity": self.model_identity,
            "training_domain": self.training_domain,
            "predicted_reference": {
                "log_gamma_B": log_gamma_B, "log_t_B_over_T": v,
                "gamma_B": ref["gamma"], "t_B_s": ref["t"],
                "exposure_constrained": True, "u": u, "v": v,
                "log_gamma_exposure": log_gamma_exposure},
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
                "exposure": "the exact exposure Gamma is a regression feature "
                            "and constrains gamma_B * t_B = Gamma_B",
                "reference_state": "simulated (a fresh integration of the "
                                   "canonical family at the predicted "
                                   "coordinates)",
                "correction": "reconstructed (a scalar along the frozen fitted "
                              "direction)",
                "temperature": "solved by enthalpy inversion"},
            "uncertainty_indicator": {
                "measured_time_norm": meas,
                "inside_declared_domain":
                    bool(meas <= self.training_domain["max_measured_time_norm"]),
                "regression_leverage": _leverage(self.beta_regressor, eta)},
        }


def fit_local_generator(cstr, library, train_cases, durations, gamma_ref,
                        T_horizon, scaling, decoder, order: int = 1,
                        ridge: float = 1e-8,
                        normal_rel_tol: float = 1e-10) -> tuple:
    """Fit the generator on the training cases.

    train_cases: dicts with
        eta                 log control perturbation
        q_target            the true endpoint
        q_B                 the oracle located family member
        located_reference   the oracle reference coordinates (training only)
    The model (complexity, regularization, normal direction, reference-coordinate
    parametrization) is FROZEN here and must not be refit after test data exists.
    """
    etas = [np.asarray(c["eta"], dtype=float) for c in train_cases]
    durs = np.asarray(durations, dtype=float)

    # ---- reference coordinates in the exposure-constrained parametrization ----
    # EMPIRICAL, not invariant-matching (Revision 8): at this anchor the initial
    # state is the HP equilibrium of the same feed, so h0 = h_in and b0 = b_in
    # and the closed-form balance law is degenerate - it cannot identify Gamma.
    # The observed gamma_B * t_B ~ Gamma is nonetheless a strong empirical
    # regularity of the located references (rms 2.7e-3 in log space, two orders
    # below a free fit of the two coordinates), so it is retained as an
    # inductive bias.  The regression learns BOTH the split v = log(t_B/T) and
    # the residual u = log(Gamma_B/Gamma); u is not forced to zero.
    def _targets(c):
        lg = float(c["located_reference"]["log_gamma_B"])
        lt = float(c["located_reference"]["log_t_B_over_T"])
        log_gamma_ex = exposure_log_feature(np.asarray(c["eta"], dtype=float),
                                            durs, T_horizon, gamma_ref)
        u = (lg + lt) - log_gamma_ex          # log(Gamma_B / Gamma), ~ 0
        v = lt                                # log(t_B / T)
        return [u, v]

    Y_beta = np.array([_targets(c) for c in train_cases])
    beta_reg = LinearRegressor(
        lambda e, o: generator_features(e, durs, T_horizon, gamma_ref, o),
        order=order, ridge=ridge).fit(etas, Y_beta)

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
    # ---- the structural face: which species can the correction move? -------
    # A structurally absent species (not in the feed, no reaction forming it) and
    # a fixed inert (in the feed at its initial value) have EXACTLY constant mass
    # fractions.  A conservation-compatible direction already zeroes them, but
    # only to SVD numerical noise; against an exactly zero Y_k that noise becomes
    # an active bound a >= 0 and silently truncates every NEGATIVE correction.
    # Embedding exact zeros restores a two-sided feasible interval.
    species_report = active_species_report(cstr, library.q0)

    U, S, _ = np.linalg.svd(R, full_matrices=False)
    w_Y = U[1:, 0]
    rep = project_onto_active_conservation_subspace(
        w_Y, cstr.E, species_report["active_mask"],
        y_interval=y_interval, tol_rel=normal_rel_tol)
    if not rep["resolved"]:
        raise RuntimeError(f"the normal field is unresolved: {rep['reason']}")
    n_Y = np.asarray(rep["n_Y"], dtype=float)
    assert all(n_Y[k] == 0.0 for k in species_report["absent_indices"]),         "structurally absent species must have exact zeros in the direction"
    assert all(n_Y[k] == 0.0 for k in species_report["fixed_indices"]),         "fixed inerts must have exact zeros in the direction"

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
        a_targets.append(a_o.get("a") if a_o.get("resolved") else 0.0)
    a_reg = LinearRegressor(
        lambda e, o: generator_features(e, durs, T_horizon, gamma_ref, o),
        order=order, ridge=ridge).fit(etas, np.asarray(a_targets, dtype=float))

    model = LocalGenerator(
        cstr=cstr, library=library, decoder=decoder, n_Y=n_Y.tolist(),
        species_report=species_report,
        beta_regressor=beta_reg, a_regressor=a_reg,
        gamma_ref=gamma_ref, T_horizon=T_horizon, durations=list(durs),
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
                        "reference_coordinate_parametrization":
                            "exposure_constrained empirical inductive bias "
                            "(log gamma_B = log Gamma - v + u, log t_B = v; "
                            "u and v are both regressed, u is not forced to "
                            "zero; at this anchor h0 = h_in and b0 = b_in so "
                            "the balance law cannot identify Gamma)",
                        "mechanism": cstr.cfg.mechanism,
                        "gamma_ref": gamma_ref, "T_horizon_s": T_horizon,
                        "n_segments": int(len(durations)),
                        "order": order, "ridge": ridge,
                        "state_scaling": {
                            "T_interval_K": float(1.0 / W[0]),
                            "Y_interval": float(1.0 / W[1]),
                            "n_species": int(len(W) - 1),
                            "note": ("an increment of T_INTERVAL kelvin and "
                                     "Y_INTERVAL in every mass fraction are "
                                     "declared comparably significant; singular "
                                     "values in the scaled space count such "
                                     "increments")},
                        "problem_definition": {
                            "initial_state": "hot HP equilibrium of the feed",
                            "mechanism": cstr.cfg.mechanism,
                            "pressure_Pa": float(cstr.cfg.pressure),
                            "T_in_K": float(cstr.cfg.inlet_temperature),
                            "fuel": cstr.cfg.fuel, "oxidizer": cstr.cfg.oxidizer,
                            "equivalence_ratio": float(cstr.cfg.equivalence_ratio),
                            "horizon_s": float(T_horizon),
                            "durations_s": [float(x) for x in durs],
                            "n_segments": int(len(durations)),
                            "gamma_ref": float(gamma_ref),
                            "control_bounds": {
                                "gamma_lo": float(cstr.cfg.gamma_bounds[0])
                                if hasattr(cstr.cfg, "gamma_bounds") else None,
                                "gamma_hi": float(cstr.cfg.gamma_bounds[1])
                                if hasattr(cstr.cfg, "gamma_bounds") else None},
                            "reactor": "adiabatic fixed-pressure CSTR"},
                        "library_identity": library.identity()})
    return model, {"beta_train_rms": beta_reg.train_rms_,
                   "a_train_rms": a_reg.train_rms_,
                   "cond_XtX": beta_reg.cond_XtX_,
                   "oracle_a": a_targets,
                   "u_rms": float(np.sqrt(np.mean(Y_beta[:, 0] ** 2))),
                   "v_rms": float(np.sqrt(np.mean(Y_beta[:, 1] ** 2))),
                   "normal_report": model.normal_report,
                   "species_report": species_report}


# ---------------------------------------------------------------------------
# 7. serialization: the reloadable frozen model (Revision 8 section D)
# ---------------------------------------------------------------------------

FEATURE_NAME = "generator_features"
"""The single feature map used by the frozen model.  A record carries its full
specification (durations, horizon, gamma_ref, order), so a fresh process can
rebuild the design matrix without refitting or reading any training endpoint."""


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def source_hashes(directory=None) -> dict:
    """Content hashes of the modules that reproduce a prediction, so a reloaded
    model can state whether the code it is running matches the code that fitted
    it.  A mismatch is REPORTED to the caller, never silently accepted."""
    import hashlib
    from pathlib import Path
    base = Path(directory) if directory is not None else Path(
        __file__).resolve().parent
    out = {}
    for name in ("generator.py", "reactor.py", "chemresponse.py",
                 "admissibility.py", "controls.py"):
        fp = base / name
        if fp.is_file():
            out[name] = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
    return out


def model_record(model, *, training_case_ids, validation_case_ids, config,
                 selection) -> dict:
    """The COMPLETE reloadable model: coefficient arrays, feature specification,
    direction, structural classification, metric, decoder policy, problem
    definition and provenance.  A fresh process must be able to reproduce
    predictions from this record alone - without refitting and without reading
any training endpoint."""
    beta = np.asarray(model.beta_regressor.coef_, dtype=float)
    a_coef = np.asarray(model.a_regressor.coef_, dtype=float)
    return {
        "model_kind": "canonical_reference_plus_one_correction",
        "frozen_at_utc": _utc_now(),
        "feature_specification": {
            "feature": FEATURE_NAME,
            "durations_s": [float(x) for x in model.durations],
            "T_horizon_s": float(model.T_horizon),
            "gamma_ref": float(model.gamma_ref),
            "order": int(model.beta_regressor.order),
            "n_features": int(beta.shape[0]),
            "n_beta_targets": int(beta.shape[1]),
            "n_a_targets": int(a_coef.shape[0]),
        },
        "beta_coefficients": beta.tolist(),
        "a_coefficients": a_coef.flatten().tolist(),
        "ridge": float(model.beta_regressor.ridge),
        "n_Y": [float(x) for x in model.n_Y],
        "species_report": model.species_report,
        "state_scaling": model.model_identity.get("state_scaling", {}),
        "decoder_policy": {
            "y_floor": float(model.decoder.y_floor),
            "sum_atol": float(model.decoder.sum_atol),
            "policy": ("invalid compositions are REJECTED and counted, never "
                       "clipped to look admissible; the temperature is solved "
                       "by enthalpy inversion at the reference's own enthalpy")},
        "reference_coordinates": {
            "parametrization": model.model_identity.get(
                "reference_coordinate_parametrization"),
            "targets": ["u = log(Gamma_B / Gamma)", "v = log(t_B / T)"],
            "log_gamma_B_rule": "log gamma_B = log Gamma - v + u",
            "log_t_B_rule": "log t_B = T_horizon * exp(v)"},
        "training_domain": model.training_domain,
        "normal_report": model.normal_report,
        "model_identity": model.model_identity,
        "library_identity": model.model_identity.get("library_identity", {}),
        "problem_definition": model.model_identity.get("problem_definition", {}),
        "selection": selection,
        "config": config,
        "training_case_ids": list(training_case_ids),
        "validation_case_ids": list(validation_case_ids),
        "code_sha256_16": source_hashes(),
        "declaration": ("the model above - complexity, regularization, the "
                        "correction direction and both coefficient arrays - was "
                        "frozen BEFORE any test history of this revision was "
                        "generated; the test set is disjoint from training and "
                        "validation"),
    }


def load_local_generator(record: dict, cstr, library, decoder) -> "LocalGenerator":
    """Rebuild a frozen model from its record.  No refitting takes place and no
    training endpoint is read: the design matrix is rebuilt from the stored
    feature specification and multiplied by the STORED coefficient arrays.

    The mechanism, the initial state and the source code are VERIFIED against the
    record; any mismatch is collected in `reload_warnings` and returned to the
    caller rather than silently ignored.
    """
    warnings = []
    spec = record["feature_specification"]
    if spec["feature"] != FEATURE_NAME:
        raise ValueError(f"unknown feature map in the record: {spec['feature']}")

    def _feature(eta, order):
        return generator_features(eta, np.asarray(spec["durations_s"],
                                                  dtype=float),
                                  float(spec["T_horizon_s"]),
                                  float(spec["gamma_ref"]), int(order))

    beta_reg = LinearRegressor(_feature, order=int(spec["order"]),
                               ridge=float(record["ridge"]))
    beta_reg.coef_ = np.asarray(record["beta_coefficients"], dtype=float)
    a_reg = LinearRegressor(_feature, order=int(spec["order"]),
                            ridge=float(record["ridge"]))
    a_raw = np.asarray(record["a_coefficients"], dtype=float)
    a_reg.coef_ = a_raw.reshape(-1, 1)
    for reg, name, want in ((beta_reg, "beta", spec["n_features"]),
                            (a_reg, "a", spec["n_features"])):
        if int(reg.coef_.shape[0]) != int(want):
            warnings.append(f"{name} coefficient count {reg.coef_.shape[0]} != "
                            f"stored n_features {want}")

    ident = record.get("model_identity", {})
    mech = ident.get("mechanism")
    if mech is not None and mech != cstr.cfg.mechanism:
        warnings.append(f"mechanism mismatch: record {mech} vs loaded "
                        f"{cstr.cfg.mechanism}")
    lib_ident = record.get("library_identity") or {}
    q0_hash = lib_ident.get("q0_sha256_16")
    if q0_hash:
        import hashlib
        got = hashlib.sha256(np.asarray(library.q0, dtype=float)
                             .tobytes()).hexdigest()[:16]
        if got != q0_hash:
            warnings.append(f"initial-state hash mismatch: record {q0_hash} "
                            f"vs loaded {got}")
    stored_hashes = record.get("code_sha256_16") or {}
    if stored_hashes:
        now = source_hashes()
        for name, h in stored_hashes.items():
            if now.get(name) not in (None, h):
                warnings.append(f"code content mismatch in {name}: record {h} "
                                f"vs running {now.get(name)}")

    model = LocalGenerator(
        cstr=cstr, library=library, decoder=decoder,
        n_Y=record["n_Y"], species_report=record["species_report"],
        beta_regressor=beta_reg, a_regressor=a_reg,
        gamma_ref=float(spec["gamma_ref"]),
        T_horizon=float(spec["T_horizon_s"]),
        durations=list(spec["durations_s"]),
        training_domain=record["training_domain"],
        normal_report=record["normal_report"],
        model_identity=ident)
    model.reload_warnings = warnings
    return model


def oracle_correction_coefficient(q_target, Y_B, n_Y, h_J_kg: float,
                                  decoder: PhysicalDecoder, scaling=None,
                                  n_scales: int = 25,
                                  xatol: float = 1e-16) -> dict:
    """The best single scalar a given the TRUE endpoint: a representational
    measurement, not a prediction.

    The feasible interval can be ORDERS OF MAGNITUDE wider than the optimal a
    (the interval is set by the least abundant species, while the optimum is set
    by the transverse excursion), so a uniform grid over the interval is far too
    coarse to resolve it.  The search is therefore MULTI-SCALE: a log-spaced scan
    of both signs of a down to 1e-12 of the interval width, then a bounded golden
    refinement around the best scale.  Evaluated through the physical decoder, so
    the temperature is enthalpy-inverted, never assumed additive.
    """
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

    # the optimum is expected to be small, so probe a wide range of SCALES on
    # both sides of zero, plus zero itself
    width = hi - lo
    scales = np.unique(np.concatenate([
        [0.0],
        np.logspace(-12.0, 0.0, n_scales) * width,
        -np.logspace(-12.0, 0.0, n_scales) * width]))
    # the candidates are offsets from ZERO (both signs), clipped to the interval
    cands = np.unique(np.clip(np.asarray(scales, dtype=float), lo, hi))
    vals = np.array([obj(a) for a in cands])
    finite = np.isfinite(vals)
    if not finite.any():
        return {"a": None, "resolved": False,
                "reason": "no admissible correction on the feasible interval",
                "interval": interval}
    i = int(np.argmin(np.where(finite, vals, np.inf)))
    a_grid, d_grid = float(cands[i]), float(vals[i])
    # bounded refinement between the NEIGHBOURING candidates, so the bracket is
    # at the scale the optimum lives at rather than an arbitrary multiple of it
    b_lo = max(lo, float(cands[max(i - 1, 0)]))
    b_hi = min(hi, float(cands[min(i + 1, cands.size - 1)]))
    if not (b_lo < a_grid < b_hi):
        step = max(abs(a_grid) * 0.1, width * 1e-9)
        b_lo, b_hi = max(lo, a_grid - step), min(hi, a_grid + step)
    r = minimize_scalar(obj, bounds=(b_lo, b_hi), method="bounded",
                        options={"xatol": xatol})
    if r.success and np.isfinite(r.fun) and r.fun <= d_grid:
        a_best, d_best = float(r.x), float(r.fun)
    else:
        a_best, d_best = a_grid, d_grid

    # ---- the LOCAL linear estimate and the active-bound status ------------
    # j = (dT/da, n_Y) at a = 0, with dT/da from enthalpy conservation at fixed
    # (h, p).  a_lin is the minimizer of the linearized objective; comparing it
    # with a_best separates a genuinely curved objective from one that the
    # feasible interval truncated.
    lin = {"a_lin": None, "resolved": False}
    try:
        sens = decoder.sensitivity(Y_B, n_Y, h_J_kg)
        if sens.get("resolved"):
            j = np.asarray(sens["j"], dtype=float)
            q_B = np.concatenate([[float(sens["T_K"])], np.asarray(Y_B,
                                                                  dtype=float)])
            resid = W * (np.asarray(q_target, dtype=float) - q_B)
            jW2 = W * j
            den = float(jW2 @ j)
            lin = {"a_lin": float(jW2 @ resid / den) if den > 0 else None,
                   "resolved": True, "dT_over_da": float(sens["dT_over_da"]),
                   "cp_J_kg_K": float(sens["cp_J_kg_K"]), "denominator": den}
    except Exception as exc:                    # noqa: BLE001
        lin = {"a_lin": None, "resolved": False, "reason": repr(exc)}

    tol_active = max(1e-12, 1e-9 * width)
    active = ("low" if a_best <= lo + tol_active else
              "high" if a_best >= hi - tol_active else None)
    return {"a": a_best, "resolved": True, "dist_scaled_raw": d_best,
            "interval": interval,
            "n_candidates": int(cands.size),
            "at_coarse_grid": d_grid,
            "refinement_improved": bool(d_best < d_grid),
            "linear_estimate": lin,
            "active_bound": active,
            "interior": active is None}
