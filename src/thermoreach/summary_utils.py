"""Small pure-math summary helpers shared by the phase scripts.

Kept free of any Cantera import so the summary and gain arithmetic can be unit
tested on any machine - an expensive server run must not be wasted on a bug in
the aggregation.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["quantiles", "same_pair_gain"]


def quantiles(xs):
    """Min/quartiles/max/mean over the finite entries, counting the unavailable
    ones as missing rather than filtering failures out of the denominator."""
    xs = np.asarray([x for x in xs if x is not None and math.isfinite(x)],
                    dtype=float)
    if xs.size == 0:
        return {"n": 0}
    q = np.quantile(xs, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"n": int(xs.size), "min": float(q[0]), "q25": float(q[1]),
            "median": float(q[2]), "q75": float(q[3]), "max": float(q[4]),
            "mean": float(xs.mean())}


def same_pair_gain(chem_row) -> dict:
    """The future-state difference at each diagnostic dt relative to the SAME
    pair's difference at dt = 0.

    A shared equilibrium does NOT imply monotone error decay: a difference can
    grow under the chemical map.  Gains above 1 are therefore reported as
    AMPLIFICATION, never as contraction, and a zero dt = 0 difference yields no
    gain (null) rather than a spurious division.
    """
    if not chem_row:
        return {}
    d0 = chem_row.get("state_difference", {})
    denom_T = float(d0.get("dT_K", 0.0))
    dY0 = np.asarray(d0.get("dY", []), dtype=float)
    denom_Y = float(np.max(np.abs(dY0))) if dY0.size else 0.0
    out = []
    for row in chem_row.get("dt_results", []):
        if not row.get("success"):
            out.append({"dt": row.get("dt"), "success": False,
                        "failures": row.get("failures")})
            continue
        num_T = float(row["future_state_dT_K"])
        dY = np.asarray(row.get("future_state_dY", []), dtype=float)
        num_Y = float(np.max(np.abs(dY))) if dY.size else 0.0
        out.append({
            "dt": row["dt"], "success": True,
            "gain_T": (num_T / denom_T) if denom_T != 0.0 else None,
            "gain_Y": (num_Y / denom_Y) if denom_Y != 0.0 else None,
            "amplifying": bool((denom_T != 0.0 and abs(num_T) > abs(denom_T))
                               or (denom_Y != 0.0 and num_Y > denom_Y)),
            "future_dT_K": num_T, "dt0_dT_K": denom_T})
    return {"same_pair_gain_relative_to_dt0": out}
