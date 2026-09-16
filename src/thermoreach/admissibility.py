"""Control-history admissibility (audit A07/A1).

Phase 3 declared a control bound gamma in [GAMMA_LO, GAMMA_HI] but included a
``const_zero_seg`` history with gamma=0, which is outside that class. An
inadmissible history silently widens the declared class and can manufacture
results (see the research_log incident where an out-of-bound gamma produced a
false switching excursion).

This module makes the admissible class explicit and validates *every* history
(structured, random, loaded, or optimized) before execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controls import History


@dataclass(frozen=True)
class AdmissibleControls:
    """The declared admissible class of control histories."""

    gamma_lo: float           # lower bound on every gamma (inclusive)
    gamma_hi: float           # upper bound on every gamma (inclusive)
    horizon: float            # total simulated time
    max_segments: int | None = None   # optional cap on the number of segments
    label: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.gamma_lo) or not np.isfinite(self.gamma_hi):
            raise ValueError("gamma bounds must be finite")
        if self.gamma_lo < 0.0:
            raise ValueError("gamma lower bound must be non-negative")
        if self.gamma_hi <= self.gamma_lo:
            raise ValueError("gamma upper bound must exceed the lower bound")
        if not np.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError("horizon must be finite and positive")
        if self.max_segments is not None and self.max_segments < 1:
            raise ValueError("max_segments must be >= 1")

    def validate(self, h: History, strict_horizon: bool = True) -> tuple[bool, str]:
        """Validate one history against this admissible class.

        Returns ``(ok, reason)``. ``reason`` is empty when ok. Non-finite values,
        non-positive durations, and bound violations are rejected.
        """
        vals = np.asarray(h.values, dtype=float)
        durs = np.asarray(h.durations, dtype=float)
        if vals.size == 0:
            return False, "empty control history"
        if not np.all(np.isfinite(vals)):
            return False, f"non-finite gamma: {vals[~np.isfinite(vals)]}"
        if not np.all(np.isfinite(durs)):
            return False, f"non-finite durations: {durs[~np.isfinite(durs)]}"
        if not np.all(durs > 0.0):
            return False, "all durations must be positive"
        lo_bad = vals[vals < self.gamma_lo]
        if lo_bad.size:
            return False, f"gamma below lower bound {self.gamma_lo}: {lo_bad}"
        hi_bad = vals[vals > self.gamma_hi]
        if hi_bad.size:
            return False, f"gamma above upper bound {self.gamma_hi}: {hi_bad}"
        if strict_horizon:
            total = float(durs.sum())
            if abs(total - self.horizon) > 1e-9 * max(1.0, self.horizon):
                return False, (f"total duration {total} differs from declared "
                               f"horizon {self.horizon}")
        if self.max_segments is not None and vals.size > self.max_segments:
            return False, (f"{vals.size} segments exceeds maximum "
                           f"{self.max_segments}")
        return True, ""

    def check_all(self, histories: dict[str, History],
                  strict_horizon: bool = True) -> dict[str, dict]:
        """Validate a labelled collection; returns a per-history report."""
        out = {}
        for label, h in histories.items():
            ok, reason = self.validate(h, strict_horizon=strict_horizon)
            out[label] = {"admissible": bool(ok), "reason": reason}
        return out

    def to_record(self) -> dict:
        return {"gamma_lo": self.gamma_lo, "gamma_hi": self.gamma_hi,
                "horizon": self.horizon, "max_segments": self.max_segments,
                "label": self.label}


def stable_child_seed(root_seed: int, label: str) -> int:
    """Deterministic child seed for a study label.

    Replaces ``abs(hash(label)) % 1000`` (audit A12): Python string hashes are
    salted per interpreter run, so the declared root seed did not reproduce the
    same child seeds across processes. hashlib is stable.
    """
    import hashlib
    digest = hashlib.sha256(f"{root_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)
