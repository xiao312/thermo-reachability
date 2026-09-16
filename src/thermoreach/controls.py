"""Admissible control-history classes for the bounded-control reachability study.

Every history is a piecewise-constant function gamma: [0, t_f] -> [0, G] with a
declared number of equal-duration segments. This is a strict subset of the
measurable admissible class U = {gamma : 0 <= gamma(t) <= G}; enlarging the
segment count approaches that class but never equals it for finite resolution.

Histories are recorded exactly (segment values + durations) so every trajectory
is replayable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class History:
    """A piecewise-constant control history.

    values[i] is the control applied on [t[i], t[i]+durations[i]).
    """

    values: np.ndarray
    durations: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        d = np.asarray(self.durations, dtype=float)
        if v.shape != d.shape or v.ndim != 1 or v.size == 0:
            raise ValueError("values and durations must be nonempty 1-D arrays of equal length")
        if np.any(v < 0.0):
            raise ValueError("controls must be nonnegative")
        if np.any(d <= 0.0):
            raise ValueError("durations must be positive")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "durations", d)

    @property
    def horizon(self) -> float:
        return float(self.durations.sum())

    @property
    def n_segments(self) -> int:
        return int(self.values.size)

    def bounds_check(self, gamma_max: float, tol: float = 0.0) -> bool:
        return bool(np.max(self.values) <= gamma_max + tol)

    def to_record(self) -> dict:
        return {
            "values": np.asarray(self.values).tolist(),
            "durations": np.asarray(self.durations).tolist(),
            "horizon": self.horizon,
            "n_segments": self.n_segments,
        }


def equal_durations(n_segments: int, horizon: float) -> np.ndarray:
    return np.full(n_segments, horizon / n_segments, dtype=float)


def constant(gamma: float, n_segments: int, horizon: float) -> History:
    """Constant-control baseline."""
    return History(np.full(n_segments, gamma, dtype=float), equal_durations(n_segments, horizon))


def bang_bang(low: float, high: float, n_segments: int, horizon: float, start_high: bool = True) -> History:
    """Alternating bang-bang history between two levels."""
    idx = np.arange(n_segments)
    mask = (idx % 2 == 0) if start_high else (idx % 2 == 1)
    vals = np.where(mask, high, low)
    return History(vals.astype(float), equal_durations(n_segments, horizon))


def pulse(level: float, n_segments: int, horizon: float, pulse_segment: int = 0, base: float = 0.0) -> History:
    """Isolated single-segment pulse of ``level`` on an otherwise ``base`` history."""
    if not 0 <= pulse_segment < n_segments:
        raise ValueError("pulse_segment out of range")
    vals = np.full(n_segments, float(base), dtype=float)
    vals[pulse_segment] = float(level)
    return History(vals, equal_durations(n_segments, horizon))


def ramp(low: float, high: float, n_segments: int, horizon: float, increasing: bool = True) -> History:
    """Structured slow-to-fast (or fast-to-slow) ramp of the control level."""
    frac = (np.arange(n_segments) + 0.5) / n_segments
    if not increasing:
        frac = 1.0 - frac
    vals = low + (high - low) * frac
    return History(vals, equal_durations(n_segments, horizon))


def random_history(rng: np.random.Generator, gamma_max: float, n_segments: int, horizon: float,
                   zero_prob: float = 0.25, log_low: float = -3.0, log_high: float = 0.0) -> History:
    """Randomized history with a declared law.

    Law: each segment is 0 with probability ``zero_prob``, else
    gamma_max * 10^U(log_low, log_high). With the default exponents this is
    log-uniform in [gamma_max*10^-3, gamma_max] and strictly respects the
    admissible bound gamma <= gamma_max.
    """
    u = rng.random(n_segments)
    mag = gamma_max * np.power(10.0, rng.uniform(log_low, log_high, n_segments))
    vals = np.where(u < zero_prob, 0.0, mag)
    return History(vals.astype(float), equal_durations(n_segments, horizon))


def random_dwell(rng: np.random.Generator, gamma_max: float, n_levels: int, horizon: float,
                 levels: np.ndarray | None = None) -> History:
    """Random dwell history: random levels with random (unequal) dwell durations.

    Durations are drawn from Dirichlet(1,...,1) * horizon so the total horizon is
    exact; levels are log-uniform as in ``random_history``.
    """
    if levels is None:
        u = rng.random(n_levels)
        levels = np.where(u < 0.25, 0.0,
                          gamma_max * np.power(10.0, rng.uniform(-3.0, 0.0, n_levels)))
    dur = rng.dirichlet(np.ones(n_levels)) * horizon
    return History(np.asarray(levels, dtype=float), dur)


def from_record(rec: dict) -> History:
    return History(np.asarray(rec["values"], dtype=float), np.asarray(rec["durations"], dtype=float))
