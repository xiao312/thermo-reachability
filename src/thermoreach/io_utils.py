"""Run manifests, trajectory persistence, and hashing helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

import numpy as np


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_revision(cwd: Path) -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=15).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                               text=True, timeout=15).stdout.strip()
        return f"{rev}{' (dirty)' if dirty else ''}"
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"<git-unavailable: {exc}>"


def manifest(run_id: str, out_dir: Path, config: dict, *, cwd: Path, extra: dict | None = None) -> dict:
    """Write and return a run manifest. Collects no credentials."""
    m = {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "git_revision": git_revision(cwd),
        "config": config,
    }
    if extra:
        m.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


def save_trajectory(out_path: Path, *, times: np.ndarray, states: np.ndarray, history: dict,
                    scalars: dict | None = None) -> None:
    """Save one trajectory as NPZ plus a small JSON sidecar with the control record.

    ``states`` may be 2-D (n_states, n_times) or 3-D for composition arrays.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"times": np.asarray(times), "states": np.asarray(states)}
    if scalars:
        for k, v in scalars.items():
            payload[k] = np.asarray(v) if isinstance(v, np.ndarray) else v
    np.savez_compressed(out_path, **payload)
    side = out_path.with_suffix(".json")
    side.write_text(json.dumps({"history": history, "npz": out_path.name}, indent=2) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"unserializable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
