"""End-to-end CLI smoke tests (audit A15).

Each advertised script must actually reach result serialization with a tiny
budget - this is exactly the class of error that left the phase3c `regime`
KeyError undetected (A02). Scripts requiring Cantera are skipped when Cantera is
absent; the toy-only scripts run regardless.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (script, extra args, results subdir, expected result files) - tiny budgets
CANTERA_SCRIPTS = [
    ("scripts/phase0_smoke.py", [], "out", ["smoke_results.json"]),
    ("scripts/phase2_reactor_validate.py", ["--horizon", "0.01", "--quick"],
     "out", ["phase2_results.json", "manifest.json"]),
    ("scripts/phase3c_splitting.py",
     ["--horizon", "0.02", "--levels", "8", "16"], "out",
     ["phase3c_results.json", "manifest.json"]),
    ("scripts/phase3d_sensitivity.py", ["--quick"], "out",
     ["phase3d_results.json", "manifest.json"]),
]
TOY_SCRIPTS = [
    ("scripts/phase1_reachability.py",
     ["--horizon", "0.5", "--n-random", "4", "--n-heldout", "2"],
     "phase1_tiny", ["phase1_results.json", "manifest.json"]),
    ("scripts/phase1b_toy_correction.py", ["--n-hist", "6"],
     "phase1b_tiny", ["phase1b_results.json", "manifest.json"]),
]


def _run(script: str, args: list[str], results_dir: Path) -> subprocess.CompletedProcess:
    cmd = [PY, str(ROOT / script), "--results-dir", str(results_dir)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(ROOT), timeout=1200)


@pytest.mark.parametrize("script,args,subdir,expected", CANTERA_SCRIPTS)
def test_cantera_scripts_serialize_results(tmp_path, script, args, subdir, expected):
    pytest.importorskip("cantera")
    res_dir = tmp_path / subdir
    res = _run(script, args, res_dir)
    assert res.returncode == 0, f"{script} failed:\n{res.stderr[-2000:]}"
    for f in expected:
        assert (res_dir / f).is_file(), f"{script} did not write {f}"


@pytest.mark.parametrize("script,args,subdir,expected", TOY_SCRIPTS)
def test_toy_scripts_serialize_results(tmp_path, script, args, subdir, expected):
    res_dir = tmp_path / subdir
    res = _run(script, args, res_dir)
    assert res.returncode == 0, f"{script} failed:\n{res.stderr[-2000:]}"
    for f in expected:
        assert (res_dir / f).is_file(), f"{script} did not write {f}"
