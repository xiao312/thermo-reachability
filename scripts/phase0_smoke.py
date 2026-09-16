"""Phase 0: environment audit and fresh local execution of the smoke tests.

Reproduces and extends the historical ``thermochemical_compute_check`` package:
  1. environment audit (OS, CPU quota/affinity, memory, disk, package versions,
     Cantera installation and mechanism provenance) -> environment.json
  2. toy A -> B -> C checks: symbolic barrier-derivative identity, exact
     propagator vs Radau/BDF, barrier and positivity under random histories
     -> smoke_results.json
  3. stiff three-variable (Robertson) ODE cross-method consistency.

The historical results.json from the handoff package is preserved unchanged and
is NEVER relabelled as a new run. This script always writes a new file.

Usage:
    python scripts/phase0_smoke.py --results-dir results/phase0
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoreach import toy  # noqa: E402
from thermoreach.io_utils import sha256_file, utc_now, write_json  # noqa: E402


def audit_environment() -> dict:
    res: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "host": platform.node(),
        "visible_logical_cpus": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "packages": {},
        "executables": {s: shutil.which(s) for s in (
            "gcc", "g++", "cmake", "git", "mpirun", "nvidia-smi", "rsync")},
    }
    for field, name in (("cpu_max", "cpu.max"), ("memory_max", "memory.max")):
        f = Path("/sys/fs/cgroup") / name
        res[field] = f.read_text().strip() if f.exists() else None
    if res["cpu_max"] and res["cpu_max"] != "max":
        q, p = res["cpu_max"].split()
        res["cpu_quota_equivalent_cores"] = float(q) / float(p)
    if res["memory_max"] and res["memory_max"] != "max":
        res["memory_limit_GiB"] = int(res["memory_max"]) / 2**30
    for name in ("numpy", "scipy", "sympy", "matplotlib", "pandas", "cantera",
                 "pytest", "CoolProp", "torch", "numba", "cvxpy"):
        try:
            res["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            res["packages"][name] = None
    res["exposed_nvidia_devices"] = [str(p) for p in Path("/dev").glob("nvidia*")]

    # Disk headroom for the working volume
    try:
        stv = os.statvfs("/data2/kexiao/ThermoReachability")
        res["working_volume_free_GiB"] = stv.f_bavail * stv.f_frsize / 2**30
    except OSError:
        res["working_volume_free_GiB"] = None

    # Cantera mechanism provenance
    ct_info: dict = {"available": bool(res["packages"]["cantera"])}
    if res["packages"]["cantera"]:
        import cantera as ct
        ct_info["cantera_version"] = ct.__version__
        ct_info["data_directories"] = list(ct.get_data_directories())
        mech = None
        for d in ct.get_data_directories():
            cand = Path(d) / "h2o2.yaml"
            if cand.exists():
                mech = cand
                break
        ct_info["h2o2_yaml"] = {
            "path": str(mech) if mech is not None else None,
            "sha256": sha256_file(mech) if mech is not None else None,
        }
        if mech is not None:
            gas = ct.Solution("h2o2.yaml")
            ct_info["h2o2_yaml"]["species"] = list(gas.species_names)
            ct_info["h2o2_yaml"]["n_reactions"] = gas.n_reactions
            ct_info["h2o2_yaml"]["elements"] = list(gas.element_names)
            ct_info["h2o2_yaml"]["thermo_valid_range_K"] = None  # filled below if available
            try:
                lo = min(sp.thermo.min_temp for sp in gas.species())
                hi = max(sp.thermo.max_temp for sp in gas.species())
                ct_info["h2o2_yaml"]["thermo_valid_range_K"] = [float(lo), float(hi)]
            except Exception:
                pass
    res["cantera"] = ct_info
    return res


def _yaml_phase_names(path: Path) -> dict:
    try:
        import yaml  # PyYAML is a Cantera dependency
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        return {"phases": [p.get("name") for p in doc.get("phases", [])]}
    except Exception:
        return {"phases": []}


# ---------------------------------------------------------------------------
# Toy checks (fresh local execution; equations identical to the handoff package)
# ---------------------------------------------------------------------------


def toy_checks() -> dict:
    x, y = sp.symbols("x y", positive=True)
    gamma = sp.symbols("gamma", nonnegative=True)
    V = y + x * sp.log(x)
    dx = -x + gamma * (1 - x)
    dy = x - y - gamma * y
    boundary_derivative = sp.simplify(
        (sp.diff(V, x) * dx + sp.diff(V, y) * dy).subs(y, -x * sp.log(x)))
    identity_ok = sp.simplify(boundary_derivative - gamma * (sp.log(x) + 1 - x)) == 0

    # Additional audit: the steady-state locus y = x(1-x).
    g = sp.symbols("gamma", positive=True)
    x_star, y_star = g / (1 + g), g / (1 + g) ** 2
    locus_ok = sp.simplify(sp.cancel(y_star - x_star * (1 - x_star))) == 0
    # And the batch trajectory y = -x log x solves the gamma = 0 system.
    t = sp.symbols("t", nonnegative=True)
    xb = sp.exp(-t)
    yb = -xb * sp.log(xb)
    batch_ok = sp.simplify(sp.diff(xb, t) - (-xb)) == 0 and sp.simplify(
        sp.diff(yb, t) - (xb - yb)) == 0

    def exact(q0, rate, t):
        a = 1.0 + rate
        xs, ys = rate / a, rate / a**2
        e = np.exp(-a * np.asarray(t))
        return np.array([xs + (q0[0] - xs) * e,
                         ys + ((q0[1] - ys) + (q0[0] - xs) * np.asarray(t)) * e])

    start = time.perf_counter()
    batch_times = np.linspace(0, 12, 401)
    batch = solve_ivp(lambda t, q: [-q[0], q[0] - q[1]], (0, 12), [1.0, 0.0],
                      method="Radau", t_eval=batch_times, rtol=1e-10, atol=1e-12)
    assert batch.success, batch.message
    batch_err = float(np.max(np.abs(batch.y - exact([1.0, 0.0], 0.0, batch_times))))
    bdf = solve_ivp(lambda t, q: [-q[0], q[0] - q[1]], (0, 12), [1.0, 0.0],
                    method="BDF", t_eval=batch_times, rtol=1e-10, atol=1e-12)
    assert bdf.success, bdf.message
    batch_err_bdf = float(np.max(np.abs(bdf.y - exact([1.0, 0.0], 0.0, batch_times))))

    rng = np.random.default_rng(20260915)
    n_histories, n_segments = 32, 8
    max_err = 0.0
    max_barrier_violation = 0.0
    min_fraction = 0.0
    tested_states = 0
    total_nfev = 0
    for _ in range(n_histories):
        q = np.array([1.0, 0.0])
        for _ in range(n_segments):
            rate = 0.0 if rng.random() < 0.25 else float(10 ** rng.uniform(-3, 2))
            duration = float(rng.uniform(0.02, 0.5))
            # Dense within-segment sampling (not only endpoints)
            times = np.linspace(0.0, duration, 9)
            initial = q.copy()
            sol = solve_ivp(lambda t, q: [-q[0] + rate * (1 - q[0]), q[0] - (1 + rate) * q[1]],
                            (0, duration), initial, method="Radau", t_eval=times,
                            jac=np.array([[-1 - rate, 0.0], [1.0, -1 - rate]]),
                            rtol=1e-9, atol=1e-12)
            assert sol.success, sol.message
            ref = exact(initial, rate, times)
            max_err = max(max_err, float(np.max(np.abs(sol.y - ref))))
            xx, yy = sol.y
            barrier_vals = toy.barrier(np.vstack([xx, yy]))
            max_barrier_violation = max(max_barrier_violation, float(np.max(barrier_vals)))
            min_fraction = min(min_fraction, float(np.min(np.array([xx, yy, 1 - xx - yy]))))
            tested_states += sol.t.size
            total_nfev += sol.nfev
            q = sol.y[:, -1]

    out = {
        "model": "isothermal A -> B -> C, k1=k2=1, pure-A feed",
        "barrier": "V(x,y)=y+x*log(x) <= 0",
        "symbolic_boundary_derivative": str(boundary_derivative),
        "symbolic_identity_verified": bool(identity_ok),
        "steady_locus_y_equals_x1_minus_x_verified": bool(locus_ok),
        "batch_trajectory_y_equals_minus_x_log_x_verified": bool(batch_ok),
        "sign_argument": "gamma >= 0 and log(x)+1-x <= 0 for 0<x<=1 (log(x) <= x-1)",
        "batch_max_abs_error_radau_vs_exact": batch_err,
        "batch_max_abs_error_bdf_vs_exact": batch_err_bdf,
        "random_seed": 20260915,
        "controlled_histories": n_histories,
        "segments_per_history": n_segments,
        "sampled_states_including_repeated_segment_endpoints": tested_states,
        "gamma_range_sampled": [0, 100],
        "segment_duration_range": [0.02, 0.5],
        "controlled_radau_max_abs_error_against_exact_segment_solution": max_err,
        "maximum_positive_barrier_violation": max_barrier_violation,
        "minimum_species_fraction_or_zero": min_fraction,
        "total_controlled_radau_rhs_evaluations": total_nfev,
        "wall_seconds": time.perf_counter() - start,
        "scope": ("Tests an upper invariant bound and the exact propagator. Random "
                  "histories do not establish complete reachability or attainability "
                  "of every point in the bound."),
        "passed": bool(max_err < 1e-7 and max_barrier_violation < 1e-8 and min_fraction > -1e-8),
    }
    return out


def stiff_checks() -> dict:
    def rhs(t, q):
        a, b, c = q
        v1, v2, v3 = 0.04 * a, 1.0e4 * b * c, 3.0e7 * b * b
        return [-v1 + v2, v1 - v2 - v3, v3]

    def jac(t, q):
        a, b, c = q
        return np.array([
            [-0.04, 1.0e4 * c, 1.0e4 * b],
            [0.04, -1.0e4 * c - 6.0e7 * b, -1.0e4 * b],
            [0.0, 6.0e7 * b, 0.0],
        ])

    times = np.r_[0.0, np.geomspace(1e-8, 1e5, 400)]
    runs, solutions = {}, {}
    specs = (
        ("Radau", "Radau", 1e-9, [1e-12, 1e-16, 1e-12]),
        ("BDF", "BDF", 1e-9, [1e-12, 1e-16, 1e-12]),
        ("Radau_tighter_reference", "Radau", 1e-11, [1e-14, 1e-18, 1e-14]),
    )
    for name, method, rtol, atol in specs:
        start = time.perf_counter()
        sol = solve_ivp(rhs, (0.0, 1e5), [1.0, 0.0, 0.0], method=method, jac=jac,
                        rtol=rtol, atol=atol, t_eval=times)
        elapsed = time.perf_counter() - start
        assert sol.success, sol.message
        cons = float(np.max(np.abs(sol.y.sum(axis=0) - 1)))
        minimum = float(sol.y.min())
        solutions[name] = sol.y
        runs[name] = {
            "success": bool(sol.success), "method": method, "rtol": rtol, "atol": atol,
            "wall_seconds": elapsed, "nfev": sol.nfev, "njev": sol.njev, "nlu": sol.nlu,
            "maximum_conservation_error": cons, "minimum_concentration": minimum,
            "final_state": sol.y[:, -1].tolist(),
        }
    ref = solutions["Radau_tighter_reference"]
    for name in ("Radau", "BDF"):
        err = float(np.max(np.abs(solutions[name] - ref)))
        runs[name]["maximum_absolute_difference_from_tighter_reference"] = err
    difference = float(np.max(np.abs(solutions["Radau"] - solutions["BDF"])))
    return {
        "model": "three-variable stiff reaction system (Robertson); not detailed combustion",
        "integration_time_interval": [0, 1e5],
        "common_output_time_count": len(times),
        "runs": runs,
        "maximum_absolute_radau_bdf_difference": difference,
        "interpretation": ("Cross-method and tighter-tolerance agreement are numerical "
                           "checks, not a rigorous interval enclosure."),
        "passed": bool(difference < 1e-6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/phase0"))
    args = ap.parse_args()

    out: dict = {"timestamp_utc": utc_now(), "script": str(Path(__file__))}
    out["environment"] = audit_environment()
    write_json(args.results_dir / "environment.json", out["environment"])

    out["cstr"] = toy_checks()
    out["stiff_ode"] = stiff_checks()
    out["all_tests_passed"] = bool(out["cstr"]["passed"] and out["stiff_ode"]["passed"])
    write_json(args.results_dir / "smoke_results.json", out)

    print(f"[phase0] all_tests_passed = {out['all_tests_passed']}")
    print(f"[phase0] toy: exact-propagator err {out['cstr']['controlled_radau_max_abs_error_against_exact_segment_solution']:.2e}, "
          f"barrier violation {out['cstr']['maximum_positive_barrier_violation']:.2e}")
    print(f"[phase0] stiff: Radau-BDF max diff {out['stiff_ode']['maximum_absolute_radau_bdf_difference']:.2e}")
    if not out["all_tests_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
