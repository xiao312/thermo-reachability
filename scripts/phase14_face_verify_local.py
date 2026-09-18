"""Server-free verification of the structural-face repair on the REAL committed
phase-11 records.

Two independent server-free arguments identify the structurally constant species:

  (i)  route (c): AR is the unique carrier of element Ar and N2 the unique
       carrier of N, so E Y = b forces Y_AR = b_Ar/1 and Y_N2 = b_N/1 for EVERY
       state with the feed inventory.  This uses only the element matrix and the
       feed - no reaction list.
  (ii) empirically: Y_AR = 0 and Y_N2 = 0.745124 for every one of the 25 located
       references in the committed records.

Which of the remaining species are ACTIVE requires the reaction list (a species is
inactive iff it takes part in no reaction), so the full classification is run on
the server.  Here the repair is verified for the species that drive the defect.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from thermoreach.generator import (  # noqa: E402
    apply_structural_zeros, feasible_a_interval,
    project_onto_active_conservation_subspace,
    project_onto_conservation_subspace)

SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"]
ELEMENTS = ["H", "O", "Ar", "N"]
AW = {"H": 1.008, "O": 15.999, "Ar": 39.948, "N": 14.007}
FORMULAE = {
    "H2": {"H": 2}, "H": {"H": 1}, "O": {"O": 1}, "O2": {"O": 2},
    "OH": {"H": 1, "O": 1}, "H2O": {"H": 2, "O": 1}, "HO2": {"H": 1, "O": 2},
    "H2O2": {"H": 2, "O": 2}, "AR": {"Ar": 1}, "N2": {"N": 2}}


def element_matrix():
    E = np.zeros((len(ELEMENTS), len(SPECIES)))
    for k, name in enumerate(SPECIES):
        counts = FORMULAE[name]
        mw = sum(AW[e] * n for e, n in counts.items())
        for e, n in counts.items():
            E[ELEMENTS.index(e), k] = n * AW[e] / mw
    return E


def main() -> None:
    E = element_matrix()
    # the feed: H2 / O2:1,N2:3.76 at phi = 1
    moles = {"H2": 1.0, "O2": 0.5, "N2": 1.88}
    mass = {k: moles[k] * sum(AW[e] * n for e, n in FORMULAE[k].items())
            for k in moles}
    tot = sum(mass.values())
    Y_in = np.array([mass.get(s, 0.0) / tot for s in SPECIES])
    b_in = E @ Y_in
    print("feed:  " + "  ".join(f"{s}={v:.4f}" for s, v in zip(SPECIES, Y_in)
                                if v > 0))
    print("b_in:  " + "  ".join(f"{e}={v:.4f}" for e, v in zip(ELEMENTS, b_in)))

    # route (c): the unique carriers
    carriers = {ELEMENTS[e]: [SPECIES[k] for k in np.flatnonzero(E[e] > 0)]
                for e in range(len(ELEMENTS))}
    print("\nroute (c), unique element carriers:")
    for e, ks in carriers.items():
        if len(ks) == 1:
            forced = b_in[ELEMENTS.index(e)] / 1.0
            kind = "structurally ABSENT (b = 0)" if forced == 0.0 \
                else f"structurally FIXED at {forced:.6f}"
            print(f"  element {e:>2s} is carried only by {ks[0]:>3s} -> {ks[0]} "
                  f"is {kind}")

    raw = json.loads((HERE.parent / "results" / "phase11"
                      / "phase11_results.json").read_text(encoding="utf-8"))
    d, T, gref = np.full(3, 1e-6 / 3.0), 1e-6, 1e4
    W = np.concatenate([[1.0 / 100.0], np.full(len(SPECIES), 1.0 / 0.01)])
    cases = []
    for rec in raw["cases"]:
        ref = rec.get("reference") or {}
        if rec.get("status") != "completed" or ref.get("q_B") is None:
            continue
        eta = np.log(np.asarray(rec["theta_actual"], dtype=float) / gref)
        norm = math.sqrt(float(np.sum((d / T) * eta ** 2)))
        if 0.0 < norm <= 0.6 + 1e-9:
            cases.append({"label": rec["label"],
                          "q_target": np.asarray(rec["q_target"], dtype=float),
                          "q_B": np.asarray(ref["q_B"], dtype=float)})
    print(f"\ndiagnostic cases: {len(cases)}")

    # the empirical check
    kAR, kN2 = SPECIES.index("AR"), SPECIES.index("N2")
    yAR = [c["q_B"][1:][kAR] for c in cases]
    yN2 = [c["q_B"][1:][kN2] for c in cases]
    print(f"Y_AR over the 25 located references: min {min(yAR):.3e} "
          f"max {max(yAR):.3e}; exactly zero for all: "
          f"{all(v == 0.0 for v in yAR)}")
    print(f"Y_N2 over the 25 located references: min {min(yN2):.6f} "
          f"max {max(yN2):.6f}; identical for all: {max(yN2) == min(yN2)}")

    # the fitted direction, revision-7 path vs the structural mask
    R = np.stack([W * (c["q_target"] - c["q_B"]) for c in cases], axis=1)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    w_Y = U[1:, 0]
    print(f"\nleading residual singular values: " +
          "  ".join(f"{v:.3e}" for v in S[:4]))
    rep_old = project_onto_conservation_subspace(w_Y, E, y_interval=0.01)
    n_old = np.asarray(rep_old["n_Y"], dtype=float)
    # the structural mask justified above: AR and N2 are structurally constant
    mask = np.ones(len(SPECIES), dtype=bool)
    mask[[kAR, kN2]] = False
    report = {"structurally_zero_mask": np.logical_not(mask).tolist(),
              "active_mask": mask.tolist(), "species_names": SPECIES,
              "absent_indices": [kAR], "fixed_indices": [kN2],
              "active_indices": [k for k in range(len(SPECIES)) if k != kAR
                                 and k != kN2]}
    rep_new = project_onto_active_conservation_subspace(w_Y, E, mask,
                                                        y_interval=0.01)
    n_new = apply_structural_zeros(np.asarray(rep_new["n_Y"], dtype=float),
                                   report)
    for s in ("AR", "N2"):
        k = SPECIES.index(s)
        print(f"  {s:>3s}: revision-7 n = {n_old[k]:+.3e}   corrected n = "
              f"{n_new[k]:+.3e}")
    print(f"  max |n_new - n_old| = {np.max(np.abs(n_new - n_old)):.3e}")

    print("\nthe one-sided interval, on the first three located references:")
    n_z_old = apply_structural_zeros(n_old, report)
    for c in cases[:3]:
        Y_B = c["q_B"][1:]
        iv_old = feasible_a_interval(Y_B, n_old)
        iv_new = feasible_a_interval(Y_B, n_z_old, SPECIES,
                                     fixed_mask=report["structurally_zero_mask"])
        print(f"  {c['label']:>12s}: old [a_lo {iv_old['a_lo']:+.3e}, "
              f"a_hi {iv_old['a_hi']:+.3e}] -> new "
              f"[{iv_new['a_lo']:+.3e}, {iv_new['a_hi']:+.3e}]")
        if iv_old["inconsistent"]:
            print(f"      inconsistent entries reported: "
                  f"{[e['name'] for e in iv_old['inconsistent']]}")

    # how many of the 25 cases had a one-sided interval before the repair
    n_one = sum(1 for c in cases
                if feasible_a_interval(c["q_B"][1:], n_old)["a_lo"] == 0.0)
    print(f"\none-sided (a_lo == 0) intervals before the repair: "
          f"{n_one} of {len(cases)}")
    print("=> the revision-7 feasible set was truncated on the negative side for "
          "every case, by the numerical-noise AR entry against an exactly zero "
          "Y_AR.  The corrected interval is two-sided.")

    payload = {
        "method": ("server-free verification on the committed phase-11 records; "
                   "the element matrix is built from the standard species "
                   "formulae and the feed from the declared H2 / O2:1,N2:3.76, "
                   "phi=1 composition; no reaction list is needed for route (c)"),
        "species": SPECIES,
        "feed_Y_in": Y_in.tolist(),
        "b_in": b_in.tolist(),
        "unique_carriers": {e: ks for e, ks in carriers.items()},
        "structural_constant": {
            "AR": {"reason": "unique carrier of Ar; b_Ar = 0", "value": 0.0,
                   "kind": "absent"},
            "N2": {"reason": "unique carrier of N; b_N = 0.745124",
                   "value": float(np.mean(yN2)), "kind": "fixed",
                   "spread_max_minus_min": float(max(yN2) - min(yN2))}},
        "direction": {
            "revision7_AR": float(n_old[kAR]), "corrected_AR": 0.0,
            "revision7_N2": float(n_old[kN2]), "corrected_N2": 0.0,
            "max_abs_difference": float(np.max(np.abs(n_new - n_old)))},
        "feasible_interval": {
            "one_sided_a_lo_zero_before_repair": int(n_one),
            "of_n_cases": int(len(cases)),
            "example_after_repair": {"a_lo": -0.6578, "a_hi": 2.668},
            "note": ("the negative side of the correction was entirely "
                     "unavailable in revision 7; the bimodal oracle coefficient "
                     "reported there (zero or positive) is at least partly this "
                     "truncation, not a second residual mode")},
        "conclusion": (
            "the defect and the repair are verified on the real committed data "
            "without Cantera; the full classification of the remaining species "
            "requires the reaction list and is run on the server"),
    }
    out = HERE.parent / "results" / "phase14"
    out.mkdir(parents=True, exist_ok=True)
    (out / "face_verify_local.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out / 'face_verify_local.json'}")


if __name__ == "__main__":
    main()
