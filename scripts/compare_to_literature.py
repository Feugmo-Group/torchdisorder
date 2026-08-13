"""Compare a TorchDisorder-refined structure against a published reference model.

Why
---
Agreement with the experimental scattering curve is the thing the refinement
optimises, so it cannot also serve as the test of whether the refinement worked.
This script scores a refined structure on quantities it was *not* fitted to —
coordination, bond lengths, and the intra- and inter-polyhedral bond angles — and
compares them both to a published atomistic model and to experimental values from
the literature.

A refinement that reproduces F(Q) while drifting away from these is overfitting a
1-D projection, which is exactly the failure mode that produced the withdrawn
a-SiO2 result.

Usage
-----
    poetry run python scripts/compare_to_literature.py \
        --test outputs/<run>/final_results/final_structure.cif \
        --reference data/crystal-structures/sio2_glass_gap.cif \
        --system SiO2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Literature values.  Sources are named so a disagreement can be chased down.
SYSTEMS = {
    "SiO2": {
        "central": "Si", "neighbour": "O", "cutoff": 2.2, "cn": 4.0,
        "density": 2.20,
        "bond": (1.608, 1.620),        # Si-O, neutron/X-ray diffraction
        "intra": (109.4, 109.5),       # O-Si-O, ideal tetrahedral
        "inter": (142.0, 147.0),       # Si-O-Si, Neuefeind & Liss; Mozzi & Warren
        "second": {"O-O": 2.63, "Si-Si": 3.08},
    },
    "GeO2": {
        "central": "Ge", "neighbour": "O", "cutoff": 2.4, "cn": 4.0,
        "density": 3.65,
        "bond": (1.730, 1.750),        # Ge-O
        "intra": (109.0, 109.5),       # O-Ge-O
        "inter": (130.0, 135.0),       # Ge-O-Ge, narrower than silica
        "second": {"O-O": 2.83, "Ge-Ge": 3.16},
    },
}


def _angles(atoms, centre_z, nb_z, cutoff):
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    i, j, _d, D = neighbor_list("ijdD", atoms, cutoff)
    m = (z[i] == centre_z) & (z[j] == nb_z)
    ii, DD = i[m], D[m]
    out = []
    for c in np.unique(ii):
        v = DD[ii == c]
        v = v / np.linalg.norm(v, axis=1)[:, None]
        for p in range(len(v)):
            for q in range(p + 1, len(v)):
                out.append(np.degrees(np.arccos(np.clip(v[p] @ v[q], -1.0, 1.0))))
    return np.array(out)


def characterise(path, spec) -> dict:
    from ase.data import atomic_numbers
    from ase.io import read
    from ase.neighborlist import neighbor_list

    atoms = read(str(path))
    zc, zn = atomic_numbers[spec["central"]], atomic_numbers[spec["neighbour"]]
    z = atoms.get_atomic_numbers()
    cutoff = spec["cutoff"]

    i, j, d = neighbor_list("ijd", atoms, cutoff)
    m = (z[i] == zc) & (z[j] == zn)
    cn = np.bincount(i[m], minlength=len(atoms))[z == zc]

    # Second-neighbour peak positions, read off a fine histogram.
    #
    # Search only a window around the expected position.  A global argmax is
    # meaningless here: the pair count grows as r^2, so once a first shell smears
    # out the mode simply migrates to the far edge of the range and reports a
    # "peak" at 5 A that is only the bulk density.  Restricting the window is what
    # one does by eye when reading a PDF, and it keeps the number comparable
    # between a sharp and a degraded structure.
    peaks, shell_resolved = {}, {}
    i2, j2, d2 = neighbor_list("ijd", atoms, 6.0)
    for label, ref in spec["second"].items():
        sa, sb = label.split("-")
        m2 = (z[i2] == atomic_numbers[sa]) & (z[j2] == atomic_numbers[sb])
        if m2.sum():
            hist, edges = np.histogram(d2[m2], bins=300, range=(0.5, 6.0))
            centres = 0.5 * (edges[:-1] + edges[1:])
            win = (centres > ref - 0.7) & (centres < ref + 0.7)
            peaks[label] = float(centres[win][np.argmax(hist[win])])
            # Is it a real peak, or just the shoulder of a featureless ramp?
            # Compare the in-window maximum against the r^2-normalised background.
            g = hist / np.maximum(centres ** 2, 1e-9)
            shell_resolved[label] = bool(g[win].max() > 1.3 * np.median(g[centres > 4.0]))

    intra = _angles(atoms, zc, zn, cutoff)
    inter = _angles(atoms, zn, zc, cutoff)

    return {
        "n_atoms": len(atoms),
        "n_centres": int((z == zc).sum()),
        "density": float(atoms.get_masses().sum() / atoms.get_volume() * 1.66054),
        "cn": float(cn.mean()),
        "cn_exact": float((cn == spec["cn"]).mean()),
        "bond": (float(d[m].mean()), float(d[m].std())) if m.sum() else (np.nan, np.nan),
        "min_bond": float(d[m].min()) if m.sum() else np.nan,
        "intra": (float(intra.mean()), float(intra.std())) if len(intra) else (np.nan, np.nan),
        "inter": (float(inter.mean()), float(inter.std())) if len(inter) else (np.nan, np.nan),
        "peaks": peaks,
        "shell_resolved": shell_resolved,
    }


def _verdict(value, window, tol=0.0):
    lo, hi = window
    return "ok" if (lo - tol) <= value <= (hi + tol) else "OFF"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", required=True, help="structure to assess")
    p.add_argument("--reference", help="published model to compare against")
    p.add_argument("--system", default="SiO2", choices=sorted(SYSTEMS))
    p.add_argument("--label", default=None)
    args = p.parse_args()

    spec = SYSTEMS[args.system]
    c, n = spec["central"], spec["neighbour"]

    entries = [(args.label or Path(args.test).parent.parent.name, characterise(args.test, spec))]
    if args.reference:
        entries.append(("reference (published)", characterise(args.reference, spec)))

    print("=" * 78)
    print(f"{args.system}: refined structure vs published model and experiment")
    print("=" * 78)

    for label, r in entries:
        print(f"\n{label}")
        print(f"  {r['n_atoms']} atoms ({r['n_centres']} {c}), rho = {r['density']:.3f} g/cm3")

    print(f"\n{'quantity':22s}" + "".join(f"{lab[:20]:>22s}" for lab, _ in entries)
          + f"{'literature':>20s}  verdict")
    print("-" * (22 + 22 * len(entries) + 20 + 10))

    test = entries[0][1]

    def row(name, fmt, getter, lit, window=None, tol=0.0):
        cells = "".join(f"{fmt.format(getter(r)):>22s}" for _, r in entries)
        v = _verdict(getter(test), window, tol) if window else ""
        print(f"{name:22s}{cells}{lit:>20s}  {v}")

    row(f"<CN> {c}-{n}", "{:.3f}", lambda r: r["cn"], f"{spec['cn']:.1f}",
        (spec["cn"] - 0.05, spec["cn"] + 0.05))
    row(f"fraction CN={int(spec['cn'])}", "{:.1%}", lambda r: r["cn_exact"], "~100%",
        (0.95, 1.0))
    row(f"{c}-{n} bond (A)", "{:.4f}", lambda r: r["bond"][0],
        f"{spec['bond'][0]:.3f}-{spec['bond'][1]:.3f}", spec["bond"], 0.02)
    row(f"{c}-{n} spread (A)", "{:.4f}", lambda r: r["bond"][1], "small", (0.0, 0.08))
    row(f"min {c}-{n} (A)", "{:.4f}", lambda r: r["min_bond"],
        f">{spec['bond'][0] - 0.15:.2f}", (spec["bond"][0] - 0.15, 99.0))
    row(f"{n}-{c}-{n} (deg)", "{:.2f}", lambda r: r["intra"][0],
        f"{spec['intra'][0]:.1f}-{spec['intra'][1]:.1f}", spec["intra"], 1.5)
    row(f"{n}-{c}-{n} sigma", "{:.2f}", lambda r: r["intra"][1], "3-8 (glass)", (2.0, 12.0))
    row(f"{c}-{n}-{c} (deg)", "{:.2f}", lambda r: r["inter"][0],
        f"{spec['inter'][0]:.0f}-{spec['inter'][1]:.0f}", spec["inter"], 6.0)
    for label, ref in spec["second"].items():
        row(f"{label} peak (A)", "{:.3f}",
            lambda r, l=label: r["peaks"].get(l, float("nan")),
            f"{ref:.2f}", (ref - 0.10, ref + 0.10))
        cells = "".join(f"{('yes' if rr['shell_resolved'].get(label) else 'NO'):>22s}"
                        for _, rr in entries)
        print(f"{'  ' + label + ' shell?':22s}{cells}{'yes':>20s}  "
              f"{'ok' if test['shell_resolved'].get(label) else 'OFF'}")

    print("\nNote: none of these quantities is fitted by the refinement, which targets")
    print("F(Q) only.  They are therefore an independent check, not a restatement of")
    print("the objective.  'OFF' on the angles with 'ok' coordination usually means the")
    print("network topology has been distorted while the first shell was preserved.")


if __name__ == "__main__":
    main()
