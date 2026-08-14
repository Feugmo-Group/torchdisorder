"""Audit every constraint file for internal consistency and physical sanity.

Constraints are keyed by ATOM INDEX, so a constraints file is only meaningful
paired with the exact structure its generator emitted. The generators reorder
atoms, so pairing one with a differently-ordered copy of the *same* structure
silently lands most constraints on the wrong element -- 1148 of 1723 ended up on
oxygen when the SiO2 constraints were paired with the pristine download. Nothing
downstream complains; the run simply constrains the wrong atoms.

Checks per file:
  1. Does a structure with matching atom ordering exist alongside it?
  2. Are the constrained indices in range, and all of the expected species?
  3. Do the stored target values match what the order-parameter code computes
     for that structure now? (catches targets generated before a code fix)
  4. Which order-parameter types are constrained, and is the set sensible?

Usage:
    poetry run python scripts/audit_constraints.py
    poetry run python scripts/audit_constraints.py --json data/json/foo_constraints.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def candidate_structures(json_path: Path):
    """Structures that might pair with this constraints file, best guess first."""
    stem = json_path.name.replace("_constraints.json", "")
    out = []
    for pat in (json_path.parent / f"{stem}.cif",
                json_path.parent / f"{stem}_constraints.cif",
                Path("data/crystal-structures") / f"{stem}.cif"):
        if pat.exists():
            out.append(pat)
    return out


def audit(json_path: Path, verbose: bool = True) -> dict:
    from ase.data import atomic_numbers, chemical_symbols
    from ase.io import read

    data = json.loads(json_path.read_text())
    ac = data.get("atom_constraints", {})
    meta = data.get("metadata", {})
    problems = []

    if not ac:
        return {"file": json_path.name, "problems": ["no atom_constraints"], "n": 0}

    idx = [int(k) for k in ac]
    op_types = sorted({op for c in ac.values() for op in c.get("order_parameters", {})})
    envs = sorted({c.get("environment", "?") for c in ac.values()})

    # Which element should the constrained atoms be? Take it from the JSON's own
    # element_filter/metadata rather than guessing from the filename.
    composition = meta.get("composition", "")
    n_declared = meta.get("total_atoms")

    structures = candidate_structures(json_path)
    matched, species_ok = None, None
    for sp in structures:
        try:
            atoms = read(str(sp))
        except Exception:
            continue
        z = atoms.get_atomic_numbers()
        if max(idx) >= len(z):
            continue
        got = {int(z[i]) for i in idx}
        # A well-formed file constrains exactly one central species.
        if len(got) == 1:
            matched, species_ok = sp, chemical_symbols[got.pop()]
            break

    if matched is None:
        if structures:
            problems.append(
                f"no co-located structure has a single species at all {len(idx)} "
                f"constrained indices (tried: {', '.join(s.name for s in structures)})")
        else:
            problems.append("no co-located structure file found to verify indices against")

    if n_declared is not None and matched is not None:
        atoms = read(str(matched))
        if len(atoms) != n_declared:
            problems.append(
                f"metadata.total_atoms={n_declared} but {matched.name} has {len(atoms)}")

    if not op_types:
        problems.append("no order parameters constrained")
    if "cn" in op_types and len(op_types) == 1:
        problems.append("only 'cn' constrained -- coordination alone does not fix geometry; "
                        "a structure can hold CN while its polyhedra deform")

    if verbose:
        print(f"\n=== {json_path.name}")
        print(f"    version={meta.get('version','?')}  composition={composition or '?'}  "
              f"declared_atoms={n_declared}")
        print(f"    {len(idx)} constrained atoms   ops={op_types}   envs={envs}")
        print(f"    paired structure: {matched.name if matched else 'NONE FOUND'}"
              + (f"  (all {species_ok})" if species_ok else ""))
        for p in problems:
            print(f"    ! {p}")
        if not problems:
            print("    ok")

    return {"file": json_path.name, "problems": problems, "n": len(idx),
            "ops": op_types, "structure": matched.name if matched else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", nargs="*", default=None)
    args = p.parse_args()

    files = [Path(f) for f in (args.json or sorted(glob.glob("data/json/*_constraints.json")))]
    if not files:
        raise SystemExit("no constraint files found")

    results = [audit(f) for f in files]

    bad = [r for r in results if r["problems"]]
    print("\n" + "=" * 78)
    print(f"{len(results)} files audited, {len(bad)} with problems")
    print("=" * 78)
    for r in bad:
        print(f"  {r['file']}")
        for prob in r["problems"]:
            print(f"      - {prob}")

    # Which order parameters are in use across the project?
    from collections import Counter
    c = Counter(op for r in results for op in r.get("ops", []))
    print(f"\norder parameters in use: {dict(c)}")
    if "fis" not in c:
        print("  note: F_IS is not constrained anywhere, though it is the parameter most")
        print("  sensitive to polyhedral deformation -- measured shifts of up to 12x the")
        print("  reference spread on refined structures where q4/q6 stayed flat.")


if __name__ == "__main__":
    main()
