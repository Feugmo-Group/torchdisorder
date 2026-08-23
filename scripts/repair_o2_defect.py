"""Remove a trapped O2 molecule from an otherwise-good oxide glass.

When to use this -- and when not to
-----------------------------------
Melt-quench of GeO2 with MACE-MPA-0 reliably produces a *good network* carrying
a *small* number of O2 molecules: across four replicates at identical settings
the Ge sublattice landed at 1.64-1.81x its noise floor (better than the
published NNP model at 2.38x) while the oxygen count came out 1, 2, 2, 3.  The
oxygen never reached zero.  This script repairs that last defect.

It is a **manual intervention, not a melt-quench result**, and any structure it
produces must be described that way.  Use it only when:

  - the disorder gate already passes (`assess_glass`), and
  - the count of offending molecules is small -- one or two, not seven.

If a run carries many O2, the melt was wrong and the answer is a different melt,
not surgery on the output.

Why the naive fix does not work
-------------------------------
The trapped O2 does not sit in the hole it came from.  In GeO2_mq_hot_slowq_r1
the molecule was bonded to zero Ge, floating in a void, while the single
under-coordinated Ge sat **12.5 A away** -- the network had already relaxed
around the missing pair, absorbing the deficit by over-coordinating ten Ge to
CN = 5.  So simply deleting the molecule changes stoichiometry, and simply
pulling it apart leaves both oxygens with nowhere to bond.

Two interventions, both then FIRE-relaxed, were compared on that structure:

  A  separate the two oxygens in place (3.2 A apart)
  B  additionally park one of them on the open face of the under-coordinated
     central atom, at a bond length

Both dissociate the molecule permanently -- **it does not re-form** -- so the O2
is kinetically trapped rather than energetically preferred, which is the result
that makes this repair legitimate at all.  B is better and is the default: it
also clears the under-coordinated site (CN = 3 count goes 1 -> 0), and scores
4.032 mean CN against the published model's 4.021, with O-Ge-O 109.24 +/- 8.07
deg against 109.22 +/- 8.18.

Usage
-----
    python scripts/repair_o2_defect.py \
        --input  data/crystal-structures/GeO2_mq_hot_slowq_r1_REJECTED.cif \
        --output data/crystal-structures/GeO2_glass_mq_o2repaired.cif \
        --central Ge --ligand O --system GeO2

Exits non-zero if the result does not pass `assess_glass`, so it cannot quietly
hand back a structure that is still broken.
"""

from __future__ import annotations

import argparse

import numpy as np


def _unit(v):
    return v / np.linalg.norm(v)


def find_molecules(atoms, species: str, rmax: float):
    """Index pairs of `species` closer than `rmax` -- i.e. the offending molecules."""
    from ase.neighborlist import neighbor_list

    z = np.array(atoms.get_chemical_symbols())
    i, j, _d = neighbor_list("ijd", atoms, float(rmax))
    m = (z[i] == species) & (z[j] == species) & (i < j)
    return list(zip(i[m].tolist(), j[m].tolist()))


def coordination(atoms, central: str, ligand: str, cutoff: float):
    from ase.neighborlist import neighbor_list

    z = np.array(atoms.get_chemical_symbols())
    i, j, _d = neighbor_list("ijd", atoms, float(cutoff))
    m = (z[i] == central) & (z[j] == ligand)
    return np.bincount(i[m], minlength=len(atoms))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--central", default="Ge")
    p.add_argument("--ligand", default="O")
    p.add_argument("--system", default=None, help="glass ruleset, e.g. GeO2")
    p.add_argument("--molecule-cutoff", type=float, default=1.35,
                   help="ligand-ligand distance that counts as a molecule")
    p.add_argument("--bond-cutoff", type=float, default=2.4)
    p.add_argument("--bond-length", type=float, default=1.78,
                   help="where to park a rescued ligand on an open face")
    p.add_argument("--max-molecules", type=int, default=2,
                   help="refuse to operate on a badly damaged structure")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--model", default="medium-mpa-0")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from ase.io import read, write
    from ase.optimize import FIRE
    from mace.calculators import mace_mp

    from torchdisorder.common.glass_quality import assess_glass

    atoms = read(args.input)
    pairs = find_molecules(atoms, args.ligand, args.molecule_cutoff)
    print(f"{args.input}: {len(atoms)} atoms, {len(pairs)} "
          f"{args.ligand}2 molecule(s)")
    if not pairs:
        raise SystemExit("nothing to repair")
    if len(pairs) > args.max_molecules:
        raise SystemExit(
            f"{len(pairs)} molecules is too many to call this a defect -- the "
            "melt is wrong, and surgery on the output would hide that. "
            "Re-run the melt instead, or raise --max-molecules deliberately.")

    for a_idx, b_idx in pairs:
        cn = coordination(atoms, args.central, args.ligand, args.bond_cutoff)
        z = np.array(atoms.get_chemical_symbols())
        centres = np.flatnonzero(z == args.central)
        under = centres[cn[centres] < cn[centres].max()]
        pos = atoms.get_positions()

        if len(under):
            # Park the ligand on the most open face of the nearest
            # under-coordinated centre: the site the network actually wants
            # filled.  Bare separation leaves it with nothing to bond to.
            target = min(under, key=lambda g: atoms.get_distance(g, a_idx, mic=True))
            from ase.neighborlist import neighbor_list

            i, j, _ = neighbor_list("ijd", atoms, args.bond_cutoff)
            nbrs = j[(i == target) & (z[j] == args.ligand)]
            if len(nbrs):
                open_dir = -np.sum(
                    [_unit(atoms.get_distance(target, n, vector=True, mic=True))
                     for n in nbrs], axis=0)
                pos[b_idx] = pos[target] + _unit(open_dir) * args.bond_length
                print(f"  moved {args.ligand}{b_idx} onto {args.central}{target} "
                      f"(CN {cn[target]})")
        else:
            v = _unit(atoms.get_distance(a_idx, b_idx, vector=True, mic=True))
            pos[b_idx] = pos[a_idx] + v * 3.2
            print(f"  separated {args.ligand}{a_idx}-{args.ligand}{b_idx} in place")
        atoms.set_positions(pos)

    atoms.calc = mace_mp(model=args.model, default_dtype="float64",
                         device=args.device)
    FIRE(atoms, logfile=None).run(fmax=args.fmax, steps=args.steps)

    left = find_molecules(atoms, args.ligand, args.molecule_cutoff)
    print(f"after relaxation: {len(left)} {args.ligand}2 molecule(s)")

    report = assess_glass(atoms, args.system) if args.system else None
    if report is not None:
        print(report.summary())
        if not report:
            raise SystemExit("repair did not produce a glass; not written")

    write(args.output, atoms, format="cif")
    print(f"\nwrote {args.output}")
    print("This structure was REPAIRED BY HAND after melt-quench. Report it as "
          "such -- it is not the unmodified output of a simulation.")


if __name__ == "__main__":
    main()
